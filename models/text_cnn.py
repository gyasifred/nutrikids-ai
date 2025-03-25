#!/usr/bin/env python3
import json
import os
from turtle import pd
import joblib
import torch
import numpy as np
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from typing import List, Dict, Union, Optional, Tuple
from collections import Counter
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import accuracy_score
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

class TextTokenizer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        max_vocab_size: int = 20000,
        min_frequency: int = 1,
        pad_token: str = "<PAD>",
        unk_token: str = "<UNK>",
        max_length: Optional[int] = None,
        padding: str = 'post'
    ):
        """
        Text tokenizer with sklearn-like API for CNN models in PyTorch.
        """
        self.max_vocab_size = max_vocab_size
        self.min_frequency = min_frequency
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.max_length = max_length
        self.padding = padding

        # These will be set during fit
        self.word2idx_ = None
        self.idx2word_ = None
        self.vocab_size_ = None

    @classmethod
    def from_args(cls, args):
        """Create a TextTokenizer instance from command-line arguments."""
        return cls(
            max_vocab_size=args.max_vocab_size,
            min_frequency=args.min_frequency,
            pad_token=args.pad_token,
            unk_token=args.unk_token,
            max_length=args.max_length,
            padding=args.padding
        )

    def _tokenize(self, text: str) -> List[str]:
        """Split text into tokens."""
        return text.lower().split()

    def fit(self, X: List[str], y=None):
        """Build vocabulary from list of texts."""
        self.word2idx_ = {self.pad_token: 0, self.unk_token: 1}
        self.idx2word_ = {0: self.pad_token, 1: self.unk_token}

        # Count word frequencies
        word_counts = Counter()
        max_seq_len = 0
        for text in X:
            words = self._tokenize(text)
            word_counts.update(words)
            max_seq_len = max(max_seq_len, len(words))

        if self.max_length is None:
            self.max_length = max_seq_len

        filtered_words = {word for word, count in word_counts.items()
                          if count >= self.min_frequency}

        if self.max_vocab_size > 0:
            vocab_size = min(self.max_vocab_size - 2, len(filtered_words))
            common_words = sorted(filtered_words,
                                  key=lambda x: -word_counts[x])[:vocab_size]
        else:
            common_words = sorted(filtered_words)

        for word in common_words:
            self.word2idx_[word] = len(self.word2idx_)
            self.idx2word_[len(self.idx2word_)] = word

        self.vocab_size_ = len(self.word2idx_)

        return self

    def transform(self, X: List[str]) -> np.ndarray:
        """Convert texts to padded sequences."""
        if self.word2idx_ is None:
            raise ValueError(
                "Tokenizer has not been fitted. Call fit \
                  fit_transform first.")

        # Convert texts to sequences
        sequences = []
        for text in X:
            words = self._tokenize(text)
            seq = [self.word2idx_.get(
                word, self.word2idx_[self.unk_token]) for word in words]
            sequences.append(seq)

        # Pad sequences
        padded_sequences = []
        for seq in sequences:
            if len(seq) > self.max_length:
                padded_seq = seq[:self.max_length]
            else:
                pad_size = self.max_length - len(seq)
                if self.padding == 'post':
                    padded_seq = seq + \
                        [self.word2idx_[self.pad_token]] * pad_size
                else:  # 'pre'
                    padded_seq = [self.word2idx_[
                        self.pad_token]] * pad_size + seq
            padded_sequences.append(padded_seq)

        return np.array(padded_sequences)

    def fit_transform(self, X: List[str], y=None) -> np.ndarray:
        """Fit to data, then transform it."""
        return self.fit(X).transform(X)

    def get_embedding_matrix(
        self,
        embedding_dim: int,
        pretrained_embeddings: Optional[Dict[str, np.ndarray]] = None
    ) -> np.ndarray:
        """Create an embedding matrix for the vocabulary."""
        if self.word2idx_ is None:
            raise ValueError(
                "Tokenizer has not been fitted.\
                      Call fit or fit_transform first.")

        embedding_matrix = np.random.normal(
            scale=0.1, size=(self.vocab_size_, embedding_dim)
        )

        embedding_matrix[0] = np.zeros(embedding_dim)

        if pretrained_embeddings is not None:
            for word, idx in self.word2idx_.items():
                if word in pretrained_embeddings:
                    embedding_matrix[idx] = pretrained_embeddings[word]

        return embedding_matrix

    @classmethod
    def load_pretrained_embeddings(cls, path: str) -> Dict[str, np.ndarray]:
        """Load pretrained word embeddings from a file."""
        embeddings = {}
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                values = line.rstrip().split(' ')
                word = values[0]
                vector = np.asarray(values[1:], dtype='float32')
                embeddings[word] = vector
        return embeddings

# =========================
# TextCNNDataset Class
# =========================


class TextCNNDataset(Dataset):
    def __init__(
        self,
        sequences: np.ndarray,
        labels: Union[List[int], np.ndarray]
    ):
        """Dataset for Text CNN models."""
        self.data = torch.tensor(sequences, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.float)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.data[idx], self.labels[idx]


# =========================
# TextCNN Model Class
# =========================
class TextCNN(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_dim=100,
        num_filters=100,
        kernel_sizes=[3, 4, 5],
        dropout_rate=0.5,
        pretrained_embeddings=None,
        freeze_embeddings=False
    ):
        super(TextCNN, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim)

        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(
                torch.from_numpy(pretrained_embeddings))
            if freeze_embeddings:
                self.embedding.weight.requires_grad = False

        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, num_filters, kernel_size=k)
            for k in kernel_sizes
        ])

        self.pool = nn.AdaptiveMaxPool1d(1)
        
        # Adjusted fully connected layers
        fc_input_size = num_filters * len(kernel_sizes)
        self.fc1 = nn.Linear(fc_input_size, 200)
        self.fc2 = nn.Linear(200, 1)  # Single output neuron
        
        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()

    def forward(self, x):
        """Forward pass with sigmoid activation."""
        x = self.embedding(x)
        x = x.permute(0, 2, 1)
        
        # Convolution and pooling
        x = [self.pool(conv(x)).squeeze(-1) for conv in self.convs]
        x = torch.cat(x, dim=1)
        
        # Fully connected layers
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        
        # Add sigmoid activation here
        return torch.sigmoid(x)

###################
# one epoch
################
def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion,
    optimizer,
    device: str
) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        optimizer.zero_grad()
        
        outputs = model(batch_x)
        batch_y = batch_y.float().view(-1, 1)
        
        # Directly use BCELoss
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        # Predictions based on sigmoid output
        preds = (outputs > 0.5).float().cpu().detach().numpy()
        all_preds.extend(preds)
        all_labels.extend(batch_y.cpu().numpy())

    avg_loss = total_loss / len(train_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    return avg_loss, accuracy
    
def evaluate_model(
    model: nn.Module,
    val_loader: DataLoader,
    criterion,
    device: str
) -> Tuple[float, float]:
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0.0

    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            batch_y = batch_y.float().view(-1, 1)
            loss = criterion(outputs, batch_y)
            total_loss += loss.item()

            # Apply sigmoid for predictions
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float().cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(batch_y.cpu().numpy())

    avg_loss = total_loss / len(val_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    return avg_loss, accuracy

# =========================
# Training and Saving Function
# =========================
def train_textcnn(
    train_texts: List[str],
    train_labels: List[Union[str, int, float]],
    val_texts: List[str],
    val_labels: List[Union[str, int, float]],
    config: Dict,
    num_epochs: int,
    pretrained_embeddings_dict: Optional[Dict[str, np.ndarray]] = None,
    provided_label_encoder: Optional[LabelEncoder] = None
):
    """
    End-to-end training function with flexible label handling and class weighting.
    """
    pos_weight = config.get('pos_weight', None)
    if pos_weight is not None:
        pos_weight = torch.tensor(pos_weight, dtype=torch.float).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        criterion = nn.BCEWithLogitsLoss()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = TextTokenizer(
        max_vocab_size=config.get("max_vocab_size", 20000),
        min_frequency=config.get("min_frequency", 2),
        max_length=config.get("max_length", None)
    )
    
    X_train_seq = tokenizer.fit_transform(train_texts)
    X_val_seq = tokenizer.transform(val_texts)
    train_dataset = TextCNNDataset(X_train_seq, train_labels)
    val_dataset = TextCNNDataset(X_val_seq, val_labels)
    train_loader = DataLoader(train_dataset, batch_size=config.get("batch_size", 32), shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.get("batch_size", 32))
    
    model = TextCNN(
        vocab_size=tokenizer.vocab_size_,
        embed_dim=config.get("embed_dim", 100),
        num_filters=config.get("num_filters", 100),
        kernel_sizes=config.get("kernel_sizes", [3, 4, 5]),
        dropout_rate=config.get("dropout_rate", 0.5),
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=config.get("lr", 0.001))
    
    metrics = {"train_loss": [], "train_accuracy": [], "val_loss": [], "val_accuracy": [], "val_f1": []}
    best_f1_score = 0.0
    best_model_state = None
    
    for epoch in range(num_epochs):
        train_loss, train_accuracy = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_accuracy, val_f1 = evaluate_model(model, val_loader, criterion, device, return_f1=True)
        
        metrics["train_loss"].append(train_loss)
        metrics["train_accuracy"].append(train_accuracy)
        metrics["val_loss"].append(val_loss)
        metrics["val_accuracy"].append(val_accuracy)
        metrics["val_f1"].append(val_f1)
        
        print(
            f"Epoch {epoch+1}/{num_epochs} - "
            f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, "
            f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}, Val F1: {val_f1:.4f}"
        )
        
        # Save best model state based on validation F1-score
        if val_f1 > best_f1_score:
            best_f1_score = val_f1
            best_model_state = model.state_dict().copy()
            print(f"New best model with validation F1-score: {val_f1:.4f}")
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, tokenizer, metrics
    
def predict_batch(model, tokenizer, texts,threshold):
    """
    Make batch predictions using the trained model.

    Args:
        model: Trained TextCNN model
        tokenizer: Fitted tokenizer
        texts: List of texts to predict

    Returns:
        predictions: List of predicted class indices
        probabilities: Array of class probabilities
    """
    # First, determine which device the model is on
    device = next(model.parameters()).device

    # Convert texts to sequences using the custom tokenizer's transform method
    sequences = tokenizer.transform(texts)

    # Convert to tensor and move to the same device as the model
    inputs = torch.tensor(sequences, dtype=torch.long).to(device)

    # Set model to evaluation mode and predict
    model.eval()
    with torch.no_grad():
        outputs = model(inputs)

        # Handle binary classification (single output)
        if outputs.shape[1] == 1:
            probs = torch.cat([1 - outputs, outputs], dim=1)
            preds = (outputs > threshold).long()
        # Handle multi-class classification
        else:
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

    return preds.cpu().numpy().flatten(), probs.cpu().numpy()

def load_model_artifacts(model_dir):
    """
    Load the model, tokenizer, and label encoder (if available) from the model directory.
    Handles cases where label_encoder doesn't exist or labels are already numeric.
    """
    # Load model state dict
    model_path = os.path.join(model_dir, "nutrikidaitextcnn_model.pt")
    model_state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    
    # Load the tokenizer using joblib
    tokenizer_path = os.path.join(model_dir, "tokenizer.joblib")
    tokenizer = joblib.load(tokenizer_path)
    
    # Try to load the label encoder, but don't fail if it doesn't exist
    label_encoder = None
    label_encoder_path = os.path.join(model_dir, "label_encoder.joblib")
    if os.path.exists(label_encoder_path):
        try:
            label_encoder = joblib.load(label_encoder_path)
        except Exception as e:
            print(f"Warning: Failed to load label encoder: {e}")
            print("Continuing without label encoder. Will handle numeric labels directly.")
    
    # Try to load best config from various possible filenames
    config = None
    config_filenames = ["best_config.joblib", "model_config.joblib", "config.joblib"]
    for filename in config_filenames:
        try:
            config_path = os.path.join(model_dir, filename)
            if os.path.exists(config_path):
                config = joblib.load(config_path)
                break
        except Exception:
            continue
    
    if config is None:
        config = {
            "embed_dim": 100,
            "num_filters": 100,
            "kernel_sizes": [3, 4, 5],
            "dropout_rate": 0.5,
            "max_vocab_size": 10000
        }
    
    # Get the number of output classes from the model state dict
    # This is useful when label_encoder doesn't exist
    try:
        # Extract number of classes from the fc2 layer's weights
        for key in model_state_dict:
            if key.endswith("fc2.weight"):
                num_classes = model_state_dict[key].size(0)
                config["num_classes"] = num_classes
                break
    except Exception:
        # Default to 2 classes if we can't determine from model
        config["num_classes"] = 2
    
    vocab_size = tokenizer.vocab_size_
    model = TextCNN(
        vocab_size=vocab_size,
        embed_dim=config.get("embed_dim", 100),
        num_filters=config.get("num_filters", 100),
        kernel_sizes=config.get("kernel_sizes", [3, 4, 5]),
        dropout_rate=config.get("dropout_rate", 0.5)
    )
    
    # Load state dict
    model.load_state_dict(model_state_dict)
    model.eval()
    
    return model, tokenizer, label_encoder, config
    

def generate_integrated_gradients(model, tokenizer,
                                  texts, output_dir,
                                  num_samples=5):
    """
    Generate integrated gradients explanations for text samples.
    This is an alternative to SHAP that works better with integer inputs.
    """
    results = []

    # Sample texts if there are many
    if len(texts) > num_samples:
        sample_indices = np.random.choice(len(texts),
                                          num_samples,
                                          replace=False)
        sample_texts = [texts[i] for i in sample_indices]
    else:
        sample_texts = texts[:num_samples]

    # Tokenize samples
    sequences = tokenizer.transform(sample_texts)

    # Get word mapping for interpretability
    word_index = tokenizer.word2idx_
    reverse_word_index = {v: k for k, v in word_index.items()}

    # Create a baseline of zeros (reference point)
    device = next(model.parameters()).device

    for i, text in enumerate(sample_texts):
        # Get sequence for this sample
        sequence = sequences[i]
        seq_length = len([idx for idx in sequence if idx > 0])

        # Convert sequence list into a NumPy array before making it a tensor
        sequence_array = np.array([sequence], dtype=np.int64)
        input_tensor = torch.tensor(sequence_array,
                                    dtype=torch.long,
                                    device=device)

        # Create baseline of zeros (represents absence of words)
        baseline = torch.zeros_like(input_tensor)

        # Manual implementation of integrated gradients
        steps = 20
        attr_scores = np.zeros(len(sequence))

        # For each word position, calculate its importance
        for pos in range(seq_length):
            # Create a modified input where we progressively add this word
            modified_inputs = []
            for step in range(steps + 1):
                alpha = step / steps
                modified = baseline.clone()
                for j in range(pos + 1):
                    modified[0, j] = int(alpha * input_tensor[0, j])
                modified_inputs.append(modified)

            # Stack all steps into one batch
            batch_input = torch.cat(modified_inputs, dim=0)

            # Get predictions for all steps
            with torch.no_grad():
                outputs = model(batch_input).squeeze().cpu().numpy()

            # Calculate gradient approximation using integral
            deltas = outputs[1:] - outputs[:-1]

            # Score is the sum of these differences
            attr_scores[pos] = np.sum(deltas)

        # Get words for visualization
        words = [reverse_word_index.get(idx, "<PAD>") for idx in sequence[:seq_length] if idx > 0]
        values = attr_scores[:len(words)]

        # Store results
        result = {
            "text": text,
            "words": words,
            "importance_scores": values.tolist()
        }
        results.append(result)

        # Create visualization
        plt.figure(figsize=(12, 6))
        plt.bar(range(len(words)), values)
        plt.xticks(range(len(words)), words, rotation=45)
        plt.title(f'Word Importance for Sample {i+1}')
        plt.xlabel('Words')
        plt.ylabel('Attribution')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'word_importance_{i+1}.png'))
        plt.close()

        # Also create a heatmap-style visualization with text
        plt.figure(figsize=(12, 4))
        norm_values = (values - values.min()) / (values.max() - values.min() + 1e-10)

        for j, (word, val) in enumerate(zip(words, norm_values)):
            plt.text(j, 0.5, word,
                     horizontalalignment='center',
                     verticalalignment='center',
                     fontsize=14 + val * 10,
                     color=plt.cm.RdBu(val))

        plt.xlim(-1, len(words))
        plt.ylim(0, 1)
        plt.axis('off')
        plt.title(f'Word Importance Heatmap for Sample {i+1}')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'word_heatmap_{i+1}.png'))
        plt.close()

    return results


def generate_permutation_importance(model, tokenizer, texts, output_dir, num_samples=5):
    """
    Generate feature importance by permuting inputs.
    This is a model-agnostic approach that doesn't require gradients.
    """
    results = []

    # Sample texts if there are many
    if len(texts) > num_samples:
        sample_indices = np.random.choice(
            len(texts), num_samples, replace=False)
        sample_texts = [texts[i] for i in sample_indices]
    else:
        sample_texts = texts[:num_samples]

    # Tokenize samples
    sequences = tokenizer.transform(sample_texts)

    # Get word mapping for interpretability
    word_index = tokenizer.word2idx_
    reverse_word_index = {v: k for k, v in word_index.items()}

    # Set model to eval mode
    model.eval()
    device = next(model.parameters()).device

    for i, text in enumerate(sample_texts):
        # Get sequence for this sample
        sequence = sequences[i]
        seq_length = len([idx for idx in sequence if idx > 0])

        # Skip if sequence is empty
        if seq_length == 0:
            continue

        # Create input tensor
        sequence_array = np.array([sequence], dtype=np.int64)
        input_tensor = torch.tensor(
            sequence_array, dtype=torch.long, device=device)

        # Get the original prediction
        with torch.no_grad():
            original_output = model(input_tensor).item()

        # Calculate importance of each word by permuting it
        importance_scores = np.zeros(seq_length)

        # For each word, replace it with a padding token and see effect
        pad_token = 0  # Usually the padding token is 0

        for j in range(seq_length):
            # Create modified input with this word removed
            modified = input_tensor.clone()
            modified[0, j] = pad_token

            # Get prediction
            with torch.no_grad():
                modified_output = model(modified).item()

            # Importance is how much the prediction changes
            importance_scores[j] = abs(original_output - modified_output)

        # Get words for visualization
        words = [reverse_word_index.get(idx, "<PAD>")
                 for idx in sequence[:seq_length] if idx > 0]
        values = importance_scores[:len(words)]

        # Store results
        result = {
            "text": text,
            "words": words,
            "importance_scores": values.tolist()
        }
        results.append(result)

        # Create visualization
        plt.figure(figsize=(12, 6))
        plt.bar(range(len(words)), values)
        plt.xticks(range(len(words)), words, rotation=45)
        plt.title(f'Permutation Importance for Sample {i+1}')
        plt.xlabel('Words')
        plt.ylabel('Importance')
        plt.tight_layout()
        plt.savefig(os.path.join(
            output_dir, f'permutation_importance_{i+1}.png'))
        plt.close()

        # Also create a heatmap-style visualization with text
        plt.figure(figsize=(12, 4))
        # Normalize values for better visualization
        norm_values = (values - values.min()) / \
            (values.max() - values.min() + 1e-10)

        # Create a color-mapped visualization
        for j, (word, val) in enumerate(zip(words, norm_values)):
            plt.text(j, 0.5, word,
                     horizontalalignment='center',
                     verticalalignment='center',
                     fontsize=14 + val * 10,  # Size based on importance
                     color=plt.cm.RdBu(val))  # Color based on importance

        plt.xlim(-1, len(words))
        plt.ylim(0, 1)
        plt.axis('off')
        plt.title(f'Word Importance Heatmap for Sample {i+1}')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'permutation_heatmap_{i+1}.png'))
        plt.close()

    return results


def generate_occlusion_importance(model, tokenizer, texts, output_dir, num_samples=5):
    """
    Generate feature importance using occlusion (similar to permutation but with sliding window).
    This is a model-agnostic approach that doesn't require gradients.
    """
    results = []

    # Sample texts if there are many
    if len(texts) > num_samples:
        sample_indices = np.random.choice(
            len(texts), num_samples, replace=False)
        sample_texts = [texts[i] for i in sample_indices]
    else:
        sample_texts = texts[:num_samples]

    # Tokenize samples
    sequences = tokenizer.transform(sample_texts)

    # Get word mapping for interpretability
    word_index = tokenizer.word2idx_
    reverse_word_index = {v: k for k, v in word_index.items()}

    # Set model to eval mode
    model.eval()
    device = next(model.parameters()).device

    # Occlusion window size (1 for single tokens, 2 for pairs, etc.)
    window_size = 1

    for i, text in enumerate(sample_texts):
        # Get sequence for this sample
        sequence = sequences[i]
        seq_length = len([idx for idx in sequence if idx > 0])

        # Skip if sequence is empty
        if seq_length == 0:
            continue

        # Create input tensor
        sequence_array = np.array([sequence], dtype=np.int64)
        input_tensor = torch.tensor(
            sequence_array, dtype=torch.long, device=device)

        # Get the original prediction
        with torch.no_grad():
            original_output = model(input_tensor).item()

        # Calculate importance of each word by occluding it
        importance_scores = np.zeros(seq_length)

        # For each position, create a sliding window and replace with pad tokens
        pad_token = 0  # Usually the padding token is 0

        for j in range(seq_length - window_size + 1):
            # Create modified input with this window occluded
            modified = input_tensor.clone()
            modified[0, j:j+window_size] = pad_token

            # Get prediction
            with torch.no_grad():
                modified_output = model(modified).item()

            # Importance is how much the prediction changes
            delta = abs(original_output - modified_output)

            # For window_size > 1, we attribute the change to each position
            for k in range(window_size):
                if j+k < seq_length:
                    importance_scores[j+k] += delta / window_size

        # Get words for visualization
        words = [reverse_word_index.get(idx, "<PAD>")
                 for idx in sequence[:seq_length] if idx > 0]
        values = importance_scores[:len(words)]

        # Store results
        result = {
            "text": text,
            "words": words,
            "importance_scores": values.tolist()
        }
        results.append(result)

        # Create visualization
        plt.figure(figsize=(12, 6))
        plt.bar(range(len(words)), values)
        plt.xticks(range(len(words)), words, rotation=45)
        plt.title(f'Occlusion Importance for Sample {i+1}')
        plt.xlabel('Words')
        plt.ylabel('Importance')
        plt.tight_layout()
        plt.savefig(os.path.join(
            output_dir, f'occlusion_importance_{i+1}.png'))
        plt.close()

        # Also create a heatmap-style visualization with text
        plt.figure(figsize=(12, 4))
        # Normalize values for better visualization
        norm_values = (values - values.min()) / \
            (values.max() - values.min() + 1e-10)

        # Create a color-mapped visualization
        for j, (word, val) in enumerate(zip(words, norm_values)):
            plt.text(j, 0.5, word,
                     horizontalalignment='center',
                     verticalalignment='center',
                     fontsize=14 + val * 10,
                     color=plt.cm.RdBu(val))

        plt.xlim(-1, len(words))
        plt.ylim(0, 1)
        plt.axis('off')
        plt.title(f'Word Importance Heatmap for Sample {i+1}')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'occlusion_heatmap_{i+1}.png'))
        plt.close()

    return results


def generate_prediction_summary(predictions, probabilities, labels, output_path):
    """Generate a summary of predictions and save as CSV."""
    # Count label frequencies
    label_counts = Counter(labels)

    # Calculate average probabilities per class
    avg_probs = {}
    for i, label in enumerate(np.unique(labels)):
        mask = np.array(labels) == label
        avg_probs[label] = np.mean(np.array(probabilities)[mask, i])

    # Prepare summary data for CSV
    summary_data = {
        'Label': list(label_counts.keys()),
        'Count': list(label_counts.values()),
        'Percentage': [count/len(predictions)*100 for count in label_counts.values()],
        'Average Probability': [avg_probs[label] for label in label_counts.keys()]
    }

    # Convert to DataFrame and save as CSV
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(output_path, index=False)

    # Additional metadata CSV
    metadata_path = os.path.join(os.path.dirname(output_path), 'prediction_summary_metadata.csv')
    metadata_df = pd.DataFrame({
        'Total Predictions': [len(predictions)],
        'Timestamp': [pd.Timestamp.now()]
    })
    metadata_df.to_csv(metadata_path, index=False)

    # Create bar chart of label distribution
    plt.figure(figsize=(10, 6))
    plt.bar(label_counts.keys(), label_counts.values())
    plt.title('Prediction Distribution')
    plt.xlabel('Labels')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(
        output_path), 'prediction_distribution.png'))
    plt.close()

    # Return the summary for potential further use
    return summary_df
