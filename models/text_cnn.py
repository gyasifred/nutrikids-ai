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
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
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
        
        # Initialize tokens_to_remove to ensure it exists
        self.tokens_to_remove = ["</s>", "_decnum_", "_lgnum_", "_date_", "_time_"]

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

    def _clean_text(self, text: str) -> str:
        """Clean text by removing standard tokens."""
        # Ensure tokens_to_remove exists (will handle loading from older saved versions)
        if not hasattr(self, 'tokens_to_remove'):
            self.tokens_to_remove = ["</s>", "_decnum_", "_lgnum_", "_date_", "_time_"]
            
        for token in self.tokens_to_remove:
            text = text.replace(token, "")
        return text

    def _tokenize(self, text: str) -> List[str]:
        """Split text into tokens after cleaning."""
        # First clean the text
        cleaned_text = self._clean_text(text)
        # Then tokenize
        return cleaned_text.lower().split()

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
                "Tokenizer has not been fitted. Call fit or fit_transform first.")

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
                "Tokenizer has not been fitted. Call fit or fit_transform first.")

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

    def __getstate__(self):
        """Custom state for pickle to ensure all attributes are saved."""
        state = self.__dict__.copy()
        # Ensure tokens_to_remove is included in the state
        if not hasattr(self, 'tokens_to_remove'):
            state['tokens_to_remove'] = ["</s>", "_decnum_", "_lgnum_", "_date_", "_time_"]
        return state

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
        return x


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion,
    optimizer,
    device: str,
    weight_decay: float = 0.0  # Added L2 regularization parameter
) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        optimizer.zero_grad()
        
        outputs = model(batch_x)
        batch_y = batch_y.float().view(-1, 1)
        
        loss = criterion(outputs, batch_y)
        
        # Add L2 regularization manually if not using optimizer's weight_decay
        if weight_decay > 0:
            l2_reg = torch.tensor(0., device=device)
            for param in model.parameters():
                l2_reg += torch.norm(param, p=2)
            loss += weight_decay * l2_reg
            
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        total_loss += loss.item()
        
        # Use appropriate threshold for binary classification
        preds = (torch.sigmoid(outputs) > 0.5).float().cpu().detach().numpy()
        all_preds.extend(preds)
        all_labels.extend(batch_y.cpu().numpy())

    avg_loss = total_loss / len(train_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    return avg_loss, accuracy
    
    
def evaluate_model(
    model: nn.Module,
    val_loader: DataLoader,
    criterion,
    device: str,
    return_metrics: bool = True 
) -> Tuple[float, float, float, float]:
    model.eval()
    all_preds, all_labels = [], []
    all_probs = []  # Store probabilities for AUC calculation
    total_loss = 0.0

    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            batch_y = batch_y.float().view(-1, 1)
            loss = criterion(outputs, batch_y)
            total_loss += loss.item()
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float().cpu().numpy()
            
            all_probs.extend(probs.cpu().numpy())  # Store probabilities for AUC
            all_preds.extend(preds)
            all_labels.extend(batch_y.cpu().numpy())

    avg_loss = total_loss / len(val_loader)
    
    if return_metrics:
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds)
        auc = roc_auc_score(all_labels, all_probs)  # Calculate AUC
        return avg_loss, accuracy, f1, auc
    else:
        return avg_loss, 0.0, 0.0, 0.0


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
    End-to-end training function with optimized training process.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Initialize best model tracking
    best_val_auc = 0  # AUC score is best when highest
    patience_counter = 0
    early_stopping_patience = config.get('early_stopping_patience', 10)

    # Use pos_weight for class imbalance if provided
    pos_weight = config.get('pos_weight', None)
    if pos_weight is not None:
        # Ensure pos_weight is on the correct device
        pos_weight = pos_weight.to(device) if isinstance(pos_weight, torch.Tensor) else torch.tensor(pos_weight, dtype=torch.float).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        criterion = nn.BCEWithLogitsLoss()

    # Tokenizer setup
    tokenizer = TextTokenizer(
        max_vocab_size=config.get("max_vocab_size", 20000),
        min_frequency=config.get("min_frequency", 2),
        max_length=config.get("max_length", None)
    )

    # Transform texts
    X_train_seq = tokenizer.fit_transform(train_texts)
    X_val_seq = tokenizer.transform(val_texts)
    
    # Label processing
    if isinstance(train_labels, np.ndarray) and provided_label_encoder is not None:
        # Labels are already processed
        train_encoded_labels = train_labels
        val_encoded_labels = val_labels
        label_encoder = provided_label_encoder
    elif isinstance(train_labels, np.ndarray) and provided_label_encoder is None:
        # Numeric labels, no encoder needed
        train_encoded_labels = train_labels
        val_encoded_labels = val_labels
        label_encoder = None
    else:
        # Process labels based on type
        is_numeric = all(isinstance(label, (int, float)) or 
                        (isinstance(label, str) and label.strip().isdigit()) 
                        for label in train_labels)
        
        if is_numeric:
            # Convert string numbers to integers if needed
            train_encoded_labels = np.array([int(label) if isinstance(label, str) else int(label) 
                                            for label in train_labels])
            val_encoded_labels = np.array([int(label) if isinstance(label, str) else int(label) 
                                          for label in val_labels])
            label_encoder = None
        else:
            # Use provided encoder or create a new one for text labels
            if provided_label_encoder is not None:
                label_encoder = provided_label_encoder
                train_encoded_labels = label_encoder.transform(train_labels)
                val_encoded_labels = label_encoder.transform(val_labels)
            else:
                # Create and fit a new label encoder
                label_encoder = LabelEncoder()
                train_encoded_labels = label_encoder.fit_transform(train_labels)
                val_encoded_labels = label_encoder.transform(val_labels)
                
                # Ensure 'yes' or 'positive' maps to 1 if present
                positive_terms = ['yes', 'positive', 'true', '1']
                for pos_term in positive_terms:
                    if pos_term in train_labels or pos_term.capitalize() in train_labels:
                        try:
                            pos_idx = next(i for i, label in enumerate(train_labels) 
                                          if str(label).lower() == pos_term)
                            pos_encoded = train_encoded_labels[pos_idx]
                            if pos_encoded != 1:
                                train_encoded_labels = 1 - train_encoded_labels
                                val_encoded_labels = 1 - val_encoded_labels
                            break
                        except StopIteration:
                            continue

    # Create datasets and data loaders
    train_dataset = TextCNNDataset(X_train_seq, train_encoded_labels)
    val_dataset = TextCNNDataset(X_val_seq, val_encoded_labels)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.get("batch_size", 32), 
        shuffle=True,
        pin_memory=True if device == "cuda" else False,
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.get("batch_size", 32),
        pin_memory=True if device == "cuda" else False,
        num_workers=0
    )
    
    # Initialize model
    model = TextCNN(
        vocab_size=tokenizer.vocab_size_,
        embed_dim=config.get("embed_dim", 100),
        num_filters=config.get("num_filters", 100),
        kernel_sizes=config.get("kernel_sizes", [3, 4, 5]),
        dropout_rate=config.get("dropout_rate", 0.5),
    ).to(device)

    # Get weight decay parameter
    weight_decay = config.get("weight_decay", 0.0)

    # Initialize optimizer with weight decay
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=config.get("lr", 0.001),
        weight_decay=weight_decay,
        betas=(0.9, 0.999)
    )
    
    # Learning rate scheduler for better convergence
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min',
        factor=0.5,
        patience=3
    )

    # Tracking metrics
    metrics = {
        "train_loss": [], 
        "train_accuracy": [], 
        "val_loss": [], 
        "val_accuracy": [], 
        "val_f1": [],
        "val_auc": []  # Added AUC metric
    }
    best_model_state = None

    for epoch in range(num_epochs):
        # Train one epoch
        train_loss, train_accuracy = train_one_epoch(
            model, 
            train_loader, 
            criterion, 
            optimizer, 
            device,
            weight_decay
        )
        
        # Evaluate model with AUC included
        val_loss, val_accuracy, val_f1, val_auc = evaluate_model(
            model, 
            val_loader, 
            criterion, 
            device, 
            return_metrics=True
        )

        # Update learning rate based on validation loss
        scheduler.step(val_loss)

        # Track metrics
        metrics["train_loss"].append(train_loss)
        metrics["train_accuracy"].append(train_accuracy)
        metrics["val_loss"].append(val_loss)
        metrics["val_accuracy"].append(val_accuracy)
        metrics["val_f1"].append(val_f1)
        metrics["val_auc"].append(val_auc)  # Add AUC to tracked metrics

        print(
            f"Epoch {epoch+1}/{num_epochs} - "
            f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, "
            f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}, "
            f"Val F1: {val_f1:.4f}, Val AUC: {val_auc:.4f}"  # Print AUC
        )

        # Save best model state based on validation AUC instead of F1
        if val_auc > best_val_auc:  
            best_val_auc = val_auc
            best_model_state = model.state_dict().copy()
            patience_counter = 0
            print(f"New best model with validation AUC: {val_auc:.4f}")
        else:
            patience_counter += 1
            print(f"No improvement for {patience_counter} epochs")
            
        # Early stopping
        if patience_counter >= early_stopping_patience:
            print(f"Early stopping after {epoch+1} epochs")
            break

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Loaded best model with validation AUC: {best_val_auc:.4f}")

    return model, tokenizer, label_encoder, metrics    
    
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
    

import torch
import numpy as np
import pandas as pd
import os

def generate_integrated_gradients(model, tokenizer, texts, output_dir, num_samples=5):
    """
    Generate a word-level importance matrix using integrated gradients.
    Output: CSV where rows = patient samples, columns = words, values = IG scores or 0.
    """
    results = []

    # Sample texts
    if len(texts) > num_samples:
        sample_indices = np.random.choice(len(texts), num_samples, replace=False)
        sample_texts = [texts[i] for i in sample_indices]
    else:
        sample_texts = texts[:num_samples]

    sequences = tokenizer.transform(sample_texts)
    word_index = tokenizer.word2idx_
    reverse_word_index = {v: k for k, v in word_index.items()}

    device = next(model.parameters()).device
    all_words = set()
    ig_matrix = []

    for i, text in enumerate(sample_texts):
        sequence = sequences[i]
        seq_length = len([idx for idx in sequence if idx > 0])
        if seq_length == 0:
            continue

        sequence_array = np.array([sequence], dtype=np.int64)
        input_tensor = torch.tensor(sequence_array, dtype=torch.long, device=device)
        baseline = torch.zeros_like(input_tensor)

        steps = 20
        attr_scores = np.zeros(len(sequence))

        for pos in range(seq_length):
            modified_inputs = []
            for step in range(steps + 1):
                alpha = step / steps
                modified = baseline.clone()
                for j in range(pos + 1):
                    modified[0, j] = int(alpha * input_tensor[0, j])
                modified_inputs.append(modified)

            batch_input = torch.cat(modified_inputs, dim=0)
            with torch.no_grad():
                outputs = model(batch_input).squeeze().cpu().numpy()
            deltas = outputs[1:] - outputs[:-1]
            attr_scores[pos] = np.sum(deltas)

        # Create a dict of word:score
        word_scores = {}
        for idx, score in zip(sequence[:seq_length], attr_scores[:seq_length]):
            word = reverse_word_index.get(idx, "<PAD>")
            word_scores[word] = score
            all_words.add(word)

        ig_matrix.append(word_scores)

    # Create DataFrame: rows = samples, cols = all words, fill missing with 0
    df = pd.DataFrame(ig_matrix).fillna(0)
    df.index.name = "sample_id"
    df = df.sort_index(axis=1)  # optional: sort columns alphabetically

    # Save as CSV
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(os.path.join(output_dir, "integrated_gradients_matrix.csv"))

    return df
    
def generate_permutation_importance(model, tokenizer, texts, output_dir, num_samples=5):
    """
    Generate a word-level importance matrix using permutation importance.
    Output: CSV where rows = samples, columns = words, values = importance scores (or 0).
    """
    results = []

    # Sample texts if there are many
    if len(texts) > num_samples:
        sample_indices = np.random.choice(len(texts), num_samples, replace=False)
        sample_texts = [texts[i] for i in sample_indices]
    else:
        sample_texts = texts[:num_samples]

    # Tokenize samples
    sequences = tokenizer.transform(sample_texts)

    # Word index maps
    word_index = tokenizer.word2idx_
    reverse_word_index = {v: k for k, v in word_index.items()}

    model.eval()
    device = next(model.parameters()).device
    all_words = set()
    perm_matrix = []

    for i, text in enumerate(sample_texts):
        sequence = sequences[i]
        seq_length = len([idx for idx in sequence if idx > 0])
        if seq_length == 0:
            continue

        # Create input tensor
        sequence_array = np.array([sequence], dtype=np.int64)
        input_tensor = torch.tensor(sequence_array, dtype=torch.long, device=device)

        # Original prediction
        with torch.no_grad():
            original_output = model(input_tensor).item()

        importance_scores = np.zeros(seq_length)
        pad_token = 0  # Typically padding token

        for j in range(seq_length):
            modified = input_tensor.clone()
            modified[0, j] = pad_token

            with torch.no_grad():
                modified_output = model(modified).item()

            importance_scores[j] = abs(original_output - modified_output)

        # Store as word:score mapping
        word_scores = {}
        for idx, score in zip(sequence[:seq_length], importance_scores[:seq_length]):
            word = reverse_word_index.get(idx, "<PAD>")
            word_scores[word] = score
            all_words.add(word)

        perm_matrix.append(word_scores)

    # Create word-level importance matrix
    df = pd.DataFrame(perm_matrix).fillna(0)
    df.index.name = "sample_id"
    df = df.sort_index(axis=1)  # Optional: sort columns alphabetically

    # Save to CSV
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(os.path.join(output_dir, "permutation_importance_matrix.csv"))

    return df

def generate_occlusion_importance(model, tokenizer, texts, output_dir, num_samples=5):
    """
    Generate word-level importance matrix using occlusion method.
    Output: CSV where rows = samples, columns = words, values = importance scores (or 0).
    """
    results = []

    # Sample a subset of texts
    if len(texts) > num_samples:
        sample_indices = np.random.choice(len(texts), num_samples, replace=False)
        sample_texts = [texts[i] for i in sample_indices]
    else:
        sample_texts = texts[:num_samples]

    # Tokenize
    sequences = tokenizer.transform(sample_texts)
    word_index = tokenizer.word2idx_
    reverse_word_index = {v: k for k, v in word_index.items()}

    model.eval()
    device = next(model.parameters()).device

    window_size = 1
    all_words = set()
    occlusion_matrix = []

    for i, text in enumerate(sample_texts):
        sequence = sequences[i]
        seq_length = len([idx for idx in sequence if idx > 0])
        if seq_length == 0:
            continue

        sequence_array = np.array([sequence], dtype=np.int64)
        input_tensor = torch.tensor(sequence_array, dtype=torch.long, device=device)

        with torch.no_grad():
            original_output = model(input_tensor).item()

        importance_scores = np.zeros(seq_length)
        pad_token = 0

        for j in range(seq_length - window_size + 1):
            modified = input_tensor.clone()
            modified[0, j:j+window_size] = pad_token

            with torch.no_grad():
                modified_output = model(modified).item()

            delta = abs(original_output - modified_output)

            for k in range(window_size):
                if j + k < seq_length:
                    importance_scores[j + k] += delta / window_size

        # Convert to {word: score} format
        word_scores = {}
        for idx, score in zip(sequence[:seq_length], importance_scores[:seq_length]):
            word = reverse_word_index.get(idx, "<PAD>")
            word_scores[word] = score
            all_words.add(word)

        occlusion_matrix.append(word_scores)

    # Create and save word-importance matrix
    df = pd.DataFrame(occlusion_matrix).fillna(0)
    df.index.name = "sample_id"
    df = df.sort_index(axis=1)  # Optional sorting

    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(os.path.join(output_dir, "occlusion_importance_matrix.csv"))

    return df

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
