#!/usr/bin/env python3
import os
import torch
import numpy as np
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import List, Dict, Union, Optional, Tuple
from collections import Counter
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import accuracy_score
from torch.utils.data import Dataset
from utils import encode_labels
import matplotlib.pyplot as plt

# =========================
# TextTokenizer Class
# =========================
class TextTokenizer(BaseEstimator, TransformerMixin):
    def __init__(
        self, 
        max_vocab_size: int = 20000, 
        min_frequency: int = 2,
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
        
    def _tokenize(self, text: str) -> List[str]:
        """Split text into tokens."""
        return text.lower().split()
    
    def fit(self, X: List[str], y=None):
        """Build vocabulary from list of texts."""
        # Initialize word mappings
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
            raise ValueError("Tokenizer has not been fitted. Call fit or fit_transform first.")
        
        # Convert texts to sequences
        sequences = []
        for text in X:
            words = self._tokenize(text)
            seq = [self.word2idx_.get(word, self.word2idx_[self.unk_token]) for word in words]
            sequences.append(seq)
        
        # Pad sequences
        padded_sequences = []
        for seq in sequences:
            if len(seq) > self.max_length:
                padded_seq = seq[:self.max_length]
            else:
                pad_size = self.max_length - len(seq)
                if self.padding == 'post':
                    padded_seq = seq + [self.word2idx_[self.pad_token]] * pad_size
                else:  # 'pre'
                    padded_seq = [self.word2idx_[self.pad_token]] * pad_size + seq
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
            raise ValueError("Tokenizer has not been fitted. Call fit or fit_transform first.")
            
        embedding_matrix = np.random.normal(
            scale=0.1, size=(self.vocab_size_, embedding_dim)
        )
        
        embedding_matrix[0] = np.zeros(embedding_dim)
        
        if pretrained_embeddings is not None:
            for word, idx in self.word2idx_.items():
                if word in pretrained_embeddings:
                    embedding_matrix[idx] = pretrained_embeddings[word]
        
        return embedding_matrix

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
        """
        Text CNN model with support for pre-trained embeddings.
        
        Args:
            vocab_size: Size of vocabulary
            embed_dim: Dimension of embeddings
            num_filters: Number of filters for each kernel size
            kernel_sizes: List of kernel sizes for convolutions
            dropout_rate: Dropout rate
            pretrained_embeddings: Pre-trained embedding matrix (numpy array)
            freeze_embeddings: Whether to freeze the embedding layer during training
        """
        super(TextCNN, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_embeddings))

            if freeze_embeddings:
                self.embedding.weight.requires_grad = False
        
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, num_filters, kernel_size=k) for k in kernel_sizes
        ])
        
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc1 = nn.Linear(num_filters * len(kernel_sizes), 200)
        self.fc2 = nn.Linear(200, 1)
        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        """Forward pass."""
        x = self.embedding(x)
        x = x.permute(0, 2, 1)
        x = [self.pool(conv(x)).squeeze(-1) for conv in self.convs]
        x = torch.cat(x, dim=1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

# =========================
# Training and Evaluation Functions
# =========================
def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion,
    optimizer,
    device: str
) -> Tuple[float, float]:
    """
    Train the model for one epoch.

    Args:
        model: The neural network model.
        train_loader: DataLoader for training data.
        criterion: Loss function.
        optimizer: Optimizer.
        device: Device string ('cuda' or 'cpu').

    Returns:
        Tuple of (average training loss, training accuracy) for the epoch.
    """
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []
    
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y.unsqueeze(1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
        # Calculate accuracy
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
    """
    Evaluate the model on the validation set.

    Args:
        model: The neural network model.
        val_loader: DataLoader for validation data.
        criterion: Loss function.
        device: Device string ('cuda' or 'cpu').

    Returns:
        Tuple of (validation loss, validation accuracy).
    """
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0.0
    
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y.unsqueeze(1))
            total_loss += loss.item()
            
            preds = (outputs > 0.5).float().cpu().numpy()
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
    train_labels: List[str],  
    val_texts: List[str],
    val_labels: List[str],
    config: Dict,
    num_epochs: int,
    pretrained_embeddings_dict: Optional[Dict[str, np.ndarray]] = None
):
    """
    End-to-end training function with label encoding.
    
    Args:
        train_texts: Training texts.
        train_labels: Training labels as text ('yes'/'no').
        val_texts: Validation texts.
        val_labels: Validation labels as text ('yes'/'no').
        config: Dictionary with hyperparameters.
        num_epochs: Number of training epochs.
        pretrained_embeddings_dict: Dictionary of pretrained word embeddings.
        
    Returns:
        model: Trained TextCNN model.
        tokenizer: Fitted TextTokenizer.
        label_encoder: Fitted LabelEncoder.
        metrics: Dictionary with training metrics.
    """
    train_encoded_labels, label_encoder = encode_labels(train_labels)
    val_encoded_labels = label_encoder.transform(val_labels)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    tokenizer = TextTokenizer(
        max_vocab_size=config.get("max_vocab_size", 20000),
        min_frequency=config.get("min_frequency", 2),
        max_length=config.get("max_length", None)
    )
    
    X_train_seq = tokenizer.fit_transform(train_texts)
    X_val_seq = tokenizer.transform(val_texts)
    
    train_dataset = TextCNNDataset(X_train_seq, train_encoded_labels)
    val_dataset = TextCNNDataset(X_val_seq, val_encoded_labels)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.get("batch_size", 32), 
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.get("batch_size", 32)
    )
    
    # Prepare pretrained embedding matrix if provided
    pretrained_embedding_matrix = None
    if pretrained_embeddings_dict is not None:
        pretrained_embedding_matrix = tokenizer.get_embedding_matrix(
            config.get("embed_dim", 100), 
            pretrained_embeddings_dict
        )
    
    model = TextCNN(
        vocab_size=tokenizer.vocab_size_,
        embed_dim=config.get("embed_dim", 100),
        num_filters=config.get("num_filters", 100),
        kernel_sizes=config.get("kernel_sizes", [3, 4, 5]),
        dropout_rate=config.get("dropout_rate", 0.5),
        pretrained_embeddings=pretrained_embedding_matrix,
        freeze_embeddings=config.get("freeze_embeddings", False)
    ).to(device)
    
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.get("lr", 0.001))
    
    metrics = {
        "train_loss": [], 
        "train_accuracy": [], 
        "val_loss": [], 
        "val_accuracy": []
    }
    
    best_val_loss = float('inf')
    best_model_state = None
    
    for epoch in range(num_epochs):
        train_loss, train_accuracy = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_accuracy = evaluate_model(model, val_loader, criterion, device)
        
        metrics["train_loss"].append(train_loss)
        metrics["train_accuracy"].append(train_accuracy)
        metrics["val_loss"].append(val_loss)
        metrics["val_accuracy"].append(val_accuracy)
        
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
        
        # Save best model state in memory based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            print(f"New best model with validation loss: {val_loss:.4f}")
    
    # Load the best model state
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, tokenizer, label_encoder, metrics


def predict_batch(model, tokenizer, texts):
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
    
    # Convert texts to sequences
    sequences = tokenizer.transform(texts)
    
    # Convert to tensor and move to the same device as the model
    inputs = torch.tensor(sequences, dtype=torch.long).to(device)
    
    # Predict
    model.eval()
    with torch.no_grad():
        outputs = model(inputs)
        
        # Handle binary classification (single output)
        if outputs.shape[1] == 1:
            probs = torch.cat([1 - outputs, outputs], dim=1)
            preds = (outputs > 0.5).long()
        # Handle multi-class classification
        else:
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
    
    return preds.cpu().numpy(), probs.cpu().numpy()
