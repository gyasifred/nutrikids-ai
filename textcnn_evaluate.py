#!/usr/bin/env python3

from collections import Counter
import json
import numpy as np
import pandas as pd
import torch
import os
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_curve, auc,
    classification_report
)
from models.text_cnn import predict_batch, load_model_artifacts
from models.text_cnn import train_textcnn, TextTokenizer


def process_labels(labels, label_encoder=None):
    """
    Process labels that could be either numeric or text.
    If label_encoder is provided, use it to transform labels.
    If not, handle numeric labels directly.

    Args:
        labels: List of labels that could be numeric or text
        label_encoder: Optional LabelEncoder to use

    Returns:
        Tuple of (processed_labels, label_encoder or None, unique_classes)
    """
    # Check if all labels are numeric (integers or floats)
    is_numeric = all(isinstance(label, (int, float, np.integer, np.floating)) or
                     (isinstance(label, str) and label.strip().isdigit())
                     for label in labels)

    if is_numeric:
        # Convert string numbers to integers if needed
        numeric_labels = [int(float(label)) if isinstance(label, str) else int(label)
                          for label in labels]
        # Get unique classes in sorted order
        unique_classes = sorted(set(numeric_labels))
        # Create a mapping for class indices
        label_map = {val: idx for idx, val in enumerate(unique_classes)}
        # Map original labels to indices
        y_true = np.array([label_map[label] for label in numeric_labels])
        return y_true, None, np.array(unique_classes)
    else:
        # Handle text labels
        if label_encoder is None:
            label_encoder = LabelEncoder()
            y_true = label_encoder.fit_transform(labels)
        else:
            # Use existing label encoder
            y_true = label_encoder.transform(labels)

        return y_true, label_encoder, label_encoder.classes_


def generate_confusion_matrix(y_true, y_pred, classes, output_dir):
    """Generate and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)

    # Normalize confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Convert class labels to strings for display
    str_classes = [str(c) for c in classes]

    # Create non-normalized confusion matrix plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=str_classes, yticklabels=str_classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()

    # Create normalized confusion matrix plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=str_classes, yticklabels=str_classes)
    plt.title('Normalized Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix_normalized.png'))
    plt.close()

    # Save confusion matrix data
    cm_data = {
        'confusion_matrix': cm.tolist(),
        'confusion_matrix_normalized': cm_normalized.tolist(),
        'class_labels': [str(c) for c in classes]
    }

    with open(os.path.join(output_dir, 'confusion_matrix.json'), 'w') as f:
        json.dump(cm_data, f, indent=2)
        
    # Save confusion matrix to CSV
    cm_df = pd.DataFrame(cm, index=str_classes, columns=str_classes)
    cm_df.to_csv(os.path.join(output_dir, 'confusion_matrix.csv'))
    
    cm_norm_df = pd.DataFrame(cm_normalized, index=str_classes, columns=str_classes)
    cm_norm_df.to_csv(os.path.join(output_dir, 'confusion_matrix_normalized.csv'))


def generate_roc_curve(y_true, y_proba, classes, output_dir):
    """Generate and save ROC curve and AUC score."""
    plt.figure(figsize=(10, 8))

    roc_data = {}
    str_classes = [str(c) for c in classes]
    
    # DataFrame for storing all ROC data
    roc_df_data = []

    if len(classes) == 2:
        # Binary classification
        fpr, tpr, thresholds = roc_curve(y_true, y_proba[:, 1])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')

        # Save ROC data
        roc_data['binary'] = {
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist(),
            'thresholds': thresholds.tolist(),
            'auc': roc_auc
        }
        
        # Add to DataFrame data
        for i in range(len(fpr)):
            roc_df_data.append({
                'class': 'binary',
                'fpr': fpr[i],
                'tpr': tpr[i],
                'threshold': thresholds[i] if i < len(thresholds) else np.nan,
                'auc': roc_auc
            })
    else:
        # Multi-class classification (one-vs-rest)
        roc_data['multiclass'] = {}

        for i, class_name in enumerate(str_classes):
            y_true_binary = (np.array(y_true) == i).astype(int)
            fpr, tpr, thresholds = roc_curve(y_true_binary, y_proba[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=2,
                     label=f'Class {class_name} (AUC = {roc_auc:.2f})')

            # Save ROC data
            roc_data['multiclass'][class_name] = {
                'fpr': fpr.tolist(),
                'tpr': tpr.tolist(),
                'thresholds': thresholds.tolist(),
                'auc': roc_auc
            }
            
            # Add to DataFrame data
            for j in range(len(fpr)):
                roc_df_data.append({
                    'class': class_name,
                    'fpr': fpr[j],
                    'tpr': tpr[j],
                    'threshold': thresholds[j] if j < len(thresholds) else np.nan,
                    'auc': roc_auc
                })

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(output_dir, 'roc_curve.png'))
    plt.close()

    # Save ROC data as JSON
    with open(os.path.join(output_dir, 'roc_data.json'), 'w') as f:
        json.dump(roc_data, f, indent=2)
    
    # Save ROC data as CSV
    roc_df = pd.DataFrame(roc_df_data)
    roc_df.to_csv(os.path.join(output_dir, 'roc_curve_data.csv'), index=False)


def generate_precision_recall_curve(y_true, y_proba, classes, output_dir):
    """Generate and save Precision-Recall curve and Average Precision score."""
    plt.figure(figsize=(10, 8))

    pr_data = {}
    str_classes = [str(c) for c in classes]
    
    # DataFrame for storing all PR curve data
    pr_df_data = []

    if len(classes) == 2:
        # Binary classification
        precision, recall, thresholds = precision_recall_curve(
            y_true, y_proba[:, 1])
        avg_precision = average_precision_score(y_true, y_proba[:, 1])
        plt.plot(recall, precision, lw=2,
                 label=f'PR curve (AP = {avg_precision:.2f})')

        # Save PR data
        pr_data['binary'] = {
            'precision': precision.tolist(),
            'recall': recall.tolist(),
            'thresholds': thresholds.tolist(),
            'average_precision': avg_precision
        }
        
        # Add to DataFrame data
        for i in range(len(precision)):
            pr_df_data.append({
                'class': 'binary',
                'precision': precision[i],
                'recall': recall[i],
                'threshold': thresholds[i] if i < len(thresholds) else np.nan,
                'average_precision': avg_precision
            })
    else:
        # Multi-class classification (one-vs-rest)
        pr_data['multiclass'] = {}

        for i, class_name in enumerate(str_classes):
            y_true_binary = (np.array(y_true) == i).astype(int)
            precision, recall, thresholds = precision_recall_curve(
                y_true_binary, y_proba[:, i])
            avg_precision = average_precision_score(
                y_true_binary, y_proba[:, i])
            plt.plot(recall, precision, lw=2,
                     label=f'Class {class_name} (AP = {avg_precision:.2f})')

            # Save PR data
            pr_data['multiclass'][class_name] = {
                'precision': precision.tolist(),
                'recall': recall.tolist(),
                'thresholds': thresholds.tolist(),
                'average_precision': avg_precision
            }
            
            # Add to DataFrame data
            for j in range(len(precision)):
                pr_df_data.append({
                    'class': class_name,
                    'precision': precision[j],
                    'recall': recall[j],
                    'threshold': thresholds[j] if j < len(thresholds) else np.nan,
                    'average_precision': avg_precision
                })

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.savefig(os.path.join(output_dir, 'precision_recall_curve.png'))
    plt.close()

    # Save PR data as JSON
    with open(os.path.join(output_dir, 'precision_recall_data.json'), 'w') as f:
        json.dump(pr_data, f, indent=2)
    
    # Save PR data as CSV
    pr_df = pd.DataFrame(pr_df_data)
    pr_df.to_csv(os.path.join(output_dir, 'precision_recall_curve_data.csv'), index=False)


def generate_class_distribution(y_true, y_pred, classes, output_dir):
    """Generate and save class distribution plots."""
    # Convert classes to strings for plotting and counting
    str_classes = [str(c) for c in classes]

    # Create mapping from index to string class label
    class_mapping = {i: str_classes[i] for i in range(len(str_classes))}

    # Actual label distribution
    plt.figure(figsize=(10, 6))
    # Convert numpy arrays to scalar values if needed
    y_true_list = [i.item() if hasattr(i, 'item') else i for i in y_true]
    y_true_counts = Counter([class_mapping[i] for i in y_true_list])
    sns.barplot(x=list(y_true_counts.keys()), y=list(y_true_counts.values()))
    plt.title('Actual Class Distribution')
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'actual_class_distribution.png'))
    plt.close()

    # Predicted label distribution
    plt.figure(figsize=(10, 6))
    # Convert numpy arrays to scalar values if needed
    y_pred_list = [i.item() if hasattr(i, 'item') else i for i in y_pred]
    y_pred_counts = Counter([class_mapping[i] for i in y_pred_list])
    sns.barplot(x=list(y_pred_counts.keys()), y=list(y_pred_counts.values()))
    plt.title('Predicted Class Distribution')
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'predicted_class_distribution.png'))
    plt.close()

    # Save distribution data
    distribution_data = {
        'actual': {k: v for k, v in sorted(y_true_counts.items())},
        'predicted': {k: v for k, v in sorted(y_pred_counts.items())}
    }

    with open(os.path.join(output_dir, 'class_distribution.json'), 'w') as f:
        json.dump(distribution_data, f, indent=2)
        
    # Save distribution data as CSV
    dist_df = pd.DataFrame({
        'class': list(distribution_data['actual'].keys()),
        'actual_count': list(distribution_data['actual'].values()),
        'predicted_count': [distribution_data['predicted'].get(k, 0) for k in distribution_data['actual'].keys()]
    })
    dist_df.to_csv(os.path.join(output_dir, 'class_distribution.csv'), index=False)


def generate_integrated_gradients(model, tokenizer, texts, output_dir, num_samples=5):
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
    
    # DataFrame to store feature importance data
    importance_data = []

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
        
        # Add to DataFrame data
        for word_idx, (word, importance) in enumerate(zip(words, values)):
            importance_data.append({
                'sample_id': i,
                'text': text,
                'method': 'integrated_gradients',
                'word': word,
                'position': word_idx,
                'importance_score': importance
            })

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

    # Save feature importance data
    with open(os.path.join(output_dir, 'integrated_gradients_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save feature importance to CSV
    importance_df = pd.DataFrame(importance_data)
    importance_df.to_csv(os.path.join(output_dir, 'feature_importance.csv'), index=False)
    
    return results


def generate_permutation_importance(model, tokenizer, texts, output_dir, num_samples=5):
    """
    Generate feature importance by permuting inputs.
    This is a model-agnostic approach that doesn't require gradients.
    """
    results = []
    
    # DataFrame to store feature importance data
    importance_data = []

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
        
        # Add to DataFrame data
        for word_idx, (word, importance) in enumerate(zip(words, values)):
            importance_data.append({
                'sample_id': i,
                'text': text,
                'method': 'permutation',
                'word': word,
                'position': word_idx,
                'importance_score': importance
            })

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

    # Save feature importance data
    with open(os.path.join(output_dir, 'permutation_importance_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save feature importance to CSV
    importance_df = pd.DataFrame(importance_data)
    if os.path.exists(os.path.join(output_dir, 'feature_importance.csv')):
        # Append to existing file
        existing_df = pd.read_csv(os.path.join(output_dir, 'feature_importance.csv'))
        combined_df = pd.concat([existing_df, importance_df])
        combined_df.to_csv(os.path.join(output_dir, 'feature_importance.csv'), index=False)
    else:
        importance_df.to_csv(os.path.join(output_dir, 'feature_importance.csv'), index=False)
    
    return results


def generate_occlusion_importance(model, tokenizer, texts, output_dir, num_samples=5):
    """
    Generate feature importance using occlusion (similar to permutation but with sliding window).
    This is a model-agnostic approach that doesn't require gradients.
    """
    results = []
    
    # DataFrame to store feature importance data
    importance_data = []

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
        
        # Add to DataFrame data
        for word_idx, (word, importance) in enumerate(zip(words, values)):
            importance_data.append({
                'sample_id': i,
                'text': text,
                'method': 'occlusion',
                'word': word,
                'position': word_idx,
                'importance_score': importance
            })

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

    # Save feature importance data
    with open(os.path.join(output_dir, 'occlusion_importance_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save feature importance to CSV
    importance_df = pd.DataFrame(importance_data)
    csv_path = os.path.join(output_dir, 'feature_importance.csv')
    if os.path.exists(csv_path):
        existing_df = pd.read_csv(csv_path)
        combined_df = pd.concat([existing_df, importance_df], ignore_index=True)
        combined_df.to_csv(csv_path, index=False)
    else:
        importance_df.to_csv(csv_path, index=False)

    return results

def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate TextCNN model on test data')
    parser.add_argument('--test_data', type=str, required=True,
                        help='Path to test CSV file')
    parser.add_argument('--text_column', type=str, default='txt',
                        help='Name of the text column in CSV (default: txt)')
    parser.add_argument('--label_column', type=str,
                        default='label',
                        help='Name of the label column in CSV')
    parser.add_argument('--id_column', type=str, default="DEID",
                        help='Name of the column containing IDs')
    parser.add_argument('--model_dir', type=str, default='CNN',
                        help='Directory containing model and artifacts')
    parser.add_argument('--output_dir', type=str, default='CNN/evaluation',
                        help='Directory to save evaluation artifacts')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for prediction (default: 32)')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Threshold for binary classification \
                            (default: 0.5)')
    parser.add_argument('--explanation_method', type=str, default='all',
                        choices=['integrated_gradients', 'permutation', 'occlusion', 'all', 'none'],
                        help='Method for generating feature importance explanations')
    parser.add_argument('--explanation_samples', type=int, default=200,
                        help='Number of samples to use for explanations (default: 200)')
    return parser.parse_args()


def main():
    # Parse command line arguments
    args = parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    print(f"Loading Test Data from {args.test_data}...")
    test_df = pd.read_csv(args.test_data)

    # Check if required columns exist
    if args.text_column not in test_df.columns:
        raise ValueError(
            f"Text column '{args.text_column}' not found in the test file")
    if args.label_column not in test_df.columns:
        raise ValueError(
            f"Label column '{args.label_column}' not found in the test file")

    test_texts = test_df[args.text_column].fillna("").tolist()
    test_labels = test_df[args.label_column].tolist()

    print(f"Test data: {len(test_texts)} examples")

    # Load model and artifacts
    print(f"Loading model and artifacts from {args.model_dir}...")
    model, tokenizer, label_encoder, config = load_model_artifacts(
        args.model_dir)

    # Process labels based on whether they're numeric or not
    # and whether a label_encoder is available
    y_true, label_encoder, unique_classes = process_labels(
        test_labels, label_encoder)

    # Make predictions
    print(f"Making predictions on {len(test_texts)} texts...")
    all_predictions = []
    all_probabilities = []

    for i in range(0, len(test_texts), args.batch_size):
        batch_texts = test_texts[i:i+args.batch_size]
        batch_preds, batch_probs = predict_batch(model, tokenizer, batch_texts, args.threshold)
        if isinstance(batch_preds, torch.Tensor):
            batch_preds = batch_preds.cpu().numpy().tolist()
        if isinstance(batch_probs, torch.Tensor):
            batch_probs = batch_probs.cpu().numpy().tolist()
        all_predictions.extend(batch_preds)
        all_probabilities.extend(batch_probs)

    # Convert to numpy arrays for analysis functions
    y_pred = np.array(all_predictions)
    y_proba = np.array(all_probabilities)

    # Generate classification report
    print("Generating classification report...")
    str_classes = [str(c) for c in unique_classes]
    report = classification_report(
        y_true, y_pred, target_names=str_classes, output_dict=True)

    print("Classification Report:")
    for class_label, metrics in report.items():
        if isinstance(metrics, dict):  # For individual classes
            print(f"Class {class_label}:")
            for metric_name, value in metrics.items():
                print(f"  {metric_name}: {value:.4f}")
        else:  # For accuracy, macro avg, and weighted avg
            print(f"{class_label}: {metrics:.4f}")

    with open(os.path.join(args.output_dir, 'classification_report.json'), 'w') as f:
        json.dump(report, f, indent=4)

    # Generate confusion matrix
    print("Generating confusion matrix...")
    generate_confusion_matrix(y_true, y_pred, unique_classes, args.output_dir)

    # Generate ROC curve
    print("Generating ROC curve...")
    generate_roc_curve(y_true, y_proba, unique_classes, args.output_dir)

    # Generate precision-recall curve
    print("Generating precision-recall curve...")
    generate_precision_recall_curve(y_true, y_proba, unique_classes,
                                    args.output_dir)

    # Generate class distribution
    print("Generating class distribution plots...")
    generate_class_distribution(y_true, y_pred, unique_classes,
                                args.output_dir)

    print("Saving predictions...")
    pred_df = test_df[[args.id_column]].copy()
    pred_df["true_label"] = test_labels

    # Map predictions back to original label format
    if label_encoder is not None:
        pred_df["predicted_label"] = label_encoder.inverse_transform(y_pred)
    else:
        # If no label encoder, map from indices to original class values
        pred_df["predicted_label"] = [unique_classes[idx] for idx in y_pred]

    # Add probability columns for each class
    for i, class_name in enumerate(str_classes):
        pred_df[f'prob_{class_name}'] = y_proba[:, i]

    output_path = os.path.join(args.output_dir, 'predictions.csv')
    pred_df.to_csv(output_path, index=False)
    
    # Generate explanations for model predictions
    if args.explanation_method != 'none':
        print(f"Generating feature importance explanations using {args.explanation_method}...")
        
        if args.explanation_method == 'integrated_gradients' or args.explanation_method == 'all':
            print("Generating integrated gradients explanations...")
            generate_integrated_gradients(model, tokenizer, test_texts, args.output_dir, args.explanation_samples)
            
        if args.explanation_method == 'permutation' or args.explanation_method == 'all':
            print("Generating permutation importance explanations...")
            generate_permutation_importance(model, tokenizer, test_texts, args.output_dir, args.explanation_samples)
            
        if args.explanation_method == 'occlusion' or args.explanation_method == 'all':
            print("Generating occlusion importance explanations...")
            generate_occlusion_importance(model, tokenizer, test_texts, args.output_dir, args.explanation_samples)
    
    # Save evaluation configuration
    config_data = {
        'test_data': args.test_data,
        'model_dir': args.model_dir,
        'text_column': args.text_column,
        'label_column': args.label_column,
        'id_column': args.id_column,
        'threshold': args.threshold,
        'batch_size': args.batch_size,
        'explanation_method': args.explanation_method,
        'evaluation_time': pd.Timestamp.now().isoformat(),
        'num_samples': len(test_texts)
    }
    
    with open(os.path.join(args.output_dir, 'evaluation_config.json'), 'w') as f:
        json.dump(config_data, f, indent=4)

    print(f"Evaluation complete. Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
