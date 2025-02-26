#!/usr/bin/env python3

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import os
import argparse
import json
import matplotlib.pyplot as plt
from models.text_cnn import TextCNN, predict_batch

def parse_args():
    parser = argparse.ArgumentParser(description='Make predictions with TextCNN model')
    parser.add_argument('--input', type=str, required=True, 
                        help='Path to input CSV file or a text string for prediction')
    parser.add_argument('--text-column', type=str, default='Note_Column',
                        help='Name of the text column in CSV (default: Note_Column)')
    parser.add_argument('--model-dir', type=str, default='model_output',
                        help='Directory containing model and artifacts (default: model_output)')
    parser.add_argument('--output', type=str, default='predictions.csv',
                        help='Path to output predictions file (default: predictions.csv)')
    parser.add_argument('--explain', action='store_true',
                        help='Generate basic explanations for predictions')
    parser.add_argument('--explain-method', type=str, choices=['integrated', 'permutation', 'occlusion', 'all'],
                        default='integrated', help='Explanation method to use (default: integrated)')
    parser.add_argument('--explanation-dir', type=str, default='explanations',
                        help='Directory to save explanation visualizations (default: explanations)')
    parser.add_argument('--num-samples', type=int, default=5,
                        help='Number of samples to explain for batch methods (default: 5)')
    return parser.parse_args()

def load_model_artifacts(model_dir):
    model_path = os.path.join(model_dir, "model.pt")
    model_state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    
    tokenizer_path = os.path.join(model_dir, "tokenizer.joblib")
    tokenizer = joblib.load(tokenizer_path)

    label_encoder_path = os.path.join(model_dir, "label_encoder.joblib")
    label_encoder = joblib.load(label_encoder_path)

    try:
        config = joblib.load(os.path.join(model_dir, "best_config.joblib"))
    except:
        config = {
            "embed_dim": 100,
            "num_filters": 100,
            "kernel_sizes": [3, 4, 5],
            "dropout_rate": 0.5,
            "max_vocab_size": 10000
        }
    
    vocab_size = tokenizer.vocab_size_
    
    model = TextCNN(
        vocab_size=vocab_size,
        embed_dim=config.get("embed_dim", 100),
        num_filters=config.get("num_filters", 100),
        kernel_sizes=config.get("kernel_sizes", [3, 4, 5]),
        dropout_rate=config.get("dropout_rate", 0.5)
    )
    
    model.load_state_dict(model_state_dict)
    model.eval()
    
    return model, tokenizer, label_encoder, config

def generate_integrated_gradients(model, tokenizer, texts, output_dir, num_samples=5):
    """
    Generate integrated gradients explanations for text samples.
    This is an alternative to SHAP that works better with integer inputs.
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
    
    # Get word mapping for interpretability
    word_index = tokenizer.word2idx_
    reverse_word_index = {v: k for k, v in word_index.items()}
    
    # Create a baseline of zeros (reference point)
    device = next(model.parameters()).device
    
    for i, text in enumerate(sample_texts):
        # Get sequence for this sample
        sequence = sequences[i]
        seq_length = len([idx for idx in sequence if idx > 0])
        
        # Create input tensor
        input_tensor = torch.tensor([sequence], dtype=torch.long, device=device)
        
        # Create baseline of zeros (represents absence of words)
        baseline = torch.zeros_like(input_tensor)
        
        # Manual implementation of integrated gradients
        steps = 20
        attr_scores = np.zeros(len(sequence))
        
        # Get the original prediction
        with torch.no_grad():
            original_output = model(input_tensor).item()
            
        # For each word position, calculate its importance
        for pos in range(seq_length):
            # Create a modified input where we progressively add this word
            modified_inputs = []
            for step in range(steps + 1):
                alpha = step / steps
                # Create a copy of baseline
                modified = baseline.clone()
                # Only for positions up to current, interpolate between baseline and input
                for j in range(pos + 1):
                    modified[0, j] = int(alpha * input_tensor[0, j])
                modified_inputs.append(modified)
            
            # Stack all steps into one batch
            batch_input = torch.cat(modified_inputs, dim=0)
            
            # Get predictions for all steps
            with torch.no_grad():
                outputs = model(batch_input).squeeze().cpu().numpy()
            
            # Calculate gradient approximation using integral
            # (difference between consecutive outputs)
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
        # Normalize values for better visualization
        norm_values = (values - values.min()) / (values.max() - values.min() + 1e-10)
        
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
        sample_indices = np.random.choice(len(texts), num_samples, replace=False)
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
        input_tensor = torch.tensor([sequence], dtype=torch.long, device=device)
        
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
        words = [reverse_word_index.get(idx, "<PAD>") for idx in sequence[:seq_length] if idx > 0]
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
        plt.savefig(os.path.join(output_dir, f'permutation_importance_{i+1}.png'))
        plt.close()
        
        # Also create a heatmap-style visualization with text
        plt.figure(figsize=(12, 4))
        # Normalize values for better visualization
        norm_values = (values - values.min()) / (values.max() - values.min() + 1e-10)
        
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
        sample_indices = np.random.choice(len(texts), num_samples, replace=False)
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
        input_tensor = torch.tensor([sequence], dtype=torch.long, device=device)
        
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
        words = [reverse_word_index.get(idx, "<PAD>") for idx in sequence[:seq_length] if idx > 0]
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
        plt.savefig(os.path.join(output_dir, f'occlusion_importance_{i+1}.png'))
        plt.close()
        
        # Also create a heatmap-style visualization with text
        plt.figure(figsize=(12, 4))
        # Normalize values for better visualization
        norm_values = (values - values.min()) / (values.max() - values.min() + 1e-10)
        
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

def main():
    args = parse_args()
    print(f"Loading model and artifacts from {args.model_dir}...")
    model, tokenizer, label_encoder, config = load_model_artifacts(args.model_dir)
    
    # Create explanation directory if needed
    if args.explain:
        os.makedirs(args.explanation_dir, exist_ok=True)
    
    if os.path.isfile(args.input):
        print(f"Loading input data from {args.input}...")
        input_df = pd.read_csv(args.input)
        texts = input_df[args.text_column].tolist()
        
        print("Making predictions...")
        predictions, probabilities = predict_batch(model, tokenizer, texts)
        predicted_labels = label_encoder.inverse_transform(predictions)
        print("Predicted labels:", predicted_labels)
        
        output_df = input_df.copy()
        output_df['predicted_label'] = predicted_labels
        for i, class_name in enumerate(label_encoder.classes_):
            output_df[f'prob_{class_name}'] = probabilities[:, i]
        
        if args.explain:
            print("Generating explanations...")
            explanation_results = {}
            
            if args.explain_method in ['integrated', 'all']:
                print("Generating integrated gradients explanations...")
                integrated_dir = os.path.join(args.explanation_dir, 'integrated_gradients')
                os.makedirs(integrated_dir, exist_ok=True)
                integrated_results = generate_integrated_gradients(model, tokenizer, texts, integrated_dir, args.num_samples)
                explanation_results['integrated_gradients'] = integrated_results
                print(f"Integrated gradients explanations saved to {integrated_dir}")
            
            if args.explain_method in ['permutation', 'all']:
                print("Generating permutation importance explanations...")
                permutation_dir = os.path.join(args.explanation_dir, 'permutation')
                os.makedirs(permutation_dir, exist_ok=True)
                permutation_results = generate_permutation_importance(model, tokenizer, texts, permutation_dir, args.num_samples)
                explanation_results['permutation'] = permutation_results
                print(f"Permutation importance explanations saved to {permutation_dir}")
            
            if args.explain_method in ['occlusion', 'all']:
                print("Generating occlusion importance explanations...")
                occlusion_dir = os.path.join(args.explanation_dir, 'occlusion')
                os.makedirs(occlusion_dir, exist_ok=True)
                occlusion_results = generate_occlusion_importance(model, tokenizer, texts, occlusion_dir, args.num_samples)
                explanation_results['occlusion'] = occlusion_results
                print(f"Occlusion importance explanations saved to {occlusion_dir}")
            
            # Save all explanation results to a JSON file
            with open(os.path.join(args.explanation_dir, 'explanation_results.json'), 'w') as f:
                json.dump(explanation_results, f, indent=2)
        
        output_df.to_csv(args.output, index=False)
        print(f"Predictions saved to {args.output}")
        
    else:
        # Single text prediction
        text = args.input
        print(f"Making prediction for: '{text}'")
        prediction, probability = predict_batch(model, tokenizer, [text])
        predicted_label = label_encoder.inverse_transform(prediction)[0]
        print(f"Predicted label: {predicted_label}")
        print(f"Confidence: {probability[0, prediction[0]].item():.4f}")
        
        # Show probabilities for all classes
        print("\nClass probabilities:")
        for i, class_name in enumerate(label_encoder.classes_):
            print(f"{class_name}: {probability[0, i].item():.4f}")
        
        if args.explain:
            explanation_results = {}
            
            if args.explain_method in ['integrated', 'all']:
                print("\nGenerating integrated gradients explanation...")
                integrated_dir = os.path.join(args.explanation_dir, 'integrated_gradients')
                os.makedirs(integrated_dir, exist_ok=True)
                integrated_results = generate_integrated_gradients(model, tokenizer, [text], integrated_dir, 1)
                explanation_results['integrated_gradients'] = integrated_results
                print(f"Integrated gradients explanation saved to {integrated_dir}")
                
                # Print top influential words
                words = integrated_results[0]["words"]
                scores = integrated_results[0]["importance_scores"]
                word_scores = list(zip(words, scores))
                word_scores.sort(key=lambda x: abs(x[1]), reverse=True)
                
                print("\nTop influential words (integrated gradients):")
                for word, score in word_scores[:10]:  # Show top 10
                    direction = "+" if score > 0 else "-"
                    print(f"{direction} {word}: {abs(score):.4f}")
            
            if args.explain_method in ['permutation', 'all']:
                print("\nGenerating permutation importance explanation...")
                permutation_dir = os.path.join(args.explanation_dir, 'permutation')
                os.makedirs(permutation_dir, exist_ok=True)
                permutation_results = generate_permutation_importance(model, tokenizer, [text], permutation_dir, 1)
                explanation_results['permutation'] = permutation_results
                print(f"Permutation importance explanation saved to {permutation_dir}")
                
                # Print top influential words
                if permutation_results:
                    words = permutation_results[0]["words"]
                    scores = permutation_results[0]["importance_scores"]
                    word_scores = list(zip(words, scores))
                    word_scores.sort(key=lambda x: abs(x[1]), reverse=True)
                    
                    print("\nTop influential words (permutation):")
                    for word, score in word_scores[:10]:  # Show top 10
                        print(f"{word}: {score:.4f}")
            
            if args.explain_method in ['occlusion', 'all']:
                print("\nGenerating occlusion importance explanation...")
                occlusion_dir = os.path.join(args.explanation_dir, 'occlusion')
                os.makedirs(occlusion_dir, exist_ok=True)
                occlusion_results = generate_occlusion_importance(model, tokenizer, [text], occlusion_dir, 1)
                explanation_results['occlusion'] = occlusion_results
                print(f"Occlusion importance explanation saved to {occlusion_dir}")
                
                # Print top influential words
                if occlusion_results:
                    words = occlusion_results[0]["words"]
                    scores = occlusion_results[0]["importance_scores"]
                    word_scores = list(zip(words, scores))
                    word_scores.sort(key=lambda x: abs(x[1]), reverse=True)
                    
                    print("\nTop influential words (occlusion):")
                    for word, score in word_scores[:10]:
                        print(f"{word}: {score:.4f}")
            
            # Save all explanation results to a JSON file
            with open(os.path.join(args.explanation_dir, 'explanation_results.json'), 'w') as f:
                json.dump(explanation_results, f, indent=2)

if __name__ == "__main__":
    main()