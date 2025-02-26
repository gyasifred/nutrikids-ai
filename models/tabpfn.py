#!/usr/bin/env python3
import json
import os
from matplotlib import pyplot as plt
import  numpy as np
import pandas as pd
from datetime import datetime
import joblib
import seaborn as sns
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
import glob


from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve
)
from tabpfn import TabPFNClassifier

def train_tabpfn(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    model_dir: str,
    device: str = "cpu",
    model_name: str = "tabpfn_nutrikid_classifier",
    **tabpfn_kwargs
):
    """
    Train a TabPFN classifier on the provided training data.
    
    Args:
        X_train: Training features DataFrame
        y_train: Training labels
        model_dir: Directory to save the model and artifacts
        device: Device to use (cpu or cuda)
        model_name: Name to use for saved model files
        **tabpfn_kwargs: Additional arguments to pass to TabPFNClassifier
    """
    print(f"Training TabPFN model...")
    
    # Ensure the model directory exists
    os.makedirs(model_dir, exist_ok=True)
    
    # Initialize and train the TabPFN classifier
    tabpfn_model = TabPFNClassifier(
        device=device,
        **tabpfn_kwargs
    )
    tabpfn_model.fit(X_train, y_train)
    
    # Save the model with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(model_dir, f"{model_name}_model_{timestamp}.joblib")
    joblib.dump(tabpfn_model, model_path)
    print(f"Model saved to: {model_path}")
    
    return {'model_file': model_path}


def evaluate_model(
    model,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    feature_names: List[str],
    output_dir: str = "model_output/tabpfn",
    model_name: str = "tabpfn",
    label_encoder = None
) -> Dict[str, Any]:
    """
    Evaluate the TabPFN model on the test set and save evaluation metrics.
    
    Args:
        model: Trained TabPFN model
        X_test: Test features
        y_test: Test labels
        feature_names: Names of the features
        output_dir: Directory to save evaluation results
        model_name: Name of the model for saving artifacts
        label_encoder: LabelEncoder for converting numeric predictions to original labels
    
    Returns:
        Dictionary containing evaluation metrics and paths to saved files
    """
    print("Evaluating model on test set...")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate predictions and probabilities
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    # Extract probability for positive class
    if y_pred_proba.shape[1] > 1:
        pos_proba = y_pred_proba[:, 1]
    else:
        pos_proba = y_pred_proba.ravel()
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, pos_proba)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    # Create a timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save metrics to CSV
    metrics_df = pd.DataFrame({
        'metric': ['accuracy', 'auc', 'precision', 'recall', 'f1'],
        'value': [accuracy, auc, precision, recall, f1]
    })
    metrics_filename = os.path.join(output_dir, f"{model_name}_metrics_{timestamp}.csv")
    metrics_df.to_csv(metrics_filename, index=False)
    print(f"Metrics saved to: {metrics_filename}")
    
    # Plot and save confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    cm_filename = os.path.join(output_dir, f"{model_name}_confusion_matrix_{timestamp}.png")
    plt.savefig(cm_filename)
    plt.close()
    print(f"Confusion matrix saved to: {cm_filename}")
    
    # Plot and save ROC curve
    plt.figure(figsize=(8, 6))
    fpr, tpr, _ = roc_curve(y_test, pos_proba)
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    roc_filename = os.path.join(output_dir, f"{model_name}_roc_curve_{timestamp}.png")
    plt.savefig(roc_filename)
    plt.close()
    print(f"ROC curve saved to: {roc_filename}")
    
    # Feature importance analysis (for interpretability)
    # TabPFN doesn't provide direct feature importances, but we can analyze using a simple permutation approach
    if X_test.shape[1] <= 100:  # Only compute for reasonable number of features
        print("Computing feature importance via permutation...")
        n_repeats = 5
        feature_importance = np.zeros(X_test.shape[1])
        baseline_auc = auc
        
        for i in range(X_test.shape[1]):
            auc_decreases = []
            for _ in range(n_repeats):
                # Create a copy of the test data
                X_test_permuted = X_test.copy()
                
                # Shuffle the values in the current feature
                X_test_permuted.iloc[:, i] = np.random.permutation(X_test_permuted.iloc[:, i])
                
                # Predict with the permuted feature
                permuted_proba = model.predict_proba(X_test_permuted)
                if permuted_proba.shape[1] > 1:
                    permuted_pos_proba = permuted_proba[:, 1]
                else:
                    permuted_pos_proba = permuted_proba.ravel()
                
                # Calculate the decrease in AUC
                permuted_auc = roc_auc_score(y_test, permuted_pos_proba)
                auc_decrease = baseline_auc - permuted_auc
                auc_decreases.append(auc_decrease)
            
            # Average the AUC decreases
            feature_importance[i] = np.mean(auc_decreases)
        
        # Create feature importance DataFrame
        feature_imp_df = pd.DataFrame({
            'feature': feature_names,
            'importance': feature_importance
        })
        feature_imp_df = feature_imp_df.sort_values('importance', ascending=False)
        
        # Save feature importance to CSV
        feature_imp_filename = os.path.join(output_dir, f"{model_name}_feature_importance_{timestamp}.csv")
        feature_imp_df.to_csv(feature_imp_filename, index=False)
        print(f"Feature importance saved to: {feature_imp_filename}")
        
        # Plot top 15 features (or all if less than 15)
        n_top_features = min(15, len(feature_names))
        plt.figure(figsize=(10, 8))
        sns.barplot(x='importance', y='feature', data=feature_imp_df.head(n_top_features))
        plt.title('Feature Importance (Permutation-based)')
        plt.tight_layout()
        feature_imp_plot_filename = os.path.join(output_dir, f"{model_name}_feature_importance_plot_{timestamp}.png")
        plt.savefig(feature_imp_plot_filename)
        plt.close()
        print(f"Feature importance plot saved to: {feature_imp_plot_filename}")
    else:
        feature_imp_df = None
        feature_imp_filename = None
        feature_imp_plot_filename = None
    
    # Generate classification report
    cr = classification_report(y_test, y_pred, output_dict=True)
    cr_df = pd.DataFrame(cr).transpose()
    cr_filename = os.path.join(output_dir, f"{model_name}_classification_report_{timestamp}.csv")
    cr_df.to_csv(cr_filename)
    print(f"Classification report saved to: {cr_filename}")
    
    # If we have a label encoder, create a prediction mapping
    if label_encoder is not None:
        class_mapping = {i: label for i, label in enumerate(label_encoder.classes_)}
        mapping_df = pd.DataFrame(list(class_mapping.items()), columns=['encoded_value', 'original_label'])
        mapping_filename = os.path.join(output_dir, f"{model_name}_class_mapping_{timestamp}.csv")
        mapping_df.to_csv(mapping_filename, index=False)
        print(f"Class mapping saved to: {mapping_filename}")
    
    # Create results dictionary
    results = {
        'accuracy': accuracy,
        'auc': auc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm,
        'feature_importance': feature_imp_df,
        'metrics_file': metrics_filename,
        'confusion_matrix_file': cm_filename,
        'roc_curve_file': roc_filename,
        'classification_report_file': cr_filename,
        'feature_importance_file': feature_imp_filename,
        'feature_importance_plot_file': feature_imp_plot_filename,
        'class_mapping_file': mapping_filename if label_encoder is not None else None
    }
    
    print(f"\nModel Evaluation Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"ROC AUC: {auc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    return results


def load_artifacts(model_dir: str):
    """ 
    Load all model artifacts (model, feature dict, pipeline) from the given directory.

    Args:
        model_dir (str): Path to the directory containing model artifacts.

    Returns:
        model, feature_dict, pipeline
    """
    # Define the file patterns to match the latest .joblib files
    model_pattern = os.path.join(model_dir, "tabpfn_nutrikid_classifier_model_*.joblib")
    label_encoder_pattern = os.path.join(model_dir, "tabpfn_nutrikid_classifier_label_encoder_*.joblib")
    pipeline_pattern = os.path.join(model_dir, "pipeline.joblib")

    # List the files that match the patterns
    model_files = glob.glob(model_pattern)
    label_encoder_files = glob.glob(label_encoder_pattern)
    pipeline_files = glob.glob(pipeline_pattern)

    # Debugging prints to check the found files
    print(f"Found model files: {model_files}")
    print(f"Found Label Encoder files: {label_encoder_files}")
    print(f"Found pipeline files: {pipeline_files}")

    # Ensure that there are files found for each pattern
    if not model_files:
        raise ValueError(f"No model files found matching pattern: {model_pattern}")
    if not label_encoder_files:
        raise ValueError(f"No feature dictionary files found matching pattern: {label_encoder_files}")
    if not pipeline_files:
        raise ValueError(f"No pipeline files found matching pattern: {pipeline_pattern}")

    # Get the latest model file by sorting the files based on the modification time
    model_path = max(model_files, key=os.path.getmtime)
    print(f"Loading model from {model_path}...")
    model = joblib.load(model_path)

    # Get the latest feature dictionary file
    label_encoder_path = max(label_encoder_files, key=os.path.getmtime)
    print(f"Loading label Encoder from {label_encoder_path}...")
    label_encoder = joblib.load(label_encoder_path)

    # Get the latest pipeline file
    pipeline_path = max(pipeline_files, key=os.path.getmtime)
    print(f"Loading pipeline from {pipeline_path}...")
    pipeline = joblib.load(pipeline_path)

    return model, label_encoder, pipeline

from sklearn.inspection import permutation_importance

def get_feature_importance(
    model,
    X: pd.DataFrame,
    y: np.ndarray,
    feature_names: List[str],
    n_top_features: int = 20,
    random_state: int = 42
) -> Tuple[pd.DataFrame, plt.Figure]:
    """
    Calculate and visualize feature importance using permutation importance.
    
    Args:
        model: The trained model
        X: Feature matrix
        y: Target values
        feature_names: Names of features
        n_top_features: Number of top features to display
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple containing:
            - DataFrame with feature importance scores
            - Figure with feature importance visualization
    """
    print("Calculating feature importance...")
    
    # Calculate permutation importance
    try:
        perm_importance = permutation_importance(
            model, X, y, n_repeats=10, random_state=random_state
        )
        
        # Create a DataFrame with feature importance scores
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': perm_importance.importances_mean,
            'Std_Dev': perm_importance.importances_std
        })
        
        # Sort by importance
        importance_df = importance_df.sort_values('Importance', ascending=False)
        
        # Create visualization
        plt.figure(figsize=(10, 8))
        
        # Select top N features
        top_features = importance_df.head(n_top_features)
        
        # Create horizontal bar chart
        ax = sns.barplot(
            x='Importance',
            y='Feature',
            data=top_features,
            palette='viridis'
        )
        
        # Add error bars
        plt.errorbar(
            x=top_features['Importance'],
            y=np.arange(len(top_features)),
            xerr=top_features['Std_Dev'],
            fmt='none',
            ecolor='black',
            capsize=3
        )
        
        plt.title(f'Top {n_top_features} Features by Importance')
        plt.tight_layout()
        
        return importance_df, plt.gcf()
    
    except Exception as e:
        print(f"Warning: Could not calculate feature importance: {str(e)}")
        return pd.DataFrame(), None