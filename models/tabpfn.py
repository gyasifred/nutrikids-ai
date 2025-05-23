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
    model_name: str,
    device: str = "cpu",
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
        ignore_pretraining_limits=True,
        **tabpfn_kwargs
    )
    tabpfn_model.fit(X_train, y_train)
    
    # Save the model with timestamp
    #timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(model_dir, f"{model_name}.joblib")
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
    label_encoder=None
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
    
    # Save raw confusion matrix data to CSV
    cm_df = pd.DataFrame(cm, columns=['Predicted_Negative', 'Predicted_Positive'], 
                         index=['Actual_Negative', 'Actual_Positive'])
    cm_data_filename = os.path.join(output_dir, f"{model_name}_confusion_matrix_data_{timestamp}.csv")
    cm_df.to_csv(cm_data_filename)
    print(f"Confusion matrix data saved to: {cm_data_filename}")
    
    # Calculate and save ROC curve data
    fpr, tpr, thresholds_roc = roc_curve(y_test, pos_proba)
    roc_data = pd.DataFrame({
        'false_positive_rate': fpr,
        'true_positive_rate': tpr,
        'thresholds': np.append(thresholds_roc, [1.0])  # Add 1.0 as the last threshold
    })
    roc_data_filename = os.path.join(output_dir, f"{model_name}_roc_curve_data_{timestamp}.csv")
    roc_data.to_csv(roc_data_filename, index=False)
    print(f"ROC curve data saved to: {roc_data_filename}")
    
    # Plot and save ROC curve
    plt.figure(figsize=(8, 6))
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
    
    # Calculate and save Precision-Recall curve data
    precision_curve, recall_curve, thresholds_pr = precision_recall_curve(y_test, pos_proba)
    avg_precision = average_precision_score(y_test, pos_proba)
    pr_data = pd.DataFrame({
        'precision': precision_curve,
        'recall': recall_curve,
        'thresholds': np.append(thresholds_pr, [1.0])  # Add 1.0 as the last threshold
    })
    pr_data_filename = os.path.join(output_dir, f"{model_name}_precision_recall_curve_data_{timestamp}.csv")
    pr_data.to_csv(pr_data_filename, index=False)
    print(f"Precision-Recall curve data saved to: {pr_data_filename}")
    
    # Plot and save Precision-Recall curve
    plt.figure(figsize=(8, 6))
    plt.plot(recall_curve, precision_curve, label=f'PR Curve (AP = {avg_precision:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='lower left')
    pr_filename = os.path.join(output_dir, f"{model_name}_precision_recall_curve_{timestamp}.png")
    plt.savefig(pr_filename)
    plt.close()
    print(f"Precision-Recall curve saved to: {pr_filename}")
    
    # Save calibration curve data (reliability diagram)
    n_bins = 10
    bin_counts, bin_edges = np.histogram(pos_proba, bins=n_bins, range=(0, 1))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Calculate fraction of positives in each bin
    bin_positives = np.zeros(n_bins)
    for i in range(n_bins):
        bin_mask = (pos_proba >= bin_edges[i]) & (pos_proba < bin_edges[i+1])
        if np.sum(bin_mask) > 0:
            bin_positives[i] = np.mean(y_test[bin_mask])
        else:
            bin_positives[i] = np.nan
    
    calibration_data = pd.DataFrame({
        'bin_centers': bin_centers,
        'fraction_of_positives': bin_positives,
        'bin_counts': bin_counts
    })
    calibration_data_filename = os.path.join(output_dir, f"{model_name}_calibration_curve_data_{timestamp}.csv")
    calibration_data.to_csv(calibration_data_filename, index=False)
    print(f"Calibration curve data saved to: {calibration_data_filename}")
    
    # Plot and save calibration curve
    plt.figure(figsize=(8, 6))
    plt.plot(bin_centers, bin_positives, marker='o', label='Calibration Curve')
    plt.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives')
    plt.title('Calibration Curve (Reliability Diagram)')
    plt.legend(loc='lower right')
    calibration_filename = os.path.join(output_dir, f"{model_name}_calibration_curve_{timestamp}.png")
    plt.savefig(calibration_filename)
    plt.close()
    print(f"Calibration curve saved to: {calibration_filename}")
    
    # Save raw prediction data
    prediction_data = pd.DataFrame({
        'actual': y_test,
        'predicted': y_pred,
        'probability': pos_proba
    })
    predictions_filename = os.path.join(output_dir, f"{model_name}_predictions_{timestamp}.csv")
    prediction_data.to_csv(predictions_filename, index=False)
    print(f"Raw predictions saved to: {predictions_filename}")
    
    # Feature importance analysis (for interpretability)
    # TabPFN doesn't provide direct feature importances,
    # but we can analyze using a simple permutation approach
    if X_test.shape[1] <= 100:
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
        
        # Plot top 20 features (or all if less than 20)
        n_top_features = min(20, len(feature_names))
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
        'confusion_matrix_data_file': cm_data_filename,
        'roc_curve_file': roc_filename,
        'roc_curve_data_file': roc_data_filename,
        'precision_recall_curve_file': pr_filename,
        'precision_recall_curve_data_file': pr_data_filename,
        'calibration_curve_file': calibration_filename,
        'calibration_curve_data_file': calibration_data_filename,
        'predictions_file': predictions_filename,
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

from sklearn.inspection import permutation_importance


from sklearn.inspection import permutation_importance
from typing import List, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def get_feature_importance(
    model,
    X: pd.DataFrame,
    y: np.ndarray,
    feature_names: List[str],
    n_top_features: int = 20,
    random_state: int = 42,
    output_dir: str = None,
    model_name: str = 'model'
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
        output_dir: Directory to save CSV output (optional)
        model_name: Name of the model for file naming (optional)
        
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
        
        # Save to CSV if output directory is provided
        if output_dir:
            import os
            os.makedirs(output_dir, exist_ok=True)
            csv_path = os.path.join(output_dir, f'{model_name}_feature_importance.csv')
            importance_df.to_csv(csv_path, index=False)
            print(f"Feature importance saved to {csv_path}")
            
            # Save raw permutation importance data with all repeats
            raw_importance_data = []
            for feature_idx, feature_name in enumerate(feature_names):
                for repeat_idx in range(perm_importance.importances.shape[1]):
                    raw_importance_data.append({
                        'Feature': feature_name,
                        'Repeat': repeat_idx,
                        'Importance': perm_importance.importances[feature_idx, repeat_idx]
                    })
            
            raw_importance_df = pd.DataFrame(raw_importance_data)
            raw_csv_path = os.path.join(output_dir, f'{model_name}_feature_importance_raw.csv')
            raw_importance_df.to_csv(raw_csv_path, index=False)
            print(f"Raw feature importance data saved to {raw_csv_path}")
        
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
        
        # Save figure if output directory is provided
        if output_dir:
            import os
            fig_path = os.path.join(output_dir, f'{model_name}_feature_importance.png')
            plt.savefig(fig_path, bbox_inches='tight', dpi=300)
            print(f"Feature importance plot saved to {fig_path}")
        
        return importance_df, plt.gcf()
    except Exception as e:
        print(f"Warning: Could not calculate feature importance: {str(e)}")
        return pd.DataFrame(), None
