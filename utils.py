#!/usr/bin/env python3

from datetime import datetime
import glob
import logging
import re
import os
from typing import  List, Union, Tuple, Optional
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
import joblib
from sklearn.metrics import (
    average_precision_score, precision_recall_curve, roc_curve, auc,
    confusion_matrix
)
import shap
from dotenv import load_dotenv, find_dotenv
import xgboost as xgb
from scipy.sparse import csr_matrix

# Download NLTK stopwords (if not already downloaded)
nltk.download('stopwords', quiet=True)


############################################################
# Text Preprocessing Classes
############################################################

class ClinicalTextPreprocessor(BaseEstimator, TransformerMixin):
    """Custom transformer for preprocessing clinical text."""

    def __init__(self,
                 remove_punctuation: bool = True,
                 lowercase: bool = True,
                 standardize_numbers: bool = False,
                 standardize_dates: bool = True):
        self.remove_punctuation = remove_punctuation
        self.lowercase = lowercase
        self.standardize_numbers = standardize_numbers
        self.standardize_dates = standardize_dates
        self.date_pattern = (r'\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b|'
                             r'\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]* \d{1,2},? \d{4}\b')
        self.number_pattern = r'\b\d+(?:,\d{3})*(?:\.\d+)?\b'

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, pd.Series):
            X = X.tolist()
        return [self._preprocess_text(str(text)) for text in X]

    def _preprocess_text(self, text: str) -> str:
        if self.lowercase:
            text = text.lower()
        if self.standardize_dates:
            text = re.sub(self.date_pattern, '', text, flags=re.IGNORECASE)
        if self.standardize_numbers:
            text = re.sub(self.number_pattern, '', text)
        if self.remove_punctuation:
            text = re.sub(r'[^\w\s<>]', '', text)
        return ' '.join(text.split())


class StopWordsRemover(BaseEstimator, TransformerMixin):
    """Remove stop words using NLTK's stop words list."""

    def __init__(self, language: str = 'english'):
        self.language = language
        self.stop_words = set(stopwords.words(language))

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        result = []
        for text in X:
            if not isinstance(text, str):
                text = str(text)
            tokens = text.split()
            tokens = [
                token for token in tokens if token not in self.stop_words
            ]
            result.append(" ".join(tokens))
        return result


class TextStemmer(BaseEstimator, TransformerMixin):
    """Apply Porter Stemming to reduce words to their root form."""

    def __init__(self):
        self.stemmer = PorterStemmer()

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        result = []
        for text in X:
            if not isinstance(text, str):
                text = str(text)
            tokens = text.split()
            tokens = [self.stemmer.stem(token) for token in tokens]
            result.append(" ".join(tokens))
        return result


############################################################
# Text Preprocessing Pipeline
############################################################

def process_csv(
    file_path: str,
    text_column: str,
    label_column: str,
    id_column: str,
    model_name: str,
    max_features: int = 8000,
    remove_stop_words: bool = True,
    apply_stemming: bool = False,
    vectorization_mode: str = 'tfidf',
    ngram_range: tuple = (1, 1),
    save_path: str = '.',
):
    """
    Process the CSV file containing clinical notes into 
    features with n-gram support.
    Parameters:
      - file_path: Path to the CSV file containing the data
      - text_column: Name of the column containing text to analyze
      - label_column: Name of the column containing labels
      - id_column: Name of the column containing unique identifiers
      - max_features: Maximum number of features to extract
      - remove_stop_words: Whether to remove stop words
      - apply_stemming: Whether to apply stemming
      - vectorization_mode: 'count' for CountVectorizer,
        'tfidf' for TF-IDF Vectorizer
      - ngram_range: Tuple (min_n, max_n) for n-gram range
      - save_path: Directory path to save the text preprocessing pipeline
    Returns:
      - X_df: DataFrame with extracted features
      - complete_df: DataFrame with features and label
      - y: Series with labels
      - pipeline: Fitted preprocessing pipeline
      - feature_dict: Dictionary mapping feature names to indices
    """
    try:
        df = pd.read_csv(file_path, usecols=[text_column, label_column, id_column])
        required_columns = [text_column, label_column, id_column]
        missing_columns = [
            col for col in required_columns if col not in df.columns
        ]
        if missing_columns:
            print(f"Available columns: {list(df.columns)}")
            raise ValueError(f"Missing required columns: {missing_columns}")

        # Validate ngram_range input
        if not isinstance(ngram_range,
                          tuple) or len(ngram_range) != 2\
                or ngram_range[0] > ngram_range[1]:
            raise ValueError(f"Invalid ngram_range:\
                 {ngram_range}. Must be a tuple (min_n, max_n)\
                      where min_n <= max_n.")

        # Build preprocessing steps
        preprocessing_steps = []
        preprocessing_steps.append(('preprocessor',
                                    ClinicalTextPreprocessor()))
        if remove_stop_words:
            preprocessing_steps.append(('stopword_remover',
                                        StopWordsRemover()))
        if apply_stemming:
            preprocessing_steps.append(('stemmer', TextStemmer()))

        # Process labels intelligently - detect if already numeric or text
        # First, save the original labels
        y_original = df[label_column]
        
        # Determine if we need a label encoder
        unique_values = y_original.unique()
        print(f"Unique label values found: {unique_values}")
        
        # Check if values are already binary (0/1) or need encoding
        if set(unique_values) <= {0, 1, '0', '1'}:
            # Already binary, just convert to int
            print("Labels are already binary (0/1), no encoding needed")
            y = y_original.astype(int)
            label_encoder = None
        else:
            # Need to encode non-binary labels (like yes/no)
            print(f"Encoding non-binary labels: {unique_values}")
            label_encoder = LabelEncoder()
            
            # Make sure all values are strings for consistent encoding
            y_str = y_original.astype(str).str.lower().str.strip()
            
            label_encoder.fit(y_str)
            y = label_encoder.transform(y_str)
            
            # Check that we have exactly 2 classes for binary classification
            if len(label_encoder.classes_) != 2:
                print(f"Warning: Found {len(label_encoder.classes_)} classes, expected 2 for binary classification")
                print(f"Classes: {label_encoder.classes_}")
            else:
                print(f"Successfully encoded to binary: {label_encoder.classes_} -> {np.unique(y)}")
                
            # Save the label encoder
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            encoder_filename = os.path.join(
                save_path,
                f'{model_name}_nutrikidai_classifier_label_encoder_{timestamp}.joblib')
            joblib.dump(label_encoder, encoder_filename)
            print(f"Label encoder saved to '{encoder_filename}'. Classes: {label_encoder.classes_}")
        
        # Ensure the save_path directory exists
        os.makedirs(save_path, exist_ok=True)
        
        # Configure the appropriate vectorizer based on mode
        if vectorization_mode == 'count':
            vectorizer = CountVectorizer(max_features=max_features,
                                         ngram_range=ngram_range)
            pipeline_filename = os.path.join(
                save_path, f'{model_name}_nutrikidai_pipeline.joblib')
        elif vectorization_mode == 'tfidf':
            vectorizer = TfidfVectorizer(max_features=max_features,
                                         ngram_range=ngram_range)
            pipeline_filename = os.path.join(
                save_path, f'{model_name}_nutrikidai_pipeline.joblib')
        else:
            raise ValueError("Invalid vectorization_mode."
                             "Choose from 'count' or 'tfidf'.")
        
        # Add vectorizer to pipeline
        preprocessing_steps.append(('vectorizer', vectorizer))
        pipeline = Pipeline(preprocessing_steps)
        
        # Fit and transform with the chosen vectorizer
        matrix = pipeline.fit_transform(df[text_column])
        feature_names = pipeline.named_steps['vectorizer'].\
            get_feature_names_out()
        X_df = pd.DataFrame(matrix.toarray(),
                            columns=feature_names, index=df.index)
        
        # Create complete DataFrame with features and label
        complete_df = pd.concat([X_df,
                                pd.DataFrame({label_column: y})], axis=1)
        
        # Save the pipeline
        joblib.dump(pipeline, pipeline_filename)
        print(
            f"{vectorization_mode.capitalize()} vectorizer pipeline with n-grams {ngram_range} saved to '{pipeline_filename}'.")
        
        # Create feature dictionary
        feature_dict = {name: idx for idx, name in enumerate(feature_names)}
        
        return X_df, complete_df, y, pipeline, feature_dict, label_encoder

    except Exception as e:
        print(f"Error processing CSV file: {str(e)}")
        raise

# =========================
# Label Encoding Function
# =========================


def process_labels(labels: List[Union[str, int, float]]) -> Tuple[np.ndarray, Optional[LabelEncoder]]:
    """
    Process labels that could be numeric categories or text categories.
    
    Args:
        labels: List of labels that could be numeric or text
    
    Returns:
        Tuple of (processed_labels, label_encoder or None)
    """
    # Convert labels to a consistent type for processing
    labels = [str(label).strip() for label in labels]
    
    # Check if labels are already numeric
    def is_numeric_category(label_list):
        return all(label.isdigit() for label in label_list)
    
    # If all labels are digit strings, convert to integers
    if is_numeric_category(labels):
        numeric_labels = [int(label) for label in labels]
        return np.array(numeric_labels), None
    
    # If labels are text/mixed, use LabelEncoder
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    
    # Attempt to ensure 'positive' labels map to 1
    positive_terms = ['yes', 'positive', 'true', '1']
    for pos_term in positive_terms:
        try:
            pos_idx = next(i for i, label in enumerate(labels) 
                           if str(label).lower() == pos_term)
            pos_encoded = encoded_labels[pos_idx]
            if pos_encoded != 1:
                encoded_labels = 1 - encoded_labels  # Flip the encoding
            break
        except StopIteration:
            continue
    
    return encoded_labels, label_encoder


def plot_confusion_matrix(y_true, y_pred, output_path):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    return cm


def plot_roc_curve(y_true, y_pred_proba, output_path):
    """Plot and save ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    return fpr, tpr, roc_auc


def plot_precision_recall_curve(y_true, y_pred_proba, output_path):
    """
    Plots the precision-recall curve.

    Args:
        precision: Precision values
        recall: Recall values
        avg_precision: Average precision
        output_path: Path to save the plot
    """
    precision_curve, recall_curve, _ = precision_recall_curve(
                y_true, y_pred_proba)
    avg_precision = average_precision_score(y_true, y_pred_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(recall_curve, precision_curve, label=f'AP = {avg_precision:.3f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='upper right')
    plt.savefig(output_path)
    plt.close()
    return precision_curve, recall_curve, avg_precision


def plot_feature_importance(feature_names, importance, top_n, output_path):
    """
    Plots feature importance.

    Args:
        feature_names: Names of features
        importance: Importance values
        top_n: Number of top features to plot
        output_path: Path to save the plot
    """
    if len(feature_names) == 0 or len(importance) == 0:
        logging.warning(
            "No feature names or importance values provided.\
                Cannot plot feature importance.")
        return

    # Convert to numpy arrays for easier handling
    feature_names = np.array(feature_names)
    importance = np.array(importance)

    # If there are too many features, limit to top N
    if len(feature_names) > top_n:
        # Get indices of top N features by importance
        indices = np.argsort(importance)[-top_n:]
        feature_names = feature_names[indices]
        importance = importance[indices]

    plt.figure(figsize=(10, 8))
    plt.barh(range(len(feature_names)), importance, align='center')
    plt.yticks(range(len(feature_names)), feature_names)
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title(f'Top {len(feature_names)} Feature Importance')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def create_shap_plots(model, X, feature_names, output_dir, num_samples=100):
    """Create and save SHAP summary and dependency plots."""
    # Limit to a subset for computational efficiency if X is large
    if X.shape[0] > num_samples:
        X_sample = X.sample(num_samples, random_state=42)
    else:
        X_sample = X
    # Create explainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    # Summary plot
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_sample,
                      feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir,
                             'shap_summary.png'), dpi=300, bbox_inches='tight')
    plt.close()
    # Dependency plots for top features
    shap_sum = np.abs(shap_values).mean(axis=0)
    top_indices = np.argsort(-shap_sum)[:5]  # Get top 5 features
    for i in top_indices:
        plt.figure(figsize=(12, 8))
        shap.dependence_plot(i, shap_values,
                             X_sample, feature_names=feature_names, show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir,
                                 f'shap_dependence_{feature_names[i]}.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()
    return shap_values, explainer


def get_api_key(env_variable):
    _ = load_dotenv(find_dotenv())
    return os.getenv(env_variable)


def load_tabfnartifacts(model_dir: str, model_name: str):
    """
    Load all model artifacts (model, feature dict, pipeline) from the given directory.
    
    Args:
        model_dir (str): Path to the directory containing model artifacts.
        model_name (str): Base name for model artifacts
    
    Returns:
        Tuple of (model, label_encoder, pipeline)
    """
    # Define the file patterns to match the latest .joblib files
    model_pattern = os.path.join(model_dir, f"{model_name}_nutrikidai_model.joblib")
    label_encoder_pattern = os.path.join(
        model_dir, f"{model_name}_nutrikidai_classifier_label_encoder_*.joblib")
    pipeline_pattern = os.path.join(model_dir, f"{model_name}_nutrikidai_pipeline.joblib")
    
    # List the files that match the patterns
    model_files = glob.glob(model_pattern)
    label_encoder_files = glob.glob(label_encoder_pattern)
    pipeline_files = glob.glob(pipeline_pattern)
    
    # Ensure that there are files found for the model and pipeline
    if not model_files:
        raise ValueError(f"No model files found matching pattern: {model_pattern}")
    if not pipeline_files:
        raise ValueError(f"No pipeline files found matching pattern: {pipeline_pattern}")
    
    # Get the latest model file by sorting based on modification time
    model_path = max(model_files, key=os.path.getmtime)
    print(f"Loading model from {model_path}...")
    model = joblib.load(model_path)
    
    # Get the latest label encoder file if exists
    label_encoder = None
    if label_encoder_files:
        label_encoder_path = max(label_encoder_files, key=os.path.getmtime)
        print(f"Loading Label Encoder from {label_encoder_path}...")
        label_encoder = joblib.load(label_encoder_path)
    else:
        print("No label encoder found. Assuming numeric or binary labels.")
    
    # Get the latest pipeline file
    pipeline_path = max(pipeline_files, key=os.path.getmtime)
    print(f"Loading pipeline from {pipeline_path}...")
    pipeline = joblib.load(pipeline_path)
    
    return model, label_encoder, pipeline


def load_xgbartifacts(model_dir: str, model_name: str):
    """ 
    Load all model artifacts (model, label encoder, pipeline, feature names)
    from the given directory.
    
    Args:
        model_dir (str): Path to the directory containing model artifacts.
        model_name (str): Name of the model.
    
    Returns:
        tuple: (model, label_encoder, pipeline, feature_names)
    """
    # Define the file patterns to match the latest files
    model_pattern = os.path.join(
        model_dir, f"{model_name}_nutrikidai_model.json")
    label_encoder_pattern = os.path.join(
        model_dir, f"{model_name}_nutrikidai_classifier_label_encoder_*.joblib")
    pipeline_pattern = os.path.join(
        model_dir, f"{model_name}_nutrikidai_pipeline.joblib")
    
    # List the files that match the patterns
    model_files = glob.glob(model_pattern)
    label_encoder_files = glob.glob(label_encoder_pattern)
    pipeline_files = glob.glob(pipeline_pattern)
    
    # Logging found files
    logging.info(f"Found model files: {model_files}")
    logging.info(f"Found Label Encoder files: {label_encoder_files}")
    logging.info(f"Found pipeline files: {pipeline_files}")
    
    # Validate file existence
    if not model_files:
        raise ValueError(f"No model files found matching pattern: {model_pattern}")
    if not pipeline_files:
        raise ValueError(f"No pipeline files found matching pattern: {pipeline_pattern}")
    
    # Load XGBoost model
    logging.info(f"Loading model from {model_dir}...")
    model = xgb.XGBClassifier()
    model.load_model(model_files[0])
    
    # Load label encoder if exists
    label_encoder = None
    if label_encoder_files:
        label_encoder_path = max(label_encoder_files, key=os.path.getmtime)
        logging.info(f"Loading label encoder from {label_encoder_path}...")
        label_encoder = joblib.load(label_encoder_path)
    else:
        logging.info("No label encoder found. Assuming binary or numeric labels.")
    
    # Load pipeline
    pipeline_path = max(pipeline_files, key=os.path.getmtime)
    logging.info(f"Loading pipeline from {pipeline_path}...")
    pipeline = joblib.load(pipeline_path)
    
    # Extract feature names
    feature_names = None
    try:
        vectorizer = pipeline.named_steps.get('vectorizer')
        if vectorizer and hasattr(vectorizer, 'get_feature_names_out'):
            feature_names = vectorizer.get_feature_names_out()
            logging.info(f"Extracted {len(feature_names)} feature names from pipeline")
        else:
            logging.warning("Could not find vectorizer or get_feature_names_out method")
    except Exception as e:
        logging.warning(f"Error extracting feature names: {str(e)}")
    
    return model, label_encoder, pipeline, feature_names


def ensure_xgbfeatures_match(X_test, feature_names):
    """
    Ensure the test feature matrix has all the features expected by the model,
    in the correct order.

    Args:
        X_test: Test features DataFrame or sparse matrix
        feature_names: List of expected feature names

    Returns:
        X_test_aligned: A matrix with columns aligned to feature_names
    """
    if feature_names is None:
        logging.warning(
            "No feature names provided. Cannot ensure feature alignment.")
        return X_test

    logging.info(
        f"Ensuring feature alignment (expected {len(feature_names)} features)")

    # Convert to DataFrame if it's a sparse matrix
    if isinstance(X_test, csr_matrix):
        X_test_dense = pd.DataFrame.sparse.from_spmatrix(X_test)
    elif isinstance(X_test, pd.DataFrame):
        X_test_dense = X_test
    else:
        X_test_dense = pd.DataFrame(X_test)

    # Create empty DataFrame with training features
    aligned_df = pd.DataFrame(0, index=X_test_dense.index, columns=feature_names)

    # Fill matching features
    for col in X_test_dense.columns:
        if col in feature_names:
            aligned_df[col] = X_test_dense[col]
        elif isinstance(col, int) and col < len(feature_names):
            aligned_df[feature_names[col]] = X_test_dense[col]

    logging.info(
        f"Feature alignment complete. Matrix shape: {aligned_df.shape}")
    return aligned_df.values
