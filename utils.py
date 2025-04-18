#!/usr/bin/env python3

# Standard libraries
import os
import re
import glob
import logging
import warnings
from collections import Counter
from datetime import datetime
from typing import List, Union, Tuple, Optional

# Third-party libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from dotenv import load_dotenv, find_dotenv
from pathlib import Path

# NLP libraries
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from textblob import TextBlob

# Machine learning and data processing
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics import (
    average_precision_score, precision_recall_curve, roc_curve,
    auc, confusion_matrix
)
from scipy import stats
from scipy.sparse import csr_matrix
from statsmodels.stats.proportion import proportion_confint
import xgboost as xgb
import shap
import joblib

# Download NLTK stopwords (if not already downloaded)
nltk.download('stopwords', quiet=True)
nltk.download('punkt_tab')
nltk.download('wordnet')
############################################################
# Text Preprocessing Classes
############################################################

class ClinicalTextPreprocessor(BaseEstimator, TransformerMixin):
    """Custom transformer for preprocessing clinical text."""

    def __init__(self,
                 remove_punctuation: bool = True,
                 lowercase: bool = True,
                 standardize_numbers: bool = False,
                 standardize_dates: bool = True,
                 remove_standard_tokens: bool = True):
        self.remove_punctuation = remove_punctuation
        self.lowercase = lowercase
        self.standardize_numbers = standardize_numbers
        self.standardize_dates = standardize_dates
        self.remove_standard_tokens = remove_standard_tokens
        self.date_pattern = (r'\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b|'
                             r'\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]* \d{1,2},? \d{4}\b')
        self.number_pattern = r'\b\d+(?:,\d{3})*(?:\.\d+)?\b'
        # Standard tokens pattern similar to what's in the R code
        self.standard_tokens_pattern = r'</s>|_decnum_|_lgnum_|_date_|_time_'

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, pd.Series):
            X = X.tolist()
        return [self._preprocess_text(str(text)) for text in X]

    def _preprocess_text(self, text: str) -> str:
        if self.lowercase:
            text = text.lower()
        if self.remove_standard_tokens:
            text = re.sub(self.standard_tokens_pattern, '', text)
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
    remove_standard_tokens: bool = True,  # Added parameter
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
      - remove_standard_tokens: Whether to remove standard tokens like "</s>", "_decnum_", etc.
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
                                    ClinicalTextPreprocessor(remove_standard_tokens=remove_standard_tokens)))
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


def detect_label_type(labels):
    """
    Detect the type of labels: numeric categories, binary, or text categories.
    
    Args:
        labels: List or Series of labels
    
    Returns:
        str: 'numeric_category', 'binary', or 'text_category'
    """
    # Convert to strings for consistent processing
    labels_str = [str(label).strip().lower() for label in labels]
    
    # Check for numeric categories
    if all(label.isdigit() for label in labels_str):
        unique_numeric_labels = set(int(label) for label in labels_str)
        return 'numeric_category' if len(unique_numeric_labels) > 2 else 'binary'
    
    # Check for binary text labels
    binary_sets = [
        {'yes', 'no'}, 
        {'true', 'false'}, 
        {'positive', 'negative'}, 
        {'1', '0'}, 
        {'y', 'n'}
    ]
    
    if any(set(labels_str).issubset(binary_set) for binary_set in binary_sets):
        return 'binary'
    
    # If not numeric and not binary, assume text category
    return 'text_category'


def load_and_filter_data(file_path):
    """
    Load the dataset and create filtered DataFrames based on prediction correctness.
    
    Args:
        file_path (str): Path to the CSV file containing prediction data.
        
    Returns:
        dict: A dictionary containing:
        - 'full_df': Original dataset with prediction_result column
        - 'TP_data': True Positives
        - 'TN_data': True Negatives
        - 'FP_data': False Positives
        - 'FN_data': False Negatives
        - 'correct_predictions': All correct predictions
        - 'incorrect_predictions': All incorrect predictions
    """
    # Load dataset
    df = pd.read_csv(file_path)

    # Ensure required columns exist
    required_columns = {'patient_id', 'true_label', 'predicted_label', 'explanation', 'original_note'}
    if not required_columns.issubset(df.columns):
        missing_cols = required_columns - set(df.columns)
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Add prediction_result column
    def classify_prediction(row):
        if row['predicted_label'] == 1 and row['true_label'] == 1:
            return 'TP'
        elif row['predicted_label'] == 0 and row['true_label'] == 0:
            return 'TN'
        elif row['predicted_label'] == 1 and row['true_label'] == 0:
            return 'FP'
        elif row['predicted_label'] == 0 and row['true_label'] == 1:
            return 'FN'
        else:
            return 'Unknown'
            
    df['prediction_result'] = df.apply(classify_prediction, axis=1)
    
    # Create filtered dataframes based on prediction result
    TP_data = df[df['prediction_result'] == 'TP'].reset_index(drop=True)
    TN_data = df[df['prediction_result'] == 'TN'].reset_index(drop=True)
    FP_data = df[df['prediction_result'] == 'FP'].reset_index(drop=True)
    FN_data = df[df['prediction_result'] == 'FN'].reset_index(drop=True)
    
    # Group by correct/incorrect for backward compatibility
    correct_predictions = pd.concat([TP_data, TN_data]).reset_index(drop=True)
    incorrect_predictions = pd.concat([FP_data, FN_data]).reset_index(drop=True)

    return {
        'full_df': df,
        'TP_data': TP_data,
        'TN_data': TN_data,
        'FP_data': FP_data,
        'FN_data': FN_data,
        'correct_predictions': correct_predictions,
        'incorrect_predictions': incorrect_predictions
    }


def extract_criteria_mentions(explanations, criteria_dict=None):
    """
    Extract mentions of specific criteria categories from explanations with improved detection.
    
    Args:
        explanations: Series of explanation texts
        criteria_dict: Dictionary mapping categories to keywords (optional)
        
    Returns:
        DataFrame with binary indicators for each criteria category and confidence scores.
    """
    # Comprehensive criteria dictionary if not provided
    if criteria_dict is None:
        criteria_dict = {
            # Anthropometric measurements
            'BMI': ['bmi', 'body mass index'],
            'weight_for_height': ['weight for height', 'weight-for-height', 'wfh'],
            'BMI_for_age': ['bmi for age', 'bmi-for-age'],
            'MUAC': ['muac', 'mid upper arm circumference', 'mid-upper arm circumference'],
            'weight_loss': ['weight loss', 'lost weight', 'decrease in weight', 'declining weight'],
            
            # Clinical symptoms
            'muscle_wasting': ['muscle wasting', 'muscle loss', 'decreased muscle mass', 'muscle atrophy'],
            'fatigue': ['fatigue', 'weakness', 'tired', 'low energy', 'lethargy'],
            'skin_changes': ['skin changes', 'dry skin', 'thin skin', 'skin breakdown', 'poor skin turgor'],
            'hair_changes': ['hair changes', 'hair loss', 'thin hair', 'brittle hair'],
            'edema': ['edema', 'swelling', 'fluid retention'],
            
            # Dietary intake
            'inadequate_intake': ['inadequate intake', 'poor intake', 'decreased intake', 'reduced appetite'],
            'caloric_deficit': ['caloric deficit', 'insufficient calories', 'low calorie', 'calorie restriction'],
            'protein_deficit': ['protein deficit', 'low protein', 'insufficient protein'],
            'food_insecurity': ['food insecurity', 'limited access to food', 'cannot afford food'],
            
            # Medical conditions
            'chronic_illness': ['chronic illness', 'chronic disease', 'comorbidity'],
            'gi_disorders': ['gi disorder', 'gastrointestinal disorder', 'malabsorption', 'diarrhea', 'vomiting'],
            'infection': ['infection', 'sepsis', 'inflammatory'],
            'cancer': ['cancer', 'malignancy', 'oncology', 'tumor'],
            
            # Risk factors
            'medications': ['medication', 'drug induced', 'steroid', 'chemotherapy'],
            'mental_health': ['depression', 'dementia', 'cognitive impairment', 'psychiatric'],
            'socioeconomic': ['socioeconomic', 'homeless', 'poverty', 'financial'],
            'functional_status': ['functional decline', 'immobility', 'bed bound', 'decreased activity'],
            
            # Lab markers
            'lab_markers': ['albumin', 'prealbumin', 'transferrin', 'hemoglobin', 'lymphocyte', 'protein']
        }
    
    # Initialize DataFrame for criteria mentions with confidence scores
    criteria_df = pd.DataFrame(0.0, index=range(len(explanations)),
                               columns=list(criteria_dict.keys()))
    
    # Add columns for confidence scores
    confidence_cols = [f"{col}_conf" for col in criteria_dict.keys()]
    for col in confidence_cols:
        criteria_df[col] = 0.0
        
    # Enhanced extraction with context awareness
    for i, exp in enumerate(explanations):
        if pd.isna(exp):
            continue
            
        exp_lower = exp.lower()
        sentences = re.split(r'[.!?]', exp_lower)
        
        for category, keywords in criteria_dict.items():
            max_confidence = 0.0
            found = False
            
            # Check for direct mentions
            for keyword in keywords:
                if keyword.lower() in exp_lower:
                    criteria_df.loc[i, category] = 1
                    found = True
                    
                    # Calculate confidence based on context
                    for sentence in sentences:
                        if keyword.lower() in sentence:
                            # Higher confidence when keyword appears with assessment terms
                            if any(term in sentence for term in ['diagnosis', 'assessment', 'presents', 'finding']):
                                confidence = 0.9
                            # Higher confidence when used with measurements
                            elif any(term in sentence for term in ['measured', 'value', 'score', 'shows']):
                                confidence = 0.8
                            # Medium confidence for descriptive mentions
                            elif any(term in sentence for term in ['mild', 'moderate', 'severe', 'significant']):
                                confidence = 0.7
                            # Lower confidence for general mentions
                            else:
                                confidence = 0.5
                                
                            max_confidence = max(max_confidence, confidence)
            
            # Check for indirect mentions and related concepts
            if not found:
                for sentence in sentences:
                    # Anthropometric context detection
                    if category in ['BMI', 'weight_for_height', 'MUAC']:
                        if re.search(r'\b\d+\.?\d*\s*(kg/m2|cm|%|percentile)', sentence):
                            if any(term in sentence for term in ['low', 'below', 'reduced', 'decreased']):
                                criteria_df.loc[i, category] = 1
                                max_confidence = 0.6
                    
                    # Clinical symptoms context
                    elif category in ['muscle_wasting', 'fatigue', 'skin_changes', 'edema']:
                        symptom_terms = {
                            'muscle_wasting': ['thin appearance', 'cachectic', 'loss of mass'],
                            'fatigue': ['decreased energy', 'easily tired', 'exhaustion'],
                            'skin_changes': ['poor healing', 'pressure injury', 'skin integrity'],
                            'edema': ['pitting', 'anasarca', 'peripheral swelling']
                        }
                        
                        if category in symptom_terms:
                            if any(term in sentence for term in symptom_terms[category]):
                                criteria_df.loc[i, category] = 1
                                max_confidence = 0.6
            
            # Record confidence score
            criteria_df.loc[i, f"{category}_conf"] = max_confidence
    
    return criteria_df
    
def extract_clinical_measurements(explanations):
    """
    Extract clinical measurements and their values from explanations with enhanced context.

    Args:
        explanations: Series of explanation texts

    Returns:
        DataFrame with extracted measurements, values, and context indicators
    """
    # Enhanced patterns with more clinical values and flexibile formatting
    patterns = {
        'BMI': r'(?:BMI|body mass index)\s*(?:is|of|:|=|at)?\s*([\d.]+)(?:\s*kg/?m2)?',
        'weight_for_height': r'(?:weight[- ]for[- ]height|WFH)\s*(?:z[- ]score)?\s*(?:is|of|:|=|at)?\s*([-\d.]+)',
        'BMI_for_age': r'(?:BMI[- ]for[- ]age)\s*(?:z[- ]score)?\s*(?:is|of|:|=|at)?\s*([-\d.]+)',
        'MUAC': r'(?:MUAC|mid[- ]upper arm circumference)\s*(?:is|of|:|=|at)?\s*([\d.]+)(?:\s*cm)?',
        'albumin': r'(?:albumin|serum albumin)\s*(?:is|of|:|=|at)?\s*([\d.]+)(?:\s*g/dL)?',
        'prealbumin': r'(?:prealbumin|transthyretin)\s*(?:is|of|:|=|at)?\s*([\d.]+)(?:\s*mg/dL)?',
        'transferrin': r'(?:transferrin|TIBC)\s*(?:is|of|:|=|at)?\s*([\d.]+)(?:\s*mg/dL)?',
        'hemoglobin': r'(?:hemoglobin|haemoglobin|Hgb|Hb)\s*(?:is|of|:|=|at)?\s*([\d.]+)(?:\s*g/dL)?',
        'weight': r'(?:weight|body weight)\s*(?:is|of|:|=|at)?\s*([\d.]+)\s*(?:kg|kilograms)?',
        'height': r'(?:height|body height)\s*(?:is|of|:|=|at)?\s*([\d.]+)\s*(?:cm|meters|m)?',
        'weight_loss': r'(?:weight loss|lost)\s*(?:is|of|:|=|at)?\s*([\d.]+)\s*(?:%|percent|kg)?',
        'length_height_for_age': r'(?:length/height[- ]for[- ]age)\s*(?:z[- ]score)?\s*(?:is|of|:|=|at)?\s*([-\d.]+)',
        'total_protein': r'(?:total protein|serum protein|protein level)\s*(?:is|of|:|=|at)?\s*([\d.]+)(?:\s*g/dL)?',
        'lymphocyte_count': r'(?:lymphocyte count|lymphocytes|ALC)\s*(?:is|of|:|=|at)?\s*([\d.]+)(?:\s*cells/mm3)?',
        'CRP': r'(?:CRP|C-reactive protein)\s*(?:is|of|:|=|at)?\s*([\d.]+)(?:\s*mg/L)?',
        'caloric_intake': r'(?:caloric intake|calorie intake|calories)\s*(?:is|of|:|=|at)?\s*([\d.]+)(?:\s*kcal)?',
        'protein_intake': r'(?:protein intake|dietary protein)\s*(?:is|of|:|=|at)?\s*([\d.]+)(?:\s*g)?'
    }

    # Initialize result list with context indicators
    measurements = []

    for i, exp in enumerate(explanations):
        if pd.isna(exp):
            continue

        row = {'explanation_id': i}
        text = exp.lower()

        # Extract measurements
        for measure, pattern in patterns.items():
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    value = float(match.group(1))
                    row[measure] = value
                    
                    # Extract context around the measurement
                    match_position = match.start()
                    context_start = max(0, match_position - 100)
                    context_end = min(len(text), match_position + 100)
                    context = text[context_start:context_end]
                    row[f"{measure}_context"] = context
                    
                    # Determine if value is flagged as abnormal
                    abnormal_terms = ['low', 'high', 'abnormal', 'insufficient', 'deficient', 
                                      'elevated', 'decreased', 'increased', 'below', 'above']
                    row[f"{measure}_abnormal"] = int(any(term in context for term in abnormal_terms))
                    
                except ValueError:
                    continue

        # Detect severity levels with finer granularity
        severity_patterns = {
            'mild_malnutrition': r'mild(?:\s+(?:acute|chronic))?\s+malnutrition',
            'moderate_malnutrition': r'moderate(?:\s+(?:acute|chronic))?\s+malnutrition',
            'severe_malnutrition': r'severe(?:\s+(?:acute|chronic))?\s+malnutrition',
            'malnutrition_risk': r'(?:at risk for|risk of|potential)\s+malnutrition',
            'protein_calorie_malnutrition': r'protein[- ]calorie malnutrition|PCM',
            'acute_malnutrition': r'acute malnutrition|(?<!severe )acute malnutrition',
            'chronic_malnutrition': r'chronic malnutrition|(?<!severe )chronic malnutrition',
            'severe_acute_malnutrition': r'severe acute malnutrition|SAM',
            'moderate_acute_malnutrition': r'moderate acute malnutrition|MAM'
        }

        for severity, pattern in severity_patterns.items():
            row[severity] = int(bool(re.search(pattern, text, re.IGNORECASE)))
            
        # Extract references to classification systems
        classification_patterns = {
            'who_criteria': r'WHO|World Health Organization',
            'glim_criteria': r'GLIM|Global Leadership Initiative on Malnutrition',
            'aspen_criteria': r'ASPEN|American Society for Parenteral and Enteral Nutrition',
            'academy_criteria': r'Academy of Nutrition and Dietetics',
            'pediatric_criteria': r'pediatric growth chart|growth percentile'
        }
        
        for classif, pattern in classification_patterns.items():
            row[classif] = int(bool(re.search(pattern, text, re.IGNORECASE)))

        if len(row) > 1:  # At least one measurement or classification
            measurements.append(row)

    return pd.DataFrame(measurements)

def analyze_clinical_symptoms(explanations, malnutrition_status):
    """
    Analyze clinical symptoms mentioned in the explanations.
    
    Args:
        explanations: Series of explanation texts
        malnutrition_status: Series with malnutrition decisions
        
    Returns:
        DataFrame with clinical symptoms analysis
    """
    # Define symptom categories and their related terms
    symptom_categories = {
        'muscle_wasting': ['muscle wasting', 'muscle loss', 'decreased muscle mass', 'muscle atrophy', 
                           'sarcopenia', 'cachexia', 'wasted appearance', 'loss of muscle tone'],
        'fatigue': ['fatigue', 'weakness', 'tiredness', 'low energy', 'lethargy', 'exhaustion', 
                    'easily fatigued', 'decreased stamina'],
        'skin_changes': ['dry skin', 'thin skin', 'skin breakdown', 'poor skin turgor', 'skin integrity',
                         'flaky skin', 'delayed wound healing', 'pressure injury', 'pressure ulcer'],
        'hair_changes': ['hair loss', 'thin hair', 'brittle hair', 'dry hair', 'hair color changes',
                         'easy hair pluckability', 'alopecia'],
        'edema': ['edema', 'swelling', 'fluid retention', 'ascites', 'anasarca', 'pitting edema',
                  'peripheral edema', 'dependent edema'],
        'decreased_functionality': ['functional decline', 'decreased activity', 'reduced mobility',
                                   'inability to perform ADLs', 'weakness', 'bed bound'],
        'oral_symptoms': ['mouth sores', 'glossitis', 'stomatitis', 'cheilosis', 'oral lesions',
                         'angular cheilitis', 'difficulty swallowing', 'dysphagia'],
        'gi_symptoms': ['diarrhea', 'nausea', 'vomiting', 'decreased appetite', 'early satiety',
                       'malabsorption', 'constipation', 'bloating'],
        'cognitive_symptoms': ['confusion', 'disorientation', 'irritability', 'apathy',
                              'decreased concentration', 'mood changes'],
        'vitamin_deficiency': ['night blindness', 'peripheral neuropathy', 'paresthesias',
                              'vitamin deficiency', 'scurvy', 'beriberi', 'pellagra']
    }
    
    # Initialize result DataFrame
    symptom_df = pd.DataFrame(0, index=range(len(explanations)), 
                             columns=list(symptom_categories.keys()))
    
    # Extract symptoms
    for i, exp in enumerate(explanations):
        if pd.isna(exp):
            continue
            
        exp_lower = exp.lower()
        
        for category, terms in symptom_categories.items():
            if any(term.lower() in exp_lower for term in terms):
                symptom_df.loc[i, category] = 1
    
    # Calculate symptom statistics
    symptom_stats = []
    
    for symptom in symptom_df.columns:
        # Count occurrences
        total_count = symptom_df[symptom].sum()
        
        # Calculate prevalence in malnutrition vs non-malnutrition
        mal_idx = malnutrition_status[malnutrition_status == 'yes'].index
        non_mal_idx = malnutrition_status[malnutrition_status == 'no'].index
        
        mal_count = symptom_df.loc[mal_idx, symptom].sum()
        non_mal_count = symptom_df.loc[non_mal_idx, symptom].sum()
        
        mal_prevalence = mal_count / len(mal_idx) if len(mal_idx) > 0 else 0
        non_mal_prevalence = non_mal_count / len(non_mal_idx) if len(non_mal_idx) > 0 else 0
        
        # Calculate prevalence ratio
        prevalence_ratio = mal_prevalence / non_mal_prevalence if non_mal_prevalence > 0 else float('inf')
        
        # Calculate attributable risk
        attr_risk = mal_prevalence - non_mal_prevalence
        
        # Calculate sensitivity and specificity
        true_pos = mal_count
        false_pos = non_mal_count
        false_neg = len(mal_idx) - mal_count
        true_neg = len(non_mal_idx) - non_mal_count
        
        sensitivity = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
        specificity = true_neg / (true_neg + false_pos) if (true_neg + false_pos) > 0 else 0
        
        # Calculate positive predictive value and negative predictive value
        ppv = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
        npv = true_neg / (true_neg + false_neg) if (true_neg + false_neg) > 0 else 0
        
        symptom_stats.append({
            'symptom': symptom,
            'total_count': total_count,
            'malnutrition_prevalence': mal_prevalence,
            'non_malnutrition_prevalence': non_mal_prevalence,
            'prevalence_ratio': prevalence_ratio,
            'attributable_risk': attr_risk,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'ppv': ppv,
            'npv': npv
        })
    
    return pd.DataFrame(symptom_stats)

def analyze_dietary_factors(explanations, malnutrition_status):
    """
    Analyze dietary intake factors mentioned in explanations.
    
    Args:
        explanations: Series of explanation texts
        malnutrition_status: Series with malnutrition decisions
        
    Returns:
        DataFrame with dietary factors analysis
    """
    # Define dietary categories and related terms
    dietary_categories = {
        'inadequate_intake': ['inadequate intake', 'poor intake', 'decreased intake', 'reduced appetite',
                             'poor appetite', 'anorexia', 'not eating', 'minimal intake'],
        'caloric_deficit': ['caloric deficit', 'insufficient calories', 'low calorie', 'calorie restriction',
                           'not meeting caloric needs', 'caloric requirement not met'],
        'protein_deficit': ['protein deficit', 'low protein', 'insufficient protein', 'protein requirement not met',
                           'low protein intake', 'inadequate protein'],
        'food_insecurity': ['food insecurity', 'limited access to food', 'cannot afford food', 'food scarcity',
                           'economic barriers to food', 'lack of food access'],
        'poor_diet_quality': ['poor diet quality', 'nutritional deficiency', 'imbalanced diet', 'micronutrient deficiency',
                             'vitamin deficiency', 'mineral deficiency'],
        'feeding_difficulties': ['feeding difficulty', 'difficulty eating', 'swallowing problem', 'dysphagia',
                               'chewing problem', 'feeding dependence'],
        'specialized_diet': ['tube feeding', 'enteral nutrition', 'parenteral nutrition', 'TPN', 'NPO',
                            'nothing by mouth', 'liquid diet', 'pureed diet'],
        'meal_skipping': ['skipping meals', 'missing meals', 'irregular eating', 'not eating regularly'],
        'NPO_status': ['NPO', 'nil per os', 'nothing by mouth', 'fasting'],
        'dietary_restrictions': ['dietary restriction', 'limited diet', 'restricted diet', 'specific diet']
    }
    
    # Initialize results DataFrame
    dietary_df = pd.DataFrame(0, index=range(len(explanations)),
                             columns=list(dietary_categories.keys()))
    
    # Extract dietary mentions
    for i, exp in enumerate(explanations):
        if pd.isna(exp):
            continue
            
        exp_lower = exp.lower()
        
        for category, terms in dietary_categories.items():
            if any(term.lower() in exp_lower for term in terms):
                dietary_df.loc[i, category] = 1
    
    # Calculate dietary factor statistics
    dietary_stats = []
    
    for factor in dietary_df.columns:
        # Count occurrences
        total_count = dietary_df[factor].sum()
        
        # Calculate odds ratio
        mal_idx = malnutrition_status[malnutrition_status == 'yes'].index
        non_mal_idx = malnutrition_status[malnutrition_status == 'no'].index
        
        # Calculate odds ratio components
        exposed_cases = dietary_df.loc[mal_idx, factor].sum()
        exposed_controls = dietary_df.loc[non_mal_idx, factor].sum()
        unexposed_cases = len(mal_idx) - exposed_cases
        unexposed_controls = len(non_mal_idx) - exposed_controls
        
        # Calculate odds ratio (with handling for zero division)
        odds_numerator = (exposed_cases / unexposed_cases) if unexposed_cases > 0 else float('inf')
        odds_denominator = (exposed_controls / unexposed_controls) if unexposed_controls > 0 else float('inf')
        odds_ratio = odds_numerator / odds_denominator if odds_denominator != 0 and odds_denominator != float('inf') else float('inf')
        
        # Calculate prevalence in each group
        mal_prevalence = exposed_cases / len(mal_idx) if len(mal_idx) > 0 else 0
        non_mal_prevalence = exposed_controls / len(non_mal_idx) if len(non_mal_idx) > 0 else 0
        
        dietary_stats.append({
            'factor': factor,
            'total_count': total_count,
            'malnutrition_prevalence': mal_prevalence,
            'non_malnutrition_prevalence': non_mal_prevalence,
            'odds_ratio': odds_ratio
        })
    
    return pd.DataFrame(dietary_stats)

def analyze_risk_factors(explanations, malnutrition_status):
    """
    Analyze risk factors for malnutrition mentioned in explanations.
    
    Args:
        explanations: Series of explanation texts
        malnutrition_status: Series with malnutrition decisions
        
    Returns:
        DataFrame with risk factors analysis
    """
    # Define risk factor categories and related terms
    risk_categories = {
        'chronic_illness': ['chronic illness', 'chronic disease', 'comorbidity', 'chronic condition',
                           'long-term illness', 'chronic health condition'],
        'gi_disorders': ['gi disorder', 'gastrointestinal disorder', 'malabsorption', 'diarrhea', 'vomiting',
                        'IBD', 'Crohn', 'colitis', 'celiac', 'pancreatitis', 'GI surgery'],
        'infection': ['infection', 'sepsis', 'inflammatory', 'inflammation', 'infectious process', 
                     'infectious disease', 'acute illness'],
        'cancer': ['cancer', 'malignancy', 'oncology', 'tumor', 'carcinoma', 'metastasis', 'neoplasm',
                  'chemotherapy', 'radiation therapy'],
        'medications': ['medication effect', 'drug induced', 'steroid', 'chemotherapy', 'drug side effect',
                       'medication side effect', 'immunosuppressants'],
        'mental_health': ['depression', 'dementia', 'cognitive impairment', 'psychiatric', 'mental illness',
                         'anxiety', 'confusion', 'altered mental status'],
        'socioeconomic': ['socioeconomic', 'homeless', 'poverty', 'financial', 'economic factors',
                         'social factors', 'lack of resources', 'poor living conditions'],
        'functional_status': ['functional decline', 'immobility', 'bed bound', 'decreased activity',
                             'impaired mobility', 'reduced function', 'disability'],
        'advanced_age': ['elderly', 'geriatric', 'advanced age', 'older adult', 'age-related',
                        'frailty', 'senile'],
        'surgical': ['post-surgical', 'post-operative', 'after surgery', 'surgical patient',
                    'recent surgery', 'surgical complication'],
        'trauma': ['trauma', 'injury', 'burn', 'accident', 'physical trauma'],
        'alcohol_substance': ['alcohol use', 'substance abuse', 'drug abuse', 'alcoholism',
                             'alcohol dependence', 'addiction']
    }
    
    # Initialize results DataFrame
    risk_df = pd.DataFrame(0, index=range(len(explanations)),
                          columns=list(risk_categories.keys()))
    
    # Extract risk factor mentions
    for i, exp in enumerate(explanations):
        if pd.isna(exp):
            continue
            
        exp_lower = exp.lower()
        
        for category, terms in risk_categories.items():
            if any(term.lower() in exp_lower for term in terms):
                risk_df.loc[i, category] = 1
    
    # Calculate risk factor statistics
    risk_stats = []
    
    for factor in risk_df.columns:
        # Count occurrences
        total_count = risk_df[factor].sum()
        
        # Calculate relative risk
        mal_idx = malnutrition_status[malnutrition_status == 'yes'].index
        non_mal_idx = malnutrition_status[malnutrition_status == 'no'].index
        
        # Calculate risk ratio components
        exposed_cases = risk_df.loc[mal_idx, factor].sum()
        exposed_controls = risk_df.loc[non_mal_idx, factor].sum()
        
        risk_in_exposed = exposed_cases / len(mal_idx) if len(mal_idx) > 0 else 0
        risk_in_unexposed = exposed_controls / len(non_mal_idx) if len(non_mal_idx) > 0 else 0
        
        # Calculate relative risk
        relative_risk = risk_in_exposed / risk_in_unexposed if risk_in_unexposed > 0 else float('inf')
        
        # Calculate attributable risk
        attributable_risk = risk_in_exposed - risk_in_unexposed
        
        risk_stats.append({
            'factor': factor,
            'total_count': total_count,
            'risk_in_malnourished': risk_in_exposed,
            'risk_in_non_malnourished': risk_in_unexposed,
            'relative_risk': relative_risk,
            'attributable_risk': attributable_risk
        })
    
    return pd.DataFrame(risk_stats)
    
def extract_patient_demographics(explanations):
    """
    Extract patient demographic and characteristic information from explanations.
    
    Args:
        explanations: Series of explanation texts
        
    Returns:
        DataFrame with extracted patient demographics
    """
    import re
    import pandas as pd
    
    # Initialize result list
    demographics = []
    
    # Define patterns for demographics extraction
    patterns = {
        'age': r'(?:age|aged|year[\s-]old)[^\d]*(\d+)',
        'pediatric': r'(?:child|infant|pediatric|newborn|baby)',
        'gender_male': r'(?:male|man|boy|gentleman)',
        'gender_female': r'(?:female|woman|girl|lady)',
        'height_cm': r'height[^\d]*(\d+)(?:\.\d+)?(?:\s*cm)',
        'height_m': r'height[^\d]*(\d+)(?:\.\d+)?(?:\s*m)',
        'weight_kg': r'weight[^\d]*(\d+)(?:\.\d+)?(?:\s*kg)',
        'hospital_status': r'(inpatient|outpatient|hospitalized|admitted)'
    }
    
    # Clinical settings
    clinical_settings = [
        'hospital', 'clinic', 'emergency', 'ER', 'ICU', 'intensive care', 
        'nursing home', 'long-term care', 'palliative', 'hospice', 'rehabilitation'
    ]
    
    # Specialty areas
    specialty_areas = [
        'pediatrics', 'geriatrics', 'oncology', 'surgery', 'internal medicine',
        'cardiology', 'pulmonology', 'neurology', 'psychiatry', 'gastroenterology'
    ]
    
    for i, exp in enumerate(explanations):
        if pd.isna(exp):
            continue
            
        text = exp.lower()
        row = {'explanation_id': i}
        
        # Extract patterns
        for key, pattern in patterns.items():
            match = re.search(pattern, text, re.IGNORECASE)
            if match and key in ['age', 'height_cm', 'height_m', 'weight_kg']:
                try:
                    row[key] = float(match.group(1))
                except ValueError:
                    pass
            elif match:
                row[key] = 1
        
        # Convert height from cm to m if needed
        if 'height_cm' in row and 'height_m' not in row:
            row['height_m'] = row['height_cm'] / 100
            
        # Calculate BMI if both height and weight are available
        if 'height_m' in row and 'weight_kg' in row and row['height_m'] > 0:
            row['calculated_bmi'] = row['weight_kg'] / (row['height_m'] ** 2)
        
        # Check for pediatric indicators
        row['pediatric'] = 1 if ('pediatric' in row and row['pediatric']) or ('age' in row and row['age'] < 18) else 0
        
        # Extract clinical settings
        for setting in clinical_settings:
            if setting.lower() in text:
                row[f'setting_{setting.replace(" ", "_")}'] = 1
        
        # Extract specialty areas
        for specialty in specialty_areas:
            if specialty.lower() in text:
                row[f'specialty_{specialty.replace(" ", "_")}'] = 1
        
        # Add to demographics list
        demographics.append(row)
    
    # Convert to DataFrame
    df = pd.DataFrame(demographics)
    
    # Fill any missing binary features with 0
    binary_columns = ['pediatric', 'gender_male', 'gender_female', 'hospital_status']
    binary_columns += [f'setting_{s.replace(" ", "_")}' for s in clinical_settings]
    binary_columns += [f'specialty_{s.replace(" ", "_")}' for s in specialty_areas]
    
    for col in binary_columns:
        if col in df.columns:
            df[col] = df[col].fillna(0)
    
    return df


def analyze_criteria_correlation(criteria_df, malnutrition_status):
    """
    Analyze association of criteria mentions with malnutrition status using multiple metrics.
    
    Args:
        criteria_df: DataFrame with binary indicators for criteria categories
        malnutrition_status: Series with malnutrition decisions
        
    Returns:
        DataFrame with correlation metrics including OR, p-value, and CI.
    """
    y = (malnutrition_status == 'yes').astype(int)
    results = []
    
    # Filter to only include binary criteria columns (exclude confidence scores)
    binary_criteria_cols = [col for col in criteria_df.columns if not col.endswith('_conf')]
    
    for criteria in binary_criteria_cols:
        # Point-biserial correlation
        clean_df = pd.DataFrame({'criteria': criteria_df[criteria], 'malnutrition': y}).dropna()
        if len(clean_df) < 2 or clean_df['criteria'].nunique() < 2 or clean_df['malnutrition'].nunique() < 2:
            continue
            
        # Calculate point-biserial correlation
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            corr = stats.pointbiserialr(clean_df['criteria'], clean_df['malnutrition'])
        
        # Odds ratio calculation
        table = pd.crosstab(clean_df['criteria'], clean_df['malnutrition'])
        
        # Handle cases where table doesn't have expected shape
        if 0 not in table.index or 1 not in table.index or 0 not in table.columns or 1 not in table.columns:
            # Reconstruct missing cells with zeros
            full_table = pd.DataFrame(0, index=[0, 1], columns=[0, 1])
            for i in table.index:
                for j in table.columns:
                    full_table.loc[i, j] = table.loc[i, j]
            table = full_table
        
        # Calculate odds ratio with Haldane-Anscombe correction for zero cells
        a = table.loc[1, 1] + 0.5  # Exposed cases
        b = table.loc[1, 0] + 0.5  # Exposed non-cases
        c = table.loc[0, 1] + 0.5  # Unexposed cases
        d = table.loc[0, 0] + 0.5  # Unexposed non-cases
        
        odds_ratio = (a/c) / (b/d)
        
        # Fisher's exact test for significance
        _, pval = stats.fisher_exact(table)
        
        # Calculate confidence intervals for odds ratio (log method)
        log_or = np.log(odds_ratio)
        se_log_or = np.sqrt(1/a + 1/b + 1/c + 1/d)
        ci_lower = np.exp(log_or - 1.96 * se_log_or)
        ci_upper = np.exp(log_or + 1.96 * se_log_or)
        
        # Calculate prevalence in each group
        prevalence_mal = clean_df.groupby('malnutrition')['criteria'].mean().get(1, 0)
        prevalence_non_mal = clean_df.groupby('malnutrition')['criteria'].mean().get(0, 0)
        
        # Add confidence score if available
        conf_score = criteria_df[f"{criteria}_conf"].mean() if f"{criteria}_conf" in criteria_df.columns else None
        
        results.append({
            'criteria': criteria,
            'correlation': corr[0],
            'p_value': corr[1],
            'odds_ratio': odds_ratio,
            'OR_p_value': pval,
            'prevalence_malnourished': prevalence_mal,
            'prevalence_non_malnourished': prevalence_non_mal,
            'CI_lower': ci_lower,
            'CI_upper': ci_upper,
            'avg_confidence': conf_score
        })
    
    # Create DataFrame and sort by statistical significance then effect size
    result_df = pd.DataFrame(results)
    if not result_df.empty:
        result_df['significant'] = result_df['p_value'] < 0.05
        result_df = result_df.sort_values(['significant', 'odds_ratio'], ascending=[False, False])
    
    return result_df

def analyze_criteria_frequency(criteria_df, malnutrition_status):
    """
    Analyze criteria frequency with group comparisons and confidence intervals.
    
    Args:
        criteria_df: DataFrame with binary indicators for criteria categories
        malnutrition_status: Series with malnutrition decisions
        
    Returns:
        DataFrame with frequency analysis.
    """
    results = []
    y = (malnutrition_status == 'yes').astype(int)
    
    # Filter to only include binary criteria columns (exclude confidence scores)
    binary_criteria_cols = [col for col in criteria_df.columns if not col.endswith('_conf')]
    
    for criteria in binary_criteria_cols:
        # Overall frequency with CI
        freq = criteria_df[criteria].mean()
        count = criteria_df[criteria].sum()
        
        # Skip if no occurrences
        if count == 0:
            continue
            
        # Calculate Wilson score confidence interval
        ci = proportion_confint(count, len(criteria_df), alpha=0.05, method='wilson')
        
        # Group frequencies
        mal_indices = y[y == 1].index
        non_mal_indices = y[y == 0].index
        
        mal_values = criteria_df.loc[mal_indices, criteria].fillna(0)
        non_mal_values = criteria_df.loc[non_mal_indices, criteria].fillna(0)
        
        mal_freq = mal_values.mean() if len(mal_values) > 0 else 0
        non_mal_freq = non_mal_values.mean() if len(non_mal_values) > 0 else 0
        
        # Risk difference and risk ratio
        risk_diff = mal_freq - non_mal_freq
        risk_ratio = mal_freq / non_mal_freq if non_mal_freq > 0 else float('inf')
        
        # Statistical test - Fisher's exact test is more appropriate for binary data
        table = pd.crosstab(criteria_df[criteria], y)
        _, pval = stats.fisher_exact(table) if table.shape == (2, 2) else (1.0, 1.0)
        
        # Add confidence score if available
        conf_score = criteria_df[f"{criteria}_conf"].mean() if f"{criteria}_conf" in criteria_df.columns else None
        
        results.append({
            'criteria': criteria,
            'overall_freq': freq,
            'count': count,
            'overall_CI_lower': ci[0],
            'overall_CI_upper': ci[1],
            'malnourished_freq': mal_freq,
            'non_malnourished_freq': non_mal_freq,
            'risk_difference': risk_diff,
            'risk_ratio': risk_ratio,
            'p_value': pval,
            'avg_confidence': conf_score
        })
    
    # Create DataFrame and sort by statistical significance then effect size
    result_df = pd.DataFrame(results)
    if not result_df.empty:
        result_df['significant'] = result_df['p_value'] < 0.05
        result_df = result_df.sort_values(['significant', 'risk_difference'], ascending=[False, False])
    
    return result_df

def analyze_measurement_thresholds(measurements_df, malnutrition_status):
    """
    Enhanced threshold analysis with effect sizes and bootstrap validation.
    
    Args:
        measurements_df: DataFrame with extracted clinical measurements
        malnutrition_status: Series with malnutrition decisions
        
    Returns:
        Dictionary with threshold analysis results for each measurement.
    """
    # Convert malnutrition status to binary
    y_binary = (malnutrition_status == 'yes').astype(int)
    
    # Join measurements with malnutrition status
    joined_df = measurements_df.merge(
        pd.DataFrame({'malnutrition_binary': y_binary}), 
        left_on='explanation_id', 
        right_index=True
    )
    
    # Define measurement directionality based on clinical knowledge
    direction_map = {
        'BMI': 'lower_is_risk',
        'albumin': 'lower_is_risk',
        'prealbumin': 'lower_is_risk',
        'transferrin': 'lower_is_risk',
        'hemoglobin': 'lower_is_risk',
        'weight': 'lower_is_risk',
        'total_protein': 'lower_is_risk',
        'lymphocyte_count': 'lower_is_risk',
        'weight_loss': 'higher_is_risk',  # Higher percent weight loss is worse
        'CRP': 'higher_is_risk',
        'caloric_intake': 'lower_is_risk',
        'protein_intake': 'lower_is_risk',
        'MUAC': 'lower_is_risk',
        'weight_for_height': 'lower_is_risk',
        'BMI_for_age': 'lower_is_risk',
        'length_height_for_age': 'lower_is_risk'
    }
    
    # Select numeric columns for analysis (excluding explanation_id and context columns)
    numeric_cols = [col for col in measurements_df.columns 
                   if col != 'explanation_id' 
                   and not col.endswith('_context')
                   and not col.endswith('_abnormal')
                   and not any(x in col for x in ['mild_', 'moderate_', 'severe_', 'malnutrition_', 'who_', 'glim_', 'aspen_', 'academy_', 'pediatric_'])]
    
    thresholds = {}
    
    for measure in numeric_cols:
        # Skip if not enough data or not a numerical column
        if measure not in joined_df.columns or not pd.api.types.is_numeric_dtype(joined_df[measure]):
            continue
        
        # Get clean data for this measure
        data = joined_df[[measure, 'malnutrition_binary']].dropna()
        
        # Skip if not enough data
        if len(data) < 10 or data['malnutrition_binary'].nunique() < 2:
            continue
            
        y = data['malnutrition_binary']
        X = data[measure]
        
        # Determine direction based on map or correlation
        if measure in direction_map:
            direction = direction_map[measure]
        else:
            # Determine direction based on correlation
            corr = X.corr(y)
            direction = 'higher_is_risk' if corr > 0 else 'lower_is_risk'
        
        # Adjust values based on direction for ROC analysis
        X_adjusted = X if direction == 'higher_is_risk' else -X
        
        try:
            # ROC analysis
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fpr, tpr, thresh = roc_curve(y, X_adjusted)
                roc_auc = roc_auc_score(y, X_adjusted)
            
            # Convert thresholds back to original scale for lower_is_risk
            if direction == 'lower_is_risk':
                thresh = -thresh
            
            # Optimal threshold (Youden's index)
            youden = tpr - fpr
            idx = np.argmax(youden)
            opt_thresh = thresh[idx]
            
            # Calculate precision-recall curve
            precision, recall, pr_thresholds = precision_recall_curve(y, X_adjusted)
            
            # Calculate average precision
            avg_precision = average_precision_score(y, X_adjusted)
            
            # Find threshold that maximizes F1 score
            f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
            f1_idx = np.argmax(f1_scores)
            
            # Convert PR threshold back if needed
            f1_threshold = pr_thresholds[min(f1_idx, len(pr_thresholds)-1)]
            if direction == 'lower_is_risk':
                f1_threshold = -f1_threshold
            
            # Bootstrap validation for confidence intervals
            n_boot = 500
            boot_aucs = []
            boot_thresholds = []
            
            for _ in range(n_boot):
                # Sample with replacement
                boot_indices = np.random.choice(data.index, size=len(data), replace=True)
                boot_sample = data.loc[boot_indices]
                
                # Skip if not enough variation
                if boot_sample['malnutrition_binary'].nunique() < 2:
                    continue
                
                # Calculate AUC and threshold
                try:
                    boot_auc = roc_auc_score(boot_sample['malnutrition_binary'], 
                                           boot_sample[measure] if direction == 'higher_is_risk' else -boot_sample[measure])
                    boot_aucs.append(boot_auc)
                    
                    # Calculate optimal threshold
                    boot_fpr, boot_tpr, boot_thresh = roc_curve(
                        boot_sample['malnutrition_binary'],
                        boot_sample[measure] if direction == 'higher_is_risk' else -boot_sample[measure]
                    )
                    
                    # Convert thresholds back if needed
                    if direction == 'lower_is_risk':
                        boot_thresh = -boot_thresh
                    
                    boot_youden = boot_tpr - boot_fpr
                    boot_idx = np.argmax(boot_youden)
                    boot_opt_thresh = boot_thresh[boot_idx]
                    boot_thresholds.append(boot_opt_thresh)
                except:
                    continue
            
            # Calculate confidence intervals
            if len(boot_aucs) >= 10:
                auc_ci = np.percentile(boot_aucs, [2.5, 97.5])
                threshold_ci = np.percentile(boot_thresholds, [2.5, 97.5])
            else:
                auc_ci = [np.nan, np.nan]
                threshold_ci = [np.nan, np.nan]
            
            # Calculate group statistics
            group_stats = data.groupby('malnutrition_binary')[measure].agg(['mean', 'std', 'count']).to_dict()
            
            # Calculate Cohen's d effect size
            if 'std' in group_stats and 0 in group_stats['std'] and 1 in group_stats['std']:
                # Pooled standard deviation
                n1 = group_stats['count'][0]
                n2 = group_stats['count'][1]
                s1 = group_stats['std'][0]
                s2 = group_stats['std'][1]
                
                # Handle case when either standard deviation is zero
                if s1 == 0 and s2 == 0:
                    cohen_d = float('inf') if group_stats['mean'][1] != group_stats['mean'][0] else 0
                else:
                    pooled_std = np.sqrt(((n1-1)*s1**2 + (n2-1)*s2**2) / (n1+n2-2))
                    cohen_d = (group_stats['mean'][1] - group_stats['mean'][0]) / pooled_std
                    
                    # Adjust sign based on direction
                    if direction == 'lower_is_risk':
                        cohen_d = -cohen_d
            else:
                cohen_d = np.nan
            
            # Store results
            thresholds[measure] = {
                'direction': direction,
                'auc': roc_auc,
                'auc_CI': auc_ci,
                'avg_precision': avg_precision,
                'optimal_threshold': opt_thresh,
                'optimal_threshold_CI': threshold_ci,
                'f1_max_threshold': f1_threshold,
                'sensitivity': tpr[idx],
                'specificity': 1-fpr[idx],
                'cohen_d': cohen_d,
                'group_means': {
                    'malnourished': X[y==1].mean(),
                    'non_malnourished': X[y==0].mean()
                },
                'group_std': {
                    'malnourished': X[y==1].std(),
                    'non_malnourished': X[y==0].std()
                },
                'n_samples': len(data)
            }
        except Exception as e:
            # Handle errors gracefully
            thresholds[measure] = {
                'error': str(e),
                'n_samples': len(data)
            }
    
    return thresholds

def plot_measurement_distributions(measurements_df, malnutrition_status):
    """
    Enhanced distribution plots with statistical annotations.
    
    Args:
        measurements_df: DataFrame with extracted clinical measurements
        malnutrition_status: Series with malnutrition decisions
        
    Returns:
        Dictionary of matplotlib figures for each measurement.
    """
    # Convert malnutrition status to binary and create a dataframe
    y_binary = (malnutrition_status == 'yes').astype(int)
    status_df = pd.DataFrame({'malnutrition': malnutrition_status, 
                             'malnutrition_binary': y_binary})
    
    # Join with measurements
    joined_df = measurements_df.merge(status_df, 
                                    left_on='explanation_id', 
                                    right_index=True)
    
    # Select numeric columns for plotting (excluding explanation_id and context columns)
    numeric_cols = [col for col in measurements_df.columns 
                   if col != 'explanation_id' 
                   and not col.endswith('_context')
                   and not col.endswith('_abnormal')
                   and not any(x in col for x in ['mild_', 'moderate_', 'severe_', 'malnutrition_', 'who_', 'glim_', 'aspen_', 'academy_', 'pediatric_'])]
    
    figures = {}
    
    for measure in numeric_cols:
        # Skip if not a numerical column
        if measure not in joined_df.columns or not pd.api.types.is_numeric_dtype(joined_df[measure]):
            continue
        
        # Create clean data for plotting
        plot_data = joined_df[[measure, 'malnutrition']].dropna()
        
        # Skip if not enough data
        if len(plot_data) < 5 or plot_data['malnutrition'].nunique() < 2:
            continue
        
        try:
            # Create figure
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Enhanced plot using both violin and swarm for small datasets
            if len(plot_data) < 30:
                # Box plot for clarity
                sns.boxplot(x='malnutrition', y=measure, data=plot_data, 
                           ax=ax, width=0.5, palette="Set2")
                
                # Add individual points
                sns.stripplot(x='malnutrition', y=measure, data=plot_data,
                             ax=ax, alpha=0.6, jitter=True, size=6)
            else:
                # Violin plot with quartiles for larger datasets
                sns.violinplot(x='malnutrition', y=measure, data=plot_data, 
                              ax=ax, cut=0, inner='quartile', palette="Set2")
            
            # Add descriptive title
            pretty_measure = measure.replace('_', ' ').title()
            ax.set_title(f'Distribution of {pretty_measure} by Malnutrition Status', fontsize=14)
            
            # Improve axis labels
            ax.set_xlabel('Malnutrition Status', fontsize=12)
            ax.set_ylabel(pretty_measure, fontsize=12)
            
            # Add statistical annotations
            mal = plot_data[plot_data.malnutrition == 'yes'][measure].dropna()
            non_mal = plot_data[plot_data.malnutrition == 'no'][measure].dropna()
            
            if len(mal) > 0 and len(non_mal) > 0:
                # Use Mann-Whitney U test (non-parametric)
                try:
                    u_stat, pval = stats.mannwhitneyu(mal, non_mal)
                    stat_text = f'Mann-Whitney U p = {pval:.4f}'
                    
                    # Add effect size
                    cohen_d = (mal.mean() - non_mal.mean()) / np.sqrt(
                        ((len(mal)-1)*mal.std()**2 + (len(non_mal)-1)*non_mal.std()**2) / 
                        (len(mal) + len(non_mal) - 2)
                    )
                    stat_text += f'\nCohen\'s d = {cohen_d:.2f}'
                    
                    # Add group means
                    stat_text += f'\nMeans: Yes={mal.mean():.2f}, No={non_mal.mean():.2f}'
                    
                    ax.text(0.5, 0.02, stat_text, 
                           transform=ax.transAxes, ha='center',
                           bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
                except:
                    pass
            
            figures[measure] = fig
            
        except Exception as e:
            # Handle errors gracefully
            print(f"Error plotting {measure}: {e}")
            continue
    
    return figures

def analyze_severity_classifications(measurements_df, malnutrition_status):
    """
    Enhanced severity analysis with classification metrics.
    
    Args:
        measurements_df: DataFrame with severity classifications
        malnutrition_status: Series with malnutrition decisions
        
    Returns:
        DataFrame with severity classification performance metrics.
    """
    # Convert malnutrition status to binary
    y_binary = (malnutrition_status == 'yes').astype(int)
    
    # Join with measurements
    joined_df = measurements_df.merge(
        pd.DataFrame({'malnutrition_binary': y_binary}), 
        left_on='explanation_id', 
        right_index=True
    )
    
    # Find all severity columns
    severity_cols = [col for col in joined_df.columns if any(x in col for x in 
                                                           ['mild_malnutrition', 
                                                            'moderate_malnutrition', 
                                                            'severe_malnutrition',
                                                            'malnutrition_risk',
                                                            'protein_calorie_malnutrition',
                                                            'acute_malnutrition',
                                                            'chronic_malnutrition',
                                                            'severe_acute_malnutrition',
                                                            'moderate_acute_malnutrition'])]
    
    results = []
    
    for severity in severity_cols:
        # Skip if column not present or no positive cases
        if severity not in joined_df or joined_df[severity].sum() == 0:
            continue
            
        # Calculate confusion matrix
        try:
            tn, fp, fn, tp = confusion_matrix(
                joined_df['malnutrition_binary'], 
                joined_df[severity]
            ).ravel()
            
            # Calculate metrics
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            npv = tn / (tn + fn) if (tn + fn) > 0 else 0
            f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
            
            # Calculate accuracy
            accuracy = (tp + tn) / (tp + tn + fp + fn)
            
            # Calculate Cohen's kappa
            p_o = accuracy
            p_e = ((tp + fn) * (tp + fp) + (tn + fp) * (tn + fn)) / ((tp + tn + fp + fn) ** 2)
            kappa = (p_o - p_e) / (1 - p_e) if p_e != 1 else 0
            
            # Calculate positive and negative likelihood ratios
            plr = sensitivity / (1 - specificity) if (1 - specificity) > 0 else float('inf')
            nlr = (1 - sensitivity) / specificity if specificity > 0 else float('inf')
            
            # Calculate prevalence
            prevalence = (tp + fn) / (tp + tn + fp + fn)
            
            # Format column name for display
            pretty_name = severity.replace('_', ' ').title()
            
            results.append({
                'severity': pretty_name,
                'prevalence': prevalence,
                'sensitivity': sensitivity,
                'specificity': specificity,
                'precision': precision,
                'npv': npv,
                'f1_score': f1,
                'accuracy': accuracy,
                'kappa': kappa,
                'positive_lr': plr,
                'negative_lr': nlr,
                'support': tp + fn,
                'true_positives': tp,
                'false_positives': fp,
                'true_negatives': tn,
                'false_negatives': fn
            })
        except Exception as e:
            # Handle errors gracefully
            print(f"Error analyzing {severity}: {e}")
            continue
    
    # Convert to DataFrame and sort by F1 score
    result_df = pd.DataFrame(results)
    if not result_df.empty:
        result_df = result_df.sort_values('f1_score', ascending=False)
    
    return result_df

def analyze_measurement_criteria_alignment(measurements_df, criteria_df, malnutrition_status):
    """
    Enhanced alignment analysis with clinical threshold validation.
    
    Args:
        measurements_df: DataFrame with extracted clinical measurements
        criteria_df: DataFrame with binary indicators for criteria categories
        malnutrition_status: Series with malnutrition decisions
        
    Returns:
        DataFrame with alignment analysis.
    """
    # Define mapping between measurements and criteria
    criteria_map = {
        'BMI': 'BMI',
        'weight_for_height': 'weight_for_height',
        'BMI_for_age': 'BMI_for_age',
        'MUAC': 'MUAC',
        'weight_loss': 'weight_loss',
        'albumin': 'lab_markers',
        'prealbumin': 'lab_markers',
        'transferrin': 'lab_markers',
        'hemoglobin': 'lab_markers',
        'total_protein': 'lab_markers',
        'lymphocyte_count': 'lab_markers',
        'CRP': 'lab_markers'
    }
    
    # Define clinical thresholds
    clinical_thresholds = {
        'BMI': {'adult': 18.5, 'elderly': 20.0, 'pediatric': None},  # Different thresholds by age
        'albumin': 3.5,  # g/dL
        'prealbumin': 15.0,  # mg/dL
        'transferrin': 200.0,  # mg/dL
        'hemoglobin': {'male': 13.0, 'female': 12.0},  # g/dL
        'MUAC': {'adult': 23.0, 'pediatric': 12.5},  # cm
        'weight_loss': 5.0,  # % unintentional weight loss
        'total_protein': 6.0,  # g/dL
        'lymphocyte_count': 1500.0,  # cells/mm3
        'CRP': 5.0  # mg/L, higher is abnormal (inflammation)
    }
    
    # List for storing alignment results
    aligned = []
    
    # Get patient demographics if available to determine appropriate thresholds
    demographics = extract_patient_demographics(measurements_df['explanation_id'])
    
    # Calculate thresholds for each case based on demographics
    for measure, criteria in criteria_map.items():
        # Skip if measure not in measurements or criteria not in criteria_df
        if measure not in measurements_df.columns or criteria not in criteria_df.columns:
            continue
        
        # Get cases where both measurement and criteria are available
        combined = measurements_df[['explanation_id', measure]].merge(
            criteria_df[criteria].reset_index().rename(columns={'index': 'exp_index'}),
            left_on='explanation_id', 
            right_on='exp_index'
        )
        
        # Skip if no matches
        if len(combined) == 0:
            continue
        
        # Determine threshold for each case based on demographics
        thresholds = []
        
        for idx, row in combined.iterrows():
            # Get default threshold
            if isinstance(clinical_thresholds.get(measure), dict):
                # Handle age-specific or gender-specific thresholds
                
                # Try to find matching demographics
                demo_row = demographics[demographics['explanation_id'] == row['explanation_id']] if 'explanation_id' in demographics.columns else None
                
                if measure == 'BMI':
                    # Use age-specific BMI threshold
                    if demo_row is not None and len(demo_row) > 0:
                        if 'age' in demo_row.columns and demo_row['age'].iloc[0] is not None:
                            age = demo_row['age'].iloc[0]
                            if age < 18:
                                thresholds.append(clinical_thresholds[measure]['pediatric'] or float('nan'))
                            elif age >= 65:
                                thresholds.append(clinical_thresholds[measure]['elderly'])
                            else:
                                thresholds.append(clinical_thresholds[measure]['adult'])
                        elif 'pediatric' in demo_row.columns and demo_row['pediatric'].iloc[0] == 1:
                            thresholds.append(clinical_thresholds[measure]['pediatric'] or float('nan'))
                        else:
                            thresholds.append(clinical_thresholds[measure]['adult'])
                    else:
                        thresholds.append(clinical_thresholds[measure]['adult'])
                            
                elif measure == 'hemoglobin':
                    # Use gender-specific hemoglobin threshold
                    if demo_row is not None and len(demo_row) > 0:
                        if 'gender_male' in demo_row.columns and demo_row['gender_male'].iloc[0] == 1:
                            thresholds.append(clinical_thresholds[measure]['male'])
                        else:
                            thresholds.append(clinical_thresholds[measure]['female'])
                    else:
                        # Default to lower threshold if gender unknown
                        thresholds.append(min(clinical_thresholds[measure].values()))
                        
                elif measure == 'MUAC':
                    # Use age-specific MUAC threshold
                    if demo_row is not None and len(demo_row) > 0:
                        if ('age' in demo_row.columns and demo_row['age'].iloc[0] is not None and 
                            demo_row['age'].iloc[0] < 18) or \
                           ('pediatric' in demo_row.columns and demo_row['pediatric'].iloc[0] == 1):
                            thresholds.append(clinical_thresholds[measure]['pediatric'])
                        else:
                            thresholds.append(clinical_thresholds[measure]['adult'])
                    else:
                        thresholds.append(clinical_thresholds[measure]['adult'])
            else:
                # Use standard threshold for measures that don't vary by demographics
                thresholds.append(clinical_thresholds.get(measure, float('nan')))
        
        # Convert to DataFrame column
        combined['threshold'] = thresholds
        
        # Determine if measurement indicates malnutrition based on threshold
        # For CRP, higher is abnormal; for others, lower is abnormal
        if measure == 'CRP':
            combined['measurement_indicates_malnutrition'] = combined[measure] > combined['threshold']
        else:
            combined['measurement_indicates_malnutrition'] = combined[measure] < combined['threshold']
        
        # Check if criteria was selected (value == 1)
        combined['criteria_selected'] = combined[criteria] == 1
        
        # Merge with malnutrition status
        combined = combined.merge(
            malnutrition_status.reset_index().rename(columns={'index': 'mal_index'}),
            left_on='explanation_id',
            right_on='mal_index',
            how='left'
        )
        
        # Determine alignment: both measurement and criteria should agree
        combined['aligned'] = (
            (combined['measurement_indicates_malnutrition'] & combined['criteria_selected']) |
            (~combined['measurement_indicates_malnutrition'] & ~combined['criteria_selected'])
        )
        
        # Add measure name for reporting
        combined['measure'] = measure
        combined['criteria_category'] = criteria
        
        # Add to results
        aligned.append(combined[['explanation_id', 'measure', 'criteria_category', 
                                 measure, 'threshold', 'measurement_indicates_malnutrition',
                                 'criteria_selected', 'malnutrition_status', 'aligned']])
    
    # Combine all results
    if aligned:
        alignment_df = pd.concat(aligned, ignore_index=True)
        
        # Calculate alignment statistics
        alignment_stats = alignment_df.groupby(['measure', 'criteria_category']).agg({
            'aligned': ['mean', 'count'],
            'measurement_indicates_malnutrition': 'mean',
            'criteria_selected': 'mean',
            'malnutrition_status': 'mean'
        }).reset_index()
        
        # Calculate correlation between measurements and malnutrition status
        correlations = {}
        for measure in alignment_df['measure'].unique():
            measure_df = alignment_df[alignment_df['measure'] == measure]
            if len(measure_df) > 1:  # Need at least 2 points for correlation
                corr = measure_df['measurement_indicates_malnutrition'].corr(measure_df['malnutrition_status'])
                correlations[measure] = corr
        
        return {
            'detailed_alignment': alignment_df,
            'alignment_stats': alignment_stats,
            'correlations': correlations
        }
    else:
        return {
            'detailed_alignment': pd.DataFrame(),
            'alignment_stats': pd.DataFrame(),
            'correlations': {}
        }

def visualize_criteria_frequency(freq_df, top_n=15):
    """Create annotated visualization of criteria prevalence with statistical markers."""
    
    # Prepare data
    plot_df = freq_df.nlargest(top_n, 'count').sort_values('malnourished_freq', ascending=False)
    melted_df = pd.melt(
        plot_df,
        id_vars=['criteria', 'p_value', 'risk_ratio'],
        value_vars=['malnourished_freq', 'non_malnourished_freq'],
        var_name='group',
        value_name='frequency'
    )
    
    # Formatting
    melted_df['group'] = melted_df['group'].str.replace('_freq', '').str.replace('_', ' ').str.title()
    melted_df['frequency'] *= 100  # Convert to percentages

    # Create visualization
    plt.figure(figsize=(14, 8))
    ax = sns.barplot(
        x='frequency',
        y='criteria',
        hue='group',
        data=melted_df,
        palette={'Malnourished': '#d95f02', 'Non Malnourished': '#7570b3'},
        orient='h'
    )

    # Add statistical annotations
    for idx, (_, row) in enumerate(plot_df.iterrows()):
        y_pos = idx + 0.2
        rr = f"RR: {row['risk_ratio']:.1f}" if row['risk_ratio'] != float('inf') else "RR: ∞"
        pval = row['p_value']
        sig = '*' * sum([pval < cutoff for cutoff in [0.05, 0.01, 0.001]])
        
        ax.text(
            x=102, 
            y=y_pos,
            s=f"{rr}{sig}",
            va='center',
            fontsize=10,
            color='#2d2d2d'
        )

    # Formatting
    plt.title('Criteria Prevalence by Nutritional Status\n(RR = Risk Ratio, *p<0.05, **p<0.01, ***p<0.001)', 
             pad=20, fontsize=14)
    plt.xlabel('Prevalence (%)', labelpad=10)
    plt.ylabel('Assessment Criteria', labelpad=15)
    plt.xlim(0, 125)
    plt.legend(bbox_to_anchor=(1.02, 1), borderaxespad=0)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    
    return plt.gcf()

def generate_measurement_summary(analysis_results):
    """Generate structured clinical insights from analysis results with enhanced interpretation."""
    
    summary = ["## Comprehensive Malnutrition Assessment Summary\n"]
    
    # 1. Threshold Analysis Section
    if 'thresholds' in analysis_results:
        summary.append("### Key Biomarker Performance\n")
        for measure, data in analysis_results['thresholds'].items():
            if 'error' in data: continue
            
            summary.append(
                f"**{measure.replace('_', ' ').title()}**\n"
                f"- AUC: {data['auc']:.2f} (95% CI {data['auc_CI'][0]:.2f}-{data['auc_CI'][1]:.2f})\n"
                f"- Optimal Threshold: {data['optimal_threshold']:.1f} "
                f"(Sens: {data['sensitivity']:.1%}, Spec: {data['specificity']:.1%})\n"
                f"- Clinical Impact: Cohen's d = {data['cohen_d']:.2f} "
                f"(Malnourished: {data['group_means']['malnourished']:.1f} vs "
                f"Non-Malnourished: {data['group_means']['non_malnourished']:.1f})\n"
            )

    # 2. Criteria Alignment Section
    if 'alignment' in analysis_results and not analysis_results['alignment']['alignment_stats'].empty:
        summary.append("\n### Measurement-Criteria Concordance\n")
        alignment_stats = analysis_results['alignment']['alignment_stats']
        
        for _, row in alignment_stats.iterrows():
            try:  # Handle multi-index columns
                aligned_rate = row[('aligned', 'mean')]
                measure = row['measure']
                criteria = row['criteria_category']
                mal_prev = row[('malnutrition_status', 'mean')]
            except KeyError:
                continue
                
            summary.append(
                f"**{measure} ↔ {criteria.replace('_', ' ')}**\n"
                f"- Alignment Rate: {aligned_rate:.1%}\n"
                f"- Criteria Activation Frequency: {row[('criteria_selected', 'mean')]:.1%}\n"
                f"- Malnutrition Prevalence: {mal_prev:.1%}\n"
            )

    # 3. Severity Classification Section
    if 'severity' in analysis_results and not analysis_results['severity'].empty:
        summary.append("\n### Severity Classification Metrics\n")
        for _, row in analysis_results['severity'].iterrows():
            summary.append(
                f"**{row['severity'].replace('_', ' ').title()}**\n"
                f"- F1: {row['f1_score']:.2f} | Precision: {row['precision']:.2f} | Recall: {row['sensitivity']:.2f}\n"
                f"- Specificity: {row['specificity']:.2f} | NPV: {row['npv']:.2f}\n"
                f"- Likelihood Ratios: +{row['positive_lr']:.1f}/-{row['negative_lr']:.1f}\n"
            )

    # 4. Key Clinical Interpretations
    summary.append("\n### Clinical Implications\n")
    summary.append("- AUC >0.75 indicates strong predictive value for malnutrition screening")
    summary.append("- Alignment rates <65% suggest documentation improvement opportunities")
    summary.append("- F1 scores <0.4 indicate need for classification system review")
    
    return "\n".join(summary)

def preprocess_text(text):
    """
    Preprocess text for NLP analysis with improved handling of edge cases.

    Args:
        text (str): Input text to preprocess

    Returns:
        str: Preprocessed text with stopwords removed and lemmatization applied
    """
    if pd.isna(text) or not isinstance(text, str):
        return ""

    # Convert to lowercase
    text = text.lower()

    # Tokenize
    tokens = word_tokenize(text)

    # Remove stopwords and punctuation
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words 
              and len(token) > 2 and token.isalpha()]

    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]

    return " ".join(tokens)

def analyze_topics(explanations, malnutrition_status, n_topics=5, n_top_words=10, random_state=42):
    """
    Perform topic modeling on explanations with improved error handling and visualization.

    Args:
        explanations (pd.Series): Series of explanation texts
        malnutrition_status (pd.Series): Series with malnutrition decisions
        n_topics (int): Number of topics to extract
        n_top_words (int): Number of top words to display per topic
        random_state (int): Random seed for reproducibility

    Returns:
        dict: Dictionary with topic modeling results
    """
    # Handle empty inputs
    if explanations.empty or explanations.isna().all():
        return {
            'topics': {},
            'doc_topic_matrix': np.array([]),
            'dominant_topics': np.array([]),
            'prevalence_by_status': pd.DataFrame()
        }

    # Preprocess texts
    preprocessed_texts = explanations.apply(preprocess_text)
    
    # Remove empty texts
    valid_indices = preprocessed_texts[preprocessed_texts != ""].index
    if len(valid_indices) == 0:
        return {
            'topics': {},
            'doc_topic_matrix': np.array([]),
            'dominant_topics': np.array([]),
            'prevalence_by_status': pd.DataFrame()
        }
    
    filtered_texts = preprocessed_texts.loc[valid_indices]
    filtered_status = malnutrition_status.loc[valid_indices]

    # Adjust n_topics if we have few documents
    if len(filtered_texts) < n_topics * 2:
        n_topics = max(2, len(filtered_texts) // 2)

    # Vectorize
    try:
        vectorizer = CountVectorizer(max_features=500, min_df=2)
        X = vectorizer.fit_transform(filtered_texts)
        feature_names = vectorizer.get_feature_names_out()
        
        # Check if we have enough features
        if X.shape[1] < n_topics:
            n_topics = max(2, X.shape[1] - 1)
        
        # Run LDA
        lda = LatentDirichletAllocation(
            n_components=n_topics, 
            random_state=random_state,
            max_iter=50,
            learning_method='online'
        )
        lda.fit(X)
        
        # Get topic-term matrix
        topic_term_matrix = lda.components_
        
        # Get document-topic matrix
        doc_topic_matrix = lda.transform(X)
        
        # Extract top words for each topic
        topics = {}
        for topic_idx, topic in enumerate(topic_term_matrix):
            top_indices = topic.argsort()[:-n_top_words-1:-1]
            top_words = [feature_names[i] for i in top_indices]
            topics[topic_idx] = top_words
        
        # Get dominant topic for each document
        dominant_topics = doc_topic_matrix.argmax(axis=1)
        
        # Calculate topic prevalence by malnutrition status
        topic_prevalence = pd.DataFrame({
            'document_id': range(len(dominant_topics)),
            'dominant_topic': dominant_topics,
            'malnutrition': filtered_status.reset_index(drop=True)
        })
        
        prevalence_by_status = topic_prevalence.groupby(['dominant_topic', 'malnutrition']).size().unstack().fillna(0)
        prevalence_by_status = prevalence_by_status.div(prevalence_by_status.sum(axis=0), axis=1)
        
        return {
            'topics': topics,
            'doc_topic_matrix': doc_topic_matrix,
            'dominant_topics': dominant_topics,
            'prevalence_by_status': prevalence_by_status
        }
    
    except Exception as e:
        warnings.warn(f"Topic modeling failed: {str(e)}")
        return {
            'topics': {},
            'doc_topic_matrix': np.array([]),
            'dominant_topics': np.array([]),
            'prevalence_by_status': pd.DataFrame(),
            'error': str(e)
        }

def plot_topic_analysis(topic_results, output_dir=None, figsize_multiplier=1.0):
    """
    Plot topic modeling results with improved visualization.

    Args:
        topic_results (dict): Dictionary of topic modeling results from analyze_topics()
        output_dir (str, optional): Directory to save output files
        figsize_multiplier (float): Multiplier for figure size to adjust for different display needs

    Returns:
        dict: Dictionary of figure objects for each plot
    """
    figures = {}
    
    # Check if topic results are empty
    if not topic_results.get('topics', {}):
        return figures
    
    # Create output directory if specified
    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Plot top words for each topic
    topics = topic_results['topics']
    n_topics = len(topics)
    
    if n_topics > 0:
        fig, axes = plt.subplots(nrows=n_topics, figsize=(12 * figsize_multiplier, 3 * n_topics * figsize_multiplier))
        
        axes = [axes] if n_topics == 1 else axes
        
        for i, (topic_idx, top_words) in enumerate(topics.items()):
            ax = axes[i]
            y_pos = range(len(top_words))
            importance = [len(top_words) - j for j in range(len(top_words))]
            
            # Use sns for better styling
            bars = ax.barh(y_pos, importance, align='center', color=sns.color_palette("husl", n_topics)[i])
            ax.set_yticks(y_pos)
            ax.set_yticklabels(top_words)
            ax.invert_yaxis()
            ax.set_title(f'Topic {topic_idx}', fontweight='bold')
            ax.set_xlabel('Relative Importance')
            
            # Add value labels
            for bar in bars:
                width = bar.get_width()
                ax.text(width + 0.1, bar.get_y() + bar.get_height()/2, 
                        f'{width:.1f}', ha='left', va='center')
        
        plt.tight_layout()
        if output_dir:
            plt.savefig(f"{output_dir}/topic_top_words.png", dpi=300, bbox_inches='tight')
        
        figures['top_words'] = fig
    
    # Plot topic prevalence by malnutrition status if data exists
    prevalence = topic_results.get('prevalence_by_status')
    if prevalence is not None and not prevalence.empty:
        fig, ax = plt.subplots(figsize=(10 * figsize_multiplier, 6 * figsize_multiplier))
        
        # Use a better color palette
        prevalence.plot(kind='bar', stacked=False, ax=ax, color=sns.color_palette("Set2"))
        
        ax.set_title('Topic Prevalence by Malnutrition Status', fontsize=16, fontweight='bold')
        ax.set_xlabel('Topic', fontsize=14)
        ax.set_ylabel('Proportion of Documents', fontsize=14)
        plt.xticks(rotation=0)
        
        # Add a grid for better readability
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add legend with better positioning
        leg = ax.legend(title='Malnutrition Status', bbox_to_anchor=(1.05, 1), loc='upper left')
        leg.set_title('Malnutrition Status', prop={'weight':'bold'})
        
        plt.tight_layout()
        
        if output_dir:
            plt.savefig(f"{output_dir}/topic_prevalence_by_status.png", dpi=300, bbox_inches='tight')
        
        figures['prevalence'] = fig
    
    return figures

def create_keyword_network(explanations, malnutrition_status, min_freq=5, max_keywords=30, clinical_stopwords=None):
    """
    Create a network visualization of co-occurring keywords in explanations with improved filtering.

    Args:
        explanations (pd.Series): Series of explanation texts
        malnutrition_status (pd.Series): Series with malnutrition decisions
        min_freq (int): Minimum frequency for a keyword to be included
        max_keywords (int): Maximum number of keywords to include
        clinical_stopwords (set, optional): Set of clinical terms to exclude from analysis

    Returns:
        networkx.Graph: NetworkX graph of keyword co-occurrences
    """
    # Handle empty inputs
    if explanations.empty or explanations.isna().all():
        return nx.Graph()

    # Default clinical stopwords if not provided
    if clinical_stopwords is None:
        clinical_stopwords = {
            'patient', 'diagnosis', 'medical', 'clinical', 'hospital',
            'assessment', 'history', 'present', 'condition', 'status',
            'note', 'report', 'examination', 'chart', 'record',
            'documented', 'documentation', 'reported', 'states', 'noted'
        }

    # Preprocess texts
    preprocessed_texts = explanations.apply(preprocess_text)
    
    # Skip empty texts
    preprocessed_texts = preprocessed_texts[preprocessed_texts != ""]
    if preprocessed_texts.empty:
        return nx.Graph()
    
    # Get corresponding malnutrition status
    filtered_status = malnutrition_status.loc[preprocessed_texts.index]

    # Tokenize
    all_tokens = []
    for text in preprocessed_texts:
        all_tokens.extend(text.split())

    # Count token frequencies
    token_counts = Counter(all_tokens)

    # Get top keywords
    top_keywords = [w for w, c in token_counts.most_common(100)
                   if c >= min_freq and w not in clinical_stopwords][:max_keywords]
    
    if not top_keywords:
        return nx.Graph()

    # Create co-occurrence matrix
    co_occurrence = np.zeros((len(top_keywords), len(top_keywords)))
    keyword_to_idx = {word: i for i, word in enumerate(top_keywords)}

    # Count co-occurrences with optimized approach
    for text in preprocessed_texts:
        tokens = set(text.split())
        relevant_tokens = [t for t in tokens if t in keyword_to_idx]
        
        for i, word1 in enumerate(relevant_tokens):
            idx1 = keyword_to_idx[word1]
            for word2 in relevant_tokens[i+1:]:  # Only process unique pairs
                idx2 = keyword_to_idx[word2]
                co_occurrence[idx1, idx2] += 1
                co_occurrence[idx2, idx1] += 1  # Symmetric

    # Create network
    G = nx.Graph()

    # Add nodes with more attributes
    for i, word in enumerate(top_keywords):
        # Count occurrences in malnourished vs. non-malnourished
        word_texts = [j for j, text in enumerate(preprocessed_texts) if word in text.split()]
        doc_indices = preprocessed_texts.index[word_texts]
        
        malnourished_count = sum(1 for idx in doc_indices if filtered_status.loc[idx] == 'yes')
        total_count = len(doc_indices)
        
        malnutrition_ratio = malnourished_count / total_count if total_count > 0 else 0
        
        G.add_node(
            word, 
            count=token_counts[word],
            malnutrition_ratio=malnutrition_ratio,
            malnutrition_count=malnourished_count,
            non_malnutrition_count=total_count - malnourished_count
        )

    # Add edges with pruning for readability
    edge_weights = []
    for i in range(len(top_keywords)):
        for j in range(i+1, len(top_keywords)):
            weight = co_occurrence[i, j]
            if weight >= min_freq:
                G.add_edge(top_keywords[i], top_keywords[j], weight=weight)
                edge_weights.append(weight)
    
    # Prune weak connections if there are too many edges
    if len(G.edges) > 50:  # Arbitrary threshold for readability
        median_weight = np.median(edge_weights)
        edges_to_remove = [(u, v) for u, v, d in G.edges(data=True) if d['weight'] < median_weight]
        G.remove_edges_from(edges_to_remove)
    
    return G

def plot_keyword_network(G, output_dir=None, figsize=(14, 10)):
    """
    Plot keyword co-occurrence network with improved visualization.

    Args:
        G (networkx.Graph): NetworkX graph of keyword co-occurrences
        output_dir (str, optional): Directory to save output files
        figsize (tuple): Figure size

    Returns:
        matplotlib.figure.Figure: Figure object for keyword network visualization
    """
    if not G or G.number_of_nodes() == 0:
        return None
    
    # Create output directory if specified
    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Set node colors based on malnutrition ratio with better color mapping
    node_colors = [plt.cm.RdYlBu_r(G.nodes[node]['malnutrition_ratio']) for node in G.nodes()]
    
    # Set node sizes based on frequency with better scaling
    counts = [G.nodes[node]['count'] for node in G.nodes()]
    min_count, max_count = min(counts), max(counts)
    size_range = (50, 500)  # Min and max node sizes
    
    # Scale node sizes with sqrt for better visual representation
    if max_count > min_count:
        node_sizes = [
            size_range[0] + (size_range[1] - size_range[0]) * 
            np.sqrt((G.nodes[node]['count'] - min_count) / (max_count - min_count))
            for node in G.nodes()
        ]
    else:
        node_sizes = [size_range[0] for _ in G.nodes()]
    
    # Set edge weights based on co-occurrence with better scaling
    edge_weights = [0.1 + 2.0 * G[u][v]['weight'] / max([G[a][b]['weight'] for a, b in G.edges()]) 
                   for u, v in G.edges()]
    
    # Position nodes using force-directed layout with better parameters
    pos = nx.spring_layout(G, k=0.4, iterations=100, seed=42)
    
    # Draw the network with improved aesthetics
    edges = nx.draw_networkx_edges(
        G, pos, width=edge_weights, alpha=0.3, 
        edge_color='gray', style='solid', ax=ax
    )
    
    nodes = nx.draw_networkx_nodes(
        G, pos, node_color=node_colors, 
        node_size=node_sizes, alpha=0.9, 
        edgecolors='white', linewidths=1, ax=ax
    )
    
    # Add labels with better positioning and formatting
    label_pos = {k: (v[0], v[1] + 0.01) for k, v in pos.items()}
    nx.draw_networkx_labels(
        G, label_pos, font_size=9, 
        font_weight='bold', font_family='sans-serif',
        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1),
        ax=ax
    )
    
    # Add colorbar with better formatting
    sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlBu_r, norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.8)
    cbar.set_label('Association with Malnutrition', rotation=270, labelpad=20, fontweight='bold')
    
    # Add legend for node sizes
    size_legend_values = [min_count, (min_count + max_count) // 2, max_count]
    size_legend_sizes = [
        size_range[0], 
        (size_range[0] + size_range[1]) / 2, 
        size_range[1]
    ]
    
    # Create separate legend for sizes
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', 
                  label=f'Count: {val}',
                  markerfacecolor='gray', 
                  markersize=np.sqrt(size/5))  # Scale down for legend
        for val, size in zip(size_legend_values, size_legend_sizes)
    ]
    ax.legend(handles=legend_elements, loc='upper left', title='Keyword Frequency')
    
    ax.axis('off')
    ax.set_title('Keyword Co-occurrence Network in Malnutrition Explanations', 
                fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    
    if output_dir:
        plt.savefig(f"{output_dir}/keyword_network.png", dpi=300, bbox_inches='tight')
    
    return fig

def analyze_explanation_sentiment(explanations, malnutrition_status):
    """
    Analyze sentiment in explanation texts by malnutrition status with improved error handling.

    Args:
        explanations (pd.Series): Series of explanation texts
        malnutrition_status (pd.Series): Series with malnutrition decisions

    Returns:
        dict: Dictionary with sentiment analysis results
    """
    # Handle empty inputs
    if explanations.empty or explanations.isna().all():
        return {
            'sentiment_df': pd.DataFrame(),
            'sentiment_stats': pd.DataFrame(),
            't_stat': None,
            'p_value': None
        }
    
    # Calculate sentiment for each explanation
    sentiment_scores = []
    for exp in explanations:
        try:
            if pd.isna(exp) or not isinstance(exp, str) or exp.strip() == "":
                sentiment_scores.append(np.nan)
                continue
            
            blob = TextBlob(exp)
            sentiment_scores.append(blob.sentiment.polarity)
        except Exception:
            sentiment_scores.append(np.nan)
    
    # Create DataFrame with sentiment and malnutrition status
    sentiment_df = pd.DataFrame({
        'sentiment': sentiment_scores,
        'malnutrition': malnutrition_status.reset_index(drop=True)
    })
    
    # Remove NaN values
    sentiment_df = sentiment_df.dropna()
    
    if sentiment_df.empty or len(sentiment_df['malnutrition'].unique()) < 2:
        return {
            'sentiment_df': sentiment_df,
            'sentiment_stats': pd.DataFrame(),
            't_stat': None,
            'p_value': None
        }
    
    # Calculate statistics by malnutrition status
    sentiment_stats = sentiment_df.groupby('malnutrition')['sentiment'].agg([
        'mean', 'median', 'std', 'count', 
        lambda x: x.quantile(0.25), 
        lambda x: x.quantile(0.75)
    ])
    sentiment_stats.rename(columns={
        '<lambda_0>': 'q25',
        '<lambda_1>': 'q75'
    }, inplace=True)
    
    # Add effect size (Cohen's d)
    try:
        mal_values = sentiment_df[sentiment_df['malnutrition'] == 'yes']['sentiment']
        non_mal_values = sentiment_df[sentiment_df['malnutrition'] == 'no']['sentiment']
        
        # Calculate pooled standard deviation
        n1, n2 = len(mal_values), len(non_mal_values)
        s1, s2 = mal_values.std(), non_mal_values.std()
        pooled_std = np.sqrt(((n1-1)*s1**2 + (n2-1)*s2**2) / (n1+n2-2))
        
        # Calculate Cohen's d
        d = (mal_values.mean() - non_mal_values.mean()) / pooled_std if pooled_std != 0 else np.nan
        
        # T-test between groups
        t_stat, p_value = stats.ttest_ind(mal_values, non_mal_values, equal_var=False)
        
        effect_size = pd.DataFrame({
            'cohen_d': [d, d],
            't_stat': [t_stat, t_stat],
            'p_value': [p_value, p_value]
        }, index=['yes', 'no'])
        
        sentiment_stats = pd.concat([sentiment_stats, effect_size], axis=1)
    except Exception as e:
        warnings.warn(f"Error calculating effect size: {str(e)}")
        t_stat, p_value = None, None
    
    return {
        'sentiment_df': sentiment_df,
        'sentiment_stats': sentiment_stats,
        't_stat': t_stat,
        'p_value': p_value
    }

def plot_sentiment_analysis(sentiment_results, output_dir=None):
    """
    Plot sentiment analysis results with enhanced visualizations.

    Args:
        sentiment_results (dict): Dictionary with sentiment analysis results
        output_dir (str, optional): Directory to save output files

    Returns:
        dict: Dictionary of figure objects for each plot
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    from pathlib import Path
    import numpy as np
    
    figures = {}
    
    sentiment_df = sentiment_results.get('sentiment_df')
    if sentiment_df is None or sentiment_df.empty:
        return figures
    
    # Create output directory if specified
    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Check which values are in the 'malnutrition' column
    unique_values = sentiment_df['malnutrition'].unique()
    
    # Determine if we have numeric or string categories
    if all(val in ['0', '1'] or val in [0, 1] for val in unique_values):
        # Convert numeric values to strings if needed
        sentiment_df['malnutrition'] = sentiment_df['malnutrition'].astype(str)
        # Map numeric values to descriptive strings
        sentiment_df['malnutrition_label'] = sentiment_df['malnutrition'].map({'0': 'no', '1': 'yes'})
        # Set a consistent color palette
        palette = {"yes": "#E63946", "no": "#457B9D"}
        # Use the label for visualization
        hue_col = 'malnutrition_label'
    else:
        # Assume we already have string values
        hue_col = 'malnutrition'
        # Set a consistent color palette
        palette = {"yes": "#E63946", "no": "#457B9D"}
    
    # Distribution plot with improved aesthetics
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Use violin plot with explicit hue parameter
    sns.violinplot(
        data=sentiment_df, x=hue_col, y='sentiment',
        hue=hue_col, inner='quartile', palette=palette, ax=ax,
        legend=False  # Avoid duplicate legend
    )
    
    # Add individual data points with jitter
    sns.stripplot(
        data=sentiment_df, x=hue_col, y='sentiment',
        hue=hue_col, size=4, alpha=0.3, jitter=True, 
        palette=palette, ax=ax, legend=False
    )
    
    # Add mean markers
    for i, status in enumerate(['yes', 'no']):
        status_value = status
        if hue_col == 'malnutrition_label':
            # Get the corresponding data
            group_data = sentiment_df[sentiment_df['malnutrition_label'] == status]
        else:
            # Get the corresponding data
            group_data = sentiment_df[sentiment_df['malnutrition'] == status]
        
        if not group_data.empty:
            mean_val = group_data['sentiment'].mean()
            ax.scatter(
                i, mean_val,
                marker='o', s=100, color='white', edgecolor='black', zorder=3,
                label=f'Mean ({status}): {mean_val:.3f}'
            )
    
    # Add p-value annotation if available
    p_value = sentiment_results.get('p_value')
    if p_value is not None:
        p_text = f"p = {p_value:.4f}" if p_value >= 0.0001 else "p < 0.0001"
        significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
        
        y_max = sentiment_df['sentiment'].max()
        ax.annotate(
            f"{p_text} {significance}", 
            xy=(0.5, y_max + 0.05), 
            xycoords='data',
            horizontalalignment='center', 
            fontsize=12,
            fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
        )
    
    # Improve axis labels and styling
    ax.set_title('Sentiment Comparison by Malnutrition Status', fontsize=16, fontweight='bold')
    ax.set_xlabel('Malnutrition Status', fontsize=14)
    ax.set_ylabel('Sentiment Score (-1: Negative, +1: Positive)', fontsize=14)
    
    # Add reference lines
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    # Add gridlines for better readability
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    
    # Add legend
    ax.legend(title="", loc="upper right")
    
    plt.tight_layout()
    
    if output_dir:
        plt.savefig(f"{output_dir}/sentiment_comparison.png", dpi=300, bbox_inches='tight')
    
    figures['comparison'] = fig
    
    # Create scatter plot of sentiment vs. length
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    
    # Add text length as a feature
    sentiment_df_with_length = sentiment_df.copy()
    
    # Fix the explanations variable reference - assuming it's from sentiment_results
    explanations = sentiment_results.get('explanations', sentiment_df['explanation'] if 'explanation' in sentiment_df.columns else None)
    
    if explanations is not None:
        sentiment_df_with_length['text_length'] = explanations.apply(
            lambda x: len(x) if isinstance(x, str) else 0
        ).reset_index(drop=True)
        
        # Create scatter plot
        sns.scatterplot(
            data=sentiment_df_with_length, 
            x='text_length', 
            y='sentiment',
            hue=hue_col,
            palette=palette,
            alpha=0.7,
            s=80,
            ax=ax2
        )
        
        # Add trend lines
        for i, status in enumerate(['yes', 'no']):
            if hue_col == 'malnutrition_label':
                group_data = sentiment_df_with_length[sentiment_df_with_length['malnutrition_label'] == status]
            else:
                group_data = sentiment_df_with_length[sentiment_df_with_length['malnutrition'] == status]
                
            if len(group_data) > 1:  # Need at least 2 points for regression
                sns.regplot(
                    x='text_length',
                    y='sentiment',
                    data=group_data,
                    scatter=False,
                    color=palette[status],
                    line_kws={'linewidth': 2},
                    ax=ax2
                )
        
        # Add statistics
        ax2.set_title('Sentiment vs. Text Length by Malnutrition Status', fontsize=16, fontweight='bold')
        ax2.set_xlabel('Text Length (characters)', fontsize=14)
        ax2.set_ylabel('Sentiment Score', fontsize=14)
        ax2.grid(alpha=0.3, linestyle='--')
        
        if output_dir:
            plt.savefig(f"{output_dir}/sentiment_vs_length.png", dpi=300, bbox_inches='tight')
        
        figures['length_correlation'] = fig2
    
    # Add a word cloud visualization if we have text data
    if explanations is not None:
        try:
            from wordcloud import WordCloud
            import matplotlib.colors as mcolors
            
            # Create word clouds by malnutrition status
            fig3, (ax3, ax4) = plt.subplots(1, 2, figsize=(16, 8))
            
            # Function to generate wordcloud
            def generate_wordcloud(text_series, ax, title, color):
                # Combine all text
                text = ' '.join([str(t) for t in text_series if isinstance(t, str)])
                
                # Create color gradient
                colors = list(mcolors.CSS4_COLORS.values())
                color_map = mcolors.LinearSegmentedColormap.from_list("custom", [color, "#FFFFFF"])
                
                # Generate word cloud
                if text:
                    wordcloud = WordCloud(
                        background_color='white',
                        max_words=100,
                        colormap=color_map,
                        width=800,
                        height=400,
                        contour_width=1,
                        contour_color='gray'
                    ).generate(text)
                    
                    # Display word cloud
                    ax.imshow(wordcloud, interpolation='bilinear')
                    ax.set_title(title, fontsize=16, fontweight='bold')
                    ax.axis('off')
                else:
                    ax.text(0.5, 0.5, "No text data available", 
                            horizontalalignment='center',
                            verticalalignment='center',
                            fontsize=14)
                    ax.axis('off')
            
            # Generate word clouds for each group
            if hue_col == 'malnutrition_label':
                malnutrition_yes = sentiment_df[sentiment_df['malnutrition_label'] == 'yes']
                malnutrition_no = sentiment_df[sentiment_df['malnutrition_label'] == 'no']
            else:
                malnutrition_yes = sentiment_df[sentiment_df['malnutrition'] == 'yes']
                malnutrition_no = sentiment_df[sentiment_df['malnutrition'] == 'no']
            
            # Get explanation text for each group
            if isinstance(explanations, dict):
                yes_texts = explanations.get('yes', [])
                no_texts = explanations.get('no', [])
            else:
                # If explanations is a Series, filter by malnutrition status
                yes_indices = malnutrition_yes.index
                no_indices = malnutrition_no.index
                yes_texts = explanations.iloc[yes_indices] if hasattr(explanations, 'iloc') else []
                no_texts = explanations.iloc[no_indices] if hasattr(explanations, 'iloc') else []
                
            generate_wordcloud(yes_texts, ax3, "Word Cloud - Malnutrition: Yes", palette["yes"])
            generate_wordcloud(no_texts, ax4, "Word Cloud - Malnutrition: No", palette["no"])
            
            plt.tight_layout()
            
            if output_dir:
                plt.savefig(f"{output_dir}/wordclouds.png", dpi=300, bbox_inches='tight')
            
            figures['wordclouds'] = fig3
        except ImportError:
            print("WordCloud package not available. Skipping word cloud visualization.")
    
    # Create sentiment distribution plots
    fig4, ax5 = plt.subplots(figsize=(10, 6))
    
    # Plot sentiment distributions as kernel density estimates
    sns.kdeplot(
        data=sentiment_df, 
        x='sentiment',
        hue=hue_col,
        fill=True,
        common_norm=False,
        palette=palette,
        alpha=0.5,
        linewidth=2,
        ax=ax5
    )
    
    # Add vertical lines for medians
    for i, status in enumerate(['yes', 'no']):
        if hue_col == 'malnutrition_label':
            group_data = sentiment_df[sentiment_df['malnutrition_label'] == status]
        else:
            group_data = sentiment_df[sentiment_df['malnutrition'] == status]
            
        if not group_data.empty:
            median_val = group_data['sentiment'].median()
            ax5.axvline(
                x=median_val,
                color=palette[status],
                linestyle='--',
                linewidth=2,
                label=f'Median ({status}): {median_val:.3f}'
            )
    
    ax5.set_title('Sentiment Distribution by Malnutrition Status', fontsize=16, fontweight='bold')
    ax5.set_xlabel('Sentiment Score', fontsize=14)
    ax5.set_ylabel('Density', fontsize=14)
    ax5.grid(alpha=0.3, linestyle='--')
    ax5.legend(title="", loc="upper right")
    
    if output_dir:
        plt.savefig(f"{output_dir}/sentiment_distribution.png", dpi=300, bbox_inches='tight')
    
    figures['distribution'] = fig4
    
    # Create a combined figure with descriptive stats
    fig5, ax6 = plt.subplots(figsize=(10, 8))
    
    # Calculate descriptive statistics
    stats_data = []
    for status in ['yes', 'no']:
        if hue_col == 'malnutrition_label':
            group_data = sentiment_df[sentiment_df['malnutrition_label'] == status]['sentiment']
        else:
            group_data = sentiment_df[sentiment_df['malnutrition'] == status]['sentiment']
            
        if not group_data.empty:
            stats_data.append({
                'Status': status,
                'Mean': group_data.mean(),
                'Median': group_data.median(),
                'Std Dev': group_data.std(),
                'Min': group_data.min(),
                'Max': group_data.max(),
                'Count': len(group_data)
            })
    
    # Remove axis
    ax6.axis('off')
    
    if stats_data:
        # Create a table with statistics
        table_data = [[f"{stat}" if idx == 0 else f"{value:.3f}" if isinstance(value, float) else f"{value}" 
                      for idx, (stat, value) in enumerate(row.items())] 
                     for row in stats_data]
        
        # Add headers
        table_headers = list(stats_data[0].keys())
        
        # Create table
        table = ax6.table(
            cellText=table_data,
            colLabels=table_headers,
            loc='center',
            cellLoc='center',
            colColours=[palette.get(status, '#F0F0F0') for status in ['yes', 'no', '#F0F0F0', '#F0F0F0', '#F0F0F0', '#F0F0F0', '#F0F0F0']][:len(table_headers)]
        )
        
        # Style the table
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.2, 2)
        
        # Add a title
        plt.suptitle('Descriptive Statistics - Sentiment by Malnutrition Status', 
                    fontsize=16, fontweight='bold', y=0.98)
    else:
        ax6.text(0.5, 0.5, "No data available for statistics", 
                horizontalalignment='center',
                verticalalignment='center',
                fontsize=14)
    
    if output_dir:
        plt.savefig(f"{output_dir}/sentiment_statistics.png", dpi=300, bbox_inches='tight')
    
    figures['statistics'] = fig5
    
    # Handle the SmallSampleWarning by adding a check for minimum sample size
    # This prevents the warning but allows the function to continue
    for status in unique_values:
        group = sentiment_df[sentiment_df['malnutrition'] == status]
        if len(group) < 2:  # Most statistical tests require at least 2 data points
            print(f"Warning: Sample size for malnutrition={status} is too small ({len(group)}). Some statistical visualizations may be limited.")
    
    # Close all figures to prevent display in notebooks if not needed
    if not plt.isinteractive():
        plt.close('all')
    
    return figures
def get_named_entities(explanations):
    """
    Extract medical named entities from explanations.
    
    Args:
        explanations (pd.Series or list): Series or list of explanation texts
        
    Returns:
        dict: Dictionary with entity counts by type
    """
    import spacy
    import pandas as pd
    from collections import Counter
    
    # Define entity category mappings
    entity_categories = {
        'DISEASE': ['DISEASE', 'DISORDER', 'SYNDROME'],
        'CONDITION': ['CONDITION', 'FINDING', 'SYMPTOM', 'SIGN'],
        'TREATMENT': ['TREATMENT', 'PROCEDURE', 'THERAPY'],
        'MEDICATION': ['MEDICATION', 'DRUG', 'SUPPLEMENT'],
        'TEST': ['TEST', 'DIAGNOSTIC', 'EXAMINATION'],
        'MEASUREMENT': ['MEASUREMENT', 'VALUE', 'NUMBER', 'QUANTITY'],
        'PERSON': ['PERSON', 'PATIENT', 'DOCTOR', 'CLINICIAN'],
        'DATE': ['DATE', 'TIME', 'DURATION', 'FREQUENCY'],
        'AGE': ['AGE'],
        'NUTRITION': ['FOOD', 'NUTRIENT', 'DIET'],
        'OTHER': []
    }
    
    # Load appropriate spaCy model with error handling
    try:
        nlp = spacy.load("en_core_sci_md")
    except OSError:
        try:
            nlp = spacy.load("en_core_web_sm")
        except OSError:
            raise ImportError(
                "Neither medical nor standard spaCy models available. "
                "Install with: python -m spacy download en_core_web_sm or pip install scispacy en_core_sci_md"
            )
    
    # Initialize categories dictionary
    categories = {category: [] for category in entity_categories}
    all_entities = []
    
    # Convert to list if pandas Series
    if hasattr(explanations, 'tolist'):
        explanations = explanations.tolist()
    
    # Process each explanation
    for exp in explanations:
        if exp is None or pd.isna(exp) or not isinstance(exp, str):
            continue
            
        # Process with spaCy
        doc = nlp(exp)
        
        # Extract entities
        for ent in doc.ents:
            entity_tuple = (ent.text, ent.label_)
            all_entities.append(entity_tuple)
            
            # Categorize the entity
            categorized = False
            for category, labels in entity_categories.items():
                if ent.label_ in labels:
                    categories[category].append(ent.text)
                    categorized = True
                    break
                    
            # Default to OTHER if not matched
            if not categorized:
                categories['OTHER'].append(ent.text)
    
    # Count entities by category
    category_counts = {cat: Counter(items) for cat, items in categories.items() if items}
    
    return {
        'all_entities': Counter(all_entities),
        'category_counts': category_counts
    }

def plot_entity_analysis(entity_results, top_n=10, figsize=(10, 6), output_dir=None):
    """
    Plot entity analysis results.
    
    Args:
        entity_results (dict): Dictionary with entity analysis results
        top_n (int): Number of top entities to display
        figsize (tuple): Figure size as (width, height)
        output_dir (str, optional): Directory to save output files
        
    Returns:
        dict: Dictionary of figure objects for each plot
    """
    import matplotlib.pyplot as plt
    import os
    
    figures = {}
    
    # Get category counts
    category_counts = entity_results.get('category_counts', {})
    
    # Plot by category
    for category, counts in category_counts.items():
        if not counts:
            continue
            
        # Get top N entities in this category
        top_entities = counts.most_common(top_n)
        if not top_entities:
            continue
            
        # Create plot
        labels, values = zip(*top_entities)
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create horizontal bar chart
        y_pos = range(len(labels))
        bars = ax.barh(y_pos, values, color='skyblue', edgecolor='navy')
        
        # Add data labels to bars
        for i, v in enumerate(values):
            ax.text(v + 0.1, i, str(v), va='center')
        
        # Configure axes
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels)
        ax.invert_yaxis()  # Highest values at the top
        ax.set_title(f'Top {len(labels)} {category} Entities', fontsize=16)
        ax.set_xlabel('Frequency', fontsize=14)
        ax.set_axisbelow(True)
        ax.grid(axis='x', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        
        # Save if output directory specified
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"entities_{category.lower()}.png")
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
        
        figures[category] = fig
    
    return figures
