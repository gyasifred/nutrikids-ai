#!/usr/bin/env python3

import os
import re
import glob
import logging
import joblib
import nltk
import shap
import xgboost as xgb
import numpy as np
import pandas as pd
import networkx as nx
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from scipy.sparse import csr_matrix
from datetime import datetime
from dotenv import load_dotenv, find_dotenv
from collections import Counter
from typing import List, Union, Tuple, Optional
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from textblob import TextBlob
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import (
    average_precision_score, precision_recall_curve, roc_curve, auc,
    confusion_matrix
)

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
    Load the malnutrition dataset and create filtered DataFrames based on prediction correctness.
    
    Args:
        file_path (str): Path to the CSV file containing prediction data.
        
    Returns:
        dict: A dictionary containing:
        - 'full_df': Original dataset
        - 'correct_predictions': DataFrame of correct predictions
        - 'incorrect_predictions': DataFrame of incorrect predictions
        - 'correct_yes': Correctly classified 'yes' (true positives)
        - 'correct_no': Correctly classified 'no' (true negatives)
        - 'incorrect_yes': Incorrectly classified as 'yes' (false positives)
        - 'incorrect_no': Incorrectly classified as 'no' (false negatives)
    """
    # Load dataset
    df = pd.read_csv(file_path)

    # Ensure required columns exist
    required_columns = {'patient_id', 'true_label', 'predicted_label', 'explanation'}
    if not required_columns.issubset(df.columns):
        raise ValueError(f"Missing required columns: {required_columns - set(df.columns)}")

    # Split into correct and incorrect predictions
    correct_predictions = df[df['true_label'] == df['predicted_label']].reset_index(drop=True)
    incorrect_predictions = df[df['true_label'] != df['predicted_label']].reset_index(drop=True)

    # Further breakdown into four groups
    correct_yes = correct_predictions[correct_predictions['true_label'] == "yes"]  # True Positives
    correct_no = correct_predictions[correct_predictions['true_label'] == "no"]    # True Negatives
    incorrect_yes = incorrect_predictions[incorrect_predictions['predicted_label'] == "yes"]  # False Positives
    incorrect_no = incorrect_predictions[incorrect_predictions['predicted_label'] == "no"]    # False Negatives

    return {
        'full_df': df,
        'correct_predictions': correct_predictions,
        'incorrect_predictions': incorrect_predictions,
        'correct_yes': correct_yes,
        'correct_no': correct_no,
        'incorrect_yes': incorrect_yes,
        'incorrect_no': incorrect_no
    }


def extract_criteria_mentions(explanations, criteria_dict):
    """
    Extract mentions of specific criteria categories from explanations.

    Args:
        explanations: Series of explanation texts
        criteria_dict: Dictionary mapping categories to keywords

    Returns:
        DataFrame with binary indicators for each criteria category.
    """
    # Initialize DataFrame for criteria mentions
    criteria_df = pd.DataFrame(0, index=range(len(explanations)),
                               columns=list(criteria_dict.keys()))

    # Extract mentions
    for i, exp in enumerate(explanations):
        if pd.isna(exp):
            continue

        exp_lower = exp.lower()
        for category, keywords in criteria_dict.items():
            if any(keyword.lower() in exp_lower for keyword in keywords):
                criteria_df.loc[i, category] = 1

    return criteria_df

def analyze_criteria_correlation(criteria_df, malnutrition_status):
    """
    Analyze correlation of criteria mentions with malnutrition status.

    Args:
        criteria_df: DataFrame with binary indicators for criteria categories
        malnutrition_status: Series with malnutrition decisions

    Returns:
        DataFrame with correlation metrics.
    """
    # Convert malnutrition status to binary (1 = Yes, 0 = No)
    y = (malnutrition_status == 'yes').astype(int)

    results = pd.DataFrame(columns=['criteria', 'correlation'])

    for criteria in criteria_df.columns:
        correlation = criteria_df[criteria].corr(y)
        results.loc[len(results)] = [criteria, correlation]

    # Sort by absolute correlation value
    results = results.sort_values('correlation', ascending=False, key=abs)

    return results

def analyze_criteria_frequency(criteria_df):
    """
    Analyze the frequency of each criteria mention.

    Args:
        criteria_df: DataFrame with binary indicators for criteria categories.

    Returns:
        DataFrame with frequency of each criteria.
    """
    results = pd.DataFrame(columns=['criteria', 'frequency'])

    for criteria in criteria_df.columns:
        frequency = criteria_df[criteria].mean()
        results.loc[len(results)] = [criteria, frequency]

    results = results.sort_values('frequency', ascending=False)

    return results

def extract_clinical_measurements(explanations):
    """
    Extract clinical measurements and their values from explanations

    Args:
        explanations: Series of explanation texts

    Returns:
        DataFrame with extracted measurements and values
    """
    # Define patterns for common clinical measurements based on prompt information
    patterns = {
        'BMI': r'BMI[\s:]*([-\d\.]+)',
        'weight_for_height': r'weight[- ]for[- ]height[:\s]*([-\d\.]+)[\s]*z[-\s]*score',
        'BMI_for_age': r'BMI[- ]for[- ]age[:\s]*([-\d\.]+)[\s]*z[-\s]*score',
        'MUAC': r'(?:mid[- ]upper arm circumference|MUAC)[:\s]*([-\d\.]+)',
        'albumin': r'(?:serum\s+)?albumin[:\s]*([\d\.]+)',
        'hemoglobin': r'(?:serum\s+)?h(?:a)?emoglobin[:\s]*([\d\.]+)',
        'weight_loss': r'(?:weight loss|lost)[:\s]*([\d\.]+)(?:\s*%)?',
        'length_height_for_age': r'length/height[- ]for[- ]age[:\s]*([-\d\.]+)[\s]*z[-\s]*score'
    }

    # Initialize results
    measurements = []

    for i, exp in enumerate(explanations):
        if pd.isna(exp):
            continue

        row = {'explanation_id': i}

        # Extract measurements
        for measure, pattern in patterns.items():
            match = re.search(pattern, exp, re.IGNORECASE)
            if match:
                try:
                    value = float(match.group(1))
                    row[measure] = value
                except ValueError:
                    pass  # Skip if not a valid number

        # Look for malnutrition severity classification mentions
        severity_patterns = {
            'mild_malnutrition': r'mild malnutrition',
            'moderate_malnutrition': r'moderate malnutrition',
            'severe_malnutrition': r'severe malnutrition'
        }

        for severity, pattern in severity_patterns.items():
            if re.search(pattern, exp, re.IGNORECASE):
                row[severity] = 1
            else:
                row[severity] = 0

        if len(row) > 1:  # Add only if at least one measurement was found
            measurements.append(row)

    # Convert to DataFrame
    measurements_df = pd.DataFrame(measurements)

    return measurements_df

def analyze_measurement_thresholds(measurements_df, malnutrition_status):
    """
    Analyze the thresholds of clinical measurements for malnutrition classification

    Args:
        measurements_df: DataFrame with clinical measurements
        malnutrition_status: Series with malnutrition decisions

    Returns:
        Dictionary with threshold analysis for each measurement
    """
    # Join measurements with malnutrition status
    joined_df = measurements_df.copy()
    joined_df['malnutrition'] = malnutrition_status.iloc[joined_df['explanation_id']].reset_index(drop=True)
    joined_df['malnutrition_binary'] = (joined_df['malnutrition'] == 'yes').astype(int)

    # Initialize results
    thresholds = {}

    # Analyze each measurement
    for measure in measurements_df.columns:
        if measure in ['explanation_id', 'mild_malnutrition', 'moderate_malnutrition',
                      'severe_malnutrition'] or joined_df[measure].count() < 5:
            continue

        # Calculate statistics
        stats = {
            'count': joined_df[measure].count(),
            'mean': joined_df[measure].mean(),
            'median': joined_df[measure].median(),
            'std': joined_df[measure].std(),
            'mean_malnutrition': joined_df[joined_df['malnutrition'] == 'yes'][measure].mean(),
            'mean_non_malnutrition': joined_df[joined_df['malnutrition'] == 'no'][measure].mean(),
            'threshold_candidates': []
        }

        # Calculate t-test for difference between groups
        mal_values = joined_df[joined_df['malnutrition'] == 'yes'][measure].dropna()
        non_mal_values = joined_df[joined_df['malnutrition'] == 'no'][measure].dropna()

        if len(mal_values) > 1 and len(non_mal_values) > 1:
            t_stat, p_value = stats_scipy.ttest_ind(mal_values, non_mal_values, equal_var=False)
            stats['t_test_p_value'] = p_value

        # Find optimal threshold using precision-recall curve
        if joined_df[measure].count() >= 10:
            # For z-scores (negative values indicate worse condition)
            if measure in ['weight_for_height', 'BMI_for_age', 'length_height_for_age', 'MUAC'] and joined_df[measure].min() < 0:
                # For these measures, lower (more negative) is worse
                feature = -joined_df[measure]
            # For measures like BMI, albumin, hemoglobin (higher is better)
            else:
                # Invert so higher values predict non-malnutrition
                feature = -joined_df[measure]

            # Only compute if we have both positive and negative examples
            if joined_df['malnutrition_binary'].nunique() > 1:
                precision, recall, thresholds_pr = precision_recall_curve(
                    joined_df['malnutrition_binary'].fillna(0),
                    feature
                )

                # Calculate F1 score for each threshold
                f1_scores = []
                for p, r in zip(precision, recall):
                    if p + r == 0:
                        f1_scores.append(0)
                    else:
                        f1_scores.append(2 * p * r / (p + r))

                # Find optimal threshold
                if len(thresholds_pr) > 0:
                    best_idx = np.argmax(f1_scores[:-1]) 
                    if best_idx < len(thresholds_pr):
                        optimal_threshold = -thresholds_pr[best_idx] 
                        stats['optimal_threshold'] = optimal_threshold
                        stats['optimal_f1'] = f1_scores[best_idx]
                        stats['optimal_precision'] = precision[best_idx]
                        stats['optimal_recall'] = recall[best_idx]

                # Calculate ROC AUC
                fpr, tpr, _ = roc_curve(
                    joined_df['malnutrition_binary'].fillna(0),
                    feature
                )
                stats['roc_auc'] = auc(fpr, tpr)

        thresholds[measure] = stats

    return thresholds

def plot_measurement_distributions(measurements_df, malnutrition_status):
    """
    Plot distributions of clinical measurements by malnutrition status

    Args:
        measurements_df: DataFrame with clinical measurements
        malnutrition_status: Series with malnutrition decisions

    Returns:
        Dictionary of figure objects
    """
    # Join measurements with malnutrition status
    joined_df = measurements_df.copy()
    joined_df['malnutrition'] = malnutrition_status.iloc[joined_df['explanation_id']].reset_index(drop=True)

    figures = {}

    # Create plots for each measurement
    for measure in measurements_df.columns:
        if measure in ['explanation_id', 'mild_malnutrition', 'moderate_malnutrition',
                      'severe_malnutrition'] or joined_df[measure].count() < 5:
            continue

        fig, ax = plt.subplots(figsize=(10, 6))

        # Create KDE plot
        sns.kdeplot(data=joined_df, x=measure, hue='malnutrition', fill=True, common_norm=False, ax=ax)

        # Add vertical lines for group means
        yes_mean = joined_df[joined_df['malnutrition'] == 'yes'][measure].mean()
        no_mean = joined_df[joined_df['malnutrition'] == 'no'][measure].mean()

        ax.axvline(yes_mean, color='blue', linestyle='--',
                  label=f'Mean (Malnourished): {yes_mean:.2f}')
        ax.axvline(no_mean, color='orange', linestyle='--',
                  label=f'Mean (Not Malnourished): {no_mean:.2f}')

        # Add reference lines for standard thresholds based on classification table
        if measure == 'weight_for_height' or measure == 'BMI_for_age':
            ax.axvline(-1.0, color='green', linestyle=':',
                      label='Mild Threshold (z = -1)')
            ax.axvline(-2.0, color='yellow', linestyle=':',
                      label='Moderate Threshold (z = -2)')
            ax.axvline(-3.0, color='red', linestyle=':',
                      label='Severe Threshold (z = -3)')
        elif measure == 'BMI' and joined_df[measure].mean() > 10:  # Adult BMI scale
            ax.axvline(18.5, color='green', linestyle=':',
                      label='Underweight Threshold (BMI = 18.5)')
            ax.axvline(16.0, color='red', linestyle=':',
                      label='Severe Underweight (BMI = 16.0)')

        ax.set_title(f'Distribution of {measure} by Malnutrition Status', fontsize=14)
        ax.set_xlabel(measure, fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.legend(title='Status')

        fig.tight_layout()
        figures[measure] = fig

    return figures

def analyze_severity_classifications(measurements_df, malnutrition_status):
    """
    Analyze severity classifications mentioned in explanations

    Args:
        measurements_df: DataFrame with severity classifications
        malnutrition_status: Series with malnutrition decisions

    Returns:
        DataFrame with analysis of severity mentions
    """
    # Join with malnutrition status
    joined_df = measurements_df.copy()
    joined_df['malnutrition'] = malnutrition_status.iloc[joined_df['explanation_id']].reset_index(drop=True)

    severity_cols = ['mild_malnutrition', 'moderate_malnutrition', 'severe_malnutrition']

    # Skip if no severity classifications found
    if not all(col in joined_df.columns for col in severity_cols):
        return pd.DataFrame()

    # Calculate overall statistics
    results = []

    for severity in severity_cols:
        if severity not in joined_df.columns:
            continue

        # Overall frequency
        frequency = joined_df[severity].mean()

        # Frequency in actual malnutrition cases
        freq_in_yes = joined_df[joined_df['malnutrition'] == 'yes'][severity].mean()

        # Frequency in non-malnutrition cases
        freq_in_no = joined_df[joined_df['malnutrition'] == 'no'][severity].mean()

        # Precision (when mentioned, how often is it correct?)
        true_positives = ((joined_df[severity] == 1) & (joined_df['malnutrition'] == 'yes')).sum()
        false_positives = ((joined_df[severity] == 1) & (joined_df['malnutrition'] == 'no')).sum()
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0

        results.append({
            'severity': severity.replace('_', ' ').title(),
            'frequency': frequency,
            'frequency_in_malnutrition': freq_in_yes,
            'frequency_in_non_malnutrition': freq_in_no,
            'precision': precision
        })

    return pd.DataFrame(results)

def analyze_measurement_criteria_alignment(measurements_df, criteria_df, malnutrition_status):
    """
    Analyze alignment between extracted measurements and criteria mentions

    Args:
        measurements_df: DataFrame with clinical measurements
        criteria_df: DataFrame with binary indicators for criteria categories
        malnutrition_status: Series with malnutrition decisions

    Returns:
        DataFrame with alignment analysis
    """
    # Map between measurement columns and criteria columns
    criteria_map = {
        'BMI': 'BMI',
        'weight_for_height': 'weight_for_height',
        'BMI_for_age': 'BMI_for_age',
        'MUAC': 'MUAC',
        'weight_loss': 'weight_loss',
        'albumin': 'lab_markers',
        'hemoglobin': 'lab_markers'
    }

    # Join explanations with malnutrition status
    results = []

    for measure, criteria in criteria_map.items():
        if measure not in measurements_df.columns or criteria not in criteria_df.columns:
            continue

        # Create indicator for having a measurement
        has_measurement = measurements_df['explanation_id'].map(
            measurements_df.set_index('explanation_id')[measure].notna()
        ).fillna(False)

        # Get corresponding criteria mentions
        has_criteria = criteria_df[criteria]

        # Calculate agreement
        agreement = (has_measurement == has_criteria).mean()

        # Calculate precision when measurement exists
        measurement_ids = measurements_df[measurements_df[measure].notna()]['explanation_id']
        predictions = malnutrition_status.iloc[measurement_ids]
        accuracy = (predictions == 'yes').mean() if len(predictions) > 0 else 0

        # Calculate precision when criteria is mentioned but no measurement
        criteria_only_mask = (has_criteria == 1) & (has_measurement == 0)
        criteria_only_ids = criteria_only_mask[criteria_only_mask].index
        criteria_only_predictions = malnutrition_status.iloc[criteria_only_ids]
        criteria_only_accuracy = (criteria_only_predictions == 'yes').mean() if len(criteria_only_predictions) > 0 else 0

        results.append({
            'measure': measure,
            'criteria': criteria,
            'measurement_count': has_measurement.sum(),
            'criteria_mention_count': has_criteria.sum(),
            'agreement': agreement,
            'measurement_precision': accuracy,
            'criteria_only_precision': criteria_only_accuracy
        })

    return pd.DataFrame(results)

def visualize_criteria_frequency(frequency_results):
    """
    Create visualization of criteria frequency across different groups

    Args:
        frequency_results: Dictionary of DataFrames with frequency results

    Returns:
        Figure object
    """
    # Prepare data for plotting
    plot_data = []

    for group, df in frequency_results.items():
        for _, row in df.iterrows():
            plot_data.append({
                'group': group,
                'criteria': row['criteria'],
                'frequency': row['frequency']
            })

    plot_df = pd.DataFrame(plot_data)

    # Select top criteria by average frequency
    top_criteria = plot_df.groupby('criteria')['frequency'].mean().nlargest(8).index.tolist()
    plot_df = plot_df[plot_df['criteria'].isin(top_criteria)]

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))

    # Create grouped bar chart
    sns.barplot(data=plot_df, x='criteria', y='frequency', hue='group', ax=ax)

    ax.set_title('Frequency of Criteria Mentions by Prediction Group', fontsize=16)
    ax.set_xlabel('Criteria', fontsize=14)
    ax.set_ylabel('Frequency', fontsize=14)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.legend(title='Group')

    fig.tight_layout()

    return fig

def generate_measurement_summary(measurement_results, threshold_results, alignment_results):
    """Generate a summary of clinical measurement analysis"""
    summary = []

    # Count measurements extracted
    measurement_counts = {key: len(df) for key, df in measurement_results.items()}
    summary.append(f"Total measurements extracted: {measurement_counts['full_df']}")
    summary.append(f"Measurements in correct predictions: {measurement_counts['correct_predictions']}")
    summary.append(f"Measurements in incorrect predictions: {measurement_counts['incorrect_predictions']}")

    # Summarize threshold analysis
    summary.append("\nMeasurement Threshold Analysis:")
    for key in ['correct_predictions']:
        if not threshold_results[key]:
            continue

        summary.append(f"\nAnalysis for {key}:")
        for measure, stats in threshold_results[key].items():
            if 'optimal_threshold' in stats:
                summary.append(f"  {measure}:")
                summary.append(f"    Mean in malnutrition: {stats['mean_malnutrition']:.2f}")
                summary.append(f"    Mean in non-malnutrition: {stats['mean_non_malnutrition']:.2f}")
                if 't_test_p_value' in stats:
                    summary.append(f"    t-test p-value: {stats['t_test_p_value']:.4f}")
                summary.append(f"    Optimal threshold: {stats['optimal_threshold']:.2f}")
                summary.append(f"    F1 score at optimal threshold: {stats['optimal_f1']:.2f}")
                if 'roc_auc' in stats:
                    summary.append(f"    ROC AUC: {stats['roc_auc']:.2f}")

    # Summarize alignment analysis
    if any(len(df) > 0 for df in alignment_results.values()):
        summary.append("\nAlignment between Measurements and Criteria Mentions:")
        for key, df in alignment_results.items():
            if len(df) == 0:
                continue

            summary.append(f"\nAnalysis for {key}:")
            for _, row in df.iterrows():
                summary.append(f"  {row['measure']} vs {row['criteria']}:")
                summary.append(f"    Measurement count: {row['measurement_count']}")
                summary.append(f"    Criteria mention count: {row['criteria_mention_count']}")
                summary.append(f"    Agreement: {row['agreement']:.2f}")
                summary.append(f"    Precision with measurement: {row['measurement_precision']:.2f}")
                summary.append(f"    Precision with criteria only: {row['criteria_only_precision']:.2f}")

    return "\n".join(summary)


def preprocess_text(text):
    """
    Preprocess text for NLP analysis.

    Args:
        text (str): Input text to preprocess

    Returns:
        str: Preprocessed text with stopwords removed and lemmatization applied
    """
    if pd.isna(text):
        return ""

    # Convert to lowercase
    text = text.lower()

    # Tokenize
    tokens = word_tokenize(text)

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words and len(token) > 2]

    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]

    return " ".join(tokens)

def analyze_topics(explanations, malnutrition_status, n_topics=5, n_top_words=10):
    """
    Perform topic modeling on explanations.

    Args:
        explanations (pd.Series): Series of explanation texts
        malnutrition_status (pd.Series): Series with malnutrition decisions
        n_topics (int): Number of topics to extract
        n_top_words (int): Number of top words to display per topic

    Returns:
        dict: Dictionary with topic modeling results including:
            - topics: Dictionary mapping topic index to list of top words
            - doc_topic_matrix: Document-topic matrix
            - dominant_topics: Series of dominant topic for each document
            - prevalence_by_status: DataFrame of topic prevalence by malnutrition status
    """
    # Preprocess texts
    preprocessed_texts = explanations.apply(preprocess_text)

    # Vectorize
    vectorizer = CountVectorizer(max_features=500, min_df=2)
    X = vectorizer.fit_transform(preprocessed_texts)
    feature_names = vectorizer.get_feature_names_out()

    # Run LDA
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
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
        'malnutrition': malnutrition_status.reset_index(drop=True)
    })

    prevalence_by_status = topic_prevalence.groupby(['dominant_topic', 'malnutrition']).size().unstack().fillna(0)
    prevalence_by_status = prevalence_by_status.div(prevalence_by_status.sum(axis=0), axis=1)

    return {
        'topics': topics,
        'doc_topic_matrix': doc_topic_matrix,
        'dominant_topics': dominant_topics,
        'prevalence_by_status': prevalence_by_status
    }

def plot_topic_analysis(topic_results, output_dir=None):
    """
    Plot topic modeling results.

    Args:
        topic_results (dict): Dictionary of topic modeling results from analyze_topics()
        output_dir (str, optional): Directory to save output files, if None, plots are only displayed

    Returns:
        dict: Dictionary of figure objects for each plot
    """
    figures = {}

    # Plot top words for each topic
    topics = topic_results['topics']
    n_topics = len(topics)

    fig, axes = plt.subplots(nrows=n_topics, figsize=(12, 4*n_topics))

    for i, (topic_idx, top_words) in enumerate(topics.items()):
        ax = axes[i] if n_topics > 1 else axes
        y_pos = range(len(top_words))
        importance = [len(top_words) - j for j in range(len(top_words))]
        ax.barh(y_pos, importance, align='center')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_words)
        ax.invert_yaxis()
        ax.set_title(f'Topic {topic_idx}')
        ax.set_xlabel('Relative Importance')

    plt.tight_layout()
    if output_dir:
        plt.savefig(f"{output_dir}/topic_top_words.png", dpi=300, bbox_inches='tight')

    figures['top_words'] = fig

    # Plot topic prevalence by malnutrition status
    prevalence = topic_results['prevalence_by_status']

    fig, ax = plt.subplots(figsize=(10, 6))
    prevalence.plot(kind='bar', stacked=False, ax=ax)
    ax.set_title('Topic Prevalence by Malnutrition Status', fontsize=16)
    ax.set_xlabel('Topic', fontsize=14)
    ax.set_ylabel('Proportion of Documents', fontsize=14)
    plt.xticks(rotation=0)
    ax.legend(title='Malnutrition Status')
    plt.tight_layout()

    if output_dir:
        plt.savefig(f"{output_dir}/topic_prevalence_by_status.png", dpi=300, bbox_inches='tight')

    figures['prevalence'] = fig

    return figures

def create_keyword_network(explanations, malnutrition_status, min_freq=5, max_keywords=30):
    """
    Create a network visualization of co-occurring keywords in explanations.

    Args:
        explanations (pd.Series): Series of explanation texts
        malnutrition_status (pd.Series): Series with malnutrition decisions
        min_freq (int): Minimum frequency for a keyword to be included
        max_keywords (int): Maximum number of keywords to include

    Returns:
        networkx.Graph: NetworkX graph of keyword co-occurrences with following attributes:
            - nodes: count (frequency), malnutrition_ratio (association with malnutrition)
            - edges: weight (co-occurrence frequency)
    """
    # Preprocess texts
    preprocessed_texts = explanations.apply(preprocess_text)

    # Tokenize
    all_tokens = []
    for text in preprocessed_texts:
        all_tokens.extend(text.split())

    # Count token frequencies
    token_counts = Counter(all_tokens)

    # Get top keywords
    clinical_stopwords = {'patient', 'diagnosis', 'medical', 'clinical', 'hospital',
                         'assessment', 'history', 'present', 'condition', 'status',
                         'note', 'report', 'examination', 'chart'}

    top_keywords = [w for w, c in token_counts.most_common(100)
                   if c >= min_freq and w not in clinical_stopwords][:max_keywords]

    # Create co-occurrence matrix
    co_occurrence = np.zeros((len(top_keywords), len(top_keywords)))
    keyword_to_idx = {word: i for i, word in enumerate(top_keywords)}

    # Count co-occurrences
    for text in preprocessed_texts:
        tokens = set(text.split())
        for word1 in tokens:
            if word1 in keyword_to_idx:
                for word2 in tokens:
                    if word2 in keyword_to_idx and word1 != word2:
                        i, j = keyword_to_idx[word1], keyword_to_idx[word2]
                        co_occurrence[i, j] += 1
                        co_occurrence[j, i] += 1  # Symmetric

    # Create network
    G = nx.Graph()

    # Add nodes
    for i, word in enumerate(top_keywords):
        # Count occurrences in malnourished vs. non-malnourished
        malnourished_count = sum(1 for j, text in enumerate(preprocessed_texts)
                                if word in text.split() and malnutrition_status.iloc[j] == 'yes')
        total_count = sum(1 for text in preprocessed_texts if word in text.split())

        G.add_node(word, count=token_counts[word],
                  malnutrition_ratio=malnourished_count / total_count if total_count > 0 else 0)

    # Add edges
    for i in range(len(top_keywords)):
        for j in range(i+1, len(top_keywords)):
            if co_occurrence[i, j] >= min_freq:
                G.add_edge(top_keywords[i], top_keywords[j], weight=co_occurrence[i, j])

    return G

def plot_keyword_network(G, output_dir=None):
    """
    Plot keyword co-occurrence network.

    Args:
        G (networkx.Graph): NetworkX graph of keyword co-occurrences from create_keyword_network()
        output_dir (str, optional): Directory to save output files, if None, plots are only displayed

    Returns:
        matplotlib.figure.Figure: Figure object for keyword network visualization
    """
    fig, ax = plt.subplots(figsize=(14, 10))

    # Set node colors based on malnutrition ratio
    node_colors = [plt.cm.RdYlBu(1 - G.nodes[node]['malnutrition_ratio']) for node in G.nodes()]

    # Set node sizes based on frequency
    node_sizes = [30 * G.nodes[node]['count'] for node in G.nodes()]

    # Set edge weights based on co-occurrence
    edge_weights = [0.5 * G[u][v]['weight'] for u, v in G.edges()]

    # Position nodes using force-directed layout
    pos = nx.spring_layout(G, k=0.3, iterations=50, seed=42)

    # Draw the network
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, alpha=0.8, ax=ax)
    nx.draw_networkx_edges(G, pos, width=edge_weights, alpha=0.3, edge_color='gray', ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold', ax=ax)

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlBu, norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Association with Malnutrition', rotation=270, labelpad=20)

    ax.axis('off')
    ax.set_title('Keyword Co-occurrence Network in Malnutrition Explanations', fontsize=16)
    plt.tight_layout()

    if output_dir:
        plt.savefig(f"{output_dir}/keyword_network.png", dpi=300, bbox_inches='tight')

    return fig

def analyze_explanation_sentiment(explanations, malnutrition_status):
    """
    Analyze sentiment in explanation texts by malnutrition status.

    Args:
        explanations (pd.Series): Series of explanation texts
        malnutrition_status (pd.Series): Series with malnutrition decisions

    Returns:
        dict: Dictionary with sentiment analysis results
    """
    from textblob import TextBlob

    # Calculate sentiment for each explanation
    sentiment_scores = []
    for exp in explanations:
        if pd.isna(exp):
            sentiment_scores.append(np.nan)
            continue

        blob = TextBlob(exp)
        sentiment_scores.append(blob.sentiment.polarity)

    # Create DataFrame with sentiment and malnutrition status
    sentiment_df = pd.DataFrame({
        'sentiment': sentiment_scores,
        'malnutrition': malnutrition_status.reset_index(drop=True)
    })

    # Remove NaN values
    sentiment_df = sentiment_df.dropna()

    # Calculate statistics by malnutrition status
    sentiment_stats = sentiment_df.groupby('malnutrition')['sentiment'].agg(['mean', 'median', 'std', 'count'])

    # T-test between groups
    mal_values = sentiment_df[sentiment_df['malnutrition'] == 'yes']['sentiment']
    non_mal_values = sentiment_df[sentiment_df['malnutrition'] == 'no']['sentiment']

    from scipy import stats
    t_stat, p_value = stats.ttest_ind(mal_values, non_mal_values, equal_var=False)

    return {
        'sentiment_df': sentiment_df,
        'sentiment_stats': sentiment_stats,
        't_stat': t_stat,
        'p_value': p_value
    }

def plot_sentiment_analysis(sentiment_results, output_dir=None):
    """
    Plot sentiment analysis results.

    Args:
        sentiment_results (dict): Dictionary with sentiment analysis results
        output_dir (str, optional): Directory to save output files, if None, plots are only displayed

    Returns:
        dict: Dictionary of figure objects for each plot
    """
    sentiment_df = sentiment_results['sentiment_df']

    figures = {}

    # Distribution plot
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(data=sentiment_df, x='sentiment', hue='malnutrition',
                 kde=True, element='step', common_norm=False, ax=ax)

    # Add vertical lines for means
    for status, color in zip(['yes', 'no'], ['blue', 'orange']):
        mean_val = sentiment_df[sentiment_df['malnutrition'] == status]['sentiment'].mean()
        ax.axvline(mean_val, color=color, linestyle='--',
                   label=f'Mean ({status}): {mean_val:.3f}')

    ax.set_title('Sentiment Distribution by Malnutrition Status', fontsize=16)
    ax.set_xlabel('Sentiment Score (-1: Negative, +1: Positive)', fontsize=14)
    ax.set_ylabel('Density', fontsize=14)
    ax.legend(title='Status')
    plt.tight_layout()

    if output_dir:
        plt.savefig(f"{output_dir}/sentiment_distribution.png", dpi=300, bbox_inches='tight')

    figures['distribution'] = fig

    # Boxplot
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.boxplot(data=sentiment_df, x='malnutrition', y='sentiment', ax=ax)

    # Add p-value annotation
    p_value = sentiment_results['p_value']
    p_text = f"p = {p_value:.4f}" if p_value >= 0.0001 else "p < 0.0001"

    y_max = sentiment_df['sentiment'].max()
    ax.annotate(p_text, xy=(0.5, y_max + 0.05), xycoords='data',
                horizontalalignment='center', fontsize=12)

    ax.set_title('Sentiment Comparison by Malnutrition Status', fontsize=16)
    ax.set_xlabel('Malnutrition Status', fontsize=14)
    ax.set_ylabel('Sentiment Score', fontsize=14)
    plt.tight_layout()

    if output_dir:
        plt.savefig(f"{output_dir}/sentiment_boxplot.png", dpi=300, bbox_inches='tight')

    figures['boxplot'] = fig

    return figures

def get_named_entities(explanations):
    """
    Extract medical named entities from explanations.

    Args:
        explanations (pd.Series): Series of explanation texts

    Returns:
        dict: Dictionary with entity counts by type
    """
    import spacy
    from collections import Counter

    # Load spaCy model (requires installing 'en_core_sci_md' for medical texts)
    # If not available, fall back to standard English model
    try:
        nlp = spacy.load("en_core_sci_md")
    except:
        nlp = spacy.load("en_core_web_sm")
        print("Warning: Using standard English NER model. For better medical entity extraction, install 'en_core_sci_md'")

    # Categories of interest
    categories = {
        'DISEASE': [],
        'CONDITION': [],
        'TREATMENT': [],
        'MEDICATION': [],
        'TEST': [],
        'MEASUREMENT': [],
        'PERSON': [],
        'DATE': [],
        'AGE': [],
        'OTHER': []
    }

    # Process each explanation
    all_entities = []

    for exp in explanations:
        if pd.isna(exp):
            continue

        doc = nlp(exp)

        # Extract entities
        for ent in doc.ents:
            all_entities.append((ent.text, ent.label_))

            # Map to our categories
            if ent.label_ in ['DISEASE', 'DISORDER']:
                categories['DISEASE'].append(ent.text)
            elif ent.label_ in ['CONDITION', 'FINDING', 'SYMPTOM']:
                categories['CONDITION'].append(ent.text)
            elif ent.label_ in ['TREATMENT', 'PROCEDURE']:
                categories['TREATMENT'].append(ent.text)
            elif ent.label_ in ['MEDICATION', 'DRUG']:
                categories['MEDICATION'].append(ent.text)
            elif ent.label_ in ['TEST', 'DIAGNOSTIC']:
                categories['TEST'].append(ent.text)
            elif ent.label_ in ['MEASUREMENT', 'VALUE', 'NUMBER']:
                categories['MEASUREMENT'].append(ent.text)
            elif ent.label_ in ['PERSON', 'PATIENT', 'DOCTOR']:
                categories['PERSON'].append(ent.text)
            elif ent.label_ in ['DATE', 'TIME']:
                categories['DATE'].append(ent.text)
            elif ent.label_ in ['AGE']:
                categories['AGE'].append(ent.text)
            else:
                categories['OTHER'].append(ent.text)

    # Count entities by category
    category_counts = {cat: Counter(items) for cat, items in categories.items()}

    return {
        'all_entities': Counter(all_entities),
        'category_counts': category_counts
    }

def plot_entity_analysis(entity_results, top_n=15, output_dir=None):
    """
    Plot entity analysis results.

    Args:
        entity_results (dict): Dictionary with entity analysis results
        top_n (int): Number of top entities to display
        output_dir (str, optional): Directory to save output files, if None, plots are only displayed

    Returns:
        dict: Dictionary of figure objects for each plot
    """
    figures = {}

    # Plot top entities by category
    category_counts = entity_results['category_counts']

    for category, counts in category_counts.items():
        if not counts:
            continue

        # Get top N entities in this category
        top_entities = counts.most_common(top_n)

        if not top_entities:
            continue

        labels, values = zip(*top_entities) if top_entities else ([], [])

        fig, ax = plt.subplots(figsize=(10, 6))
        y_pos = range(len(labels))
        ax.barh(y_pos, values)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels)
        ax.invert_yaxis()
        ax.set_title(f'Top {len(labels)} {category} Entities', fontsize=16)
        ax.set_xlabel('Frequency', fontsize=14)
        plt.tight_layout()

        if output_dir:
            plt.savefig(f"{output_dir}/entities_{category.lower()}.png", dpi=300, bbox_inches='tight')

        figures[category] = fig

    return figures
