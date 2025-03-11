#!/usr/bin/env python3

from datetime import datetime
import glob
import re
import os
from typing import List
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer 
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
import joblib
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, auc, confusion_matrix, classification_report
)
import shap
from dotenv import load_dotenv, find_dotenv


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
                 standardize_numbers: bool = True,
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
            tokens = [token for token in tokens if token not in self.stop_words]
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
    model_name:str,
    max_features: int = 8000,
    remove_stop_words: bool = True,
    apply_stemming: bool = False,
    vectorization_mode: str = 'tfidf', 
    ngram_range: tuple = (1, 1), 
    save_path: str = '.',
):
    """
    Process the CSV file containing clinical notes into features with n-gram support.
    Parameters:
      - file_path: Path to the CSV file containing the data
      - text_column: Name of the column containing text to analyze
      - label_column: Name of the column containing labels
      - id_column: Name of the column containing unique identifiers
      - max_features: Maximum number of features to extract
      - remove_stop_words: Whether to remove stop words
      - apply_stemming: Whether to apply stemming
      - vectorization_mode: 'count' for CountVectorizer, 'tfidf' for TF-IDF Vectorizer
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
        df = pd.read_csv(file_path)
        required_columns = [text_column, label_column, id_column]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Available columns: {list(df.columns)}")
            raise ValueError(f"Missing required columns: {missing_columns}")

        # Validate ngram_range input
        if not isinstance(ngram_range, tuple) or len(ngram_range) != 2 or ngram_range[0] > ngram_range[1]:
            raise ValueError(f"Invalid ngram_range: {ngram_range}. Must be a tuple (min_n, max_n) where min_n <= max_n.")

        # Build preprocessing steps
        preprocessing_steps = []
        preprocessing_steps.append(('preprocessor', ClinicalTextPreprocessor()))
        if remove_stop_words:
            preprocessing_steps.append(('stopword_remover', StopWordsRemover()))
        if apply_stemming:
            preprocessing_steps.append(('stemmer', TextStemmer()))
        
        # Process labels (convert 'yes'/'no' to 1/0 if needed)
        # First, save the original labels
        y_original = df[label_column]
        
        # Create and fit a label encoder
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y_original)
        
        # Save the label encoder
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        encoder_filename = os.path.join(save_path, f'{model_name}_nutrikidai_classifier_label_encoder_{timestamp}.joblib')
        joblib.dump(label_encoder, encoder_filename)
        print(f"Label encoder saved to '{encoder_filename}'. Classes: {label_encoder.classes_}")
        
        # Ensure the save_path directory exists
        os.makedirs(save_path, exist_ok=True)
        
        # Configure the appropriate vectorizer based on mode
        if vectorization_mode == 'count':
            vectorizer = CountVectorizer(max_features=max_features, ngram_range=ngram_range)
            pipeline_filename = os.path.join(save_path, f'{model_name}_nutrikidai_pipeline.joblib')
        elif vectorization_mode == 'tfidf':
            vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)
            pipeline_filename = os.path.join(save_path, f'{model_name}_nutrikidai_pipeline.joblib')
        else:
            raise ValueError("Invalid vectorization_mode. Choose from 'count' or 'tfidf'.")
        
        # Add vectorizer to pipeline
        preprocessing_steps.append(('vectorizer', vectorizer))
        pipeline = Pipeline(preprocessing_steps)
        
        # Fit and transform with the chosen vectorizer
        matrix = pipeline.fit_transform(df[text_column])
        feature_names = pipeline.named_steps['vectorizer'].get_feature_names_out()
        X_df = pd.DataFrame(matrix.toarray(), columns=feature_names, index=df.index)
        
        # Create complete DataFrame with features and label
        complete_df = pd.concat([X_df, pd.DataFrame({label_column: y})], axis=1)
        
        # Save the pipeline
        joblib.dump(pipeline, pipeline_filename)
        print(f"{vectorization_mode.capitalize()} vectorizer pipeline with n-grams {ngram_range} saved to '{pipeline_filename}'.")
        
        # Create feature dictionary
        feature_dict = {name: idx for idx, name in enumerate(feature_names)}
        
        return X_df, complete_df, y, pipeline, feature_dict, label_encoder

    except Exception as e:
        print(f"Error processing CSV file: {str(e)}")
        raise

# =========================
# Label Encoding Function
# =========================
def encode_labels(labels: List[str]) -> np.ndarray:
    """
    Encode text labels like 'yes'/'no' to binary values (1/0).
    
    Args:
        labels: List of text labels
    
    Returns:
        numpy array of binary encoded labels (1 for 'yes', 0 for 'no')
    """
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    
    # Ensure 'yes' is encoded as 1 if present
    if 'yes' in labels:
        yes_idx = labels.index('yes')
        yes_encoded = encoded_labels[yes_idx]
        if yes_encoded != 1:
            encoded_labels = 1 - encoded_labels
    
    return encoded_labels, label_encoder


def load_artifacts(model_dir: str, model_name:str):
    """ 
    Load all model artifacts (model, feature dict, pipeline) from the given directory.

    Args:
        model_dir (str): Path to the directory containing model artifacts.

    Returns:
        model, feature_dict, pipeline
    """
    # Define the file patterns to match the latest .joblib files
    model_pattern = os.path.join(model_dir,f"{model_name}_nutrikidai_model.joblib")
    label_encoder_pattern = os.path.join(model_dir, f"{model_name}_nutrikidai_classifier_label_encoder_*.joblib")
    pipeline_pattern = os.path.join(model_dir,  f"{model_name}_nutrikidai_pipeline.joblib")

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

def plot_roc_curve(y_true, y_pred_proba, output_path):
    """Plot and save ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_feature_importance(model, feature_names, output_path, top_n=20):
    """Plot and save feature importance."""
    # Get feature importance
    importance = model.get_booster().get_score(importance_type='gain')
    tuples = [(k, importance[k]) for k in importance]
    tuples = sorted(tuples, key=lambda x: x[1], reverse=True)
    
    # Get top N features
    if top_n > 0 and top_n < len(tuples):
        tuples = tuples[:top_n]
    
    # Extract feature names and values
    feature_indices = [int(t[0][1:]) for t in tuples]  # Remove 'f' prefix and convert to int
    feature_names_top = [feature_names[i] for i in feature_indices]
    importance_values = [t[1] for t in tuples]
    
    # Create plot
    plt.figure(figsize=(12, 8))
    y_pos = np.arange(len(feature_names_top))
    plt.barh(y_pos, importance_values, align='center')
    plt.yticks(y_pos, feature_names_top)
    plt.xlabel('Importance (Gain)')
    plt.title('Feature Importance (Top Features)')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
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
    shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'shap_summary.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Dependency plots for top features
    shap_sum = np.abs(shap_values).mean(axis=0)
    top_indices = np.argsort(-shap_sum)[:5]  # Get top 5 features
    
    for i in top_indices:
        plt.figure(figsize=(12, 8))
        shap.dependence_plot(i, shap_values, X_sample, feature_names=feature_names, show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'shap_dependence_{feature_names[i]}.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    return shap_values, explainer


def get_api_key(env_variable): 
    _ = load_dotenv(find_dotenv())
    return os.getenv(env_variable)
