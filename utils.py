#!/usr/bin/env python3

import re
import os
from typing import List
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
import pickle


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
    max_features: int = 8000,
    remove_stop_words: bool = True,
    apply_stemming: bool = False,
    vectorization_mode: str = 'tfidf', 
    ngram_range: tuple = (1, 1), 
    save_path: str = '.'  
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
        y = df[label_column]
        if y.dtype == 'object':
            y = y.map({'yes': 1, 'no': 0})
        
        # Ensure the save_path directory exists
        os.makedirs(save_path, exist_ok=True)
        
        # Configure the appropriate vectorizer based on mode
        if vectorization_mode == 'count':
            vectorizer = CountVectorizer(max_features=max_features, ngram_range=ngram_range)
            pipeline_filename = os.path.join(save_path, f'count_vectorizer_ngram_{ngram_range[0]}_{ngram_range[1]}_pipeline.joblib')
        elif vectorization_mode == 'tfidf':
            vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)
            pipeline_filename = os.path.join(save_path, f'tfidf_vectorizer_ngram_{ngram_range[0]}_{ngram_range[1]}_pipeline.joblib')
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
        
        return X_df, complete_df, y, pipeline, feature_dict

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

