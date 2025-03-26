#!/usr/bin/env python3
"""
Memory-Efficient TabPFN Training Script: Processes text data in batches and trains a TabPFN classifier.
Saves the model, text vectorization pipeline, and label encoder to disk.
Handles various label types appropriately.
"""

import os
import argparse
import numpy as np
import pandas as pd
import joblib
from datetime import datetime
from models.tabpfn import train_tabpfn
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
# Import the text processing classes from utils
from utils import ClinicalTextPreprocessor, StopWordsRemover, TextStemmer
# Import custom label processing utility
from utils import process_labels, detect_label_type


def batch_process_csv(
    file_path: str,
    text_column: str,
    label_column: str,
    id_column: str,
    model_name: str,
    batch_size: int = 1000,
    max_features: int = 8000,
    remove_stop_words: bool = False,
    apply_stemming: bool = False,
    vectorization_mode: str = 'tfidf',
    ngram_range: tuple = (1, 1),
    save_path: str = '.',
):
    """
    Process CSV data in batches to avoid memory issues.
    Handles various label types appropriately.

    Parameters:
      - file_path: Path to the CSV file containing the data
      - text_column: Name of the column containing text to analyze
      - label_column: Name of the column containing labels
      - id_column: Name of the column containing unique identifiers
      - model_name: Name of the model being trained
      - batch_size: Number of rows to process at once
      - max_features: Maximum number of features to extract
      - remove_stop_words: Whether to remove stop words
      - apply_stemming: Whether to apply stemming
      - vectorization_mode: 'count' for CountVectorizer, 'tfidf' for TF-IDF Vectorizer
      - ngram_range: Tuple (min_n, max_n) for n-gram range
      - save_path: Directory path to save the text preprocessing pipeline

    Returns:
      - data_generator: Generator yielding batches of processed data
      - feature_dict: Dictionary mapping feature indices to names
      - label_encoder: Fitted LabelEncoder or None
      - label_type: Type of labels (numeric_category, binary, text_category)
    """
    # Validate inputs
    try:
        # Sample a small chunk to verify columns exist
        sample_df = next(pd.read_csv(file_path,
                                     usecols=[text_column,
                                              label_column, id_column],
                                     chunksize=5))
        required_columns = [text_column, label_column, id_column]
        missing_columns = [
            col for col in required_columns if col not in sample_df.columns]

        if missing_columns:
            print(f"Available columns: {list(sample_df.columns)}")
            raise ValueError(f"Missing required columns: {missing_columns}")

        # Validate ngram_range input
        if not isinstance(ngram_range, tuple) or len(ngram_range) != 2 or ngram_range[0] > ngram_range[1]:
            raise ValueError(
                f"Invalid ngram_range: {ngram_range}. Must be a tuple (min_n, max_n) where min_n <= max_n.")

    except Exception as e:
        print(f"Error validating CSV file: {str(e)}")
        raise

    # First pass: count total rows and collect all labels
    print("First pass: counting rows and collecting labels...")
    total_rows = 0
    all_labels = []

    try:
        for chunk in pd.read_csv(file_path, chunksize=batch_size, usecols=[text_column, label_column, id_column]):
            total_rows += len(chunk)
            all_labels.extend(chunk[label_column].tolist())

        print(f"Total rows: {total_rows}, Total labels: {len(all_labels)}")
    except Exception as e:
        print(f"Error during first pass: {str(e)}")
        raise

    # Detect label type
    label_type = detect_label_type(all_labels)
    print(f"Detected label type: {label_type}")

    # Process labels
    processed_labels, label_encoder = process_labels(all_labels)

    # Prepare to save label encoder if needed
    encoder_filename = None
    if label_encoder is not None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        encoder_filename = os.path.join(
            save_path,
            f'{model_name}_nutrikidai_classifier_label_encoder_{timestamp}.joblib'
        )
        joblib.dump(label_encoder, encoder_filename)
        print(
            f"Label encoder saved to '{encoder_filename}'. Classes: {label_encoder.classes_}")

    # Second pass: create and fit the text processing pipeline on a sample
    print("Second pass: Creating text processing pipeline...")
    # Use at most 10k samples to fit vectorizer
    sample_size = min(10000, total_rows)
    sample_df = pd.DataFrame()

    # Collect sample data
    remaining = sample_size
    for chunk in pd.read_csv(file_path, chunksize=min(batch_size, sample_size)):
        if remaining <= 0:
            break
        sample_df = pd.concat([sample_df, chunk.head(remaining)])
        remaining -= len(chunk)
        if remaining <= 0:
            break

    # Build preprocessing steps
    preprocessing_steps = []
    preprocessing_steps.append(('preprocessor', ClinicalTextPreprocessor()))
    if remove_stop_words:
        preprocessing_steps.append(('stopword_remover', StopWordsRemover()))
    if apply_stemming:
        preprocessing_steps.append(('stemmer', TextStemmer()))

    # Configure the appropriate vectorizer based on mode
    if vectorization_mode == 'count':
        vectorizer = CountVectorizer(
            max_features=max_features, ngram_range=ngram_range)
        pipeline_filename = os.path.join(
            save_path, f'{model_name}_nutrikidai_pipeline.joblib')
    elif vectorization_mode == 'tfidf':
        vectorizer = TfidfVectorizer(
            max_features=max_features, ngram_range=ngram_range)
        pipeline_filename = os.path.join(
            save_path, f'{model_name}_nutrikidai_pipeline.joblib')
    else:
        raise ValueError(
            "Invalid vectorization_mode. Choose from 'count' or 'tfidf'.")

    # Add vectorizer to pipeline
    preprocessing_steps.append(('vectorizer', vectorizer))
    pipeline = Pipeline(preprocessing_steps)

    # Fit pipeline on sample data
    pipeline.fit(sample_df[text_column].fillna(''))

    # Get feature names
    feature_names = pipeline.named_steps['vectorizer'].get_feature_names_out()
    feature_dict = {i: name for i, name in enumerate(feature_names)}

    # Save the pipeline
    os.makedirs(save_path, exist_ok=True)
    joblib.dump(pipeline, pipeline_filename)
    print(f"{vectorization_mode.capitalize()} vectorizer pipeline with n-grams {ngram_range} saved to '{pipeline_filename}'.")

    # Third pass: process data in batches and train
    print("Third pass: Processing data in batches for training...")

    # We'll use a generator to yield batches of processed data
    def data_generator():
        for chunk in pd.read_csv(file_path, chunksize=batch_size):
            try:
                # Process text
                X_text = chunk[text_column].fillna('')
                X_processed = pipeline.transform(X_text)

                # Convert sparse matrix to dense array to fix memory management issue
                X_dense = pd.DataFrame(X_processed.toarray(
                ), index=chunk.index, columns=feature_names)

                # Get labels for this chunk
                y = chunk[label_column]

                # Process labels (this will return the integer-encoded labels)
                y_processed, _ = process_labels(y)

                # Create complete DataFrame with features and label for this batch
                complete_df = pd.concat([X_dense, pd.DataFrame(
                    {label_column: y_processed}, index=chunk.index)], axis=1)

                # Add ID column if available
                if id_column and id_column in chunk.columns:
                    complete_df[id_column] = chunk[id_column]

                yield X_dense, complete_df, y_processed, pipeline, feature_dict, label_encoder

            except Exception as e:
                print(f"Error processing batch: {str(e)}")
                continue

    return data_generator, feature_dict, label_encoder, label_type


def main():
    parser = argparse.ArgumentParser(
        description='Train a TabPFN classifier on text data with batch processing')
    # Data parameters
    parser.add_argument('--data_file', type=str, required=True,
                        help='Path to the CSV data file')
    parser.add_argument('--text_column', type=str, default="txt",
                        help='Column containing text data')
    parser.add_argument('--label_column', type=str, default="label",
                        help='Column containing labels')
    parser.add_argument('--id_column', type=str, default="DEID",
                        help='Column containing IDs')
    parser.add_argument('--batch_size', type=int, default=1000,
                        help='Number of rows to process at once')
    parser.add_argument('--max_train_samples', type=int, default=10000,
                        help='Maximum number of samples to use for training')

    # Text processing parameters
    parser.add_argument('--max_features', type=int, default=10000,
                        help='Max number of features to extract')
    parser.add_argument('--remove_stop_words', action='store_true',
                        default=False, help='Remove stop words')
    parser.add_argument('--apply_stemming', action='store_true',
                        default=False, help='Apply stemming')
    parser.add_argument('--vectorization_mode', type=str, default='tfidf',
                        choices=['count', 'tfidf'], help='Vectorization mode')
    parser.add_argument('--ngram_min', type=int, default=1,
                        help='Minimum n-gram size')
    parser.add_argument('--ngram_max', type=int, default=1,
                        help='Maximum n-gram size')

    # Model parameters
    parser.add_argument('--device', type=str, default='cpu',
                        choices=['cpu', 'cuda'], help='Device to use')
    parser.add_argument('--model_name', type=str, default="tabpfn",
                        help='Name of the type of Model being trained')

    # Output parameters
    parser.add_argument('--model_dir', type=str, default='TABPFN',
                        help='Directory to save all models and artifacts')

    args = parser.parse_args()

    # Ensure model directory exists
    os.makedirs(args.model_dir, exist_ok=True)

    # Process CSV data in batches
    print(f"Processing CSV data from {args.data_file} in batches...")
    ngram_range = (args.ngram_min, args.ngram_max)

    data_generator, feature_dict, label_encoder, label_type = batch_process_csv(
        file_path=args.data_file,
        text_column=args.text_column,
        label_column=args.label_column,
        id_column=args.id_column,
        model_name=args.model_name,
        batch_size=args.batch_size,
        max_features=args.max_features,
        remove_stop_words=args.remove_stop_words,
        apply_stemming=args.apply_stemming,
        vectorization_mode=args.vectorization_mode,
        ngram_range=ngram_range,
        save_path=args.model_dir
    )

    # Save feature dictionary
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    feature_dict_path = os.path.join(
        args.model_dir,
        f"{args.model_name}_nutrikidai_classifier_feature_dict_{timestamp}.joblib")
    joblib.dump(feature_dict, feature_dict_path)
    print(f"Feature dictionary saved to: {feature_dict_path}")

    # Collect training data up to max_train_samples
    print(f"Collecting up to {args.max_train_samples} samples for training...")
    X_train_batches = []
    y_train_batches = []
    samples_collected = 0

    # The data_generator now yields (X_dense, complete_df, y, pipeline, feature_dict, label_encoder)
    for X_batch, complete_batch, y_batch, _, _, _ in data_generator():
        batch_size = len(X_batch)
        if samples_collected + batch_size <= args.max_train_samples:
            X_train_batches.append(X_batch)
            y_train_batches.append(y_batch)
            samples_collected += batch_size
        else:
            # Take only what we need to reach max_train_samples
            samples_needed = args.max_train_samples - samples_collected
            if samples_needed > 0:
                X_train_batches.append(X_batch.iloc[:samples_needed])
                y_train_batches.append(y_batch[:samples_needed])
                samples_collected += samples_needed
            break

    X_train = pd.concat(X_train_batches) if X_train_batches else pd.DataFrame()
    y_train = np.concatenate(
        y_train_batches) if y_train_batches else np.array([])

    print(f"Collected {len(X_train)} samples for training")
    print(f"Label type: {label_type}")

    # Train model
    results = train_tabpfn(
        X_train=X_train,
        y_train=y_train,
        model_dir=args.model_dir,
        device=args.device,
        model_name=f"{args.model_name}_nutrikidai_model"
    )
    print("\nTraining completed successfully!")
    print(f"Model saved to: {results['model_file']}")


if __name__ == "__main__":
    main()
