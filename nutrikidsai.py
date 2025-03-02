#!/usr/bin/env python3
"""
This module defines the command interpreter for NutriKids AI.
It provides a user-friendly CLI with comprehensive help and error handling.
"""

import cmd
import sys
import os
import argparse
import shlex
import textwrap
from datetime import datetime


class NutrikidsAiCommand(cmd.Cmd):
    """
    Command interpreter class for NutriKids AI with improved interface and documentation.
    """
    prompt = "\033[1;36m(NUTRIKIDS)% \033[0m"

    # ----------------------- Basic Commands -------------------------
    def do_quit(self, arg):
        """Quit the program: Exits the application."""
        print("Exiting NutriKids AI. Goodbye!")
        return True

    def do_EOF(self, arg):
        """Exit the program using Ctrl+D (Unix) or Ctrl+Z (Windows)."""
        print("")
        return self.do_quit(arg)

    def do_clear(self, arg):
        """Clear the terminal screen."""
        os.system('cls' if os.name == 'nt' else 'clear')

    # -------------------- File System Commands ----------------------
    def do_ls(self, arg):
        """List directory contents with basic formatting."""
        try:
            items = os.listdir(arg or '.')
            cols = 3
            max_len = max(len(item) for item in items) if items else 0
            term_width = os.get_terminal_size().columns
            cols = max(1, term_width // (max_len + 2))

            for i, item in enumerate(sorted(items), 1):
                print(f"{item:<{max_len+2}}", end='')
                if i % cols == 0:
                    print()
            if len(items) % cols != 0:
                print()
        except Exception as e:
            print(f"\033[31mError: {str(e)}\033[0m")

    def do_pwd(self, arg):
        """Print current working directory with path validation."""
        try:
            print(os.path.abspath(os.getcwd()))
        except Exception as e:
            print(f"\033[31mError: {str(e)}\033[0m")

    def do_cd(self, arg):
        """Change directory with improved error handling."""
        try:
            target = os.path.expanduser(
                arg) if arg else os.path.expanduser("~")
            if not os.path.exists(target):
                raise FileNotFoundError(f"Directory '{target}' does not exist")
            if not os.path.isdir(target):
                raise NotADirectoryError(f"'{target}' is not a directory")
            os.chdir(target)
            print(f"Current directory: {os.getcwd()}")
        except Exception as e:
            print(f"\033[31mError: {str(e)}\033[0m")

    def do_cat(self, arg):
        """Display file contents with line numbers and pagination."""
        if not arg:
            print("Usage: cat <file_path>")
            return
        try:
            with open(arg, 'r') as f:
                lines = f.readlines()
                for num, line in enumerate(lines, 1):
                    print(f"{num:4} | {line.rstrip()}")
        except Exception as e:
            print(f"\033[31mError: {str(e)}\033[0m")

    # ------------------ Machine Learning Commands -------------------
    def do_tunetextcnn(self, arg):
        """
        Tune TextCNN hyperparameters using Ray Tune.

        Usage: tunetextcnn --train <train_file> --val <val_file> [options]

        Required Arguments:
            --train          Path to training CSV file
            --val            Path to validation CSV file

        Data Options:
            --text-column    Name of the text column in CSV (default: Note_Column)
            --label-column   Name of the label column in CSV (default: Malnutrition_Label)

        Tokenizer Options:
            --max-vocab-size  Maximum vocabulary size (default: 20000)
            --min-frequency   Minimum word frequency for vocabulary (default: 2)
            --pad-token       Token used for padding (default: <PAD>)
            --unk-token       Token used for unknown words (default: <UNK>)
            --max-length      Maximum sequence length (default: auto)
            --padding         Padding type: pre or post (default: post)
            --embedding-dim   Dimension of word embeddings (default: 100)
            --pretrained-embeddings  Path to pretrained embeddings (default: None)

        Ray Tune Options:
            --output-dir     Directory to save model artifacts (default: textcnn_model)
            --num-samples    Number of parameter settings to sample (default: 10)
            --max-epochs     Maximum epochs for tuning (default: 10)
            --grace-period   Minimum epochs per trial (default: 3)

        Example:
            tunetextcnn --train data/train.csv --val data/val.csv --max-epochs 15 --num-samples 20
        """
        args = shlex.split(arg)
        parser = argparse.ArgumentParser(
            description='Tune TextCNN hyperparameters')

        # Required arguments
        parser.add_argument('--train', type=str, required=True,
                            help='Path to training CSV file')
        parser.add_argument('--val', type=str, required=True,
                            help='Path to validation CSV file')

        # Data options
        parser.add_argument('--text-column', '--text_column', type=str, default='Note_Column',
                            help='Name of the text column in CSV (default: Note_Column)')
        parser.add_argument('--label-column', '--label_column', type=str, default='Malnutrition_Label',
                            help='Name of the label column in CSV (default: Malnutrition_Label)')

        # Tokenizer parameters
        parser.add_argument('--max-vocab-size', '--max_vocab_size', type=int, default=20000,
                            help='Maximum vocabulary size (default: 20000)')
        parser.add_argument('--min-frequency', '--min_frequency', type=int, default=2,
                            help='Minimum word frequency to include in vocabulary (default: 2)')
        parser.add_argument('--pad-token', '--pad_token', type=str, default='<PAD>',
                            help='Token used for padding (default: <PAD>)')
        parser.add_argument('--unk-token', '--unk_token', type=str, default='<UNK>',
                            help='Token used for unknown words (default: <UNK>)')
        parser.add_argument('--max-length', '--max_length', type=int, default=None,
                            help='Maximum sequence length (default: None, will use longest sequence)')
        parser.add_argument('--padding', type=str, default='post', choices=['pre', 'post'],
                            help='Padding type: pre or post (default: post)')
        parser.add_argument('--embedding-dim', '--embedding_dim', type=int, default=100,
                            help='Dimension of word embeddings (default: 100)')
        parser.add_argument('--pretrained-embeddings', '--pretrained_embeddings', type=str, default=None,
                            help='Path to pretrained word embeddings file (default: None)')

        # Ray Tune parameters
        parser.add_argument('--output-dir', '--output_dir', type=str, default='textcnn_model',
                            help='Directory to save model and artifacts (default: textcnn_model)')
        parser.add_argument('--num-samples', '--num_samples', type=int, default=10,
                            help='Number of parameter settings that are sampled (default: 10)')
        parser.add_argument('--max-epochs', '--max_epochs', type=int, default=10,
                            help='Maximum number of epochs for hyperparameter search (default: 10)')
        parser.add_argument('--grace-period', '--grace_period', type=int, default=3,
                            help='Minimum number of epochs for each trial (default: 3)')
        try:
            parsed_args = parser.parse_args(args)

            # Normalize arguments to use underscores for module imports
            cmd_args = []
            for key, value in vars(parsed_args).items():
                if value is not None:
                    key = key.replace('-', '_')
                    cmd_args.append(f"--{key}")
                    cmd_args.append(str(value))

            # Execute the tune script
            try:
                from tune_textcnn import main as tune_main
                sys.argv = ['tune_textcnn.py'] + cmd_args
                tune_main()
            except ImportError:
                print(
                    "Error: Module 'tune_textcnn' not found. Make sure it's in your PYTHONPATH.")
            except Exception as e:
                print(f"Error: Failed to execute tuning: {e}")

        except SystemExit:
            # argparse will exit if --help is called or arguments are invalid
            pass

    def do_traintextcnn(self, arg):
        """
        Train TextCNN model with best hyperparameters.

        Usage: traintextcnn --train <train_file> --val <val_file> [options]

        Required Arguments:
            --train          Path to training CSV file
            --val            Path to validation CSV file

        Data Options:
            --text-column    Name of the text column in CSV (default: Note_Column)
            --label-column   Name of the label column in CSV (default: Malnutrition_Label)

        Model Options:
            --output-dir     Directory to save model artifacts (default: textcnn_model)
            --epochs         Number of epochs for final training (default: 10)
            --pretrained-embeddings  Path to pretrained embeddings file (default: None)
            --freeze-embeddings      Whether to freeze embeddings during training (flag)

        Example:
            traintextcnn --train data/train.csv --val data/val.csv --epochs 15
        """
        args = shlex.split(arg)
        parser = argparse.ArgumentParser(
            description='Train TextCNN with best hyperparameters')
        parser.add_argument('--train', type=str, required=True,
                            help='Path to training CSV file')
        parser.add_argument('--val', type=str, required=True,
                            help='Path to validation CSV file')
        parser.add_argument('--text-column', '--text_column', type=str, default='Note_Column',
                            help='Name of the text column in CSV (default: Note_Column)')
        parser.add_argument('--label-column', '--label_column', type=str, default='Malnutrition_Label',
                            help='Name of the label column in CSV (default: Malnutrition_Label)')
        parser.add_argument('--output-dir', '--output_dir', type=str, default='textcnn_model',
                            help='Directory to save model and artifacts (default: textcnn_model)')
        parser.add_argument('--epochs', type=int, default=10,
                            help='Number of epochs for final training (default: 10)')
        parser.add_argument('--pretrained-embeddings', '--pretrained_embeddings', type=str, default=None,
                            help='Path to pretrained word embeddings file (default: None)')
        parser.add_argument('--freeze-embeddings', '--freeze_embeddings', action='store_true',
                            help='Whether to freeze embeddings during training (default: False)')

        try:
            parsed_args = parser.parse_args(args)

            # Normalize arguments to use underscores for module imports
            cmd_args = []
            for key, value in vars(parsed_args).items():
                key = key.replace('-', '_')
                if isinstance(value, bool):
                    if value:
                        cmd_args.append(f"--{key}")
                    continue
                if value is not None:
                    cmd_args.append(f"--{key}")
                    cmd_args.append(str(value))

            # Execute the train script
            try:
                from train_textcnn import main as train_main
                sys.argv = ['train_textcnn.py'] + cmd_args
                train_main()
            except ImportError:
                print(
                    "Error: Module 'train_textcnn' not found. Make sure it's in your PYTHONPATH.")
            except Exception as e:
                print(f"Error: Failed to execute training: {e}")

        except SystemExit:
            # argparse will exit if --help is called or arguments are invalid
            pass

    def do_evaluatetextcnn(self, arg):
        """
        Evaluate TextCNN model on test data.

        Usage: evaluate_textcnn --test <test_file> --model-dir <model_dir> [options]

        Required Arguments:
            --test           Path to test CSV file

        Data Options:
            --text-column    Name of the text column in CSV (default: Note_Column)
            --label-column   Name of the label column in CSV (default: Malnutrition_Label)

        Model Options:
            --model-dir      Directory containing model artifacts (default: textcnn_model)
            --output-dir     Directory to save evaluation results (default: textcnn_evaluation_output)
            --batch-size     Batch size for prediction (default: 32)
            --threshold      Threshold for binary classification (default: 0.5)

        Example:
            evaluate_textcnn --test data/test.csv --model-dir my_textcnn_model
        """
        args = shlex.split(arg)
        parser = argparse.ArgumentParser(
            description='Evaluate TextCNN model on test data')
        parser.add_argument('--test', type=str, required=True,
                            help='Path to test CSV file')
        parser.add_argument('--text-column', '--text_column', type=str, default='Note_Column',
                            help='Name of the text column in CSV (default: Note_Column)')
        parser.add_argument('--label-column', '--label_column', type=str, default='Malnutrition_Label',
                            help='Name of the label column in CSV (default: Malnutrition_Label)')
        parser.add_argument('--model-dir', '--model_dir', type=str, default='textcnn_model',
                            help='Directory containing model and artifacts (default: textcnn_model)')
        parser.add_argument('--output-dir', '--output_dir', type=str, default='textcnn_evaluation_output',
                            help='Directory to save evaluation artifacts (default: textcnn_evaluation_output)')
        parser.add_argument('--batch-size', '--batch_size', type=int, default=32,
                            help='Batch size for prediction (default: 32)')
        parser.add_argument('--threshold', type=float, default=0.5,
                            help='Threshold for binary classification (default: 0.5)')

        try:
            parsed_args = parser.parse_args(args)

            # Normalize arguments to use underscores for module imports
            cmd_args = []
            for key, value in vars(parsed_args).items():
                if value is not None:
                    key = key.replace('-', '_')
                    cmd_args.append(f"--{key}")
                    cmd_args.append(str(value))

            # Execute the evaluate script
            try:
                from textcnn_evaluate import main as evaluate_main
                sys.argv = ['textcnn_evaluate.py'] + cmd_args
                evaluate_main()
            except ImportError:
                print(
                    "Error: Module 'textcnn_evaluate' not found. Make sure it's in your PYTHONPATH.")
            except Exception as e:
                print(f"Error: Failed to execute evaluation: {e}")

        except SystemExit:
            # argparse will exit if --help is called or arguments are invalid
            pass

    def do_textcnnpredict(self, arg):
        """
        Make predictions with TextCNN model.

        Usage: textcnn_predict --input <input_file_or_text> --model-dir <model_dir> [options]

        Required Arguments:
            --input             Path to input CSV file or text string

        Model Options:
            --model-dir         Directory containing model artifacts (default: textcnn_model)
            --output-dir        Directory to save prediction results (default: text_cnn_predictions)
            --batch-size        Batch size for prediction (default: 32)
            --threshold         Probability threshold for classification (default: 0.5)

        Data Options (for CSV input):
            --text-column       Name of the text column in CSV (default: Note_Column)

        Explanation Options:
            --explain           Generate explanations for predictions (flag)
            --explain-method    Explanation method: integrated, permutation, occlusion, all (default: all)
            --explanation-dir   Directory to save explanations (default: text_cnn_predictions)
            --num-samples       Number of samples to explain (default: 10)
            --summary           Generate a summary report of predictions (flag)

        Examples:
            textcnn_predict --input "Patient shows signs of low weight for age" --model-dir my_model
            textcnn_predict --input patient_notes.csv --text-column notes --explain
        """

        args = shlex.split(arg)
        parser = argparse.ArgumentParser(
            description='Use trained TextCNN model for prediction')
        parser.add_argument('--input', type=str, required=True,
                            help='Path to input CSV file or a text string for prediction')
        parser.add_argument('--text-column', '--text_column', type=str, default='Note_Column',
                            help='Name of the text column in CSV (default: Note_Column)')
        parser.add_argument('--model-dir', '--model_dir', type=str, default="textcnn_model",
                            help='Directory containing saved model and artifacts (default: textcnn_model)')
        parser.add_argument('--output-dir', '--output_dir', type=str, default='text_cnn_predictions',
                            help='Directory to save prediction results (default: text_cnn_predictions)')
        parser.add_argument('--batch-size', '--batch_size', type=int, default=32,
                            help='Batch size for prediction (default: 32)')
        parser.add_argument('--explain', action='store_true',
                            help='Generate explanations for predictions')
        parser.add_argument('--explain-method', '--explain_method', type=str,
                            choices=['integrated', 'permutation',
                                     'occlusion', 'all'],
                            default='all', help='Explanation method to use (default: all)')
        parser.add_argument('--explanation-dir', '--explanation_dir', type=str, default='text_cnn_predictions',
                            help='Directory to save explanation visualizations (default: text_cnn_predictions)')
        parser.add_argument('--num-samples', '--num_samples', type=int, default=10,
                            help='Number of samples to explain for batch methods (default: 10)')
        parser.add_argument('--summary', action='store_true',
                            help='Generate a summary report of predictions')
        parser.add_argument('--threshold', type=float, default=0.5,
                            help='Probability threshold for binary classification (default: 0.5)')
        try:
            parsed_args = parser.parse_args(args)

            # Normalize arguments to use underscores for module imports
            cmd_args = []
            for key, value in vars(parsed_args).items():
                key = key.replace('-', '_')
                if key == 'explain' and value:
                    cmd_args.append('--explain')
                    continue
                if key == 'summary' and value:
                    cmd_args.append('--summary')
                    continue
                if isinstance(value, bool) and not value:
                    continue
                cmd_args.append(f"--{key}")
                if not isinstance(value, bool):
                    cmd_args.append(str(value))

            # Execute the predict script
            try:
                from textcnn_predict import main as predict_main
                sys.argv = ['textcnn_predict.py'] + cmd_args
                predict_main()
            except ImportError:
                print(
                    "Error: Module 'textcnn_predict' not found. Make sure it's in your PYTHONPATH.")
            except Exception as e:
                print(f"Error: Failed to execute prediction: {e}")

        except SystemExit:
            # argparse will exit if --help is called or arguments are invalid
            pass

    def do_traintabpfn(self, arg):
        """
        Train a TabPFN classifier on text data.

        Usage: traintabpfn --data-file <data_file> [options]

        Required Arguments:
            --data-file           Path to the CSV data file

        Data Options:
            --text-column         Column containing text data (default: Note_Column)
            --label-column        Column containing labels (default: Malnutrition_Label)
            --id-column           Column containing IDs (default: Patient_ID)

        Text Processing Options:
            --max-features        Max number of features to extract (default: 8000)
            --remove-stop-words   Remove stop words (flag)
            --apply-stemming      Apply stemming (flag)
            --vectorization-mode  Vectorization mode: count or tfidf (default: tfidf)
            --ngram-min           Minimum n-gram size (default: 1)
            --ngram-max           Maximum n-gram size (default: 1)

        Model Options:
            --device              Device to use: cpu or cuda (default: cpu)
            --model-dir           Directory to save models (default: tabpfn_model)
            --model-name          Name of the model (default: tabpfn)

        Example:
            traintabpfn --data-file data/patient_notes.csv --max-features 10000 --remove-stop-words
        """
        args = shlex.split(arg)
        parser = argparse.ArgumentParser(
            description='Train a TabPFN classifier on text data')
        parser.add_argument('--data-file', '--data_file', type=str, required=True,
                            help='Path to the CSV data file')
        parser.add_argument('--text-column', '--text_column', type=str, default="Note_Column",
                            help='Column containing text data (default: Note_Column)')
        parser.add_argument('--label-column', '--label_column', type=str, default="Malnutrition_Label",
                            help='Column containing labels (default: Malnutrition_Label)')
        parser.add_argument('--id-column', '--id_column', type=str, default="Patient_ID",
                            help='Column containing IDs (default: Patient_ID)')

        # Text processing parameters
        parser.add_argument('--max-features', '--max_features', type=int, default=8000,
                            help='Max number of features to extract (default: 8000)')
        parser.add_argument('--remove-stop-words', '--remove_stop_words', action='store_true',
                            help='Remove stop words')
        parser.add_argument('--apply-stemming', '--apply_stemming', action='store_true',
                            help='Apply stemming')
        parser.add_argument('--vectorization-mode', '--vectorization_mode', type=str, default='tfidf',
                            choices=['count', 'tfidf'], help='Vectorization mode (default: tfidf)')
        parser.add_argument('--ngram-min', '--ngram_min', type=int, default=1,
                            help='Minimum n-gram size (default: 1)')
        parser.add_argument('--ngram-max', '--ngram_max', type=int, default=1,
                            help='Maximum n-gram size (default: 1)')

        # Model parameters
        parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'],
                            help='Device to use (default: cpu)')
        parser.add_argument('--model-name', '--model_name', type=str, default="tabpfn",
                            help='Name of the model (default: tabpfn)')

        # Output parameters
        parser.add_argument('--model-dir', '--model_dir', type=str, default='tabpfn_model',
                            help='Directory to save models and artifacts (default: tabpfn_model)')

        try:
            parsed_args = parser.parse_args(args)

            # Normalize arguments to use underscores for module imports
            cmd_args = []
            for key, value in vars(parsed_args).items():
                key = key.replace('-', '_')
                if isinstance(value, bool):
                    if value:
                        cmd_args.append(f"--{key}")
                    continue
                cmd_args.append(f"--{key}")
                cmd_args.append(str(value))

            # Execute the train script
            try:
                from train_tabpfn import main as train_main
                sys.argv = ['train_tabpfn.py'] + cmd_args
                train_main()
            except ImportError:
                print(
                    "Error: Module 'train_tabpfn' not found. Make sure it's in your PYTHONPATH.")
            except Exception as e:
                print(f"Error: Failed to execute TabPFN training: {e}")

        except SystemExit:
            # argparse will exit if --help is called or arguments are invalid
            pass

    def do_evaluatetabpfn(self, arg):
        """
        Evaluate a trained TabPFN model.

        Usage: evaluatetabpfn --model <model_dir> --data-file <data_file> [options]

        Required Arguments:
            --model               Path to the directory containing model artifacts
            --data-file           Path to the CSV test data file

        Data Options:
            --text-column         Name of the column containing text data (default: Note_Column)
            --label-column        Name of the column containing labels (default: Malnutrition_Label)
            --id-column           Name of the column containing IDs (default: Patient_ID)

        Output Options:
            --output-dir          Directory to save evaluation results (default: model_output/tabpfn)
            --model-name          Name to use for saved artifacts (default: tabpfn)

        Example:
            evaluatetabpfn --model tabpfn_model --data-file data/test_data.csv
        """
        args = shlex.split(arg)
        parser = argparse.ArgumentParser(
            description='Evaluate a trained TabPFN model')

        # Required parameters
        parser.add_argument('--model', type=str, required=True,
                            help='Path to the directory containing model artifacts')
        parser.add_argument('--model-name', '--model_name', type=str, default="tabpfn",
                            help='Name of the model (default: tabpfn)')

        # Data parameters
        parser.add_argument('--data-file', '--data_file', type=str, required=True,
                            help='Path to the CSV test data file')
        parser.add_argument('--text-column', '--text_column', type=str, default="Note_Column",
                            help='Name of the column containing text data (default: Note_Column)')
        parser.add_argument('--label-column', '--label_column', type=str, default="Malnutrition_Label",
                            help='Name of the column containing labels (default: Malnutrition_Label)')
        parser.add_argument('--id-column', '--id_column', type=str, default="Patient_ID",
                            help='Name of the column containing IDs (default: Patient_ID)')

        # Optional parameters
        parser.add_argument('--output-dir', '--output_dir', type=str, default='model_output/tabpfn',
                            help='Directory to save evaluation results (default: model_output/tabpfn)')

        try:
            parsed_args = parser.parse_args(args)

            # Normalize arguments to use underscores for module imports
            cmd_args = []
            for key, value in vars(parsed_args).items():
                if value is not None:
                    key = key.replace('-', '_')
                    cmd_args.append(f"--{key}")
                    cmd_args.append(str(value))

            # Execute the evaluate script
            try:
                from evaluate_tabpfn import main as evaluate_main
                sys.argv = ['evaluate_tabpfn.py'] + cmd_args
                evaluate_main()
            except ImportError:
                print(
                    "Error: Module 'evaluate_tabpfn' not found. Make sure it's in your PYTHONPATH.")
            except Exception as e:
                print(f"Error: Failed to execute TabPFN evaluation: {e}")

        except SystemExit:
            # argparse will exit if --help is called or arguments are invalid
            pass

    def do_predicttabpfn(self, arg):
        """
        Make predictions using a trained TabPFN model.

        Usage: predicttabpfn --model <model_dir> (--data-file <data_file> | --text <text>) [options]

        Options:
            --model                 Path to the directory containing model artifacts (required)
            --data-file             Path to the CSV file with data to predict on
            --text                  Single text input to predict on
            --text-column           Name of the column containing text data (default: Note_Column)
            --id-column             Name of the column containing IDs (default: Patient_ID)
            --output-dir            Directory to save prediction results (default: tabpfn_predictions)
            --run-name              Name for this prediction run (default: timestamp-based)
            --include-features      Include features in output (flag)
            --calculate-importance  Calculate feature importance (flag)
            --top-features          Number of top features to display (default: 20)
        """
        args = shlex.split(arg)
        parser = argparse.ArgumentParser(
            description='Make predictions using a trained TabPFN model')

        # Required parameter
        parser.add_argument('--model', type=str, required=True,
                            help='Path to the directory containing model artifacts')

        # Input options (one of these is required)
        input_group = parser.add_mutually_exclusive_group(required=True)
        input_group.add_argument(
            '--data_file', type=str, help='Path to the CSV file with data to predict on')
        input_group.add_argument(
            '--text', type=str, help='Single text input to predict on')

        # Optional parameters
        parser.add_argument('--text_column', type=str, default='Note_Column',
                            help='Name of the column containing text data (default: Note_Column)')
        parser.add_argument('--id_column', type=str, default='Patient_ID',
                            help='Name of the column containing IDs (default: Patient_ID)')
        parser.add_argument('--output_dir', type=str, default='tabpfn_predictions',
                            help='Directory to save prediction results (default: tabpfn_predictions)')
        parser.add_argument('--run_name', type=str,
                            help='Name for this prediction run (default: timestamp-based)')
        parser.add_argument('--include_features', action='store_true',
                            help='Include features in output')
        parser.add_argument('--calculate_importance', action='store_true',
                            help='Calculate feature importance')
        parser.add_argument('--top_features', type=int, default=20,
                            help='Number of top features to display (default: 20)')
        parser.add_argument('--model_name', type=str, default="tabpfn",
                            help='Name of the model type (default: tabpfn)')

        try:
            parsed_args = parser.parse_args(args)

            # Convert namespace to command line args
            cmd_args = []
            for key, value in vars(parsed_args).items():
                if key in ['include_features', 'calculate_importance'] and value:
                    cmd_args.append(f"--{key}")
                    continue
                if isinstance(value, bool) and not value:
                    continue
                if value is not None:
                    cmd_args.append(f"--{key}")
                    if not isinstance(value, bool):
                        cmd_args.append(str(value))

            # Execute the predict script
            from predict_tabpfn import main as predict_main
            sys.argv = ['predicttabpfn.py'] + cmd_args
            predict_main()

        except SystemExit:
            # argparse will exit if --help is called or arguments are invalid
            pass
        except Exception as e:
            print(f"Error running TabPFN prediction: {e}")

    def do_tunexgb(self, arg):
        """
        Tune XGBoost hyperparameters for text classification.

        Usage: tunexgb --data-file <data_file> [options]

        Options:
            --train-data-file   Path to training data CSV file (required)
            --valid-data_file   Path to Validation data CSV file (required)
            --text-column        Name of the text column in CSV (default: Note_Column)
            --label-column       Name of the label column in CSV (default: Malnutrition_Label)
            --id-column          Name of the ID column in CSV (default: Patient_ID)
            --max-features       Maximum number of features for vectorization (default: 10000)
            --remove-stop-words  Remove stop words during preprocessing (flag)
            --apply-stemming     Apply stemming during preprocessing (flag)
            --vectorization-mode Vectorization mode: tfidf, count, binary (default: tfidf)
            --ngram-min          Minimum n-gram size (default: 1)
            --ngram-max          Maximum n-gram size (default: 1)
            --model-dir          Directory to save the model (default: ./xgb_models)
            --model-name         Name of the model (default: xgb)
        """
        args = shlex.split(arg)
        parser = argparse.ArgumentParser(
            description='Tune XGBoost hyperparameters for text classification')
        parser.add_argument('--train-data-file', '--train_data_file', type=str, required=True,
                            help='Path to training data CSV file')
        parser.add_argument('--valid-data-file', '--valid_data_file', type=str, required=True,
                            help='Path to validation data CSV file')
        parser.add_argument('--text-column', '--text_column', type=str, default="Note_Column",
                            help='Name of the text column in the CSV')
        parser.add_argument('--label-column', '--label_column', type=str, default="Malnutrition_Label",
                            help='Name of the label column in the CSV')
        parser.add_argument('--id-column', '--id_column', type=str, default="Patient_ID",
                            help='Name of the ID column in the CSV')
        parser.add_argument('--max-features', '--max_features', type=int, default=10000,
                            help='Maximum number of features for vectorization')
        parser.add_argument('--remove-stop-words', '--remove_stop_words', action='store_true', default=False,
                            help='Remove stop words during preprocessing')
        parser.add_argument('--apply-stemming', '--apply_stemming', action='store_true', default=False,
                            help='Apply stemming during preprocessing')
        parser.add_argument('--vectorization-mode', '--vectorization_mode', type=str, default='tfidf',
                            choices=['tfidf', 'count', 'binary'], help='Vectorization mode')
        parser.add_argument('--ngram-min', '--ngram_min', type=int,
                            default=1, help='Minimum n-gram size')
        parser.add_argument('--ngram-max', '--ngram_max', type=int,
                            default=1, help='Maximum n-gram size')
        parser.add_argument('--model-dir', '--model_dir', type=str,
                            default='./xgb_models', help='Directory to save the model')
        parser.add_argument('--model-name', '--model_name', type=str,
                            default="xgb", help='Name of the model')

        try:
            parsed_args = parser.parse_args(args)

            # Normalize arguments to use underscores for module imports
            cmd_args = []
            for key, value in vars(parsed_args).items():
                if value is not None:
                    key = key.replace('-', '_')
                    if isinstance(value, bool) and value:
                        cmd_args.append(f"--{key}")
                    elif not isinstance(value, bool):
                        cmd_args.append(f"--{key}")
                        cmd_args.append(str(value))

            # Execute the XGBoost tuning script
            try:
                from tune_xgb import main as tune_main
                sys.argv = ['tune_xgb.py'] + cmd_args
                tune_main()
            except ImportError:
                print(
                    "Error: Module 'tune_xgb' not found. Make sure it's in your PYTHONPATH.")
            except Exception as e:
                print(f"Error running XGBoost tuning: {e}")

        except SystemExit:
            # argparse will exit if --help is called or arguments are invalid
            pass

    def do_trainxgb(self, arg):
        """
        Train an XGBoost model for text classification.

        Usage: trainxgb --data-file <data_file> [options]

        Options:
            --data-file           Path to training data CSV file (required)
            --text-column         Name of the text column in CSV (default: Note_Column)
            --label-column        Name of the label column in CSV (default: Malnutrition_Label)
            --id-column           Name of the ID column in CSV (default: Patient_ID)
            --config-dir          Path to best hyperparameter directory (default: xgb_models)
            --max-features        Maximum number of features for vectorization (default: 10000)
            --remove-stop-words   Remove stop words during preprocessing (flag)
            --apply-stemming      Apply stemming during preprocessing (flag)
            --vectorization-mode  Vectorization mode: tfidf, count, binary (default: tfidf)
            --ngram-min           Minimum n-gram size (default: 1)
            --ngram-max           Maximum n-gram size (default: 1)
            --model-dir           Directory to save the model (default: ./xgb_models)
            --model-name          Name of the model (default: xgb)

            # XGBoost parameters
            --eta                 Learning rate (default: 0.1)
            --max-depth           Maximum depth of trees (default: 6)
            --min-child-weight    Minimum sum of instance weight needed in a child (default: 1)
            --subsample           Subsample ratio of the training instances (default: 0.8)
            --colsample-bytree    Subsample ratio of columns when constructing each tree (default: 0.8)
        """
        args = shlex.split(arg)
        parser = argparse.ArgumentParser(
            description='Train an XGBoost model')

        # Required parameters
        parser.add_argument('--data-file', '--data_file', type=str, required=True,
                            help='Path to training data CSV file')

        # Data parameters
        parser.add_argument('--text-column', '--text_column', type=str, default="Note_Column",
                            help='Name of the text column in the CSV')
        parser.add_argument('--label-column', '--label_column', type=str, default="Malnutrition_Label",
                            help='Name of the label column in the CSV')
        parser.add_argument('--id-column', '--id_column', type=str, default="Patient_ID",
                            help='Name of the ID column in the CSV')

        # Preprocessing parameters
        parser.add_argument('--config-dir', '--config_dir', type=str, default="xgb_models",
                            help='Path to best hyperparameter directory')
        parser.add_argument('--max-features', '--max_features', type=int, default=10000,
                            help='Maximum number of features for vectorization')
        parser.add_argument('--remove-stop-words', '--remove_stop_words', action='store_true',
                            help='Remove stop words during preprocessing')
        parser.add_argument('--apply-stemming', '--apply_stemming', action='store_true',
                            help='Apply stemming during preprocessing')
        parser.add_argument('--vectorization-mode', '--vectorization_mode', type=str, default='tfidf',
                            choices=['tfidf', 'count', 'binary'], help='Vectorization mode')
        parser.add_argument('--ngram-min', '--ngram_min', type=int, default=1,
                            help='Minimum n-gram size')
        parser.add_argument('--ngram-max', '--ngram_max', type=int, default=1,
                            help='Maximum n-gram size')

        # Model parameters
        parser.add_argument('--model-dir', '--model_dir', type=str, default='./xgb_models',
                            help='Directory to save the model')
        parser.add_argument('--model-name', '--model_name', type=str, default="xgb",
                            help='Name of the model')

        # XGBoost parameters
        parser.add_argument('--eta', type=float, default=0.1,
                            help='Learning rate')
        parser.add_argument('--max-depth', '--max_depth', type=int, default=6,
                            help='Maximum depth of trees')
        parser.add_argument('--min-child-weight', '--min_child_weight', type=float, default=1,
                            help='Minimum sum of instance weight needed in a child')
        parser.add_argument('--subsample', type=float, default=0.8,
                            help='Subsample ratio of the training instances')
        parser.add_argument('--colsample-bytree', '--colsample_bytree', type=float, default=0.8,
                            help='Subsample ratio of columns when constructing each tree')

        try:
            parsed_args = parser.parse_args(args)

            # Normalize arguments to use underscores for module imports
            cmd_args = []
            for key, value in vars(parsed_args).items():
                if value is not None:
                    key = key.replace('-', '_')
                    if isinstance(value, bool) and value:
                        cmd_args.append(f"--{key}")
                    elif not isinstance(value, bool):
                        cmd_args.append(f"--{key}")
                        cmd_args.append(str(value))

            # Execute the XGBoost training script
            try:
                from train_xgb import main as train_main
                sys.argv = ['train_xgb.py'] + cmd_args
                train_main()
            except ImportError:
                print(
                    "Error: Module 'train_xgb' not found. Make sure it's in your PYTHONPATH.")
            except Exception as e:
                print(f"Error running XGBoost training: {e}")

        except SystemExit:
            # argparse will exit if --help is called or arguments are invalid
            pass

    def do_predictxgb(self, arg):
        """
        Make predictions using a trained XGBoost model.

        Usage: predictxgb (--data-file <data_file> | --text <text>) --model-name <model_name> [options]

        Options:
            --data-file        Path to the CSV test data file (mutually exclusive with --text)
            --text             Raw text input for prediction (mutually exclusive with --data-file)
            --model-name       Name of the model (default: xgb)
            --text-column      Name of the column containing text data (default: Note_Column)
            --id-column        Name of the column containing IDs (default: Patient_ID)
            --model-dir        Directory containing model artifacts (default: ./xgb_models)
            --output-dir       Directory to save prediction results (default: ./xgb_predictions)
            --explain          Generate explanations for predictions (flag)
            --top-n-features   Number of top features to include in explanation (default: 5)
            --debug            Enable extra debug logging (flag)
        """
        args = shlex.split(arg)
        parser = argparse.ArgumentParser(
            description='Make predictions using a trained XGBoost model')

        # Required parameters - model name is optional with default
        parser.add_argument('--model-name', '--model_name', type=str, default="xgb",
                            help='Name of the model')

        # Data input options (either file or text)
        data_group = parser.add_mutually_exclusive_group(required=True)
        data_group.add_argument('--data-file', '--data_file', type=str,
                                help='Path to the CSV test data file')
        data_group.add_argument('--text', type=str,
                                help='Raw text input for prediction')

        # CSV-specific parameters
        parser.add_argument('--text-column', '--text_column', type=str, default="Note_Column",
                            help='Name of the column containing text data')
        parser.add_argument('--id-column', '--id_column', type=str, default="Patient_ID",
                            help='Name of the column containing IDs')

        # Optional parameters
        parser.add_argument('--model-dir', '--model_dir', type=str, default='./xgb_models',
                            help='Directory containing model artifacts')
        parser.add_argument('--output-dir', '--output_dir', type=str, default='./xgb_predictions',
                            help='Directory to save prediction results')
        parser.add_argument('--explain', action='store_true',
                            help='Generate explanations for predictions')
        parser.add_argument('--top-n-features', '--top_n_features', type=int, default=5,
                            help='Number of top features to include in explanation')
        parser.add_argument('--debug', action='store_true',
                            help='Enable extra debug logging')

        try:
            parsed_args = parser.parse_args(args)

            # Normalize arguments to use underscores for module imports
            cmd_args = []
            for key, value in vars(parsed_args).items():
                if value is not None:
                    key = key.replace('-', '_')
                    if isinstance(value, bool) and value:
                        cmd_args.append(f"--{key}")
                    elif not isinstance(value, bool):
                        cmd_args.append(f"--{key}")
                        cmd_args.append(str(value))

            # Execute the XGBoost prediction script
            try:
                from xgb_predict import main as predict_main
                sys.argv = ['xgb_predict.py'] + cmd_args
                predict_main()
            except ImportError:
                print(
                    "Error: Module 'xgb_predict' not found. Make sure it's in your PYTHONPATH.")
            except Exception as e:
                print(f"Error running XGBoost prediction: {e}")

        except SystemExit:
            # argparse will exit if --help is called or arguments are invalid
            pass

    def do_predicttabpfn(self, arg):
        """
        Make predictions using a trained TabPFN model.

        Usage: predicttabpfn --model <model_dir> (--data-file <data_file> | --text <text>) [options]

        Options:
            --model                 Path to the directory containing model artifacts (required)
            --data-file             Path to the CSV file with data to predict on
            --text                  Single text input to predict on
            --text-column           Name of the column containing text data (default: Note_Column)
            --id-column             Name of the column containing IDs (default: Patient_ID)
            --output-dir            Directory to save prediction results (default: tabpfn_predictions)
            --run-name              Name for this prediction run (default: timestamp-based)
            --include-features      Include features in output (flag)
            --calculate-importance  Calculate feature importance (flag)
            --top-features          Number of top features to display (default: 20)
        """
        args = shlex.split(arg)
        parser = argparse.ArgumentParser(
            description='Make predictions using a trained TabPFN model')

        # Required parameter
        parser.add_argument('--model', type=str, required=True,
                            help='Path to the directory containing model artifacts')

        # Input options (one of these is required)
        input_group = parser.add_mutually_exclusive_group(required=True)
        input_group.add_argument('--data-file', '--data_file', type=str,
                                 help='Path to the CSV file with data to predict on')
        input_group.add_argument('--text', type=str,
                                 help='Single text input to predict on')

        # Optional parameters
        parser.add_argument('--text-column', '--text_column', type=str, default='Note_Column',
                            help='Name of the column containing text data (default: Note_Column)')
        parser.add_argument('--id-column', '--id_column', type=str, default='Patient_ID',
                            help='Name of the column containing IDs (default: Patient_ID)')
        parser.add_argument('--output-dir', '--output_dir', type=str, default='tabpfn_predictions',
                            help='Directory to save prediction results (default: tabpfn_predictions)')
        parser.add_argument('--run-name', '--run_name', type=str,
                            help='Name for this prediction run (default: timestamp-based)')
        parser.add_argument('--include-features', '--include_features', action='store_true',
                            help='Include features in output')
        parser.add_argument('--calculate-importance', '--calculate_importance', action='store_true',
                            help='Calculate feature importance')
        parser.add_argument('--top-features', '--top_features', type=int, default=20,
                            help='Number of top features to display (default: 20)')
        parser.add_argument('--model-name', '--model_name', type=str, default="tabpfn",
                            help='Name of the model type (default: tabpfn)')

        try:
            parsed_args = parser.parse_args(args)

            # Normalize arguments to use underscores for module imports
            cmd_args = []
            for key, value in vars(parsed_args).items():
                if value is not None:
                    key = key.replace('-', '_')
                    if isinstance(value, bool) and value:
                        cmd_args.append(f"--{key}")
                    elif not isinstance(value, bool):
                        cmd_args.append(f"--{key}")
                        cmd_args.append(str(value))

            # Execute the predict script
            try:
                from predict_tabpfn import main as predict_main
                sys.argv = ['predict_tabpfn.py'] + cmd_args
                predict_main()
            except ImportError:
                print(
                    "Error: Module 'predict_tabpfn' not found. Make sure it's in your PYTHONPATH.")
            except Exception as e:
                print(f"Error running TabPFN prediction: {e}")

        except SystemExit:
            # argparse will exit if --help is called or arguments are invalid
            pass

    def do_evaluatexgb(self, arg):
        """
        Evaluate a trained XGBoost model.

        Usage: evaluatexgb --data-file <data_file> --model-name <model_name> [options]

        Options:
            --model-name          Name of the model (default: xgb)
            --data-file           Path to the CSV test data file (required)
            --text-column         Name of the column containing text data (default: Note_Column)
            --label-column        Name of the column containing labels (default: Malnutrition_Label)
            --id-column           Name of the column containing IDs (default: Patient_ID)
            --model-dir           Directory containing model artifacts (default: ./xgb_models)
            --output-dir          Directory to save evaluation results (default: ./xgb_evaluation)
            --num-shap-samples    Number of samples for SHAP explanation (default: 100)
            --top-n-features      Number of top features to plot (default: 20)
            --debug               Enable extra debug logging (flag)
        """
        args = shlex.split(arg)
        parser = argparse.ArgumentParser(
            description='Evaluate a trained XGBoost model')

        # Required parameters
        parser.add_argument('--model-name', '--model_name', type=str, default="xgb",
                            help='Name of the model')
        parser.add_argument('--data-file', '--data_file', type=str, required=True,
                            help='Path to the CSV test data file')
        parser.add_argument('--text-column', '--text_column', type=str, default="Note_Column",
                            help='Name of the column containing text data')
        parser.add_argument('--label-column', '--label_column', type=str, default="Malnutrition_Label",
                            help='Name of the column containing labels')
        parser.add_argument('--id-column', '--id_column', type=str, default="Patient_ID",
                            help='Name of the column containing IDs')

        # Optional parameters
        parser.add_argument('--model-dir', '--model_dir', type=str, default='./xgb_models',
                            help='Directory containing model artifacts')
        parser.add_argument('--output-dir', '--output_dir', type=str, default='./xgb_evaluation',
                            help='Directory to save evaluation results')
        parser.add_argument('--num-shap-samples', '--num_shap_samples', type=int, default=100,
                            help='Number of samples for SHAP explanation')
        parser.add_argument('--top-n-features', '--top_n_features', type=int, default=20,
                            help='Number of top features to plot')
        parser.add_argument('--debug', action='store_true',
                            help='Enable extra debug logging')

        try:
            parsed_args = parser.parse_args(args)

            # Normalize arguments to use underscores for module imports
            cmd_args = []
            for key, value in vars(parsed_args).items():
                if value is not None:
                    key = key.replace('-', '_')
                    if isinstance(value, bool) and value:
                        cmd_args.append(f"--{key}")
                    elif not isinstance(value, bool):
                        cmd_args.append(f"--{key}")
                        cmd_args.append(str(value))

            # Execute the XGBoost evaluation script
            try:
                from evaluate_xgb import main as evaluate_main
                sys.argv = ['evaluate_xgb.py'] + cmd_args
                evaluate_main()
            except ImportError:
                print(
                    "Error: Module 'evaluate_xgb' not found. Make sure it's in your PYTHONPATH.")
            except Exception as e:
                print(f"Error running XGBoost evaluation: {e}")

        except SystemExit:
            # argparse will exit if --help is called or arguments are invalid
            pass

    def do_llminference(self, arg):
        """
        Run inference with a malnutrition detection model (base or fine-tuned).

        Usage: llminference --input-csv <csv_file> | --input-text <text> [options]

        Input Options (one required):
            --input-csv      Path to CSV file with patient notes
            --input-text     Single patient note as text string

        Model Options:
            --model-path     Path to fine-tuned model adapter weights (optional)
            --base-model     Base model name (default: unsloth/Phi-3-mini-4k-instruct)

        CSV Column Options:
            --text-column    Column name containing patient notes (default: Note_Column)
            --id-column      Column name containing sample IDs (default: Patient_ID)
            --label-column   Column name containing true labels (default: Malnutrition_Label)

        Few-shot Options:
            --examples-data     Path to few-shot examples CSV data (optional)
            --few-shot-count    Number of few-shot examples to use (default: 0)
            --balanced-examples Whether to balance positive/negative examples

        Output Options:
            --output-dir     Directory to save results (default: ./llm_inference_results)
            --output-csv     Name of output CSV file (default: malnutrition_predictions.csv)
            --print-report   Print evaluation report if labels available

        Inference Options:
            --use-flash-attention Use Flash Attention 2 if available
            --max-new-tokens     Maximum new tokens to generate (default: 256)
            --temperature        Temperature for sampling (default: 0.1)
            --min-p              Min-P sampling parameter (default: 0.0)
            --stream-output      Stream model output to console
            --seed               Random seed for reproducibility (default: 42)

        Example:
            llminference --input-csv data/test.csv --print-report --stream-output
            llminference --input-text "Patient presents with weight loss of 10kg in 3 months"
        """
        args = shlex.split(arg)
        parser = argparse.ArgumentParser(
            description="Run inference with a malnutrition detection model")

        # Input arguments
        group = parser.add_mutually_exclusive_group(required=True)
        group.add_argument('--input-csv', '--input_csv', type=str,
                           help="Path to CSV file with patient notes")
        group.add_argument('--input-text', '--input_text', type=str,
                           help="Single patient note as text string")

        # Model arguments
        parser.add_argument('--model-path', '--model_path', type=str, default=None,
                            help="Path to fine-tuned model adapter weights (optional)")
        parser.add_argument('--base-model', '--base_model', type=str,
                            default="unsloth/Phi-3-mini-4k-instruct",
                            help="Base model that was fine-tuned or to be used for inference")

        # CSV column options
        parser.add_argument('--text-column', '--text_column', type=str, default="Note_Column",
                            help="Column name in CSV containing patient notes")
        parser.add_argument('--id-column', '--id_column', type=str, default="Patient_ID",
                            help="Column name in CSV containing sample IDs")
        parser.add_argument('--label-column', '--label_column', type=str, default='Malnutrition_Label',
                            help="Column name in CSV containing true labels (optional)")

        # Few-shot settings
        parser.add_argument('--examples-data', '--examples_data', type=str, default=None,
                            help="Path to few-shot examples CSV data (optional)")
        parser.add_argument('--few-shot-count', '--few_shot_count', type=int, default=0,
                            help="Number of few-shot examples to use (default: 0 for zero-shot)")
        parser.add_argument('--balanced-examples', '--balanced_examples', action="store_true",
                            help="Whether to balance positive/negative few-shot examples")

        # Output arguments
        parser.add_argument('--output-dir', '--output_dir', type=str, default="./llm_inference_results",
                            help="Directory to save inference results")
        parser.add_argument('--output-csv', '--output_csv', type=str, default="malnutrition_predictions.csv",
                            help="Name of output CSV file")
        parser.add_argument('--print-report', '--print_report', action="store_true",
                            help="Print evaluation report to terminal (if labels available)")

        # Model inference settings
        parser.add_argument('--use-flash-attention', '--use_flash_attention', action="store_true",
                            help="Use Flash Attention 2 if available")
        parser.add_argument('--seed', type=int, default=42,
                            help="Random seed for reproducibility")
        parser.add_argument('--max-new-tokens', '--max_new_tokens', type=int, default=256,
                            help="Maximum number of new tokens to generate")
        parser.add_argument('--temperature', type=float, default=0.1,
                            help="Temperature for sampling")
        parser.add_argument('--min-p', '--min_p', type=float, default=0.0,
                            help="Min-P sampling parameter (optional)")
        parser.add_argument('--stream-output', '--stream_output', action="store_true",
                            help="Stream model output to console during generation")

        try:
            parsed_args = parser.parse_args(args)

            # Normalize arguments to use underscores for module imports
            cmd_args = []
            for key, value in vars(parsed_args).items():
                if value is not None:
                    key = key.replace('-', '_')
                    if isinstance(value, bool):
                        if value:
                            cmd_args.append(f"--{key}")
                    else:
                        cmd_args.append(f"--{key}")
                        cmd_args.append(str(value))

            # Execute the LLM inference script
            try:
                from llm_inference import main as llm_inference_main
                sys.argv = ['llm_inference.py'] + cmd_args
                llm_inference_main()
            except ImportError:
                print(
                    "Error: Module 'models.llm_models' not found. Make sure it's in your PYTHONPATH.")
            except Exception as e:
                print(f"Error: Failed to execute LLM inference: {e}")

        except SystemExit:
            # argparse will exit if --help is called or arguments are invalid
            pass

    def do_llmtrain(self, arg):
        """
        Train a malnutrition detection model with fine-tuning.

        Usage: llmtrain --train-data <train_csv> [options]

        Required Arguments:
            --train-data      Path to training CSV data

        Model Options:
            --model-name      Base model to use for fine-tuning (default: unsloth/Phi-3-mini-4k-instruct)
            --val-data        Path to validation CSV data (optional)
            --examples-data   Path to few-shot examples CSV data (optional)

        Data Options:
            --text-column     Name of the text column in CSV (default: Note_Column)
            --label-column    Name of the label column in CSV (default: Malnutrition_Label)

        Output Options:
            --output-dir      Directory for saving training outputs (default: ./llm_train_output)
            --model-output    Path to save the final model (default: ./llm_models)

        Training Parameters:
            --batch-size           Per-device training batch size (default: 32)
            --gradient-accumulation Number of gradient accumulation steps (default: 4)
            --learning-rate        Learning rate for training (default: 2e-4)
            --max-steps            Maximum number of training steps (default: 60)
            --max-seq-length       Maximum sequence length for tokenization (default: 1024)

        LoRA Parameters:
            --lora-r          LoRA r parameter/rank (default: 8)
            --lora-alpha      LoRA alpha parameter/scaling (default: 32)

        Miscellaneous Options:
            --seed            Random seed for reproducibility (default: 42)
            --use-flash-attention  Use Flash Attention 2 if available
            --report-to       Where to report metrics: none, tensorboard, wandb (default: none)

        Example:
            llmtrain --train-data data/train.csv --val-data data/val.csv --max-steps 100
        """
        args = shlex.split(arg)
        parser = argparse.ArgumentParser(
            description="Train a malnutrition detection model")

        # Model and data arguments
        parser.add_argument('--model-name', '--model_name', type=str, default="unsloth/Phi-3-mini-4k-instruct",
                            help="Base model to use for fine-tuning")
        parser.add_argument('--train-data', '--train_data', type=str, required=True,
                            help="Path to training CSV data")
        parser.add_argument('--val-data', '--val_data', type=str, default=None,
                            help="Path to validation CSV data (optional)")
        parser.add_argument('--examples-data', '--examples_data', type=str, default=None,
                            help="Path to few-shot examples CSV data (optional)")
        parser.add_argument('--text-column', '--text_column', type=str, default="Note_Column",
                            help="Name of the text column in the CSV")
        parser.add_argument('--label-column', '--label_column', type=str, default="Malnutrition_Label",
                            help="Name of the label column in the CSV")

        # Output arguments
        parser.add_argument('--output-dir', '--output_dir', type=str, default="./llm_train_output",
                            help="Directory for saving training outputs")
        parser.add_argument('--model-output', '--model_output', type=str, default="./llm_models",
                            help="Path to save the final model")

        # Training arguments
        parser.add_argument('--batch-size', '--batch_size', type=int, default=32,
                            help="Per-device training batch size")
        parser.add_argument('--gradient-accumulation', '--gradient_accumulation', type=int, default=4,
                            help="Number of gradient accumulation steps")
        parser.add_argument('--learning-rate', '--learning_rate', type=float, default=2e-4,
                            help="Learning rate for training")
        parser.add_argument('--max-steps', '--max_steps', type=int, default=60,
                            help="Maximum number of training steps")
        parser.add_argument('--max-seq-length', '--max_seq_length', type=int, default=1024,
                            help="Maximum sequence length for tokenization")

        # LoRA parameters
        parser.add_argument('--lora-r', '--lora_r', type=int, default=8,
                            help="LoRA r parameter (rank)")
        parser.add_argument('--lora-alpha', '--lora_alpha', type=int, default=32,
                            help="LoRA alpha parameter (scaling)")

        # Miscellaneous
        parser.add_argument('--seed', type=int, default=42,
                            help="Random seed for reproducibility")
        parser.add_argument('--use-flash-attention', '--use_flash_attention', action="store_true",
                            help="Use Flash Attention 2 if available")
        parser.add_argument('--report-to', '--report_to', type=str, default="none",
                            choices=["none", "tensorboard", "wandb"],
                            help="Where to report training metrics")

        try:
            parsed_args = parser.parse_args(args)

            # Normalize arguments to use underscores for module imports
            cmd_args = []
            for key, value in vars(parsed_args).items():
                if value is not None:
                    key = key.replace('-', '_')
                    if isinstance(value, bool):
                        if value:
                            cmd_args.append(f"--{key}")
                    else:
                        cmd_args.append(f"--{key}")
                        cmd_args.append(str(value))

            # Execute the LLM training script
            try:
                from finetune_llm import main as train_main
                sys.argv = ['llm_train.py'] + cmd_args
                train_main()
            except ImportError:
                print(
                    "Error: Module 'models.llm_models' not found. Make sure it's in your PYTHONPATH.")
            except Exception as e:
                print(f"Error: Failed to execute LLM training: {e}")

        except SystemExit:
            # argparse will exit if --help is called or arguments are invalid
            pass

    def do_ollamaserve(self, arg):
        """
        Serve a fine-tuned model with Ollama.

        Usage: ollama_serve --model-path <model_path> [options]

        Required Arguments:
            --model-path          Path to the fine-tuned model adapter weights

        Model Options:
            --base-model          Base model that was fine-tuned (default: microsoft/Phi-3-mini-4k-instruct)
            --model-name          Name to assign to the Ollama model (default: custom_model)
            --system-prompt       System prompt for the Ollama model (default: "You are a helpful assistant.")

        GGUF Conversion Options:
            --quantization        Quantization method: q4_k_m, q5_k_m, q8_0, f16 (default: q8_0)
            --output-dir          Directory to save GGUF converted model (default: ./ollama_model)
            --use-safetensors     Save intermediate model in safetensors format

        Ollama Settings:
            --ollama-port         Port number for Ollama server (default: 11434)
            --test-prompt         Test prompt to verify model functionality
            --stream-output       Stream model output during testing (flag)

        Device Options:
            --cpu                 Use CPU for model loading and conversion (default: use GPU)
            --max-memory          Max memory allocation for model, e.g. '12GiB'
            --batch-size          Batch size for conversion process (default: 1)

        Examples:
            ollama_serve --model-path ./lora/adapter_model --model-name nutrition_assistant
            ollama_serve --model-path ./trained_model --base-model meta-llama/Llama-2-7b-chat --quantization q4_k_m --cpu
        """

        args = shlex.split(arg)
        parser = argparse.ArgumentParser(
            description='Convert and serve fine-tuned models with Ollama')

        # Required arguments
        parser.add_argument('--model-path', '--model_path', type=str, required=True,
                            help='Path to the fine-tuned model adapter weights')

        # Model options
        parser.add_argument('--base-model', '--base_model', type=str, default="microsoft/Phi-3-mini-4k-instruct",
                            help='Base model that was fine-tuned (default: microsoft/Phi-3-mini-4k-instruct)')
        parser.add_argument('--model-name', '--model_name', type=str, default="custom_model",
                            help='Name to assign to the Ollama model (default: custom_model)')
        parser.add_argument('--system-prompt', '--system_prompt', type=str,
                            default="You are a helpful assistant.",
                            help='System prompt for the Ollama model (default: "You are a helpful assistant.")')

        # GGUF conversion options
        parser.add_argument('--quantization', type=str, default="q8_0",
                            choices=["q4_k_m", "q5_k_m", "q8_0", "f16"],
                            help='Quantization method for GGUF conversion (default: q8_0)')
        parser.add_argument('--output-dir', '--output_dir', type=str, default="./ollama_model",
                            help='Directory to save GGUF converted model (default: ./ollama_model)')
        parser.add_argument('--use-safetensors', '--use_safetensors', action='store_true',
                            help='Save intermediate model in safetensors format')

        # Ollama settings
        parser.add_argument('--ollama-port', '--ollama_port', type=int, default=11434,
                            help='Port number for Ollama server (default: 11434)')
        parser.add_argument('--test-prompt', '--test_prompt', type=str,
                            default="Hello! How can you help me today?",
                            help='Test prompt to verify model functionality')
        parser.add_argument('--stream-output', '--stream_output', action='store_true', default=True,
                            help='Stream model output during testing (default: True)')

        # Device options
        parser.add_argument('--cpu', action='store_true', default=False,
                            help='Use CPU for model loading and conversion (default: use GPU)')
        parser.add_argument('--max-memory', '--max_memory', type=str, default=None,
                            help='Max memory allocation for model, e.g. "12GiB"')
        parser.add_argument('--batch-size', '--batch_size', type=int, default=1,
                            help='Batch size for conversion process (default: 1)')

        try:
            parsed_args = parser.parse_args(args)

            # Normalize arguments to use underscores for module imports
            cmd_args = []
            for key, value in vars(parsed_args).items():
                key = key.replace('-', '_')
                if isinstance(value, bool):
                    if value:
                        cmd_args.append(f"--{key}")
                    continue
                cmd_args.append(f"--{key}")
                if not isinstance(value, bool):
                    cmd_args.append(str(value))

            # Execute the ollama serve script
            try:
                from serve_model_with_ollama import main as ollama_serve_main
                sys.argv = ['ollama_serve.py'] + cmd_args
                ollama_serve_main()
            except ImportError:
                print(
                    "Error: Module 'ollama_serve' not found. Make sure it's in your PYTHONPATH.")
            except Exception as e:
                print(f"Error: Failed to execute Ollama serve: {e}")

        except SystemExit:
            # argparse will exit if --help is called or arguments are invalid
            pass

    # --------------------- Help System Improvements -----------------

    def print_doc(self, doc):
        """Format and print documentation with proper wrapping."""
        clean_doc = textwrap.dedent(doc).strip()
        wrapped = textwrap.fill(clean_doc, width=80,
                                initial_indent='  ', subsequent_indent='  ')
        print(f"\n{wrapped}\n")

    def do_help(self, arg):
        """Show enhanced help system with categorized commands."""
        if not arg:
            self.print_welcome_banner()
            print("\n\033[1mCommand Categories:\033[0m")
            categories = {
                "File Operations": ['ls', 'pwd', 'cd', 'cat'],
                "Model Training": ['tunetextcnn', 'traintextcnn', 'traintabpfn', 'tunexgb', 'trainxgb', 'llmtrain'],
                "Model Evaluation": ['evaluatetextcnn', 'evaluatetabpfn', 'evaluatexgb'],
                "Model Inference": ['textcnnpredict', 'predicttabpfn', 'predictxgb', 'llminference', 'ollamaserve'],
                "System": ['clear', 'quit', 'help']
            }
            for category, cmds in categories.items():
                print(f"\n\033[1;34m{category}:\033[0m")
                for cmd_name in cmds:
                    func = getattr(self, 'do_' + cmd_name)
                    summary = func.__doc__.split(
                        '\n')[0] if func.__doc__ else "No description"
                    print(f"  \033[1m{cmd_name:<12}\033[0m {summary}")
            print("\nUse 'help <command>' for detailed documentation")
        else:
            super().do_help(arg)

    def print_welcome_banner(self):
        """Display enhanced welcome banner with color and version info."""
        print("\033[1;36m")
        print("╔══════════════════════════════════════════════════╗")
        print("║             NutriKids AI Console v1.0            ║")
        print("║      Malnutrition Detection Command Interface    ║")
        print("╚══════════════════════════════════════════════════╝")
        print("\033[0m")
        print(f"  System Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("  Type 'help' for command overview or 'help <command>' for details")


if __name__ == "__main__":
    console = NutrikidsAiCommand()
    if len(sys.argv) > 1:
        command = ' '.join(sys.argv[1:])
        console.onecmd(command)
    else:
        console.print_welcome_banner()
        console.cmdloop()
