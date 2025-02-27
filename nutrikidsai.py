#!/usr/bin/env python3
"""
This module defines the command interpreter for NutriKids AI.
It allows users to interact with the application using a command-line interface.
"""

import cmd
import sys
import os
import argparse
import shlex
import textwrap

class NutrikidsAiCommand(cmd.Cmd):
    """
    Command interpreter class for NutriKids AI.
    """
    prompt = "BMIC% "
    
    def do_quit(self, arg):
        """Quit command: exit the program."""
        return True
    
    def do_EOF(self, arg):
        """EOF command: exit the program."""
        print("")
        return True
    
    def do_ls(self, arg):
        """List the contents of the current directory."""
        try:
            contents = os.listdir('.')
            for item in contents:
                print(item)
        except Exception as e:
            print(f"Error: {e}")

    def do_pwd(self, arg):
        """Print the current working directory."""
        try:
            cwd = os.getcwd()
            print(cwd)
        except Exception as e:
            print(f"Error: {e}")
    
    def do_cd(self, arg):
        """
        Change the current working directory.
        
        Usage: cd <directory_path>
        
        If no directory is specified, it will change to the user's home directory.
        """
        try:
            # If no argument is provided, change to home directory
            if not arg:
                home_dir = os.path.expanduser("~")
                os.chdir(home_dir)
                print(f"Changed directory to: {home_dir}")
            else:
                # Change to the specified directory
                os.chdir(arg)
                print(f"Changed directory to: {os.getcwd()}")
        except Exception as e:
            print(f"Error: {e}")


    def do_cat(self, arg):
        """
        Display the contents of a file.
        
        Usage: cat <file_path>
        
        Displays the entire contents of the specified file.
        """
        try:
            if not arg:
                print("Error: File path not specified. Usage: cat <file_path>")
                return
                
            # Check if file exists
            if not os.path.isfile(arg):
                print(f"Error: File '{arg}' not found.")
                return
                
            # Read and display file contents
            with open(arg, 'r') as file:
                content = file.read()
                print(content)
        except Exception as e:
            print(f"Error: {e}")


    def emptyline(self):
        """Overrides default emptyline behavior (does nothing)."""
        pass
    
    def do_tunetextcnn(self, arg):
        """
        Tune TextCNN hyperparameters using Ray Tune.
        
        Usage: tunetextcnn --train <train_file> --val <val_file> [options]
        
        Options:
            --train          Path to training CSV file (required)
            --val            Path to validation CSV file (required)
            --text-column    Name of the text column in CSV (default: Note_Column)
            --label-column   Name of the label column in CSV (default: Malnutrition_Label)
            --output-dir     Directory to save model artifacts (default: model_output)
            --num-samples    Number of parameter settings to sample (default: 10)
            --max-epochs     Maximum epochs for tuning (default: 5)
            --grace-period   Minimum epochs per trial (default: 3)
        """
        args = shlex.split(arg)
        parser = argparse.ArgumentParser(description='Tune TextCNN hyperparameters')
        parser.add_argument('--train', type=str, required=True)
        parser.add_argument('--val', type=str, required=True)
        parser.add_argument('--text-column', type=str, default='Note_Column')
        parser.add_argument('--label-column', type=str, default='Malnutrition_Label')
        parser.add_argument('--output-dir', type=str, default='model_output')
        parser.add_argument('--num-samples', type=int, default=10)
        parser.add_argument('--max-epochs', type=int, default=10)
        parser.add_argument('--grace-period', type=int, default=5)
        
        try:
            parsed_args = parser.parse_args(args)
            
            # Convert namespace to command line args
            cmd_args = []
            for key, value in vars(parsed_args).items():
                cmd_args.append(f"--{key}")
                cmd_args.append(str(value))
            
            # Execute the tune script
            from textcnn_raytune import main as tune_main
            sys.argv = ['textcnn_raytune.py'] + cmd_args
            tune_main()
            
        except SystemExit:
            # argparse will exit if --help is called or arguments are invalid
            pass
    
    def do_traintextcnn(self, arg):
        """
        Train TextCNN with best hyperparameters.
        
        Usage: traintextcnn --train <train_file> --val <val_file> --config <config_file> [options]
        
        Options:
            --train          Path to training CSV file (required)
            --val            Path to validation CSV file (required)
            --config         Path to best configuration joblib file (required)
            --text-column    Name of the text column in CSV (default: Note_Column)
            --label-column   Name of the label column in CSV (default: Malnutrition_Label)
            --output-dir     Directory to save model artifacts (default: model_output)
            --epochs         Number of epochs for final training (default: 10)
        """
        args = shlex.split(arg)
        parser = argparse.ArgumentParser(description='Train TextCNN with best hyperparameters')
        parser.add_argument('--train', type=str, required=True)
        parser.add_argument('--val', type=str, required=True)
        parser.add_argument('--config', type=str, required=True)
        parser.add_argument('--text-column', type=str, default='Note_Column')
        parser.add_argument('--label-column', type=str, default='Malnutrition_Label')
        parser.add_argument('--output-dir', type=str, default='model_output')
        parser.add_argument('--epochs', type=int, default=10)
        
        try:
            parsed_args = parser.parse_args(args)
            
            # Convert namespace to command line args
            cmd_args = []
            for key, value in vars(parsed_args).items():
                cmd_args.append(f"--{key}")
                cmd_args.append(str(value))
            
            # Execute the train script
            from traincnn import main as train_main
            sys.argv = ['traincnn.py'] + cmd_args
            train_main()
            
        except SystemExit:
            # argparse will exit if --help is called or arguments are invalid
            pass
    
    def do_evaluate_textcnn(self, arg):
        """
        Evaluate TextCNN model on test data.
        
        Usage: evaluate_textcnn --test <test_file> [options]
        
        Options:
            --test           Path to test CSV file (required)
            --text-column    Name of the text column in CSV (default: Note_Column)
            --label-column   Name of the label column in CSV (default: Malnutrition_Label)
            --model-dir      Directory containing model artifacts (default: model_output)
            --output-dir     Directory to save evaluation artifacts (default: evaluation_output)
        """
        args = shlex.split(arg)
        parser = argparse.ArgumentParser(description='Evaluate TextCNN model')
        parser.add_argument('--test', type=str, required=True)
        parser.add_argument('--text-column', type=str, default='Note_Column')
        parser.add_argument('--label-column', type=str, default='Malnutrition_Label')
        parser.add_argument('--model-dir', type=str, default='model_output')
        parser.add_argument('--output-dir', type=str, default='evaluation_output')
        
        try:
            parsed_args = parser.parse_args(args)
            
            # Convert namespace to command line args
            cmd_args = []
            for key, value in vars(parsed_args).items():
                cmd_args.append(f"--{key}")
                cmd_args.append(str(value))
            
            # Execute the evaluate script
            from evaluate_cnn import main as evaluate_main
            sys.argv = ['evaluate_cnn.py'] + cmd_args
            evaluate_main()
            
        except SystemExit:
            # argparse will exit if --help is called or arguments are invalid
            pass
    
    def do_cnnpredict(self, arg):
        """
        Make predictions with TextCNN model.
        
        Usage: cnnpredict --input <input_file_or_text> [options]
        
        Options:
            --input             Path to input CSV file or text string (required)
            --text-column       Name of the text column in CSV (default: Note_Column)
            --model-dir         Directory containing model artifacts (default: model_output)
            --output            Path to output predictions file (default: predictions.csv)
            --explain           Generate explanations for predictions (flag)
            --explain-method    Explanation method to use: integrated, permutation, occlusion, all (default: integrated)
            --explanation-dir   Directory to save explanation visualizations (default: explanations)
            --num-samples       Number of samples to explain for batch methods (default: 5)
        """
        
        args = shlex.split(arg)
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
        
        try:
            parsed_args = parser.parse_args(args)
            
            # Convert namespace to command line args
            cmd_args = []
            for key, value in vars(parsed_args).items():
                if key == 'explain' and value:
                    cmd_args.append('--explain')
                    continue
                if isinstance(value, bool) and not value:
                    continue
                cmd_args.append(f"--{key}")
                if not isinstance(value, bool):
                    cmd_args.append(str(value))
            
            # Execute the predict script
            from textcnnpredict import main as predict_main
            sys.argv = ['predict.py'] + cmd_args
            predict_main()
            
        except SystemExit:
            # argparse will exit if --help is called or arguments are invalid
            pass
    
    def do_traintabpfn(self, arg):
        """
        Train a TabPFN classifier on text data.
        
        Usage: traintabpfn --data-file <data_file> [options]
        
        Options:
            --data-file           Path to the CSV data file (required)
            --text-column         Column containing text data (default: Note_Column)
            --label-column        Column containing labels (default: Malnutrition_Label)
            --id-column           Column containing IDs (default: Patient_ID)
            --max-features        Max number of features to extract (default: 8000)
            --remove-stop-words   Remove stop words (flag)
            --apply-stemming      Apply stemming (default: True)
            --vectorization-mode  Vectorization mode: count or tfidf (default: tfidf)
            --ngram-min           Minimum n-gram size (default: 1)
            --ngram-max           Maximum n-gram size (default: 1)
            --device              Device to use: cpu or cuda (default: cpu)
            --model-dir           Directory to save models and artifacts (default: model_dir)
        """
        args = shlex.split(arg)
        parser = argparse.ArgumentParser(description='Train a TabPFN classifier on text data')
        
        # Data parameters - using hyphens consistently in argument names
        parser.add_argument('--data-file', type=str, required=True, help='Path to the CSV data file')
        parser.add_argument('--text-column', type=str, default="Note_Column", help='Column containing text data')
        parser.add_argument('--label-column', type=str, default="Malnutrition_Label", help='Column containing labels')
        parser.add_argument('--id-column', type=str, default="Patient_ID", help='Column containing IDs')
        
        # Text processing parameters
        parser.add_argument('--max-features', type=int, default=8000, help='Max number of features to extract')
        parser.add_argument('--remove-stop-words', action='store_true', help='Remove stop words')
        parser.add_argument('--apply-stemming', action='store_false', help='Apply stemming')
        parser.add_argument('--vectorization-mode', type=str, default='tfidf', choices=['count', 'tfidf'], help='Vectorization mode')
        parser.add_argument('--ngram-min', type=int, default=1, help='Minimum n-gram size')
        parser.add_argument('--ngram-max', type=int, default=1, help='Maximum n-gram size')
        
        # Model parameters
        parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'], help='Device to use')
        
        # Output parameters
        parser.add_argument('--model-dir', type=str, default='tabpfn_model', help='Directory to save all models and artifacts')
        
        try:
            parsed_args = parser.parse_args(args)
            
            # Convert namespace to command line args
            cmd_args = []
            for key, value in vars(parsed_args).items():
                if isinstance(value, bool):
                    # Handle boolean flags properly 
                    if value and key == 'remove_stop_words':
                        cmd_args.append('--remove-stop-words')
                    elif not value and key == 'apply_stemming':
                        cmd_args.append('--apply-stemming')
                    continue
                # Use the key directly without replacing underscores
                cmd_args.append(f"--{key}")
                cmd_args.append(str(value))
            
            # Execute the train script
            from train_tabpfn import main as train_main
            sys.argv = ['traintabpfn.py'] + cmd_args
            train_main()
            
        except SystemExit:
            # argparse will exit if --help is called or arguments are invalid
            pass
    
    def do_evaluatetabpfn(self, arg):
        """
        Evaluate a trained TabPFN model.
        
        Usage: evaluatetabpfn --model <model_dir> --data-file <data_file> [options]
        
        Options:
            --model               Path to the directory containing model artifacts (required)
            --data-file           Path to the CSV test data file (required)
            --text-column         Name of the column containing text data (default: Note_Column)
            --label-column        Name of the column containing labels (default: Malnutrition_Label)
            --id-column           Name of the column containing IDs (default: Patient_ID)
            --output-dir          Directory to save evaluation results (default: model_output/tabpfn)
            --model-name          Name to use for saved artifacts (default: tabpfn)
        """
        args = shlex.split(arg)
        parser = argparse.ArgumentParser(description='Evaluate a trained TabPFN model')
        
        # Required parameter
        parser.add_argument('--model', type=str, required=True, help='Path to the directory containing model artifacts')
        
        # Data parameters
        parser.add_argument('--data-file', type=str, required=True, help='Path to the CSV test data file')
        parser.add_argument('--text-column', type=str, default="Note_Column", help='Name of the column containing text data')
        parser.add_argument('--label-column', type=str, default="Malnutrition_Label", help='Name of the column containing labels')
        parser.add_argument('--id-column', type=str, default="Patient_ID", help='Name of the column containing IDs')
        
        # Optional parameters
        parser.add_argument('--output-dir', type=str, default='model_output/tabpfn', help='Directory to save evaluation results')
        parser.add_argument('--model-name', type=str, default='tabpfn', help='Name to use for saved artifacts')
        
        try:
            parsed_args = parser.parse_args(args)
            
            # Convert namespace to command line args
            cmd_args = []
            for key, value in vars(parsed_args).items():
                cmd_args.append(f"--{key}")
                cmd_args.append(str(value))
            
            # Execute the evaluate script
            from evaluate_tabpfn import main as evaluate_main
            sys.argv = ['evaluatetabpfn.py'] + cmd_args
            evaluate_main()
            
        except SystemExit:
            # argparse will exit if --help is called or arguments are invalid
            pass
    
    def do_predicttabpfn(self, arg):
        """
        Make predictions using a trained TabPFN model.
        
        Usage: predicttabpfn --model <model_dir> (--data-file <data_file> | --text <text>) [options]
        
        Options:
            --model                 Path to the directory containing model artifacts (required)
            --data-file             Path to the CSV file with data to predict on (mutually exclusive with --text)
            --text                  Single text input to predict on (mutually exclusive with --data-file)
            --text-column           Name of the column containing text data (for CSV input)
            --id-column             Name of the column containing IDs (for CSV input)
            --output-dir            Directory to save all prediction artifacts
            --run-name              Name for this prediction run (default: timestamp-based)
            --include-features      Include features in output (flag)
            --calculate-importance  Calculate feature importance (flag)
            --top-features          Number of top features to display in importance analysis (default: 20)
        """
        args = shlex.split(arg)
        parser = argparse.ArgumentParser(description='Make predictions using a trained TabPFN model')
        
        # Required parameter
        parser.add_argument('--model', type=str, required=True, help='Path to the directory containing model artifacts')
        
        # Input options (one of these is required)
        input_group = parser.add_mutually_exclusive_group(required=True)
        input_group.add_argument('--data-file', type=str, help='Path to the CSV file with data to predict on')
        input_group.add_argument('--text', type=str, help='Single text input to predict on')
        
        # Optional parameters for CSV input
        parser.add_argument('--text-column', type=str, default="Note_Column", help='Name of the column containing text data')
        parser.add_argument('--id-column', type=str, default="Patient_ID", help='Name of the column containing IDs')
        
        # Output parameters
        parser.add_argument('--output-dir', type=str, default='predictions', help='Directory to save all prediction artifacts')
        parser.add_argument('--run-name', type=str, help='Name for this prediction run (default: timestamp-based)')
        parser.add_argument('--include-features', action='store_true', help='Include features in output')
        
        # Feature importance parameters
        parser.add_argument('--calculate-importance', action='store_true', help='Calculate feature importance')
        parser.add_argument('--top-features', type=int, default=20, help='Number of top features to display in importance analysis')
        
        try:
            parsed_args = parser.parse_args(args)
            
            # Convert namespace to command line args
            cmd_args = []
            for key, value in vars(parsed_args).items():
                if key in ['include_features', 'calculate_importance'] and value:
                    cmd_args.append(f"--{key.replace('_', '-')}")
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
    
    def do_tunexgb(self, arg):
        """
        Tune XGBoost hyperparameters for text classification.
        
        Usage: tunexgb --train_data_file <train_file> --valid_data_file <valid_file> [options]
        
        Options:
            --train_data_file      Path to training data CSV file (required)
            --valid_data_file      Path to validation data CSV file (required)
            --text_column          Name of the text column in CSV (default: Note_Column)
            --label_column         Name of the label column in CSV (default: Malnutrition_Label)
            --id_column            Name of the ID column in CSV (default: Patient_ID)
            --max_features         Maximum number of features for vectorization (default: 10000)
            --remove_stop_words    Remove stop words during preprocessing (flag)
            --apply_stemming       Apply stemming during preprocessing (flag)
            --vectorization_mode   Vectorization mode: tfidf, count, binary (default: tfidf)
            --ngram_min            Minimum n-gram size (default: 1)
            --ngram_max            Maximum n-gram size (default: 1)
            --model_name           Name of the model (default: xgb)
            --model_dir            Directory to save the model (default: ./xgbtune_params)
        """
        args = shlex.split(arg)
        
        try:
            # Execute the XGBoost tuning script
            from xgbraytune import parse_arguments, main as xgb_tune_main
            sys.argv = ['xgbraytune.py'] + args
            args = parse_arguments()
            xgb_tune_main(args)
            
        except SystemExit:
            # argparse will exit if --help is called or arguments are invalid
            pass
        except Exception as e:
            print(f"Error running XGBoost tuning: {e}")

    def do_predictxgb(self, arg):
        """
        Make predictions using a trained XGBoost model.
        
        Usage: predictxgb (--data_file <data_file> | --text <text>) --model_name <model_name> [options]
        
        Options:
            --data_file        Path to the CSV test data file (mutually exclusive with --text)
            --text             Raw text input for prediction (mutually exclusive with --data_file)
            --model_name       Name of the model (default: xgb)
            --text_column      Name of the column containing text data (default: Note_Column)
            --id_column        Name of the column containing IDs (default: Patient_ID)
            --model_dir        Directory containing model artifacts (default: ./xgb_models)
            --output_dir       Directory to save prediction results (default: ./xgb_predictions)
            --explain          Generate explanations for predictions (flag)
            --top_n_features   Number of top features to include in explanation (default: 5)
            --debug            Enable extra debug logging (flag)
        """
        args = shlex.split(arg)
        
        try:
            # Execute the XGBoost prediction script
            from xgbpredict import main as xgb_predict_main
            sys.argv = ['xgbpredict.py'] + args
            xgb_predict_main()
            
        except SystemExit:
            # argparse will exit if --help is called or arguments are invalid
            pass
        except Exception as e:
            print(f"Error running XGBoost prediction: {e}")

    def do_evaluatexgb(self, arg):
        """
        Evaluate a trained XGBoost model.
        
        Usage: evaluatexgb --data_file <data_file> --model_name <model_name> [options]
        
        Options:
            --model_name          Name of the model (default: xgb)
            --data_file           Path to the CSV test data file (required)
            --text_column         Name of the column containing text data (default: Note_Column)
            --label_column        Name of the column containing labels (default: Malnutrition_Label)
            --id_column           Name of the column containing IDs (default: Patient_ID)
            --model_dir           Directory containing model artifacts (default: ./xgb_models)
            --output_dir          Directory to save evaluation results (default: ./xgb_evaluation)
            --num_shap_samples    Number of samples for SHAP explanation (default: 100)
            --top_n_features      Number of top features to plot (default: 20)
            --debug               Enable extra debug logging (flag)
        """
        args = shlex.split(arg)
        
        try:
            # Execute the XGBoost evaluation script
            from evaluate_xgb import main as xgb_evaluate_main
            sys.argv = ['evaluate_xgb.py'] + args
            xgb_evaluate_main()
            
        except SystemExit:
            # argparse will exit if --help is called or arguments are invalid
            pass
        except Exception as e:
            print(f"Error running XGBoost evaluation: {e}")
            
    
    def print_help_formatted(self, command_list, header=None):
        """Print formatted help for a list of commands with descriptions."""
        if header:
            print(header)
        
        # Find the maximum command length for proper alignment
        max_cmd_len = max(len(cmd) for cmd, _ in command_list)
        
        # Print each command with its description
        for cmd, desc in command_list:
            print(f"  {cmd:{max_cmd_len}}  - {desc}")
    
    def print_welcome_banner(self):
        """Print the welcome banner with NUTRIKIDS AI prominently displayed."""
        print("\n")
        print("╔═════════════════════════════════════════════════════════════════╗")
        print("║                                                                 ║")
        print("║   ███╗   ██╗██╗   ██╗████████╗██████╗ ██╗██╗  ██╗██╗██████╗    ║")
        print("║   ████╗  ██║██║   ██║╚══██╔══╝██╔══██╗██║██║ ██╔╝██║██╔══██╗   ║")
        print("║   ██╔██╗ ██║██║   ██║   ██║   ██████╔╝██║█████╔╝ ██║██║  ██║   ║")
        print("║   ██║╚██╗██║██║   ██║   ██║   ██╔══██╗██║██╔═██╗ ██║██║  ██║   ║")
        print("║   ██║ ╚████║╚██████╔╝   ██║   ██║  ██║██║██║  ██╗██║██████╔╝   ║")
        print("║   ╚═╝  ╚═══╝ ╚═════╝    ╚═╝   ╚═╝  ╚═╝╚═╝╚═╝  ╚═╝╚═╝╚═════╝    ║")
        print("║                                                                 ║")
        print("║                    AI Command Interface                         ║")
        print("║                                                                 ║")
        print("╚═════════════════════════════════════════════════════════════════╝\n")

    def do_help(self, arg):
        """Show help message for commands."""
        if not arg:
            
            # Group commands by category
            ml_commands = [
                ("tunetextcnn", "Tune TextCNN hyperparameters using Ray Tune"),
                ("traintextcnn", "Train TextCNN with best hyperparameters"),
                ("evaluate_textcnn", "Evaluate TextCNN model on test data"),
                ("cnnpredict", "Make predictions with TextCNN model"),
                ("traintabpfn", "Train a TabPFN classifier on text data"),
                ("evaluatetabpfn", "Evaluate a trained TabPFN model"),
                ("predicttabpfn", "Make predictions using a trained TabPFN model"),
                ("tunexgb", "Tune XGBoost hyperparameters for text classification"),
                ("predictxgb", "Make predictions using a trained XGBoost model"),
                ("evaluatexgb", "Evaluate a trained XGBoost model")
            ]
            
            file_commands = [
                ("ls", "List the contents of the current directory"),
                ("pwd", "Print the current working directory"),
                ("cd", "Change the current working directory"),
                ("cat", "Display the contents of a file")
            ]
            
            system_commands = [
                ("help", "Show this help message or help for a specific command"),
                ("quit", "Exit the program")
            ]
            
            # Print each category
            self.print_help_formatted(ml_commands, "\nMachine Learning Commands:")
            self.print_help_formatted(file_commands, "\nFile System Commands:")
            self.print_help_formatted(system_commands, "\nSystem Commands:")
            
            print("\nUse 'help <command>' for detailed information about a specific command.")
            print("\nExample:")
            print("  BMIC% tunetextcnn --train data/train.csv --val data/val.csv --max-epochs 15")
        else:
            # Get the docstring for the command
            try:
                func = getattr(self, 'do_' + arg)
                if func.__doc__:
                    # Format the docstring for better readability
                    doc = func.__doc__.strip()
                    print("\n" + textwrap.dedent(doc) + "\n")
                else:
                    print(f"\nNo detailed help available for '{arg}'.\n")
            except AttributeError:
                print(f"\nCommand '{arg}' not found. Type 'help' for a list of available commands.\n")

if __name__ == "__main__":
    nutrikids_cmd = NutrikidsAiCommand()
    if len(sys.argv) > 1:
        command = ' '.join(sys.argv[1:])
        nutrikids_cmd.onecmd(command)
    else:
        nutrikids_cmd.print_welcome_banner()
        print("Welcome to the NUTRIKIDS AI Command Interface")
        print("Type 'help' for a list of commands.")
        nutrikids_cmd.cmdloop()