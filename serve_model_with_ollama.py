#!/usr/bin/env python3
import os
import argparse
import subprocess
import time
import torch
from transformers import BitsAndBytesConfig
from unsloth import FastLanguageModel
import json
import requests


def parse_arguments():
    """Parse command line arguments for model serving."""
    parser = argparse.ArgumentParser(
        description="Convert and serve fine-tuned models with Ollama"
    )

    # Model arguments
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the fine-tuned model adapter weights")
    parser.add_argument("--base_model", type=str, default="unsloth/Phi-3-mini-4k-instruct",
                        help="Base model that was fine-tuned")

    # GGUF conversion settings
    parser.add_argument("--model_name", type=str, default="unsloth_model",
                        help="Name to assign to the Ollama model")
    parser.add_argument("--quantization", type=str, default="q8_0",
                        choices=["q4_k_m", "q5_k_m", "q8_0", "f16"],
                        help="Quantization method for GGUF conversion")
    parser.add_argument("--system_prompt", type=str, default="You are a helpful assistant.",
                        help="System prompt for the Ollama model")

    # Ollama settings
    parser.add_argument("--ollama_port", type=int, default=11434,
                        help="Port number for Ollama server")
    parser.add_argument("--test_prompt", type=str,
                        default="Analyze this patient note and determine if there are signs of malnutrition.",
                        help="Test prompt to verify model functionality")

    # Output directory
    parser.add_argument("--output_dir", type=str, default="./ollama_model",
                        help="Directory to save GGUF converted model")

    # HuggingFace integration (optional)
    parser.add_argument("--push_to_hub", action="store_true",
                        help="Push the converted model to HuggingFace Hub")
    parser.add_argument("--hf_username", type=str, default=None,
                        help="HuggingFace username if pushing to Hub")
    parser.add_argument("--hf_token", type=str, default=None,
                        help="HuggingFace token if pushing to Hub")

    return parser.parse_args()


def get_quantization_config():
    """Define quantization configuration for loading the model."""
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        llm_int8_enable_fp32_cpu_offload=True,
        llm_int8_threshold=6.0,
        llm_int8_has_fp16_weight=True
    )


def load_model_and_tokenizer(base_model, model_path, quantization_config):
    """Load base model with fine-tuned adapter weights and tokenizer.

    Args:
        base_model (str): Base model name.
        model_path (str): Path to fine-tuned model adapter weights.
        quantization_config: Quantization configuration.

    Returns:
        Tuple: (model, tokenizer)
    """
    print(
        f"Loading base model '{base_model}' with adapter weights from '{model_path}'...")
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=base_model,
            adapter_name=model_path,
            load_in_4bit=True,
            dtype=torch.float16,
            quantization_config=quantization_config,
            device_map="auto"
        )
        print("Model and tokenizer loaded successfully")
        FastLanguageModel.for_inference(model)
        return model, tokenizer
    except Exception as e:
        print(f"Error loading model: {e}")
        raise


def convert_to_gguf(model, tokenizer, args):
    """Convert model to GGUF format for Ollama.

    Args:
        model: The loaded fine-tuned model.
        tokenizer: The model tokenizer.
        args: Command line arguments.

    Returns:
        str: Path to the GGUF model directory
    """
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    print(f"Converting model to GGUF with {args.quantization} quantization...")
    try:
        model.save_pretrained_gguf(
            output_dir,
            tokenizer,
            quantization_method=args.quantization
        )
        print(f"Model converted and saved to {output_dir}")

        if args.push_to_hub and args.hf_username and args.hf_token:
            repo_id = f"{args.hf_username}/{args.model_name}"
            print(
                f"Pushing converted model to HuggingFace Hub at {repo_id}...")
            model.push_to_hub_gguf(
                repo_id,
                tokenizer,
                quantization_method=args.quantization,
                token=args.hf_token
            )
            print("Model successfully pushed to HuggingFace Hub")

        return output_dir
    except Exception as e:
        print(f"Error converting model to GGUF: {e}")
        raise


def create_custom_modelfile(model_dir, model_name, system_prompt):
    """Create a custom Modelfile for Ollama.

    Args:
        model_dir (str): Path to the GGUF model directory.
        model_name (str): Name to assign to the Ollama model.
        system_prompt (str): System prompt for the model.

    Returns:
        str: Path to the Modelfile
    """
    modelfile_path = os.path.join(model_dir, "Modelfile")

    # Find the GGUF file
    gguf_files = [f for f in os.listdir(model_dir) if f.endswith(".gguf")]
    if not gguf_files:
        raise FileNotFoundError("No GGUF files found in the output directory")
    gguf_file = gguf_files[0]

    # Use just the filename for the FROM directive, since Ollama will look for it in the same directory
    with open(modelfile_path, "w") as f:
        f.write(f"""FROM {gguf_file}
PARAMETER temperature 0.1
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER num_ctx 2048
PARAMETER stop "<|im_end|>"
PARAMETER stop "</s>"
SYSTEM "{system_prompt}"
""")

    print(f"Created custom Modelfile at {modelfile_path}")
    return modelfile_path


def start_ollama_server(port):
    """Start the Ollama server if not already running.

    Args:
        port (int): Port number for Ollama server.

    Returns:
        subprocess.Popen: Process object for the Ollama server
    """
    try:
        # Check if Ollama is already running
        response = requests.get(f"http://localhost:{port}/api/version")
        print(
            f"Ollama already running on port {port}, version: {response.json().get('version')}")
        return None
    except:
        print(f"Starting Ollama server on port {port}...")
        # Set environment variable for custom port
        env = os.environ.copy()
        env["OLLAMA_HOST"] = f"0.0.0.0:{port}"

        # Start Ollama server as a subprocess
        process = subprocess.Popen(
            ["ollama", "serve"],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        # Wait for server to start
        for _ in range(10):
            time.sleep(1)
            try:
                response = requests.get(f"http://localhost:{port}/api/version")
                if response.status_code == 200:
                    print(f"Ollama server started on port {port}")
                    return process
            except:
                continue

        print("Warning: Ollama server may not have started properly")
        return process


def create_ollama_model(model_name, modelfile_path, port):
    """Create an Ollama model from the Modelfile.

    Args:
        model_name (str): Name to assign to the Ollama model.
        modelfile_path (str): Path to the Modelfile.
        port (int): Port number for Ollama server.

    Returns:
        bool: True if successful, False otherwise
    """
    print(f"Creating Ollama model '{model_name}' from Modelfile...")
    try:
        # Get the directory containing the Modelfile
        model_dir = os.path.dirname(modelfile_path)

        # Change to that directory before running ollama create
        current_dir = os.getcwd()
        os.chdir(model_dir)

        # Run ollama create with just the filename, not the full path
        modelfile_name = os.path.basename(modelfile_path)
        result = subprocess.run(
            ["ollama", "create", model_name, "-f", modelfile_name],
            capture_output=True,
            text=True,
            check=True
        )

        # Change back to the original directory
        os.chdir(current_dir)

        print(f"Successfully created Ollama model: {model_name}")
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        # Change back to the original directory if an error occurred
        if 'current_dir' in locals():
            os.chdir(current_dir)

        print(f"Error creating Ollama model: {e}")
        print(f"Error output: {e.stderr}")
        return False


def test_ollama_model(model_name, test_prompt, port):
    """Test the Ollama model with a sample prompt.

    Args:
        model_name (str): Name of the Ollama model.
        test_prompt (str): Test prompt to use.
        port (int): Port number for Ollama server.

    Returns:
        dict: Response from the Ollama API
    """
    print(f"Testing Ollama model '{model_name}' with prompt: '{test_prompt}'")

    api_url = f"http://localhost:{port}/api/chat"
    request_data = {
        "model": model_name,
        "messages": [
            {"role": "user", "content": test_prompt}
        ]
    }

    try:
        response = requests.post(api_url, json=request_data)
        response_data = response.json()

        print("\n--- Model Response ---")
        print(response_data.get("message", {}).get("content", "No response"))
        print("---------------------\n")

        return response_data
    except Exception as e:
        print(f"Error testing Ollama model: {e}")
        return None


def main():
    """Main function to run the model serving pipeline."""
    args = parse_arguments()

    # Load model and tokenizer
    quantization_config = get_quantization_config()
    model, tokenizer = load_model_and_tokenizer(
        args.base_model, args.model_path, quantization_config
    )

    # Convert model to GGUF format
    model_dir = convert_to_gguf(model, tokenizer, args)

    # Create custom Modelfile
    modelfile_path = create_custom_modelfile(
        model_dir, args.model_name, args.system_prompt
    )

    # Start Ollama server if not already running
    server_process = start_ollama_server(args.ollama_port)

    # Create Ollama model
    success = create_ollama_model(
        args.model_name, modelfile_path, args.ollama_port)

    if success:
        # Test the model
        test_ollama_model(args.model_name, args.test_prompt, args.ollama_port)

        print(f"""
Model serving setup complete!

Your model is now available for inference through Ollama with the following details:
- Model name: {args.model_name}
- Quantization: {args.quantization}
- Ollama port: {args.ollama_port}

You can use it with the Ollama API:
curl http://localhost:{args.ollama_port}/api/chat -d '{{
    "model": "{args.model_name}",
    "messages": [
        {{"role": "user", "content": "Your prompt here"}}
    ]
}}'

Or with the Ollama CLI:
ollama run {args.model_name}
""")


if __name__ == "__main__":
    main()
