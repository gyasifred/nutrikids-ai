#!/usr/bin/env python3
import os
import argparse
import subprocess
import time
import torch
import json
import requests
import gc
from transformers import BitsAndBytesConfig, TextStreamer, AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def parse_arguments():
    """Parse command line arguments for model serving."""
    parser = argparse.ArgumentParser(
        description="Convert and serve fine-tuned models with Ollama"
    )

    # Model arguments
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the fine-tuned model adapter weights")
    parser.add_argument("--base_model", type=str, default="microsoft/Phi-3-mini-4k-instruct",
                        help="Base model that was fine-tuned")

    # GGUF conversion settings
    parser.add_argument("--model_name", type=str, default="nutrikidai_model",
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
    parser.add_argument("--stream_output", action="store_true", default=True,
                        help="Stream model output during testing")

    # Output directory
    parser.add_argument("--output_dir", type=str, default="./ollama_model",
                        help="Directory to save GGUF converted model")

    # Device options
    parser.add_argument("--cpu", action="store_true", default=False,
                        help="Use CPU for model loading and conversion (default: use GPU)")
    parser.add_argument("--use_safetensors", action="store_true",
                        help="Save intermediate model in safetensors format to avoid memory issues")

    # Memory management
    parser.add_argument("--max_memory", type=str, default=None,
                        help="Max memory allocation for model, e.g. '12GiB'")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size for conversion process")

    return parser.parse_args()


def get_quantization_config(args):
    """Define quantization configuration based on device."""
    if not args.cpu and torch.cuda.is_available():
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    return None


def load_model_and_tokenizer(base_model, model_path, args):
    """Load model and tokenizer with appropriate method based on GPU usage."""
    print(
        f"Loading base model '{base_model}' with adapter weights from '{model_path}'...")

    try:
        if not args.cpu and torch.cuda.is_available():
            return load_gpu_model(base_model, model_path, args)
        else:
            return load_cpu_model(base_model, model_path, args)
    except Exception as e:
        print(f"Error loading model: {e}")
        raise


def load_gpu_model(base_model, model_path, args):
    """Load model using Unsloth's optimizations for GPU."""
    from unsloth import FastLanguageModel

    print("Using Unsloth optimizations on GPU")

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model,
        adapter_name=model_path,
        load_in_4bit=True,
        dtype=torch.float16,
        quantization_config=quantization_config,
        device_map="auto",
        max_memory=parse_max_memory(
            args.max_memory) if args.max_memory else None,
    )
    FastLanguageModel.for_inference(model)
    return model, tokenizer


def load_cpu_model(base_model, model_path, args):
    """Load model using standard Transformers/PEFT for CPU."""
    print("Using standard Transformers/PEFT loading on CPU")

    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map="cpu",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    # Load adapter weights if provided
    if model_path and os.path.exists(model_path):
        print(f"Loading adapter weights from {model_path}")
        model = PeftModel.from_pretrained(model, model_path)
        model = model.merge_and_unload()

    return model, tokenizer


def parse_max_memory(max_memory_str):
    """Parse max memory string into device allocation dictionary."""
    return {0: max_memory_str, "cpu": "16GiB"}


def test_model_with_streaming(model, tokenizer, test_prompt, args):
    """Test the model with streaming before converting to GGUF."""
    print(f"\nTesting model with prompt: '{test_prompt}'")

    messages = [{"role": "user", "content": test_prompt}]
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    print(f"Using device: {model.device}")

    streamer = TextStreamer(tokenizer, skip_prompt=True)

    try:
        print("\n--- Model Response ---")
        _ = model.generate(
            input_ids,
            streamer=streamer,
            max_new_tokens=512,
            temperature=0.1,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
        print("\n---------------------\n")
    except Exception as e:
        print(f"Generation error: {e}")
        if not args.cpu:
            print("Attempting CPU fallback for generation test...")
            model = model.cpu()
            input_ids = input_ids.cpu()
            _ = model.generate(
                input_ids,
                streamer=streamer,
                max_new_tokens=512,
                temperature=0.1,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )


def convert_to_gguf(model, tokenizer, args):
    """Convert model to GGUF format with safety checks."""
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    try:
        # Save merged model first for CPU conversions
        if args.cpu or args.use_safetensors:
            interim_dir = os.path.join(output_dir, "merged_model")
            os.makedirs(interim_dir, exist_ok=True)

            print("Saving merged model for conversion...")
            model.save_pretrained(interim_dir, safe_serialization=True)
            tokenizer.save_pretrained(interim_dir)

            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            model = AutoModelForCausalLM.from_pretrained(
                interim_dir,
                device_map="cpu",
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
            )

        # Actual conversion logic
        print(f"Converting to GGUF with {args.quantization} quantization...")
        # Use only the supported parameters for save_pretrained_gguf
        model.save_pretrained_gguf(
            args.output_dir,
            tokenizer,
            quantization_method=args.quantization
        )
        return output_dir

    except Exception as e:
        print(f"GGUF conversion failed: {e}")
        print("Attempting fallback conversion...")
        return fallback_conversion(args)


def fallback_conversion(args):
    """Fallback conversion using llama.cpp."""
    print("Using llama.cpp for fallback conversion...")

    interim_dir = os.path.join(args.output_dir, "merged_model")
    os.makedirs(interim_dir, exist_ok=True)

    # Load and save model using CPU-only method
    model, tokenizer = load_cpu_model(args.base_model, args.model_path, args)
    model.save_pretrained(interim_dir, safe_serialization=True)
    tokenizer.save_pretrained(interim_dir)

    # Clone and build llama.cpp if needed
    llama_cpp_dir = os.path.expanduser("~/llama.cpp")
    if not os.path.exists(llama_cpp_dir):
        subprocess.run(
            ["git", "clone", "https://github.com/ggerganov/llama.cpp.git", llama_cpp_dir], check=True)
        subprocess.run(["make", "-C", llama_cpp_dir], check=True)

    # Convert to GGUF
    gguf_file = os.path.join(args.output_dir, f"{args.model_name}.gguf")
    convert_cmd = [
        "python3",
        os.path.join(llama_cpp_dir, "convert.py"),
        interim_dir,
        "--outfile", gguf_file,
        "--outtype", args.quantization
    ]

    subprocess.run(convert_cmd, check=True)
    return args.output_dir


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
    except Exception as e:
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
        max_attempts = 30
        for attempt in range(max_attempts):
            time.sleep(2)  # Wait longer between attempts
            try:
                response = requests.get(f"http://localhost:{port}/api/version")
                if response.status_code == 200:
                    print(f"Ollama server started on port {port}")
                    return process
            except Exception as e:
                if attempt == max_attempts - 1:
                    print(
                        f"Failed to connect to Ollama after {max_attempts} attempts: {e}")
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


def test_ollama_model(model_name, test_prompt, port, stream_output=True):
    """Test the Ollama model with a sample prompt.

    Args:
        model_name (str): Name of the Ollama model.
        test_prompt (str): Test prompt to use.
        port (int): Port number for Ollama server.
        stream_output (bool): Whether to stream the output.

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
    args = parse_arguments()

    # Device configuration
    if args.cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        torch.set_default_device("cpu")
        print("Using CPU for all operations")
    else:
        print(
            f"Using GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU available'}")

    try:
        # Load model with appropriate method
        model, tokenizer = load_model_and_tokenizer(
            args.base_model, args.model_path, args)

        # Test model functionality
        test_model_with_streaming(model, tokenizer, args.test_prompt, args)

        # Convert to GGUF format
        model_dir = convert_to_gguf(model, tokenizer, args)

        # Create Ollama Modelfile
        modelfile_path = create_custom_modelfile(
            model_dir, args.model_name, args.system_prompt)

        # Start Ollama server
        server_process = start_ollama_server(args.ollama_port)

        # Create and test Ollama model
        if create_ollama_model(args.model_name, modelfile_path, args.ollama_port):
            test_ollama_model(args.model_name, args.test_prompt,
                              args.ollama_port, args.stream_output)

    except Exception as e:
        print(f"Pipeline failed: {e}")
        exit(1)


if __name__ == "__main__":
    main()