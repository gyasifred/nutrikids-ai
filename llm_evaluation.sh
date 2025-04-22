#!/bin/bash
# ==================== Dual Evaluation of LLMs ====================
# This script performs a two-stage evaluation:
# 1. Pre-evaluation: Evaluates only the base models (without adapter weights)
#    and saves outputs in LLM_pre_evaluation.
# 2. Post-evaluation: If an adapter model path is provided, evaluates using both
#    the base model and the adapter weights and saves outputs in LLM_post_evaluation.
#
# The script automatically sets the quantization flag based on the model name:
# if the model name contains "4bit", then it uses 4-bit quantization;
# otherwise, it defaults to 8-bit.
#
# Adjust the default parameters below or override them via command-line arguments.

# -------------------- Default Parameters --------------------

# Test data and evaluation settings
TEST_CSV="data/notes_test.csv"
TEXT_COLUMN="txt"
LABEL_COLUMN="label"
ID_COLUMN="DEID"
EXAMPLES_DATA=""         # Optional few-shot examples CSV
FEW_SHOT_COUNT=0        # Zero-shot evaluation by default
BALANCED_EXAMPLES=false  # Set to true to balance few-shot examples

# Evaluation parameters
SEED=3407
MAX_NEW_TOKENS=256
TEMPERATURE=0.3
BATCH_SIZE=8
# Model settings
# Array of base models to evaluate
LLM_MODELS=(
     "unsloth/meta-llama-3.1-8b-instruct-unsloth-bnb-4bit"
    # "unsloth/Llama-3.2-1B-Instruct-bnb-4bit"
    "unsloth/gemma-7b-it-bnb-4bit"
    # "unsloth/Phi-4"
    # "unsloth/DeepSeek-R1-Distill-Qwen-7B-unsloth-bnb-4bit"
   )
# Adapter model path: if provided, it will be used in post-evaluation.
MODEL_PATH="trained/LLm_final"

# Output directories
PRE_EVAL_DIR="LLMs_base"
POST_EVAL_DIR="LLMs_finetuned"

# -------------------- Command-Line Arguments Parsing --------------------
while [[ "$#" -gt 0 ]]; do
  case $1 in
    --test_csv)
      TEST_CSV="$2"
      shift 2
      ;;
    --text_column)
      TEXT_COLUMN="$2"
      shift 2
      ;;
    --label_column)
      LABEL_COLUMN="$2"
      shift 2
      ;;
    --id_column)
      ID_COLUMN="$2"
      shift 2
      ;;
    --examples_data)
      EXAMPLES_DATA="$2"
      shift 2
      ;;
    --few_shot_count)
      FEW_SHOT_COUNT="$2"
      shift 2
      ;;
    --balanced_examples)
      BALANCED_EXAMPLES=true
      shift 1
      ;;
    --seed)
      SEED="$2"
      shift 2
      ;;
    --max_new_tokens)
      MAX_NEW_TOKENS="$2"
      shift 2
      ;;
    --temperature)
      TEMPERATURE="$2"
      shift 2
      ;;
    --batch_size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --model_path)
      MODEL_PATH="$2"
      shift 2
      ;;
    --pre_eval_dir)
      PRE_EVAL_DIR="$2"
      shift 2
      ;;
    --post_eval_dir)
      POST_EVAL_DIR="$2"
      shift 2
      ;;
    *)
      echo "Unknown parameter: $1"
      exit 1
      ;;
  esac
done

# Create output directories if they don't exist
mkdir -p "$PRE_EVAL_DIR"
mkdir -p "$POST_EVAL_DIR"
# -------------------- Pre-Evaluation (Base Models Only) --------------------
echo "==================== Starting Pre-Evaluation (Base Models Only) ===================="
for MODEL in "${LLM_MODELS[@]}"; do
  # Automatically determine quantization flag based on model name.
  if [[ "$MODEL" == *"4bit"* ]]; then
    BIT_FLAG="--load_in_4bit"
  else
    BIT_FLAG="--load_in_8bit"
  fi

  # Extract a short model name for output directory naming.
  MODEL_SHORT_NAME=$(echo "$MODEL" | sed 's/.*\///' | sed 's/-.*//')
  MODEL_OUTPUT_DIR="${PRE_EVAL_DIR}/${MODEL_SHORT_NAME}"
  mkdir -p "$MODEL_OUTPUT_DIR"

  echo "Pre-Evaluating Base Model: ${MODEL_SHORT_NAME}"
  CMD="python3 evaluate_llm.py \
    --base_model \"$MODEL\" \
    --test_csv \"$TEST_CSV\" \
    --text_column \"$TEXT_COLUMN\" \
    --label_column \"$LABEL_COLUMN\" \
    --id_column \"$ID_COLUMN\" \
    --output_dir \"$MODEL_OUTPUT_DIR\" \
    --few_shot_count \"$FEW_SHOT_COUNT\" \
    --temperature \"$TEMPERATURE\" \
    --batch_size \"$BATCH_SIZE\" \
    --seed \"$SEED\" $BIT_FLAG"
  
  # Append optional flags.
  if [ "$BALANCED_EXAMPLES" = true ]; then
    CMD="$CMD --balanced_examples"
  fi
  
  if [ -n "$EXAMPLES_DATA" ]; then
    CMD="$CMD --examples_data \"$EXAMPLES_DATA\""
  fi
  
  echo "Running command: $CMD"
  eval $CMD
done

# -------------------- Post-Evaluation (Base + Adapter) --------------------
# Only perform post-evaluation if a MODEL_PATH (adapter weights) is provided.
if [ -n "$MODEL_PATH" ]; then
  echo "==================== Starting Post-Evaluation (Base Model + Adapter) ===================="
  for MODEL in "${LLM_MODELS[@]}"; do
    # Automatically determine quantization flag based on model name.
    if [[ "$MODEL" == *"4bit"* ]]; then
      BIT_FLAG="--load_in_4bit"
    else
      BIT_FLAG="--load_in_8bit"
    fi

    # Extract a short model name for output directory naming.
    MODEL_SHORT_NAME=$(echo "$MODEL" | sed 's/.*\///' | sed 's/-.*//')
    MODEL_OUTPUT_DIR="${POST_EVAL_DIR}/${MODEL_SHORT_NAME}"
    mkdir -p "$MODEL_OUTPUT_DIR"

    echo "Post-Evaluating Model with Adapter: ${MODEL_SHORT_NAME}"
    CMD="python3 evaluate_llm.py \
      --base_model \"$MODEL\" \
      --model_path \"${MODEL_PATH}/${MODEL_SHORT_NAME}/final_model\" \
      --test_csv \"$TEST_CSV\" \
      --text_column \"$TEXT_COLUMN\" \
      --label_column \"$LABEL_COLUMN\" \
      --id_column \"$ID_COLUMN\" \
      --output_dir \"$MODEL_OUTPUT_DIR\" \
      --few_shot_count \"$FEW_SHOT_COUNT\" \
      --temperature \"$TEMPERATURE\" \
      --batch_size \"$BATCH_SIZE\" \
      --seed \"$SEED\" $BIT_FLAG"
    
    # Append optional flags.
    if [ "$BALANCED_EXAMPLES" = true ]; then
      CMD="$CMD --balanced_examples"
    fi
    
    if [ -n "$EXAMPLES_DATA" ]; then
      CMD="$CMD --examples_data \"$EXAMPLES_DATA\""
    fi
    
    echo "Running command: $CMD"
    eval $CMD
  done
else
  echo "No adapter model path provided. Skipping post-evaluation."
fi
echo "LLM Evaluation (Pre & Post) completed successfully!"
