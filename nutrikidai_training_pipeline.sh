#!/bin/bash

# Default values for training and validation data
TRAIN_DATA="data/notes_train_data.csv"
FULL_TRAIN_DATA="data/notes_train.csv"
VAL_DATA="data/notes_val_data.csv"

# Default values for tuning
TEXT_COLUMN="txt"
LABEL_COLUMN="label"
ID_COLUMN="DEID"
MAX_VOCAB_SIZE=20000
MIN_FREQUENCY=2
PAD_TOKEN="<PAD>"
UNK_TOKEN="<UNK>"
MAX_LENGTH="None"
PADDING="post"
EMBEDDING_DIM=100
PRETRAINED_EMBEDDINGS="None"
NUM_SAMPLES=20
MAX_EPOCHS=50
GRACE_PERIOD=5

# Set common output directory
OUTPUT_BASE_DIR="trained_models"

# Default values for textcnn tuning and training
CNN_OUTPUT_DIR="${OUTPUT_BASE_DIR}/CNN"
EPOCHS=30
FREEZE_EMBEDDINGS=false

# Default values for XGBoost tuning and training
MAX_FEATURES=10000
REMOVE_STOP_WORDS=false
APPLY_STEMMING=false
VECTORIZATION_MODE="tfidf"
NGRAM_MIN=1
NGRAM_MAX=1
MODEL_NAME="xgboost"
XGB_OUTPUT_DIR="${OUTPUT_BASE_DIR}/XGB"
ETA=0.1
MAX_DEPTH=6
MIN_CHILD_WEIGHT=1
SUBSAMPLE=1.0
COLSAMPLE_BYTREE=1.0

# Default values for TabPFN training
DEVICE="cuda"
TABPFN_MODEL_NAME="tabpfn"
TABPFN_OUTPUT_DIR="${OUTPUT_BASE_DIR}/TabPFN"
export RAY_FUNCTION_SIZE_ERROR_THRESHOLD=200
# LLM fine-tuning settings
LLM_MODELS=(
   # "unsloth/meta-llama-3.1-8b-instruct-unsloth-bnb-4bit"
   #  "unsloth/gemma-7b-it-bnb-4bit"
   "unsloth/Phi-4"
  # "unsloth/DeepSeek-R1-Distill-Qwen-7B-unsloth-bnb-4bit"
   )
LLM_BASE_DIR="${OUTPUT_BASE_DIR}/LLM_final"

# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
  case $1 in
    --train_data)
      TRAIN_DATA="$2"
      shift 2
      ;;
    --val_data)
      VAL_DATA="$2"
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
    --max_vocab_size)
      MAX_VOCAB_SIZE="$2"
      shift 2
      ;;
    --min_frequency)
      MIN_FREQUENCY="$2"
      shift 2
      ;;
    --pad_token)
      PAD_TOKEN="$2"
      shift 2
      ;;
    --unk_token)
      UNK_TOKEN="$2"
      shift 2
      ;;
    --max_length)
      MAX_LENGTH="$2"
      shift 2
      ;;
    --padding)
      PADDING="$2"
      shift 2
      ;;
    --embedding_dim)
      EMBEDDING_DIM="$2"
      shift 2
      ;;
    --pretrained_embeddings)
      PRETRAINED_EMBEDDINGS="$2"
      shift 2
      ;;
    --output_dir)
      OUTPUT_BASE_DIR="$2"
      # Update all output directories
      CNN_OUTPUT_DIR="${OUTPUT_BASE_DIR}/CNN"
      XGB_OUTPUT_DIR="${OUTPUT_BASE_DIR}/XGB"
      TABPFN_OUTPUT_DIR="${OUTPUT_BASE_DIR}/TabPFN"
      LLM_BASE_DIR="${OUTPUT_BASE_DIR}/LLM_MODELS_w1"
      shift 2
      ;;
    --num_samples)
      NUM_SAMPLES="$2"
      shift 2
      ;;
    --max_epochs)
      MAX_EPOCHS="$2"
      shift 2
      ;;
    --grace_period)
      GRACE_PERIOD="$2"
      shift 2
      ;;
    --epochs)
      EPOCHS="$2"
      shift 2
      ;;
    --freeze_embeddings)
      FREEZE_EMBEDDINGS=true
      shift 1
      ;;
    --max_features)
      MAX_FEATURES="$2"
      shift 2
      ;;
    --remove_stop_words)
      REMOVE_STOP_WORDS=true
      shift 1
      ;;
    --apply_stemming)
      APPLY_STEMMING=true 
      shift 1
      ;;
    --vectorization_mode)
      VECTORIZATION_MODE="$2"
      shift 2
      ;;
    --ngram_min)
      NGRAM_MIN="$2"
      shift 2
      ;;
    --ngram_max)
      NGRAM_MAX="$2"
      shift 2
      ;;
    --model_name)
      MODEL_NAME="$2"
      shift 2
      ;;
    --eta)
      ETA="$2"
      shift 2
      ;;
    --max_depth)
      MAX_DEPTH="$2"
      shift 2
      ;;
    --min_child_weight)
      MIN_CHILD_WEIGHT="$2"
      shift 2
      ;;
    --subsample)
      SUBSAMPLE="$2"
      shift 2
      ;;
    --colsample_bytree)
      COLSAMPLE_BYTREE="$2"
      shift 2
      ;;
    --device)
      DEVICE="$2"
      shift 2
      ;;
    --tabpfn_model_name)
      TABPFN_MODEL_NAME="$2"
      shift 2
      ;;
    *)
      echo "Unknown parameter: $1"
      exit 1
      ;;
  esac
done

# Check required parameters
if [[ -z "$TRAIN_DATA" || -z "$VAL_DATA" ]]; then
  echo "Error: --train_data and --val_data are required."
  echo "Usage example: $0 --train_data path/to/train.csv --val_data path/to/val.csv"
  exit 1
fi

# Create output directories if they don't exist
mkdir -p "$CNN_OUTPUT_DIR"
mkdir -p "$XGB_OUTPUT_DIR"
mkdir -p "$TABPFN_OUTPUT_DIR"
mkdir -p "$LLM_BASE_DIR"

# # Create a patch for the TextCNN tuning script to handle "None" values
# cat > fix_none_values.py << 'EOF'
# #!/usr/bin/env python
# import os
# import sys
# import re

# # File to patch
# file_path = sys.argv[1]

# # Read the file
# with open(file_path, 'r') as f:
#     content = f.read()

# # Function to modify argparse add_argument calls
# def modify_argparse_call(match):
#     arg_name = match.group(1)
#     arg_type = match.group(2)
#     rest = match.group(3)
    
#     # If the argument is max_length or any parameter that might be None, modify to handle None
#     if arg_name == '--max_length':
#         return f"parser.add_argument('{arg_name}', type=lambda x: None if x.lower() == 'none' else int(x), {rest}"
#     elif 'pretrained_embeddings' in arg_name:
#         return f"parser.add_argument('{arg_name}', type=lambda x: None if x.lower() == 'none' else str(x), {rest}"
#     else:
#         return match.group(0)

# # Find and replace argparse add_argument calls
# modified_content = re.sub(r"parser\.add_argument\('(--[^']+)',\s*type=([^,]+),([^\)]+)\)", modify_argparse_call, content)

# # Write the modified content back
# with open(file_path, 'w') as f:
#     f.write(modified_content)

# print(f"Applied None value fix to {file_path}")
# EOF

# # Make the script executable
# chmod +x fix_none_values.py

# # Try to patch the Python scripts to handle "None" values properly
# echo "Applying fixes to Python scripts to handle None values..."
# for script in "./tune_textcnn.py" "./train_textcnn.py" "./tune_xgb.py" "./train_xgb.py" "./train_tabpfn.py" "./train_llm.py"; do
#   if [ -f "$script" ]; then
#     python fix_none_values.py "$script"
#   fi
# done

# # echo "==================== Step 1: TextCNN Tuning ===================="
# # Run the CNN tuning script
# ./tune_textcnn.py \
#   --train_data "$TRAIN_DATA" \
#   --val_data "$VAL_DATA" \
#   --text_column "$TEXT_COLUMN" \
#   --label_column "$LABEL_COLUMN" \
#   --max_vocab_size "$MAX_VOCAB_SIZE" \
#   --min_frequency "$MIN_FREQUENCY" \
#   --pad_token "$PAD_TOKEN" \
#   --unk_token "$UNK_TOKEN" \
#   --max_length "$MAX_LENGTH" \
#   --padding "$PADDING" \
#   --embedding_dim "$EMBEDDING_DIM" \
#   --pretrained_embeddings "$PRETRAINED_EMBEDDINGS" \
#   --output_dir "$CNN_OUTPUT_DIR" \
#   --num_samples "$NUM_SAMPLES" \
#   --max_epochs "$MAX_EPOCHS" \
#   --grace_period "$GRACE_PERIOD"

# echo "==================== Step 2: TextCNN Training ===================="
# # Run the CNN training script with the tuned config
# ./train_textcnn.py \
#   --train_data "$TRAIN_DATA" \
#   --val_data "$VAL_DATA" \
#   --text_column "$TEXT_COLUMN" \
#   --label_column "$LABEL_COLUMN" \
#   --config_dir "$CNN_OUTPUT_DIR" \
#   --output_dir "$CNN_OUTPUT_DIR" \
#   --epochs "$EPOCHS" \
#   --pretrained_embeddings "$PRETRAINED_EMBEDDINGS" \
#   $( [[ "$FREEZE_EMBEDDINGS" == true && "$PRETRAINED_EMBEDDINGS" != "None" ]] && echo "--freeze_embeddings" )

# echo "==================== Step 3: XGBoost Tuning ===================="
# # Run the XGBoost tuning script with correct parameter names
# ./tune_xgb.py \
#   --train_data_file "$TRAIN_DATA" \
#   --valid_data_file "$VAL_DATA" \
#   --text_column "$TEXT_COLUMN" \
#   --label_column "$LABEL_COLUMN" \
#   --id_column "$ID_COLUMN" \
#   --max_features "$MAX_FEATURES" \
#   --vectorization_mode "$VECTORIZATION_MODE" \
#   --ngram_min "$NGRAM_MIN" \
#   --ngram_max "$NGRAM_MAX" \
#   --model_name "$MODEL_NAME" \
#   --num_samples "$NUM_SAMPLES" \
#   $( [[ "$REMOVE_STOP_WORDS" == true ]] && echo "--remove_stop_words" ) \
#   $( [[ "$APPLY_STEMMING" == true ]] && echo "--apply_stemming" ) \
#   --model_dir "$XGB_OUTPUT_DIR"

# echo "==================== Step 4: XGBoost Training ===================="
# # Run the XGBoost training script with tuned parameters
# ./train_xgb.py \
#   --data_file "$FULL_TRAIN_DATA" \
#   --text_column "$TEXT_COLUMN" \
#   --label_column "$LABEL_COLUMN" \
#   --id_column "$ID_COLUMN" \
#   --max_features "$MAX_FEATURES" \
#   --vectorization_mode "$VECTORIZATION_MODE" \
#   --ngram_min "$NGRAM_MIN" \
#   --ngram_max "$NGRAM_MAX" \
#   $( [[ "$REMOVE_STOP_WORDS" == true ]] && echo "--remove_stop_words" ) \
#   $( [[ "$APPLY_STEMMING" == true ]] && echo "--apply_stemming" ) \
#   --eta "$ETA" \
#   --max_depth "$MAX_DEPTH" \
#   --min_child_weight "$MIN_CHILD_WEIGHT" \
#   --subsample "$SUBSAMPLE" \
#   --colsample_bytree "$COLSAMPLE_BYTREE" \
#   --model_name "$MODEL_NAME" \
#   --config_dir "$XGB_OUTPUT_DIR" \
#   --model_dir "$XGB_OUTPUT_DIR"

# echo "==================== Step 5: TabPFN Training ===================="
# # Run the TabPFN training script
# ./train_tabpfn.py \
#   --data_file "$FULL_TRAIN_DATA" \
#   --text_column "$TEXT_COLUMN" \
#   --label_column "$LABEL_COLUMN" \
#   --id_column "$ID_COLUMN" \
#   --device "$DEVICE" \
#   --model_name "$TABPFN_MODEL_NAME" \
#   --max_features "$MAX_FEATURES" \
#   --vectorization_mode "$VECTORIZATION_MODE" \
#   --ngram_min "$NGRAM_MIN" \
#   --ngram_max "$NGRAM_MAX" \
#   $( [[ "$REMOVE_STOP_WORDS" == true ]] && echo "--remove_stop_words" ) \
#   $( [[ "$APPLY_STEMMING" == true ]] && echo "--apply_stemming" ) \
#   --model_dir "$TABPFN_OUTPUT_DIR" 

echo "==================== Step 6: LLM Fine-tuning ===================="
# Fine-tune LLMs with different directories for each model
for MODEL in "${LLM_MODELS[@]}"; do
  # Extract model name for directory
  MODEL_SHORT_NAME=$(echo "$MODEL" | sed 's/.*\///' | sed 's/-.*//')
  MODEL_OUTPUT_DIR="${LLM_BASE_DIR}/${MODEL_SHORT_NAME}"
  
  echo "Training $MODEL_SHORT_NAME with output directory $MODEL_OUTPUT_DIR"
  
  # Create model-specific output directory
  mkdir -p "$MODEL_OUTPUT_DIR"
  
  # Determine if model should be loaded in 4-bit (default) or 8-bit
  if [[ "$MODEL" == *"4bit"* ]]; then
    BIT_FLAG="--load_in_4bit"
  else
    BIT_FLAG="--load_in_8bit"
  fi
  
  ./finetune_llm.py \
    --model_name "$MODEL" \
    --train_data "$TRAIN_DATA" \
    --val_data "$VAL_DATA" \
    --text_column "$TEXT_COLUMN" \
    --label_column "$LABEL_COLUMN" \
    --output_dir "$MODEL_OUTPUT_DIR" \
    --model_output "${MODEL_OUTPUT_DIR}/final_model" \
    --batch_size 8 \
    --learning_rate 2e-4 \
    # --max_steps 500 \
    --max_seq_length 4096 \
    --lora_r 8 \
    --lora_alpha 32 \
    --seed 42 \
    --use_flash_attention \
    $BIT_FLAG
done

echo "All training tasks completed successfully!"
