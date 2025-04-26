#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fine-tune LLaMA-style model using Unsloth + LoRA for malnutrition assessment with reasoning.
"""

import os
os.environ['UNSLOTH_RETURN_LOGITS'] = '1'  # Critical for probability extraction
import argparse
import pandas as pd
import torch
from datasets import Dataset
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported
from unsloth import FastLanguageModel 

def parse_arguments():
    parser = argparse.ArgumentParser(description="Fine-tune LLM for pediatric malnutrition classification with reasoning")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the CSV data file")
    parser.add_argument("--model_name", type=str, default="unsloth/Meta-Llama-3.1-8B", help="Base model to use")
    parser.add_argument("--output_dir", type=str, default="./malnutrition_model", help="Where to save the model")
    parser.add_argument("--max_seq_length", type=int, default=None, 
                        help="Max sequence length (if None, uses model's native max length)")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Gradient accumulation")
    parser.add_argument("--epochs", type=float, default=10, help="Epochs")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--load_in_4bit", action="store_true", help="Use 4-bit quantization")
    parser.add_argument("--save_steps", type=int, default=50, help="Steps between checkpoints")
    parser.add_argument("--preprocess_tokens", action="store_true", help="Preprocess </s> tokens in clinical notes")
    parser.add_argument("--use_native_max_len", action="store_true", 
                        help="Use the model's native maximum sequence length")
    return parser.parse_args()

def preprocess_clinical_note(note_text):
    """Preprocess clinical notes to handle special tokens."""
    if not note_text:
        return note_text
    
    # Your original replacement logic
    processed_text = note_text.replace('</s>', '\n\n')
    special_tokens = ['<s>', '<pad>', '</pad>', '<eos>', '<bos>']
    for token in special_tokens:
        if token in processed_text:
            processed_text = processed_text.replace(token, f"[{token}]")
    
    return processed_text

def create_malnutrition_prompt(note, label="", reasoning="", tokenizer=None, max_tokens=None):
    """
    Create balanced malnutrition assessment prompt with clear criteria and structured output.
    Supports both training (with label/reasoning) and inference modes.
    """
    prompt = """[Role] Pediatric nutrition specialist analyzing growth charts and clinical documentation.
[WHO Diagnostic Framework]
<<CRITERIA FOR MALNUTRITION>>
1. **Severe Acute Malnutrition (SAM):**
   - Weight-for-height/BMI-for-age < -3 SD z-score **OR**
   - MUAC < 115 mm (6-59mo) **OR**
   - Bilateral pitting edema
2. **Moderate Acute Malnutrition (MAM):**
   - Weight-for-height/BMI-for-age -2 to -3 SD z-score
3. **Stunting (Chronic):**
   - Height-for-age < -2 SD z-score
4. **Supportive Indicators:**
   ▸ Weight loss >5% in 30 days
   ▸ Inadequate intake >5 days
   ▸ Documented growth decline >2 percentiles
<</CRITERIA FOR MALNUTRITION>>
<<CRITERIA FOR NORMAL NUTRITIONAL STATUS>>
1. **Normal Growth Parameters:**
   - Weight-for-height/BMI-for-age ≥ -2 SD z-score
   - Height-for-age ≥ -2 SD z-score
   - MUAC ≥ 115 mm (for children 6-59mo)
2. **Normal Clinical Assessment:**
   - No bilateral pitting edema
   - No significant recent weight loss (<5% in 30 days)
   - Adequate dietary intake
   - Stable or appropriate growth trajectory
3. **Note:** Growth measurements should be evaluated in context of the child's overall clinical picture and medical history
<</CRITERIA FOR NORMAL NUTRITIONAL STATUS>>
[Assessment Protocol]
1. Primary reliance on anthropometrics and clinical data
2. Differentiate acute vs chronic patterns when present
3. Confirm with clinical correlates
4. Determine "yes" ONLY when criteria for malnutrition are definitively met
5. Determine "no" when criteria for normal nutritional status are met
[Output Format]
Strictly follow this structure:
### Assessment:
<yes/no>  # FIRST TOKEN MUST BE yes/no, based ONLY on evidence meeting criteria
### Severity:
<sam/mam/stunting/none>
### Basis:
- Maximum 3 bullet points
- Cite specific z-scores or measurements when available
- Note key clinical findings
- For "no" assessments, document normal growth parameters

Clinical note for analysis:
{note}

{label_reasoning}"""

    # Handle training vs. inference mode
    if label and reasoning:
        label_reasoning = f"""### Assessment: {label}
### Severity: {get_severity_from_reasoning(reasoning)}
### Basis:
{reasoning}"""
    else:
        label_reasoning = ""
    
    # Apply token truncation if needed
    formatted_prompt = prompt.format(note=note, label_reasoning=label_reasoning)
    
    if tokenizer and max_tokens:
        tokens = tokenizer.encode(formatted_prompt)
        
        # Check if tokens is too long
        if len(tokens) > max_tokens:
            # Get template without the note to determine available tokens
            template = prompt.format(note="", label_reasoning=label_reasoning)
            template_tokens = tokenizer.encode(template)
            
            # Calculate available tokens for the note
            available_tokens = max_tokens - len(template_tokens)
            
            # Ensure at least some tokens are available
            if available_tokens <= 0:
                available_tokens = max_tokens // 2  # Fallback if template is too long
            
            # Tokenize the note separately
            note_tokens = tokenizer.encode(note)
            
            # Truncate the note tokens
            truncated_note_tokens = note_tokens[:available_tokens]
            
            # Decode the truncated tokens back to text
            truncated_note = tokenizer.decode(truncated_note_tokens)
            
            # Recreate the prompt with the truncated note
            formatted_prompt = prompt.format(note=truncated_note, label_reasoning=label_reasoning)
    
    return formatted_prompt
def get_severity_from_reasoning(reasoning):
    """Extract severity from reasoning text based on keywords."""
    reasoning_lower = reasoning.lower()
    if any(term in reasoning_lower for term in ["severe", "sam", "< -3", "<-3", "< -3sd", "<-3sd", "muac < 115", "edema"]):
        return "sam"
    elif any(term in reasoning_lower for term in ["moderate", "mam", "-2 to -3", "-2sd to -3sd"]):
        return "mam"
    elif any(term in reasoning_lower for term in ["stunting", "chronic", "height-for-age < -2", "height for age < -2"]):
        return "stunting"
    else:
        return "none"

# In your prepare_dataset function:
def prepare_dataset(data_path, tokenizer, max_seq_length, preprocess_tokens=False):
    """Prepare dataset with enhanced prompt structure"""
    df = pd.read_csv(data_path)
    if not {"DEID", "txt", "label"}.issubset(df.columns):
        raise ValueError("CSV must include 'DEID', 'txt', and 'label' columns.")

    # Your original processing flow
    has_reasoning = "reasoning" in df.columns
    prompts = []
    for _, row in df.iterrows():
        note_text = preprocess_clinical_note(row["txt"]) if preprocess_tokens else row["txt"]
        label_text = "yes" if str(row["label"]).lower() in {"1", "yes", "true"} else "no"
        reasoning_text = row.get("reasoning", "") if has_reasoning else ""
        
        prompt = create_malnutrition_prompt(
            note=note_text, 
            label=label_text, 
            reasoning=reasoning_text,
            tokenizer=tokenizer,
            max_tokens=max_seq_length - 20  # Leave room for EOS token
        )
        prompts.append(prompt + tokenizer.eos_token)

    return Dataset.from_dict({"text": prompts})


def get_model_max_length(model_name):
    """Your original max length detection"""
    try:
        from unsloth.models.llama import auto_get_max_seq_length
        return auto_get_max_seq_length(model_name)
    except (ImportError, AttributeError):
        from transformers import AutoConfig
        try:
            config = AutoConfig.from_pretrained(model_name)
            return getattr(config, "max_position_embeddings", 4096)
        except Exception as e:
            print(f"Failed to get model's max length: {e}")
            return 4096

def main():
    args = parse_arguments()

    # Your original max length logic
    if args.use_native_max_len or args.max_seq_length is None:
        max_seq_length = get_model_max_length(args.model_name)
        print(f"Using model's max length: {max_seq_length}")
    else:
        max_seq_length = args.max_seq_length

    # Your original model loading
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=args.load_in_4bit,
    )

    # Your original LoRA config
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=False,
    )

    # Your dataset prep
    dataset = prepare_dataset(args.data_path, tokenizer, max_seq_length, args.preprocess_tokens)

    # Your original trainer setup
    def formatting_prompts_func(examples):
        return examples["text"]

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        formatting_func=formatting_prompts_func, 
        args=TrainingArguments(
            output_dir=args.output_dir,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.learning_rate,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=10,
            save_steps=args.save_steps,
            num_train_epochs=args.epochs,
            report_to="none",
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            warmup_steps=10,
        )
    )

    # Your training execution
    print("Starting training...")
    trainer_stats = trainer.train()

    # Your model saving
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("Training complete!")

if __name__ == "__main__":
    main()
