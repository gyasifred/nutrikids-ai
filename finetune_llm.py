#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fine-tune LLaMA-style model using Unsloth + LoRA for malnutrition assessment
with a detailed prompt for context-limited environments.
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
    parser = argparse.ArgumentParser(description="Fine-tune LLM for pediatric malnutrition classification")
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

    # Replace problematic tokens
    processed_text = note_text.replace('</s>', '\n\n')
    special_tokens = ['<s>', '<pad>', '</pad>', '<eos>', '<bos>']
    for token in special_tokens:
        if token in processed_text:
            processed_text = processed_text.replace(token, f"[{token}]")

    return processed_text

def create_detailed_malnutrition_prompt(note, label="", tokenizer=None, max_tokens=None):
    """Create balanced malnutrition assessment prompt with clear criteria for both positive and negative determinations."""
    base_prompt = """[Role] Pediatric nutrition specialist analyzing growth charts and clinical documentation.

<<CRITERIA FOR MALNUTRITION>>
Mild malnutrition related to undernutrition is usually the result of an acute event, either due to economic circumstances or acute illness, and presents with unintentional weight loss or weight gain velocity less than expected. Moderate malnutrition related to undernutrition occurs due to undernutrition of a significant duration that results in weight-for-length/height values or BMI-for-age values that are below the normal range. Severe malnutrition related to undernutrition occurs as a result of prolonged undernutrition and is most frequently quantified by declines in rates of linear growth that result in stunting.

You should use z scores (also called z for short) for weight-for-height/length, BMI-for-age, length/height-for-age or MUAC criteria. When a child has only one data point in the records (single z score present) use the table below:

Table 1. Single data point present.
Mild Malnutrition
Weight-for-height: −1 to −1.9 z score
BMI-for-age: −1 to −1.9 z score
Length/height-for-age: No Data
Mid–upper arm circumference: Greater than or equal to −1 to −1.9 z score	

Moderate Malnutrition	
Weight-for-height: −2 to −2.9 z score
BMI-for-age: −2 to −2.9 z score
Length/height-for-age: No Data
Mid–upper arm circumference: Greater than or equal to −2 to −2.9 z score	

Severe Malnutrition
Weight-for-height:	−3 or greater z score
BMI-for-age: −3 or greater z score
Length/height-for-age: −3 z score
Mid–upper arm circumference: Greater than or equal to −3 z score

When the child has 2 or more data points (multiple z scores over time) use this table:
Table 2. Multiple data points available.
Mild Malnutrition
Weight gain velocity (<2 years of age): Less than 75% of the norm for expected weight gain
Weight loss (2–20 years of age): 5% usual body weigh
Deceleration in weight for length/height: Decline of 1 z score
Inadequate nutrient intake: 51%−75% estimated energy/protein need

Moderate Malnutrition	
Weight gain velocity (<2 years of age): Less than 50% of the norm for expected weight gain
Weight loss (2–20 years of age): 7.5% usual body weight
Deceleration in weight for length/height: Decline of 2 z score
Inadequate nutrient intake: 26%−50% estimated energy/protein need

Severe Malnutrition
Weight gain velocity (<2 years of age): Less than 25% of the normb for expected weight gain
Weight loss (2–20 years of age): 10% usual body weight
Deceleration in weight for length/height: Decline of 3 z score
Inadequate nutrient intake: less than 25% estimated energy/protein need
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

{label_part}"""

    # For training mode (with label), format the output part
    if label:
        # Map numeric labels (1/0) to yes/no format
        if label in ["1", 1, "yes", "Yes", "YES"]:
            formatted_label = "### Assessment:\nyes"
        else:
            formatted_label = "### Assessment:\nno"
        label_part = f"{formatted_label}"
    else:
        label_part = "### Assessment:\n"

    # Apply token truncation if needed
    formatted_prompt = base_prompt.format(note=note, label_part=label_part)

    if tokenizer and max_tokens:
        tokens = tokenizer.encode(formatted_prompt)

        # Check if tokens is too long
        if len(tokens) > max_tokens:
            # Get template without the note to determine available tokens
            template = base_prompt.format(note="", label_part=label_part)
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
            formatted_prompt = base_prompt.format(note=truncated_note, label_part=label_part)

    return formatted_prompt

def prepare_dataset(data_path, tokenizer, max_seq_length, preprocess_tokens=False):
    """Prepare dataset with specialized malnutrition prompt structure"""
    df = pd.read_csv(data_path)
    if not {"txt", "label"}.issubset(df.columns):
        raise ValueError("CSV must include 'txt' and 'label' columns.")

    prompts = []
    for _, row in df.iterrows():
        note_text = preprocess_clinical_note(row["txt"]) if preprocess_tokens else row["txt"]

        # Convert label format (1/0 to yes/no)
        label_text = "yes" if str(row["label"]).lower() in {"1", "yes", "true"} else "no"

        prompt = create_detailed_malnutrition_prompt(
            note=note_text, 
            label=label_text,
            tokenizer=tokenizer,
            max_tokens=max_seq_length - 20  # Leave room for EOS token
        )
        prompts.append(prompt + tokenizer.eos_token)

    return Dataset.from_dict({"text": prompts})

def get_model_max_length(model_name):
    """Get the model's maximum sequence length."""
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

    # Determine max sequence length
    if args.use_native_max_len or args.max_seq_length is None:
        max_seq_length = get_model_max_length(args.model_name)
        print(f"Using model's native max length: {max_seq_length}")
    else:
        max_seq_length = args.max_seq_length

    # Load the model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=args.load_in_4bit,
    )

    # Configure LoRA
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

    # Prepare dataset
    dataset = prepare_dataset(
        args.data_path, 
        tokenizer, 
        max_seq_length, 
        args.preprocess_tokens
    )

    # Set up trainer
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

    # Train the model
    print(f"Starting training with detailed prompt style...")
    trainer_stats = trainer.train()

    # Save the model
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("Training complete!")

if __name__ == "__main__":
    main()
