import re
import os
os.environ['UNSLOTH_RETURN_LOGITS'] = '1'
import pandas as pd
import torch
from datasets import Dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments


# Enhanced instruction for pediatric malnutrition diagnosis using small LLMs
instruction = """Read the patient's notes and determine if the patient is likely to have malnutrition according to the criteria below.

====== STEP-BY-STEP ANALYSIS PROCESS ======

STEP 1: EXTRACT KEY DATA FROM NOTES
Age: ___
Weight-for-height z-score: ___
BMI-for-age z-score: ___  
Height-for-age z-score: ___
MUAC z-score: ___
Weight change: ___% over ___ timeframe
Nutrient intake: ___% of estimated needs

STEP 2: DETERMINE APPROACH
□ Single data point available → Use Table 1
□ Multiple data points available → Use Table 2  
□ Both available → Use both, choose more severe classification

STEP 3: APPLY QUICK SCREENING
🔴 IMMEDIATE SEVERE: Any z-score ≤ -3.0 OR weight loss ≥10% OR intake <25%
🟠 IMMEDIATE MODERATE: Any z-score -2.0 to -2.9 OR weight loss 7.5-10% OR intake 26-50%
🟡 IMMEDIATE MILD: Any z-score -1.0 to -1.9 OR weight loss 5-7.5% OR intake 51-75%

====== MALNUTRITION CLASSIFICATION CRITERIA ======

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
Weight-for-height: −3 or greater z score
BMI-for-age: −3 or greater z score
Length/height-for-age: −3 z score
Mid–upper arm circumference: Greater than or equal to −3 z score

When the child has 2 or more data points (multiple z scores over time) use this table:
Table 2. Multiple data points available.
Mild Malnutrition
Weight gain velocity (<2 years of age): Less than 75% of the norm for expected weight gain
Weight loss (2–20 years of age): 5% usual body weight
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

====== CRITICAL DECISION RULES ======
⚠️ When multiple criteria conflict, choose the MORE SEVERE classification
⚠️ Height-for-age z-score indicates stunting but requires ≤-3.0 for severe malnutrition diagnosis
⚠️ A single normal measurement does NOT rule out malnutrition if other criteria are abnormal
⚠️ For infants <2 years, prioritize weight gain velocity over static measurements when both available
⚠️ ANY z-score ≤ -3.0 automatically indicates severe malnutrition
⚠️ Weight loss percentages: 5-7.5% = mild, 7.5-10% = moderate, ≥10% = severe

====== Z-SCORE INTERPRETATION HELPER ======
🟢 Normal: z-score > -1.0 (example: -0.5, -0.9)
🟡 Mild: z-score -1.0 to -1.9 (example: -1.2, -1.8)
🟠 Moderate: z-score -2.0 to -2.9 (example: -2.1, -2.7)
🔴 Severe: z-score ≤ -3.0 (example: -3.0, -3.5)

====== OUTPUT FORMAT INSTRUCTIONS ======

Provide your analysis in two parts:

PART 1: ANALYSIS
Use this template: "I extracted the following data: [list key values found in notes]. I used [single/multiple] data point criteria from [Table 1/Table 2/both tables]. The key finding is [specific measurement that determines classification]. This indicates [mild/moderate/severe/no] malnutrition because [brief clinical reasoning]."

PART 2: FINAL CLASSIFICATION
- As the VERY LAST LINE of your response, provide EXACTLY one of these two classifications:
  malnutrition=yes
  malnutrition=no

CRITICAL FORMATTING RULES:
1. Do NOT change the spelling of "malnutrition" - it must be spelled EXACTLY as shown
2. Use EXACTLY the format shown: malnutrition=yes or malnutrition=no
3. No spaces around the equals sign
4. No periods, commas or other punctuation
5. The classification MUST be on its own line at the very end of your response
6. Do NOT explain your final classification - just state it
7. Even if uncertain, you MUST choose either yes or no - do not say "maybe"

====== EXAMPLE RESPONSES ======

EXAMPLE 1:
I extracted the following data: weight loss 6% over 3 months, nutrient intake 60% of estimated needs. I used multiple data point criteria from Table 2. The key finding is weight loss of 6% which exceeds the 5% threshold for mild malnutrition. This indicates mild malnutrition because the patient meets both weight loss and inadequate intake criteria from Table 2.

malnutrition=yes

EXAMPLE 2: 
I extracted the following data: BMI-for-age z-score -0.8, weight-for-height z-score -0.5. I used single data point criteria from Table 1. The key finding is both z-scores are above -1.0 threshold. This indicates no malnutrition because all measurements are within normal range.

malnutrition=no

EXAMPLE 3:
I extracted the following data: weight-for-height z-score -2.3, BMI-for-age z-score -2.1, nutrient intake 40% of needs. I used both single and multiple data point criteria from Tables 1 and 2. The key finding is z-scores in the -2.0 to -2.9 range indicating moderate malnutrition. This indicates moderate malnutrition because both anthropometric measurements and nutritional intake criteria are met.

malnutrition=yes

====== EDGE CASE EXAMPLES ======

EDGE CASE 1 - Conflicting Data:
Input: 18-month-old with weight-for-height z-score -1.2 but weight gain velocity only 40% of expected over past 3 months.

Response: I extracted the following data: weight-for-height z-score -1.2, weight gain velocity 40% of expected. I used both single and multiple data point criteria. The key finding is weight gain velocity <50% of expected, which indicates moderate malnutrition from Table 2, overriding the mild classification from the single z-score. This indicates moderate malnutrition because longitudinal growth failure takes precedence over single measurements in infants.

malnutrition=yes

EDGE CASE 2 - Borderline Values:
Input: 5-year-old with BMI-for-age z-score exactly -2.0 and no other data available.

Response: I extracted the following data: BMI-for-age z-score -2.0. I used single data point criteria from Table 1. The key finding is z-score of exactly -2.0 which falls at the threshold for moderate malnutrition. This indicates moderate malnutrition because -2.0 meets the criteria for the -2 to -2.9 range.

malnutrition=yes

EDGE CASE 3 - Height-for-age Only:
Input: 3-year-old with height-for-age z-score -2.5, no weight data available.

Response: I extracted the following data: height-for-age z-score -2.5. I used single data point criteria from Table 1. The key finding is height-for-age z-score does not indicate acute malnutrition unless ≤-3.0. This indicates no malnutrition because height-for-age z-score of -2.5 indicates stunting but does not meet criteria for malnutrition diagnosis by itself.

malnutrition=no"""

def build_prompt(note):
    """Builds the prompt in the AlpaCa/ChatML format"""
    return f"### Instruction:\n{instruction}\n\n### Input:\n{note}\n\n### Response:"
    
def preprocess_clinical_note(note_text):
    """Enhanced preprocessing that keeps clinical data intact while removing problematic patterns."""
    if not note_text:
        return ""
    
    # Preserve clinical abbreviations and numbers while removing artifacts
    note_text = re.sub(r'[*_\-=+#~^`]{2,}', ' ', note_text)  # Remove repeating special chars
    note_text = re.sub(r'<[^>]+>', ' ', note_text)           # Remove HTML/XML tags
    note_text = re.sub(r'\s{2,}', ' ', note_text)            # Normalize whitespace
    
    # Handle special tokens without affecting clinical content
    note_text = note_text.replace('</s>', '\n\n')
    special_tokens = ['<s>', '<pad>', '</pad>', '<eos>', '<bos>']
    for token in special_tokens:
        note_text = note_text.replace(token, ' ')
    
    return note_text.strip()

def train_malnutrition_model(data_path, model_name="unsloth/meta-llama-3.1-8b-unsloth-bnb-4bit", 
                           output_dir="./malnutrition_models", max_length=8192,
                           num_epochs=3, batch_size=4, truncation_strategy="hybrid"):
    """Train a sequence-to-sequence model for malnutrition classification using Unsloth
    
    Parameters:
    -----------
    data_path : str
        Path to the CSV file containing the training data
    model_name : str, optional
        Name of the pretrained model to load, by default "unsloth/meta-llama-3.1-8b-unsloth-bnb-4bit"
    output_dir : str, optional
        Directory to save the trained model, by default "./malnutrition_models"
    max_length : int, optional
        Maximum sequence length for tokenization, by default 8192
    num_epochs : int, optional
        Number of training epochs, by default 3
    batch_size : int, optional
        Batch size for training, by default 4
    truncation_strategy : str, optional
        Strategy for handling inputs exceeding max_length: "head" (truncate beginning), 
        "tail" (truncate end), "hybrid" (preserve instruction and truncate middle), 
        or "middle" (truncate from middle), by default "hybrid"
    """
    
    # Create a log directory for tracking training progress
    log_dir = os.path.join(output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    # Load and prepare dataset
    try:
        df = pd.read_csv(data_path, usecols=['DEID','txt','label'])
        print(f"Loaded dataset with {len(df)} examples")
        
        # Save an example of the data for reference during inference
        example_df = df.head(5)
        example_df.to_csv(os.path.join(output_dir, "example_data.csv"), index=False)
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None
    
    # Ensure correct column names and preprocess the notes
    if "txt" not in df.columns or "label" not in df.columns:
        print("Error: Dataset must contain 'txt' and 'label' columns")
        return None
    
    # Fill missing values to avoid NoneType errors
    df["txt"] = df["txt"].fillna("")  # Handle NoneType values
    df["label"] = df["label"].fillna("no")  # Handle NoneType values
    
    # Analyze text length distribution to inform truncation strategy
    text_lengths = df["txt"].str.len()
    print(f"Text length statistics:")
    print(f"  Min: {text_lengths.min()}")
    print(f"  Max: {text_lengths.max()}")
    print(f"  Mean: {text_lengths.mean():.2f}")
    print(f"  Median: {text_lengths.median()}")
    print(f"  90th percentile: {text_lengths.quantile(0.9)}")
    print(f"  95th percentile: {text_lengths.quantile(0.95)}")
    
    # Preprocess notes
    df["txt"] = df["txt"].apply(preprocess_clinical_note)
    
    # Convert to HuggingFace dataset
    dataset = Dataset.from_pandas(df)
    
    # Apply formatting with AlpaCa/ChatML style prompts
    def process(example):
        prompt = build_prompt(example["txt"])
        # Ensure label is properly formatted
        label = str(example.get("label", "no")).strip().lower()  # Use get() with default
        if label not in ["yes", "no"]:
            # Check if label matches any of the positive indicators
            positive_indicators = ["1", "1.0", "true", "positive", "y"]
            is_positive = any(str(label) == indicator for indicator in positive_indicators)
            label = "yes" if is_positive else "no"
        return {"prompt": prompt, "response": f"malnutrition={label}", "txt": example["txt"]}
    
    # Add proper error handling for long text processing
    try:
        dataset = dataset.map(process)
    except Exception as e:
        print(f"Error during dataset processing: {e}")
        # Implement a fallback processing for potential errors
        def safe_process(example):
            try:
                return process(example)
            except Exception as e:
                print(f"Error processing example {example.get('DEID', 'unknown')}: {e}")
                # Return a simplified version for problematic examples
                return {
                    "prompt": "### Instruction:\n[Truncated instruction]\n\n### Input:\n[Error processing text]\n\n### Response:",
                    "response": "malnutrition=no",  # Default fallback
                    "txt": "[Error processing text]"
                }
        
        print("Attempting fallback processing...")
        dataset = dataset.map(safe_process)
    
    # Create train/validation split if needed
    if "split" in dataset.column_names:
        train_dataset = dataset.filter(lambda x: x["split"] == "train")
        eval_dataset = dataset.filter(lambda x: x["split"] == "val")
    else:
        # If no split column, use a random 90/10 split
        dataset_dict = dataset.train_test_split(test_size=0.1, seed=3407)
        train_dataset = dataset_dict["train"]
        eval_dataset = dataset_dict["test"]
    
    print(f"Training set: {len(train_dataset)} examples")
    print(f"Validation set: {len(eval_dataset)} examples")
    
    # Save tokenizer configuration for inference consistency
    tokenizer_config = {
        "model_name": model_name,
        "max_length": max_length,
        "truncation_strategy": truncation_strategy
    }
    
    with open(os.path.join(output_dir, "tokenizer_config.json"), "w") as f:
        import json
        json.dump(tokenizer_config, f)
    
    # Load model and tokenizer using Unsloth
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_length, 
        dtype=None,  # Let Unsloth decide based on hardware
        load_in_4bit=True  # Use 4-bit quantization for efficiency
    )
    
    # Ensure padding token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # Configure the tokenizer to handle lengthy inputs
    tokenizer.model_max_length = max_length
    
    # For training with hybrid approach, initially set truncation_side="tail"
    # This preserves the instruction part (usually at the beginning)
    tokenizer.truncation_side = "tail"
    
    # Save the original instruction token count for smart truncation
    instruction_tokens = tokenizer.encode(instruction)
    instruction_token_count = len(instruction_tokens)
    print(f"Instruction part uses {instruction_token_count} tokens")
    
    # Calculate approximate token buffer needed for response
    response_buffer = 50  # Tokens reserved for response
    
    # Prepare the model for instruction fine-tuning
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,  # Rank
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=32,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth", 
        random_state=3407,
        use_rslora=False,
        loftq_config=None, 
    )
    
    # Enhanced formatting function with improved truncation strategies
    def formatting_func(example):
        try:
            # Handle missing or None values
            if example is None or 'prompt' not in example or example['prompt'] is None:
                return f"### Instruction:\nPlease evaluate.\n\n### Input:\nEmpty note.\n\n### Response:\nmalnutrition=no{tokenizer.eos_token}"
                
            if 'response' not in example or example['response'] is None:
                example['response'] = "malnutrition=no"
                
            # The prompt already contains the instruction and input text
            text = f"{example['prompt']}\n{example['response']}{tokenizer.eos_token}"
            
            # Check if the tokenized length exceeds max_length
            try:
                tokens = tokenizer.encode(text)
                token_length = len(tokens)
            except Exception as e:
                print(f"Tokenization error: {e}")
                # Return a safe default if tokenization fails
                return f"### Instruction:\n[Error in tokenization]\n\n### Response:\nmalnutrition=no{tokenizer.eos_token}"
            
            # If text fits within the max_length, no truncation needed
            if token_length <= max_length:
                return text
                
            print(f"Warning: Example exceeds max length ({token_length} > {max_length})")
            
            # Apply truncation strategy based on parameter
            if truncation_strategy == "hybrid":
                # HYBRID APPROACH: Preserve instruction, truncate clinical note in middle if needed
                
                # Split into components
                instruction_part = example['prompt'].split("\n\n### Input:\n")[0] + "\n\n### Input:\n"
                input_part = example['prompt'].split("\n\n### Input:\n")[1]
                response_part = f"\n\n### Response:\n{example['response']}{tokenizer.eos_token}"
                
                # Calculate token counts
                try:
                    instruction_tokens = tokenizer.encode(instruction_part)
                    response_tokens = tokenizer.encode(response_part)
                    
                    # Available tokens for input text
                    available_tokens = max_length - len(instruction_tokens) - len(response_tokens) - 10  # safety buffer
                    
                    if available_tokens <= 0:
                        # If almost no space for input, use minimal example
                        return f"{instruction_part}[Note too long to process fully]\n\n### Response:\nmalnutrition=no{tokenizer.eos_token}"
                    
                    # Tokenize the input text
                    input_tokens = tokenizer.encode(input_part)
                    
                    if len(input_tokens) <= available_tokens:
                        # If it fits, use the whole input
                        return text
                        
                    # Smart truncation of clinical note:
                    # 1. Keep the beginning (may contain vital patient info)
                    # 2. Keep the end (may contain recent measurements/assessments)
                    # 3. Truncate the middle
                    
                    # Determine how much to keep from beginning and end
                    keep_start = available_tokens // 2
                    keep_end = available_tokens - keep_start
                    
                    # Get beginning and end tokens
                    start_tokens = input_tokens[:keep_start]
                    end_tokens = input_tokens[-keep_end:]
                    
                    # Reconstruct truncated input
                    truncated_input = tokenizer.decode(start_tokens) + " [...note truncated...] " + tokenizer.decode(end_tokens)
                    
                    # Build final text
                    return f"{instruction_part}{truncated_input}{response_part}"
                    
                except Exception as e:
                    print(f"Error in hybrid truncation: {e}")
                    # Fallback to basic truncation if hybrid fails
                    return f"{instruction_part}[Error in truncation process]\n\n### Response:\nmalnutrition=no{tokenizer.eos_token}"
                    
            elif truncation_strategy == "middle":
                # MIDDLE TRUNCATION: Similar to hybrid but different partition
                input_part = example['prompt']
                response_part = f"\n{example['response']}{tokenizer.eos_token}"
                
                try:
                    input_tokens = tokenizer.encode(input_part)
                    response_tokens = tokenizer.encode(response_part)
                except Exception as e:
                    print(f"Tokenization error during truncation: {e}")
                    return f"### Instruction:\n[Truncated due to error]\n\n### Response:\nmalnutrition=no{tokenizer.eos_token}"
                
                # Calculate how much to keep from input (leave room for response)
                available_length = max_length - len(response_tokens) - 10
                if available_length <= 0:
                    return f"### Instruction:\n[Input too long]\n\n### Response:\nmalnutrition=no{tokenizer.eos_token}"
                
                # Extract instruction and input components
                instruction_end = input_part.find("\n\n### Input:")
                if instruction_end > 0:  # If found
                    instruction = input_part[:instruction_end+12]  # Include the "### Input:" part
                    input_text = input_part[instruction_end+12:]
                    
                    try:
                        instruction_tokens = tokenizer.encode(instruction)
                        remaining_tokens = available_length - len(instruction_tokens)
                        
                        if remaining_tokens <= 0:
                            text = f"{instruction}[content truncated]{response_part}"
                        else:
                            # Keep half from start, half from end
                            keep_start = remaining_tokens // 2
                            keep_end = remaining_tokens - keep_start
                            
                            input_text_tokens = tokenizer.encode(input_text)
                            
                            if len(input_text_tokens) <= remaining_tokens:
                                # No need to truncate if it fits
                                truncated_text = input_text
                            else:
                                # Get tokens from start and end
                                start_tokens = input_text_tokens[:keep_start]
                                end_tokens = input_text_tokens[-keep_end:]
                                
                                # Convert back to text
                                truncated_text = tokenizer.decode(start_tokens) + " [...content truncated...] " + tokenizer.decode(end_tokens)
                            
                            text = f"{instruction}{truncated_text}{response_part}"
                    except Exception as e:
                        print(f"Error in middle truncation: {e}")
                        text = f"{instruction}[truncation error]{response_part}"
                else:
                    # Fallback if structure not recognized
                    text = f"### Instruction:\n[Content truncated]\n\n### Response:\n{example['response']}{tokenizer.eos_token}"
                    
            elif truncation_strategy == "head":
                # HEAD TRUNCATION: Keep the end part (truncate the beginning)
                # This is useful if the most important information is at the end
                
                # Calculate buffer to leave for instruction part
                instruction_buffer = instruction_token_count + 50  # Add some margin
                
                # Keep response and some buffer for it
                response_text = f"\n\n### Response:\n{example['response']}{tokenizer.eos_token}"
                response_token_count = len(tokenizer.encode(response_text))
                
                # Calculate available tokens for main content
                available_tokens = max_length - response_token_count - 50  # buffer
                
                if available_tokens <= instruction_buffer:
                    # Not enough space even for instruction
                    return f"### Instruction:\n[Truncated]\n\n### Input:\n[Truncated]\n\n### Response:\n{example['response']}{tokenizer.eos_token}"
                
                # Keep this much from the end
                keep_tokens = available_tokens - instruction_buffer
                
                # Ensure we have enough tokens to keep
                if keep_tokens <= 0:
                    return f"### Instruction:\n[Truncated]\n\n### Input:\n[Truncated]\n\n### Response:\n{example['response']}{tokenizer.eos_token}"
                
                # Get the last part of the text
                end_part = text[-keep_tokens*4:]  # Approximate character count
                
                # Make sure we have the response part
                if "### Response:" not in end_part:
                    end_part = f"{end_part}\n\n### Response:\n{example['response']}{tokenizer.eos_token}"
                
                # Include truncation notice
                text = f"...[content truncated]...\n{end_part}"
                
            else:  # "tail" or default
                # TAIL TRUNCATION: Keep the beginning (let tokenizer handle it)
                # Standard approach, works well when important info is at the beginning
                tokenizer.truncation_side = "right"
                text = tokenizer.decode(tokenizer.encode(text, truncation=True, max_length=max_length))
            
            return text
            
        except Exception as e:
            print(f"Error in formatting function: {e}")
            # Return safe fallback if there's an error
            return f"### Instruction:\n[Truncated due to formatting error]\n\n### Response:\nmalnutrition=no{tokenizer.eos_token}"
    
    # Create a dataset processing function that handles errors
    def prepare_dataset_for_trainer(dataset, formatting_function):
        processed_texts = []
        error_count = 0
        
        for i, example in enumerate(dataset):
            try:
                processed_text = formatting_function(example)
                processed_texts.append(processed_text)
                
                # Sample logging of processed examples
                if i < 3 or i % 1000 == 0:
                    print(f"Example {i} processed successfully")
                    # Log the first 100 and last 100 characters of processed text
                    print(f"Text preview: {processed_text[:100]}...{processed_text[-100:]}")
                    
            except Exception as e:
                error_count += 1
                print(f"Error processing example {i}: {e}")
                # Add a safe fallback example
                processed_texts.append(f"### Instruction:\n[Error processing example]\n\n### Response:\nmalnutrition=no{tokenizer.eos_token}")
        
        print(f"Processing complete with {error_count} errors out of {len(dataset)} examples")
        return {"text": processed_texts}
    
    # Prepare datasets with error handling
    print("Processing training dataset...")
    train_texts = prepare_dataset_for_trainer(train_dataset, formatting_func)
    train_dataset_processed = Dataset.from_dict(train_texts)
    
    print("Processing validation dataset...")
    eval_texts = prepare_dataset_for_trainer(eval_dataset, formatting_func)
    eval_dataset_processed = Dataset.from_dict(eval_texts)
    
    # Calculate effective max_seq_length based on observed token counts
    sample_token_lengths = [len(tokenizer.encode(text)) for text in train_texts["text"][:100]]
    effective_max_length = min(max_length, max(sample_token_lengths) + 50)  # Add small buffer
    print(f"Using effective max sequence length: {effective_max_length}")
    
    # Set up training arguments with dynamic configuration
    training_args = TrainingArguments(
        per_device_eval_batch_size=batch_size,
        per_device_train_batch_size=batch_size, 
        gradient_accumulation_steps=1,
        warmup_ratio=0.1,
        max_grad_norm=1.0,
        num_train_epochs=num_epochs, 
        learning_rate=2e-5,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=10, 
        optim="adamw_8bit",
        weight_decay=0.1,
        lr_scheduler_type="linear",
        # save_strategy="epoch",
        seed=3407,
        report_to="none",
        output_dir=output_dir,
        # Add overflow protection for long sequences
        dataloader_drop_last=True,  
        dataloader_num_workers=2,
        remove_unused_columns=False
    )
        
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset_processed,
        # eval_dataset=eval_dataset_processed,
        processing_class=tokenizer,  
        max_seq_length=effective_max_length,
        dataset_text_field="text", 
        packing=True,
        args=training_args,
    )
    
    # Save some sample training data for inference consistency checks
    with open(os.path.join(output_dir, "sample_training_data.txt"), "w") as f:
        for i, text in enumerate(train_texts["text"][:5]):
            f.write(f"=== SAMPLE {i+1} ===\n")
            f.write(text)
            f.write("\n\n")
    
    # Start training with enhanced error handling
    try:
        print("Starting training...")
        trainer.train()
        
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            print("CUDA out of memory error. Trying with smaller batch size...")
            # Clean up CUDA memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Recreate trainer with smaller batch size
            reduced_batch_size = max(1, batch_size // 2)
            training_args.per_device_train_batch_size = reduced_batch_size
            training_args.per_device_eval_batch_size = reduced_batch_size
            print(f"Retrying with batch size: {reduced_batch_size}")
            
            trainer = SFTTrainer(
                model=model,
                train_dataset=train_dataset_processed,
                # eval_dataset=eval_dataset_processed,
                processing_class=tokenizer,  
                max_seq_length=effective_max_length,
                dataset_text_field="text", 
                packing=True,
                args=training_args,
            )
            
            trainer.train()
            
        elif "token indices sequence length is longer than" in str(e):
            print("Input sequence too long. Trying with stricter truncation...")
            
            # Clean up memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Update tokenizer with more aggressive truncation
            tokenizer.truncation_side = "right"  # Always truncate from the end
            effective_max_length = effective_max_length - 100  # Add more buffer
            
            print(f"Reprocessing datasets with stricter truncation (max_length={effective_max_length})...")
            
            # Reprocess with stricter length
            train_texts = prepare_dataset_for_trainer(train_dataset, formatting_func)
            train_dataset_processed = Dataset.from_dict(train_texts)
            
            eval_texts = prepare_dataset_for_trainer(eval_dataset, formatting_func)
            eval_dataset_processed = Dataset.from_dict(eval_texts)
            
            # Create new trainer with updated settings
            trainer = SFTTrainer(
                model=model,
                train_dataset=train_dataset_processed,
                # eval_dataset=eval_dataset_processed,
                processing_class=tokenizer,  
                max_seq_length=effective_max_length,
                dataset_text_field="text", 
                packing=True,
                args=training_args,
            )
            
            print(f"Retrying with more aggressive truncation...")
            trainer.train()
            
        else:
            print(f"Training error: {e}")
            raise
    
    # Save configuration to ensure inference matches training
    config = {
        "model_name": model_name,
        "max_length": max_length,
        "truncation_strategy": truncation_strategy,
        "instruction": instruction,
        "prompt_template": "### Instruction:\n{instruction}\n\n### Input:\n{note}\n\n### Response:"
    }
    
    with open(os.path.join(output_dir, "model_config.json"), "w") as f:
        import json
        json.dump(config, f)
    
    # Save the model with optimized settings for inference
    save_path = f"{output_dir}/final_model"
    model.save_pretrained_merged(
        save_path, 
        tokenizer, 
        save_method="merged_16bit"
    )
    
    # Save tokenizer separately for inference
    tokenizer.save_pretrained(os.path.join(save_path, "tokenizer"))
    
    print(f"Model successfully saved to {save_path}")
    print(f"Configuration saved to {os.path.join(output_dir, 'model_config.json')}")
    
    return model, tokenizer
    
if __name__ == "__main__":
    # Configuration parameters
    data_path = "data/notes_train.csv"  # Path to your CSV dataset
    model_name = "unsloth/mistral-7b-instruct-v0.3-bnb-4bit" 
    output_dir = "./mistral-7b-instruct-v0.3-malnutrition_model"
    max_length = 32000
    num_epochs = 15  
    batch_size = 2 
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Train the model
    print(f"Starting training using {model_name}...")
    model, tokenizer = train_malnutrition_model(
        data_path=data_path,
        model_name=model_name,
        output_dir=output_dir,
        max_length=max_length,
        num_epochs=num_epochs,
        batch_size=batch_size,
        truncation_strategy="hybrid"
    )
    
    print("Training completed successfully!")
