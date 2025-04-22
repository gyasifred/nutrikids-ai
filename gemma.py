#!/usr/bin/env python3
from unsloth import FastLanguageModel
import torch
max_seq_length = 4096 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.


model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/gemma-7b-bnb-4bit",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
)

"""We now add LoRA adapters so we only need to update 1 to 10% of all parameters!"""

model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)

import pandas as pd
from typing import List, Dict, Optional, Tuple
from datasets import Dataset
import random
import re
import json

class MalnutritionDataset:
    """Class to handle malnutrition dataset operations."""

    def __init__(self, data_path: str, note_col: str, label_col: str):
        """Initialize dataset from a CSV file.

        Args:
            data_path (str): Path to the CSV file containing the data
            note_col (str): Name of the text column in the CSV
            label_col (str): Name of the label column in the CSV
        """
        self.df = pd.read_csv(data_path)
        self.text_col = note_col
        self.label_col = label_col
    
    def __len__(self):
        """Return the number of items in the dataset."""
        return len(self.df)

    def prepare_training_data(self, prompt_builder, tokenizer) -> List[Dict[str, str]]:
        """Prepare data in the format required for training, following Alpaca-style formatting.

        Args:
            prompt_builder: An instance of MalnutritionPromptBuilder
            tokenizer: The tokenizer to use for adding EOS token

        Returns:
            List of dictionaries with formatted text for training
        """
        EOS_TOKEN = tokenizer.eos_token
        formatted_data = []

        for _, row in self.df.iterrows():
            # Generate prompt for each example
            prompt = prompt_builder.get_training_prompt(row[self.text_col])

            # Format label as "yes" or "no"
            label_text = "yes" if str(row[self.label_col]).lower() in ["1", "yes", "true"] else "no"

            # JSON format output as per the prompt builder's specifications
            output = json.dumps({"malnutrition": label_text,
                               "explanation": f"Based on the clinical evidence in the patient notes."})

            # Create formatted text with prompt and response, similar to Alpaca format
            formatted_text = f"{prompt}\n{output}{EOS_TOKEN}"

            formatted_data.append({
                "text": formatted_text
            })

        return formatted_data

    def to_huggingface_dataset(self, prompt_builder, tokenizer) -> Dataset:
        """Convert prepared data to a HuggingFace Dataset.

        Args:
            prompt_builder: An instance of MalnutritionPromptBuilder
            tokenizer: Tokenizer for adding EOS token

        Returns:
            HuggingFace Dataset ready for model training
        """
        formatted_data = self.prepare_training_data(prompt_builder, tokenizer)

        return Dataset.from_dict({
            "text": [item["text"] for item in formatted_data]
        })
# ────────────────────────────────────────────────────────────────────────────────
# Enhanced clinical‑note cleaner
# ────────────────────────────────────────────────────────────────────────────────
def preprocess_text(text: str) -> str:
    """Light preprocessing to clean clinical text."""
    text = re.sub(r"\s+", " ", text)                    # collapse multiple spaces/newlines
    text = text.replace(" </s> ", "\n- ")               # convert separators to bullets
    text = re.sub(r"_date_|_lgnum_", "[REDACTED]", text)
    return text.strip()


# ────────────────────────────────────────────────────────────────────────────────
# Enhanced Prompt‑builder with improved prompt design
# ────────────────────────────────────────────────────────────────────────────────
class MalnutritionPromptBuilder:
    """Manage creation of malnutrition prompts for training or inference."""

    # ------------------------------------------------------------------ #
    # Initialise (optionally load CSV of few‑shot examples)
    # ------------------------------------------------------------------ #
    def __init__(self, examples_csv_path: Optional[str] = None):
        self.examples_csv_path = examples_csv_path
        self.examples_cache: Optional[pd.DataFrame] = None

        if examples_csv_path:
            try:
                self.examples_cache = pd.read_csv(examples_csv_path)
                print(f"[PromptBuilder] Loaded {len(self.examples_cache)} examples from {examples_csv_path}")
            except Exception as exc:  # noqa: BLE001
                print(f"[PromptBuilder] Error loading examples: {exc}")

    # ------------------------------------------------------------------ #
    # Public helpers
    # ------------------------------------------------------------------ #
    def get_training_prompt(self, patient_notes: str) -> str:
        """Return a zero‑shot prompt for supervised fine‑tuning."""
        return self._construct_prompt(patient_notes)

    def get_inference_prompt(
        self,
        patient_notes: str,
        note_col: str,
        label_col: str,
        *,
        num_examples: int = 0,
        specific_example_indices: Optional[List[int]] = None,
        balanced: bool = False,
    ) -> str:
        """
        Build an inference prompt with optional few‑shot examples.

        Parameters
        ----------
        patient_notes : str
            Note you want classified.
        note_col, label_col : str
            Column names in the CSV for the note text and label.
        num_examples : int
            Number of examples to prepend (0 → zero‑shot).
        specific_example_indices : list[int], optional
            Explicit rows to use as few‑shot examples.
        balanced : bool
            If True and `num_examples ≥ 2`, sample a 50/50 yes/no mix.
        """
        if num_examples == 0 or self.examples_cache is None:
            return self._construct_prompt(patient_notes)

        # balanced branch
        if balanced and num_examples >= 2:
            return self._get_balanced_prompt(
                patient_notes, note_col, label_col, num_examples=num_examples
            )

        # generic few‑shot sampling
        few_shot_examples: List[Dict[str, str]] = []
        if specific_example_indices:
            valid = [i for i in specific_example_indices if 0 <= i < len(self.examples_cache)]
            for idx in valid[: num_examples]:
                few_shot_examples.append(
                    {
                        "text": self.examples_cache.at[idx, note_col],
                        "label": self.examples_cache.at[idx, label_col],
                    }
                )
        else:
            chosen = random.sample(
                range(len(self.examples_cache)), k=min(num_examples, len(self.examples_cache))
            )
            for idx in chosen:
                few_shot_examples.append(
                    {
                        "text": self.examples_cache.at[idx, note_col],
                        "label": self.examples_cache.at[idx, label_col],
                    }
                )

        return self._construct_prompt(patient_notes, few_shot_examples)

    def get_balanced_inference_prompt(
        self,
        patient_notes: str,
        text_col: str,
        label_col: str,
        *,
        num_examples: int = 4,
    ) -> str:
        """Return a prompt with a balanced yes/no example mix."""
        return self._get_balanced_prompt(
            patient_notes, text_col, label_col, num_examples=num_examples
        )

    # ------------------------------------------------------------------ #
    # Few‑shot example formatter - enhanced for clarity
    # ------------------------------------------------------------------ #
    def _format_example(self, example: Dict[str, str]) -> str:
        """Return one example block that matches the JSON output spec."""
        notes = preprocess_text(example["text"])
        raw_label = str(example["label"]).lower()
        label = "yes" if raw_label in {"1", "yes", "true"} else "no"

        # Enhanced explanations based on label
        if label == "yes":
            explanations = [
                "Evidence of significant weight loss and reduced dietary intake.",
                "Z-scores below -2 SD with clinical signs of wasting.",
                "Poor nutritional intake with anthropometric measurements in malnutrition range.",
                "Clinical evidence of muscle wasting with documented inadequate intake.",
                "Height-for-age z-score below -3 SD indicating severe chronic malnutrition."
            ]
        else:
            explanations = [
                "Anthropometry within normal limits with adequate intake documented.",
                "No significant weight loss or reduced intake reported.",
                "All nutritional parameters within normal range.",
                "Growth and development on track with no clinical signs of malnutrition.",
                "Z-scores within normal limits and no reported feeding difficulties."
            ]

        explanation = random.choice(explanations)

        return (
            "Patient notes:\n"
            f"{notes}\n"
            "Output:\n"
            f'{{"malnutrition":"{label}","explanation":"{explanation}"}}\n'
        )

    # ------------------------------------------------------------------ #
    # ENHANCED PROMPT CONSTRUCTION - improved readability and effectiveness
    # ------------------------------------------------------------------ #
    def _construct_prompt(
        self,
        patient_notes: str,
        few_shot_examples: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        notes_clean = preprocess_text(patient_notes)

        header = (
            "You are a board‑certified clinical dietitian with expertise in pediatric malnutrition assessment."
        )

        task = (
            "# ASSESSMENT TASK\n"
            "Evaluate the patient notes to determine if there is clinical evidence of malnutrition.\n\n"
            "Classification guidelines:\n"
            "• \"yes\" - Patient meets at least MILD criteria for malnutrition\n"
            "• \"no\" - Patient does not meet ANY malnutrition criteria\n"
            "• IMPORTANT: If evidence is borderline or ambiguous, classify as \"no\"\n"
        )

        checklist = (
            "# MALNUTRITION DIAGNOSTIC CRITERIA\n\n"
            "1. **ANTHROPOMETRY**\n"
            "   ✓ Mild:     z-score -1.0 to -1.9 SD\n"
            "   ✓ Moderate: z-score -2.0 to -2.9 SD\n"
            "   ✓ Severe:   z-score ≤ -3.0 SD\n"
            "   ✓ Weight-for-height, BMI-for-age, or weight-for-age\n"
            "   ✓ Height-for-age z-score ≤ -3 SD indicates severe stunting\n"
            "   ✓ MUAC (Mid-Upper Arm Circumference) follows same cutoffs\n\n"

            "2. **WEIGHT LOSS**\n"
            "   ✓ Documented involuntary weight loss\n"
            "   ✓ Failure to gain expected weight/height in child\n"
            "   ✓ Declining percentile crossing on growth charts\n\n"

            "3. **REDUCED INTAKE/ABSORPTION**\n"
            "   ✓ Decreased appetite or food intake\n"
            "   ✓ Feeding difficulties or dysphagia\n"
            "   ✓ Restricted diet or food insecurity\n"
            "   ✓ Malabsorption conditions\n\n"

            "4. **CLINICAL ASSESSMENT**\n"
            "   ✓ Muscle wasting (temporal, extremities)\n"
            "   ✓ Subcutaneous fat loss\n"
            "   ✓ Edema (can mask weight loss)\n"
            "   ✓ Poor wound healing, skin/hair changes\n\n"

            "5. **COMPLICATING FACTORS**\n"
            "   ✓ Chronic illness/inflammation\n"
            "   ✓ Increased metabolic demand (infection, trauma)\n"
            "   ✓ Medication effects on intake/absorption\n"
            "   ✓ Psychosocial factors\n"
        )

        output_spec = (
            "# OUTPUT REQUIREMENTS\n"
            "Return a valid JSON object with these exact fields:\n"
            "```json\n"
            "{\n"
            '  "malnutrition": "yes" | "no",\n'
            '  "confidence": <decimal between 0.1 and 0.9>,\n'
            '  "evidence": {\n'
            '    "anthropometry": "<Specific anthropometric findings from notes>",\n'
            '    "weight_trajectory": "<Comments on weight loss/gain patterns>",\n'
            '    "intake_absorption": "<Details on eating patterns/absorption issues>",\n'
            '    "clinical_signs": "<Observable clinical indicators>",\n'
            '    "complicating_factors": "<Relevant comorbidities or risk factors>"\n'
            '  },\n'
            '  "explanation": "<2-3 sentences synthesizing the key evidence supporting your conclusion>"\n'
            "}\n"
            "```\n"
            "- Provide only the complete JSON object without additional text\n"
            "- Base assessment solely on evidence present in the notes\n"
            "- Include specific metrics/findings for each evidence category (use \"None documented\" if no information available)\n"
            "- Confidence reflects your certainty (0.1=very uncertain, 0.9=very certain)\n"
            "- If evidence is sparse or ambiguous, use lower confidence values\n"
            "- Explanations should directly cite the strongest evidence points\n"
        )

        # Enhanced few-shot examples block with more detailed examples
        few_shot_block = ""
        if few_shot_examples:
            few_shot_block = (
                "# EXAMPLES\n"
                + "\n".join(self._format_detailed_example(ex) for ex in few_shot_examples)
                + "\n---\n"
            )

        # assemble prompt
        return (
            f"{header}\n\n{task}\n{checklist}\n{output_spec}\n"
            f"{few_shot_block}"
            "# PATIENT NOTES\n"
            f"{notes_clean}\n\n"
            "# ASSESSMENT"
        )

    def _format_detailed_example(self, example: Dict[str, str]) -> str:
        """Return one example block with detailed evidence categorization."""
        notes = preprocess_text(example["text"])
        raw_label = str(example["label"]).lower()
        label = "yes" if raw_label in {"1", "yes", "true"} else "no"
        
        # Set confidence based on label (could be randomized within ranges)
        confidence = 0.75 if label == "yes" else 0.65
        
        # Generate plausible evidence categories based on label
        if label == "yes":
            anthropometry = "Weight-for-height z-score -2.3 SD (moderate); BMI 12.1 kg/m² (<1st percentile)"
            weight_trajectory = "Lost 4.2 kg (10% body weight) over past 2 months; dropping from 25th to 5th percentile"
            intake = "Poor oral intake (<50% of meals); refusing high-calorie foods; early satiety reported"
            clinical = "Visible temporal wasting; reduced subcutaneous fat; reduced muscle mass in extremities"
            factors = "Concurrent inflammatory bowel disease; increased energy requirements due to infection"
            explanation = "Patient shows moderate malnutrition with weight-for-height z-score of -2.3 SD, significant weight loss of 10% over 2 months, and poor oral intake (<50% of meals). Clinical examination confirms temporal wasting and reduced muscle mass consistent with protein-energy malnutrition."
        else:
            anthropometry = "Weight-for-height z-score -0.8 SD (normal range); BMI 15.2 kg/m² (15th percentile)"
            weight_trajectory = "Stable growth pattern; maintaining 25th percentile on growth chart"
            intake = "Adequate caloric intake; eating 80-100% of meals; good variety in diet"
            clinical = "No visible muscle wasting; normal fat stores; no edema"
            factors = "Well-controlled chronic condition; normal activity levels"
            explanation = "Patient does not meet malnutrition criteria with weight-for-height z-score within normal range (-0.8 SD) and stable growth pattern maintaining the 25th percentile. Clinical assessment shows no muscle wasting or fat loss, and dietary intake is reported as adequate with 80-100% of meals consumed."
        
        # Format detailed JSON response
        response = {
            "malnutrition": label,
            "confidence": confidence,
            "evidence": {
                "anthropometry": anthropometry,
                "weight_trajectory": weight_trajectory,
                "intake_absorption": intake,
                "clinical_signs": clinical,
                "complicating_factors": factors
            },
            "explanation": explanation
        }
        
        return (
            "Patient notes:\n"
            f"{notes}\n"
            "Output:\n"
            f"{json.dumps(response, indent=2)}\n"
        )
        # ------------------------------------------------------------------ #
    # Improved balanced few‑shot sampling helper
    # ------------------------------------------------------------------ #
    def _get_balanced_prompt(
        self,
        patient_notes: str,
        note_col: str,
        label_col: str,
        *,
        num_examples: int = 4,
    ) -> str:
        """Return a prompt with a balanced yes/no example mix."""
        if self.examples_cache is None:
            return self._construct_prompt(patient_notes)

        yes_mask = self.examples_cache[label_col].astype(str).str.lower().isin({"1", "yes", "true"})
        yes_df = self.examples_cache[yes_mask]
        no_df = self.examples_cache[~yes_mask]

        # Calculate how many examples to get from each class
        yes_count = len(yes_df)
        no_count = len(no_df)
        total_available = yes_count + no_count

        if total_available < num_examples:
            # Not enough examples, use all available
            few_shot_examples = []
            for i in range(yes_count):
                few_shot_examples.append({
                    "text": yes_df.iloc[i][note_col],
                    "label": yes_df.iloc[i][label_col]
                })
            for i in range(no_count):
                few_shot_examples.append({
                    "text": no_df.iloc[i][note_col],
                    "label": no_df.iloc[i][label_col]
                })
        else:
            # We have enough examples, try to balance
            half = num_examples // 2
            yes_needed = min(half, yes_count)
            no_needed = min(num_examples - yes_needed, no_count)

            # If we couldn't get enough of one class, get more from the other
            if yes_needed < half and no_count > no_needed:
                no_needed = min(num_examples - yes_needed, no_count)
            elif no_needed < (num_examples - half) and yes_count > yes_needed:
                yes_needed = min(num_examples - no_needed, yes_count)

            # Sample
            few_shot_examples = []
            if yes_needed > 0:
                yes_indices = random.sample(range(yes_count), k=yes_needed)
                for i in yes_indices:
                    few_shot_examples.append({
                        "text": yes_df.iloc[i][note_col],
                        "label": yes_df.iloc[i][label_col]
                    })

            if no_needed > 0:
                no_indices = random.sample(range(no_count), k=no_needed)
                for i in no_indices:
                    few_shot_examples.append({
                        "text": no_df.iloc[i][note_col],
                        "label": no_df.iloc[i][label_col]
                    })

        # Shuffle to avoid order bias
        random.shuffle(few_shot_examples)
        return self._construct_prompt(patient_notes, few_shot_examples)

def extract_malnutrition_decision(response: str):
    """
    Extract the malnutrition decision and explanation from model output.
    Handles both structured JSON responses and free-text responses.
    
    Args:
        response: The model's generated response text
        
    Returns:
        tuple: (decision, explanation) where:
            - decision: "yes" or "no" string
            - explanation: Explanation text string
    """
    # Clean the input string
    cleaned = response.strip()
    
    # Initialize default values
    decision = "unknown"
    explanation = ""
    
    # Try to parse as JSON first
    try:
        # Check for JSON code blocks
        if "```json" in cleaned or "```" in cleaned:
            pattern = r'```(?:json)?\s*(.*?)\s*```'
            matches = re.findall(pattern, cleaned, re.DOTALL)
            if matches:
                for match in matches:
                    try:
                        parsed = json.loads(match.strip())
                        cleaned = match.strip()
                        break
                    except:
                        continue
        
        # Try direct JSON parsing
        parsed = json.loads(cleaned)
        
        # Extract decision
        if "malnutrition" in parsed:
            decision = str(parsed["malnutrition"]).lower()
            if decision not in ["yes", "no"]:
                decision = "unknown"
        
        # Extract explanation
        if "explanation" in parsed:
            explanation = str(parsed["explanation"]).strip()
            
        return decision, explanation
        
    except json.JSONDecodeError:
        # Fallback to regex extraction if JSON parsing fails
        pass
    
    # Extract decision with regex patterns
    decision_patterns = [
        r'"malnutrition"\s*:\s*"(yes|no)"',
        r'malnutrition\s*[=:]\s*(yes|no)',
        r"patient (does|doesn['']t) meet.*?malnutrition",
        r'(evidence|signs|indications) of malnutrition',
        r'assessment:\s*(malnutrition|no malnutrition)',
        r'diagnosis:\s*(malnutrition|no malnutrition)'
    ]

    for pattern in decision_patterns:
        decision_match = re.search(pattern, cleaned, flags=re.I)
        if decision_match:
            matched_text = decision_match.group(1).lower()
            if matched_text in ["yes", "does", "evidence", "signs", "indications", "malnutrition"]:
                decision = "yes"
                break
            elif matched_text in ["no", "doesn't", "doesn't", "doesnt", "no malnutrition"]:
                decision = "no"
                break
            elif matched_text in ["yes", "no"]:
                decision = matched_text
                break

    # Extract explanation
    explanation_patterns = [
        r'"explanation"\s*:\s*"((?:[^"\\]|\\.)*)"',
        r'explanation\s*[:=]\s*["\'](.*?)["\']',
        r'explanation:\s*(.*?)(?=\n\n|\}|$)',
        r'malnutrition.*?because\s+(.*?)(?:$|(?=\n\n|\}))',
        r'assessment:.*?([^.]*?(?:malnutrition|nutritional)[^.]*\.)'
    ]

    for pattern in explanation_patterns:
        expl_match = re.search(pattern, cleaned, flags=re.I | re.DOTALL)
        if expl_match:
            candidate_explanation = expl_match.group(1).strip()
            if candidate_explanation:
                explanation = candidate_explanation
                break
    
    return decision, explanation

# Create dataset and prepare data
dataset_handler = MalnutritionDataset(data_path="data/notes_train.csv", note_col="txt", label_col="label")
prompt_builder = MalnutritionPromptBuilder()

# Convert to HuggingFace Dataset - this is what we need to pass to SFTTrainer
hf_dataset = dataset_handler.to_huggingface_dataset(prompt_builder, tokenizer)

# Now use the HuggingFace Dataset with the SFTTrainer
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported

# Capture start memory for stats
start_gpu_memory = round(torch.cuda.memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(torch.cuda.get_device_properties(0).total_memory / 1024 / 1024 / 1024, 3)

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = hf_dataset,  # Use the HuggingFace Dataset here, not the MalnutritionDataset
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False,
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        num_train_epochs = 10,
        learning_rate = 2e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
        report_to = "none",
    ),
)

trainer_stats = trainer.train()

# # Show final memory and time stats
# used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
# used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
# used_percentage = round(used_memory / max_memory * 100, 3)
# lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
# print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
# print(f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training.")
# print(f"Peak reserved memory = {used_memory} GB.")
# print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
# print(f"Peak reserved memory % of max memory = {used_percentage} %.")
# print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")

# # For inference
# FastLanguageModel.for_inference(model)  # Enable native 2x faster inference

# # Define a function for malnutrition inference
# def predict_malnutrition(patient_notes):
#     # Create the prompt
#     prompt = prompt_builder.get_inference_prompt(patient_notes, "txt", "label")
    
#     # Tokenize and generate prediction
#     inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
#     outputs = model.generate(**inputs, max_new_tokens=128, use_cache=True)
#     response = tokenizer.batch_decode(outputs)[0]
    
#     # Extract decision and explanation
#     decision, explanation = extract_malnutrition_decision(response)
#     return {"malnutrition": decision, "explanation": explanation}

# Save the model
model.save_pretrained("trained/LLm_final/gemma/final_model")
tokenizer.save_pretrained("trained/LLm_final/gemma/final_model")
