import json
import os
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model, TaskType
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)

# Load HF token from environment or .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("HF_TOKEN not found in environment. Please set it in your .env file.")
TRAIN_FILE = Path("mlx_data/train.jsonl")
VALID_FILE = Path("mlx_data/valid.jsonl")
MODEL_NAME_OR_PATH = "google/gemma-2b"
OUTPUT_DIR = Path("output/gemma2b-lora-windows")
MAX_LENGTH = 2048
PER_DEVICE_TRAIN_BATCH_SIZE = 1
PER_DEVICE_EVAL_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 4
LEARNING_RATE = 3e-4
NUM_TRAIN_EPOCHS = 1.0
LOGGING_STEPS = 20
SAVE_STEPS = 100
EVAL_STEPS = 100
FP16 = True
SEED = 42

# MODEL_NAME_OR_PATH = "google/gemma-7b"
# NUM_TRAIN_EPOCHS = 2.0  # Train twice for better convergence
# # Modify prepareLora.py:
# TOTAL = 48000


def load_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def build_examples(conversations):
    examples = []
    for conv in conversations:
        messages = conv.get("messages", [])
        if not messages:
            continue

        history = []
        for m in messages:
            role = m.get("role")
            content = (m.get("content") or "").strip()
            if not content:
                continue

            if role == "user":
                history.append(f"USER: {content}")
            elif role == "assistant":
                prompt = "\n\n".join(history)
                if prompt:
                    prompt = prompt + "\n\nASSISTANT: "
                else:
                    prompt = "ASSISTANT: "
                response = content
                examples.append({"prompt": prompt, "response": response})
                history.append(f"ASSISTANT: {content}")
            else:
                continue
    return examples


class ChatDataset(torch.utils.data.Dataset):
    def __init__(self, examples, tokenizer, max_length):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]
        prompt = example["prompt"]
        response = example["response"]

        prompt_ids = self.tokenizer(prompt, add_special_tokens=False).input_ids
        response_ids = self.tokenizer(response, add_special_tokens=False).input_ids
        if self.tokenizer.eos_token_id is not None:
            response_ids = response_ids + [self.tokenizer.eos_token_id]

        input_ids = prompt_ids + response_ids
        labels = [-100] * len(prompt_ids) + response_ids

        if len(input_ids) > self.max_length:
            overflow = len(input_ids) - self.max_length
            input_ids = input_ids[overflow:]
            labels = labels[overflow:]

        attention_mask = [1] * len(input_ids)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fp16 = FP16 and device.type == "cuda"

    print(f"Using device: {device}")
    if FP16 and device.type != "cuda":
        print("FP16 disabled because CUDA is not available.")

    print(f"Loading tokenizer for {MODEL_NAME_OR_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_OR_PATH, trust_remote_code=True, token=HF_TOKEN)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print(f"Loading base model: {MODEL_NAME_OR_PATH}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME_OR_PATH,
        device_map="auto",
        torch_dtype=torch.float16 if fp16 else torch.float32,
        trust_remote_code=True,
        token=HF_TOKEN,
    )

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
    )
    model = get_peft_model(model, lora_config)

    print("Building examples from JSONL data")
    train_data = load_jsonl(TRAIN_FILE)
    valid_data = load_jsonl(VALID_FILE)

    train_examples = build_examples(train_data)
    valid_examples = build_examples(valid_data)

    if not train_examples:
        raise ValueError(f"No training examples found in {TRAIN_FILE}")
    if not valid_examples:
        raise ValueError(f"No validation examples found in {VALID_FILE}")

    train_dataset = ChatDataset(train_examples, tokenizer, MAX_LENGTH)
    valid_dataset = ChatDataset(valid_examples, tokenizer, MAX_LENGTH)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8,
    )

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        eval_steps=EVAL_STEPS,
        eval_strategy="steps",
        save_strategy="steps",
        fp16=fp16,
        warmup_steps=10,
        logging_dir=str(OUTPUT_DIR / "logs"),
        report_to="none",
        load_best_model_at_end=True,
        greater_is_better=False,
        lr_scheduler_type="cosine",
        ddp_find_unused_parameters=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=data_collator,
    )

    print("Starting training")
    trainer.train()
    print("Saving final model")
    trainer.save_model(str(OUTPUT_DIR))


if __name__ == "__main__":
    main()
