#!/usr/bin/env -S uv run --script
# /// script
# dependencies = [
#   "torch",
#   "transformers",
#   "datasets",
#   "accelerate",
#   "peft",
#   "bitsandbytes",
#   "trl",
#   "sentencepiece",
#   "protobuf",
#   "huggingface_hub",
# ]
# ///
import argparse
import os
from pathlib import Path

import torch
from datasets import DatasetDict, load_dataset
from huggingface_hub import login as hf_login
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import SFTConfig, SFTTrainer

# QLoRA parameters
LORA_R = 128
LORA_ALPHA = 32
LORA_DROPOUT = 0.1

# Training parameters
EPOCHS = 3
BATCH_SIZE = 6
GRADIENT_ACCUMULATION_STEPS = 2
LEARNING_RATE = 2e-4
MAX_SEQ_LENGTH = 2048
VALIDATION_SPLIT = 0.05  # 5% for validation
MODEL_NAME = "mistralai/Mistral-7B-v0.3"
DATASET_NAME = "mprpic/rh-vex-data"


def main():
    parser = argparse.ArgumentParser(description="Fine-tune a model with LoRA")
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default="./data",
        help="Base path for output and logs (default: ./data)",
    )
    args = parser.parse_args()

    base_path = Path(args.output_dir)
    out_dir = base_path / "model"
    logs_dir = base_path / "logs"
    out_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {out_dir}")
    print(f"Logs directory: {logs_dir}")

    token = os.getenv("HF_TOKEN")
    if not token:
        try:
            token = Path("~/.cache/huggingface/token").expanduser().read_text().strip()
        except (FileNotFoundError, PermissionError):
            token = None
    hf_login(token=token)

    # BitsAndBytes config for 4-bit quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    # Load model and tokenizer
    print(f"Loading model: {MODEL_NAME}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Configure LoRA
    peft_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    print(f"GPU memory reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")

    # Load dataset
    print(f"Loading dataset: {DATASET_NAME}")
    dataset = load_dataset(DATASET_NAME, split="train")

    # Create a smaller sample of 20k records
    print(f"Original dataset size: {len(dataset)}")
    dataset = dataset.shuffle(seed=42).select(range(20000))
    print(f"New dataset size: {len(dataset)}")

    # Create train/validation split
    dataset_dict = dataset.train_test_split(test_size=VALIDATION_SPLIT, seed=42)
    dataset_dict = DatasetDict(
        {"train": dataset_dict["train"], "validation": dataset_dict["test"]}
    )

    # Print dataset info
    print(f"Training samples: {len(dataset_dict['train'])}")
    print(f"Validation samples: {len(dataset_dict['validation'])}")

    training_args = SFTConfig(
        output_dir=str(out_dir),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        warmup_steps=100,
        learning_rate=LEARNING_RATE,
        fp16=True,
        logging_dir=str(logs_dir),
        logging_steps=25,
        save_strategy="epoch",
        eval_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        report_to="none",
        remove_unused_columns=False,
        # Memory optimization
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        # Parameters for data handling
        max_seq_length=MAX_SEQ_LENGTH,
        dataset_text_field="text",
        packing=False,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset_dict["train"],
        eval_dataset=dataset_dict["validation"],
        args=training_args,
        peft_config=peft_config,
    )

    # Train
    print("Starting training...")
    trainer.train()

    # Save the fine-tuned model
    print(f"Saving model to {out_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(str(out_dir))
    print("Training complete!")


if __name__ == "__main__":
    main()
