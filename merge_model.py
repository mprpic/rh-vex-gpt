#!/usr/bin/env -S uv run --script
# /// script
# dependencies = [
#   "torch",
#   "transformers",
#   "peft",
#   "sentencepiece",
#   "protobuf",
# ]
# ///
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base_model_name = "mistralai/Mistral-7B-v0.3"
base_path = Path(__file__).parent
adapter_path = base_path / "training/model_20250614/"
merged_model_path = base_path / "training/merged_model_20250614/"
merged_model_path.mkdir(parents=True, exist_ok=True)

print(f"Loading base model: {base_model_name}")
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype=torch.bfloat16,
    device_map="cpu",
)

print(f"Loading tokenizer from: {adapter_path}")
tokenizer = AutoTokenizer.from_pretrained(adapter_path)

print("Loading PEFT model to merge...")
model = PeftModel.from_pretrained(base_model, adapter_path)

print("Merging adapter weights...")
merged_model = model.merge_and_unload()
print("Merge complete.")

print(f"Saving merged model to: {merged_model_path}")
merged_model.save_pretrained(merged_model_path)
tokenizer.save_pretrained(merged_model_path)
print("Merged model saved successfully!")
