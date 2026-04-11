"""
src/models/qwen_peft.py
------------------------
Qwen2.5-1.5B-Instruct + LoRA (or DoRA) adapter via PEFT.

Usage:
  from src.models.qwen_peft import load_model_and_tokenizer, get_lora_model
  model, tokenizer = load_model_and_tokenizer()
  peft_model = get_lora_model(model, use_dora=False)
"""
from __future__ import annotations

from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    PeftModel,
)

MODEL_PATH = str(Path("models/Qwen2.5-1.5B-Instruct"))

LORA_CONFIG = dict(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)


def load_model_and_tokenizer(
    model_path: str = MODEL_PATH,
    load_in_4bit: bool = False,
    device_map: str = "auto",
) -> tuple:
    """Load base model and tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_cfg = None
    if load_in_4bit:
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map=device_map,
        quantization_config=bnb_cfg,
    )
    model.config.use_cache = False
    return model, tokenizer


def get_lora_model(
    model,
    use_dora: bool = False,
    **lora_kwargs,
) -> "PeftModel":
    """Wrap base model with LoRA (or DoRA) adapter."""
    cfg_dict = {**LORA_CONFIG, **lora_kwargs}
    if use_dora:
        cfg_dict["use_dora"] = True
    config = LoraConfig(**cfg_dict)
    peft_model = get_peft_model(model, config)
    peft_model.print_trainable_parameters()
    return peft_model


def load_peft_model(
    base_model,
    adapter_path: str,
) -> "PeftModel":
    """Load a saved PEFT adapter on top of a base model."""
    return PeftModel.from_pretrained(base_model, adapter_path)


@torch.inference_mode()
def generate(
    model,
    tokenizer,
    system_msg: str,
    user_msg: str,
    max_new_tokens: int = 256,
    temperature: float = 0.1,
) -> str:
    """Generate a response given system + user messages."""
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user",   "content": user_msg},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=temperature > 0,
        temperature=temperature if temperature > 0 else None,
        pad_token_id=tokenizer.pad_token_id,
    )
    # Decode only new tokens
    new_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_ids, skip_special_tokens=True)
