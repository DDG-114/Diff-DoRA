"""
src/models/dora_adapter.py
---------------------------
DoRA (Weight-Decomposed Low-Rank Adaptation) helper.

PEFT >= 0.11 supports DoRA via use_dora=True in LoraConfig.
This module exposes a convenience function that wraps the PEFT call
and documents the decomposition for reference.

Paper: DoRA decomposes W = m · (V / ||V||) and adapts both
magnitude m and direction V/||V|| separately.

In practice: just pass use_dora=True to get_lora_model() in qwen_peft.py.
This file is here to make the dependency explicit and to document
the DoRA-specific hyper-parameters used in this project.
"""
from __future__ import annotations

from peft import LoraConfig, TaskType, get_peft_model

DORA_CONFIG = dict(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    use_dora=True,          # ← key difference vs LoRA
)


def get_dora_model(base_model, **overrides):
    """Wrap base model with a DoRA adapter."""
    cfg = {**DORA_CONFIG, **overrides}
    config = LoraConfig(**cfg)
    model = get_peft_model(base_model, config)
    model.print_trainable_parameters()
    return model
