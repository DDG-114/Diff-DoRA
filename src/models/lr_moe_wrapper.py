"""
src/models/lr_moe_wrapper.py
-----------------------------
LR-MoE inference wrapper: selects the correct expert adapter at runtime
based on the hard router label for each node.

Usage:
  wrapper = LRMoEWrapper(base_model, tokenizer, labels,
                         expert_paths=["outputs/.../expert_0/adapter",
                                       "outputs/.../expert_1/adapter"])
  response = wrapper.generate(sample, node_idx=3, horizon=6)
"""
from __future__ import annotations

from pathlib import Path

import torch

from peft import PeftModel
from transformers import AutoTokenizer

from src.models.qwen_peft import generate as _generate
from src.routing.hard_router import HardRouter
from src.prompts.prompt_vanilla import build_vanilla_prompt


class LRMoEWrapper:
    def __init__(
        self,
        base_model,
        tokenizer,
        labels,
        expert_paths: list[str],
    ):
        """
        Parameters
        ----------
        base_model    : loaded base AutoModelForCausalLM
        tokenizer     : corresponding tokenizer
        labels        : (N,) int array from build_routing_labels
        expert_paths  : list of adapter directories, indexed by expert id
        """
        self.tokenizer = tokenizer
        self.router    = HardRouter(labels)
        self.experts: dict[int, PeftModel] = {}

        for eid, path in enumerate(expert_paths):
            p = Path(path)
            if p.exists():
                self.experts[eid] = PeftModel.from_pretrained(base_model, str(p))
                self.experts[eid].eval()
                print(f"[LRMoE] Expert {eid} loaded from {path}")
            else:
                print(f"[LRMoE] Warning: expert path not found: {path}")

    def generate(
        self,
        sample: dict,
        node_idx: int,
        horizon: int = 6,
        system_msg: str | None = None,
        user_msg:   str | None = None,
        max_new_tokens: int = 128,
    ) -> str:
        """Route sample to the correct expert and generate."""
        eid = self.router.route(node_idx)
        model = self.experts.get(eid)
        if model is None:
            raise ValueError(f"Expert {eid} not loaded.")

        if system_msg is None or user_msg is None:
            system_msg, user_msg = build_vanilla_prompt(sample, node_idx, horizon)

        return _generate(model, self.tokenizer, system_msg, user_msg,
                         max_new_tokens=max_new_tokens)
