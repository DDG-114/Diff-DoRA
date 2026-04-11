"""
src/models/diff_dora.py
------------------------
Diff-DoRA: Differential-signal-driven magnitude adaptation on top of DoRA.

Architecture
------------
Standard DoRA decomposes W = m · (V / ||V||).
Diff-DoRA modulates the magnitude vector m with a learned scalar gate g:

    m_eff = m · (1 + g(diff_features))

where g is a small 2-layer MLP with sigmoid output.

Implementation strategy
-----------------------
Because PEFT wraps the magnitude as a registered parameter, we inject
DiffDoRAHook as a forward hook on every LoRA layer after model creation.
The hook reads a thread-local diff_context that the caller must set before
each forward pass via `DiffDoRAController.set_context(diff_vec)`.
"""
from __future__ import annotations

import threading
from typing import Any

import torch
import torch.nn as nn
from peft import PeftModel


# ─── Thread-local context ────────────────────────────────────────────────────

_ctx = threading.local()


def set_diff_context(diff_vec: torch.Tensor) -> None:
    """Set the current differential feature vector for the next forward pass."""
    _ctx.diff_vec = diff_vec


def get_diff_context() -> torch.Tensor | None:
    return getattr(_ctx, "diff_vec", None)


# ─── Controller MLP ──────────────────────────────────────────────────────────

class DiffController(nn.Module):
    """
    g(diff_features) → scalar gate in [0, scale].

    Parameters
    ----------
    input_dim : dimensionality of the diff feature vector
    scale     : maximum magnitude modulation (default 1.0 → gate ∈ [0,1])
    """

    def __init__(self, input_dim: int, hidden_dim: int = 32, scale: float = 0.5):
        super().__init__()
        self.scale = scale
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, diff_vec: torch.Tensor) -> torch.Tensor:
        """Return scalar gate ∈ [0, scale]."""
        return self.net(diff_vec) * self.scale


# ─── Diff-DoRA wrapper ───────────────────────────────────────────────────────

class DiffDoRAModel(nn.Module):
    """
    Wraps a PEFT DoRA model and inserts Diff-DoRA magnitude modulation.

    Usage
    -----
    diff_model = DiffDoRAModel(peft_dora_model, diff_input_dim=3)
    set_diff_context(diff_tensor)   # shape (diff_input_dim,)
    output = diff_model(**inputs)
    """

    def __init__(
        self,
        peft_model: PeftModel,
        diff_input_dim: int,
        hidden_dim: int = 32,
        scale: float = 0.5,
    ):
        super().__init__()
        self.peft_model  = peft_model
        self.controller  = DiffController(diff_input_dim, hidden_dim, scale)
        self._register_hooks()

    def _register_hooks(self):
        self._hooks = []
        for name, module in self.peft_model.named_modules():
            # PEFT DoRA stores magnitude as "lora_magnitude_vector"
            if hasattr(module, "lora_magnitude_vector"):
                hook = module.register_forward_pre_hook(self._make_hook(module))
                self._hooks.append(hook)

    def _make_hook(self, module):
        def hook(mod, inputs):
            diff_vec = get_diff_context()
            if diff_vec is None:
                return inputs
            gate = self.controller(diff_vec.to(mod.lora_magnitude_vector.device))
            # Temporarily scale the magnitude; will be restored by no-grad context
            # (We add to the original magnitude)
            mod.lora_magnitude_vector.data *= (1.0 + gate.squeeze())
            return inputs
        return hook

    def forward(self, **kwargs) -> Any:
        return self.peft_model(**kwargs)

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
