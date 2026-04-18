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
Because PEFT wraps DoRA magnitude in a dedicated module, we patch the DoRA
layer forward path after model creation and inject a differentiable
sample-conditioned scale factor there.
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import PeftModel
from peft.utils.integrations import dequantize_module_weight
from peft.utils.other import transpose


# ─── Global context (checkpoint-safe) ───────────────────────────────────────

# NOTE:
# Using thread-local context can break gradient-checkpoint recomputation,
# because recompute may run on a different worker thread and fail to read the
# same diff vector, causing forward/recompute graph mismatch.
_diff_ctx: torch.Tensor | None = None


def set_diff_context(diff_vec: torch.Tensor | None) -> None:
    """Set the current differential feature vector for the next forward pass."""
    global _diff_ctx
    _diff_ctx = diff_vec


def get_diff_context() -> torch.Tensor | None:
    return _diff_ctx


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
    set_diff_context(diff_tensor)   # shape (diff_input_dim,) or (batch, diff_input_dim)
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
        self.peft_model = peft_model
        self.controller = DiffController(diff_input_dim, hidden_dim, scale)
        self._patched_layers: list[tuple[nn.Module, Any]] = []
        self._patch_dora_layers()

    def _get_scale_factor(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor | None:
        diff_vec = get_diff_context()
        if diff_vec is None:
            return None
        diff_vec = diff_vec.to(device=device)
        gate = self.controller(diff_vec)
        return (1.0 + gate.squeeze(-1)).to(device=device, dtype=dtype)

    @staticmethod
    def _broadcast_mag_norm_scale(scale_factor: torch.Tensor, dora_layer: nn.Module, weight_norm: torch.Tensor, base_result: torch.Tensor) -> torch.Tensor:
        base_mag = (dora_layer.weight / weight_norm).to(dtype=base_result.dtype)
        if scale_factor.ndim == 0:
            return (base_mag * scale_factor).view(*([1] * (base_result.dim() - 1)), -1)

        # One gate per sample; broadcast across sequence/time dims and output channels.
        base_view = (1,) + (1,) * max(base_result.dim() - 2, 0) + (-1,)
        sample_view = (-1,) + (1,) * max(base_result.dim() - 2, 0) + (1,)
        return base_mag.view(*base_view) * scale_factor.view(*sample_view)

    def _patch_dora_layers(self):
        for module in self.peft_model.modules():
            mag = getattr(module, "lora_magnitude_vector", None)
            if not isinstance(mag, nn.ModuleDict):
                continue
            for dora_layer in mag.values():
                self._patch_single_dora_layer(dora_layer)

    def _patch_single_dora_layer(self, dora_layer: nn.Module):
        if getattr(dora_layer, "_diffdora_patched", False):
            return

        original_forward = dora_layer.forward
        outer = self

        def forward_with_diff(x, *args, **kwargs):
            scale_factor = outer._get_scale_factor(dora_layer.weight.device, dora_layer.weight.dtype)
            if scale_factor is None:
                return original_forward(x, *args, **kwargs)

            lora_A = kwargs["lora_A"]
            lora_B = kwargs["lora_B"]
            scaling = kwargs["scaling"]
            base_layer = kwargs["base_layer"]

            # Linear path: this is the one used by Qwen target modules.
            if "embed_fn" not in kwargs and not hasattr(dora_layer, "conv_fn"):
                base_result = kwargs.get("base_result")
                x_eye = torch.eye(lora_A.weight.shape[1], device=lora_A.weight.device, dtype=x.dtype)
                lora_weight = lora_B(lora_A(x_eye)).T
                weight = dequantize_module_weight(base_layer).to(x.dtype)
                weight_norm = dora_layer.get_weight_norm(weight, lora_weight.detach(), scaling).detach()

                lora_result = lora_B(lora_A(x))
                if base_result is not None:
                    bias = base_layer.bias
                    if bias is not None:
                        base_result = base_result - bias
                else:
                    base_result = F.linear(x, transpose(weight, dora_layer.fan_in_fan_out))

                mag_norm_scale = outer._broadcast_mag_norm_scale(scale_factor, dora_layer, weight_norm, base_result)
                return (mag_norm_scale - 1) * base_result + mag_norm_scale * lora_result * scaling

            # Embedding / conv paths are not used by the current Qwen target modules.
            return original_forward(x, *args, **kwargs)

        dora_layer.forward = forward_with_diff
        dora_layer._diffdora_original_forward = original_forward
        dora_layer._diffdora_patched = True
        self._patched_layers.append((dora_layer, original_forward))

    def forward(self, *args, **kwargs) -> Any:
        return self.peft_model(*args, **kwargs)

    @property
    def device(self):
        if hasattr(self.peft_model, "device"):
            return self.peft_model.device
        return next(self.peft_model.parameters()).device

    def generate(self, *args, **kwargs):
        return self.peft_model.generate(*args, **kwargs)

    def remove_hooks(self):
        for layer, original_forward in self._patched_layers:
            layer.forward = original_forward
            layer._diffdora_patched = False
        self._patched_layers.clear()
