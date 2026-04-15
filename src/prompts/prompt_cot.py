"""
src/prompts/prompt_cot.py
--------------------------
Three-stage Chain-of-Thought prompt template (paper Section 3.x):

  Stage 1 – Gap Analysis     : Compare current occupancy vs. retrieved historical
  Stage 2 – Spatial Logic    : Incorporate neighbour influence
  Stage 3 – Numerical Pred.  : Output the final list

The parser only reads the last "Numerical Prediction: [...]" line.
"""
from __future__ import annotations

import numpy as np

from src.prompts.prompt_vanilla import _format_series, format_static_context
from src.retrieval.diff_features import format_diff_block

COT_SYSTEM_MSG = (
    "You are an expert EV charging demand forecaster. "
    "You will reason step by step through three stages: "
    "(1) Gap Analysis, (2) Spatial Logic, (3) Numerical Prediction. "
    "Always end with exactly:\n"
    "Numerical Prediction: [v1, v2, ..., vH]\n"
    "where H is the forecast horizon."
)

COT_TARGET_TEMPLATE = """\
## Stage 1 – Gap Analysis
Current mean occupancy: {curr_occ:.3f}
Retrieved mean occupancy: {retr_occ:.3f}
{diff_block}
The gap suggests {gap_desc}.

## Stage 2 – Spatial Logic
{static_block}Neighbour mean occupancy (last step): {nbr_occ:.3f}
Spatial influence: {spatial_desc}.

## Stage 3 – Numerical Prediction
Based on the above analysis:
Numerical Prediction: {pred_list}"""


def build_cot_prompt(
    sample: dict,
    retrieved_samples: list[dict],
    diff_features: dict,
    node_idx: int = 0,
    horizon: int = 6,
    domain_label: str | None = None,
    static_context: dict | None = None,
) -> tuple[str, str]:
    """
    Build (system_msg, user_msg) including retrieved context, diff features, and optional domain label.
    
    The domain_label (e.g., "CBD" or "Residential") acts as a metadata tag to guide the MoE router,
    as described in the LR-MoE paper: "In the inference phase, the model activates the corresponding
    expert based on the metadata label in the input prompt."
    """
    x = sample["x_hist"][:, node_idx]   # (12,)
    nbr = sample["nbr_feat"][-1, node_idx]  # last step neighbour mean

    # Retrieved history
    retr_lines = []
    for i, rs in enumerate(retrieved_samples):
        rx = rs["x_hist"][:, node_idx]
        retr_lines.append(f"  Ref {i+1}: {_format_series(rx)}")
    retr_block = "\n".join(retr_lines) if retr_lines else "  (none)"

    diff_str = format_diff_block(diff_features)

    # Add domain label as metadata prefix if provided
    meta_parts = []
    if domain_label:
        meta_parts.append(f"[Domain: {domain_label}]")
    static_block = format_static_context(static_context)
    if static_block:
        meta_parts.append(static_block)
    meta_prefix = "\n\n".join(meta_parts)
    if meta_prefix:
        meta_prefix += "\n\n"

    user_msg = (
        f"{meta_prefix}"
        f"Current historical occupancy (12 steps): {_format_series(x)}\n\n"
        f"Retrieved similar windows:\n{retr_block}\n\n"
        f"Environmental differentials: {diff_str}\n\n"
        f"Neighbour occupancy (last step): {nbr:.3f}\n\n"
        f"Forecast horizon: {horizon} steps\n\n"
        "Reason through the three stages and output the forecast.\n"
        "Numerical Prediction:"
    )
    return COT_SYSTEM_MSG, user_msg


def build_cot_target(
    sample: dict,
    retrieved_samples: list[dict],
    diff_features: dict,
    node_idx: int = 0,
    horizon: int = 6,
    static_context: dict | None = None,
) -> str:
    """
    Build the supervised target string (for training).
    """
    x       = sample["x_hist"][:, node_idx]
    nbr     = sample["nbr_feat"][-1, node_idx]
    y       = sample["y"][:horizon, node_idx]

    curr_occ = float(x.mean())
    retr_occ = float(np.mean([rs["x_hist"][:, node_idx].mean() for rs in retrieved_samples])) \
               if retrieved_samples else curr_occ
    diff_occ = diff_features.get("diff_occ", 0.0)
    gap_desc  = "higher demand" if diff_occ > 0.05 else ("lower demand" if diff_occ < -0.05 else "stable demand")
    spatial_desc = "increasing pressure" if nbr > curr_occ + 0.05 else "stable neighbourhood"
    static_block = format_static_context(static_context)
    if static_block:
        static_block = static_block.replace("Static station context:\n", "")
        static_block = static_block + "\n"

    return COT_TARGET_TEMPLATE.format(
        curr_occ    = curr_occ,
        retr_occ    = retr_occ,
        diff_block  = format_diff_block(diff_features),
        gap_desc    = gap_desc,
        static_block= static_block,
        nbr_occ     = nbr,
        spatial_desc= spatial_desc,
        pred_list   = _format_series(y),
    )
