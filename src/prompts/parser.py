"""
src/prompts/parser.py
----------------------
Parse model output into a list of floats.

The model is expected to output one of:
  Numerical Prediction: [0.3, 0.4, 0.5]
  Numerical Prediction: [[0.3, 0.4], [0.5, 0.6]]

Falls back to regex extraction when JSON parsing fails.
"""
from __future__ import annotations

import json
import re
import numpy as np


_LIST_RE = re.compile(r"\[[\d\s,.\-eE]+\]")
_NESTED_RE = re.compile(r"\[\s*\[[\d\s,.\-eE]+\](?:\s*,\s*\[[\d\s,.\-eE]+\])*\s*\]")


def parse_output(text: str, expected_len: int | None = None) -> np.ndarray | None:
    """
    Parse model output text into a numpy array.

    Returns
    -------
    np.ndarray with shape (horizon,) or (N, horizon), or None on failure.
    """
    # Strip everything before "Numerical Prediction:"
    if "Numerical Prediction:" in text:
        text = text.split("Numerical Prediction:")[-1].strip()

    # Guard against template placeholders, e.g. "[v1, v2, v3]".
    # These should be treated as parse failure rather than [1, 2, 3].
    if re.search(r"\bv\d+\b", text):
        return None

    # Try JSON parse of first [ ... ] block
    # 1. nested list
    m = _NESTED_RE.search(text)
    if m:
        try:
            arr = np.array(json.loads(m.group()), dtype=np.float32)
            return arr
        except (json.JSONDecodeError, ValueError):
            pass

    # 2. flat list
    m = _LIST_RE.search(text)
    if m:
        try:
            arr = np.array(json.loads(m.group()), dtype=np.float32)
            if expected_len is not None and len(arr) != expected_len:
                # Try to truncate / pad gracefully
                if len(arr) > expected_len:
                    arr = arr[:expected_len]
                else:
                    arr = np.pad(arr, (0, expected_len - len(arr)), constant_values=arr[-1] if len(arr) else 0.0)
            return arr
        except (json.JSONDecodeError, ValueError):
            pass

    # 3. bare numbers
    # Avoid extracting accidental digits from non-numeric placeholders.
    if re.search(r"[A-DF-Za-df-z]", text):
        return None
    nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
    if nums:
        arr = np.array([float(n) for n in nums], dtype=np.float32)
        if expected_len is not None:
            arr = arr[:expected_len]
        return arr

    return None


def parse_success_rate(texts: list[str], expected_len: int) -> float:
    """Return fraction of texts that parse successfully."""
    ok = sum(1 for t in texts if parse_output(t, expected_len) is not None)
    return ok / max(len(texts), 1)
