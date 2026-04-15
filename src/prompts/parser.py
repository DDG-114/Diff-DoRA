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
_PLACEHOLDER_EQ_RE = re.compile(r"\[[^\]]*\bv\d+\b[^\]]*\]\s*=\s*", re.IGNORECASE)


def _coerce_expected_len(arr: np.ndarray, expected_len: int | None) -> np.ndarray:
    if expected_len is None or arr.ndim != 1 or len(arr) == expected_len:
        return arr
    if len(arr) > expected_len:
        return arr[:expected_len]
    if len(arr) == 0:
        return np.zeros(expected_len, dtype=np.float32)
    return np.pad(arr, (0, expected_len - len(arr)), constant_values=arr[-1])


def _iter_json_candidates(text: str):
    for regex in (_NESTED_RE, _LIST_RE):
        for match in regex.finditer(text):
            candidate = match.group().strip()
            if re.search(r"\bv\d+\b", candidate, flags=re.IGNORECASE):
                continue
            yield candidate


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

    # Common paper/prompt format: "[v1, ..., vH] = [0.1, 0.2, ...]"
    text = _PLACEHOLDER_EQ_RE.sub("", text)

    # Guard against template placeholders, e.g. "[v1, v2, v3]".
    # These should be treated as parse failure rather than [1, 2, 3].
    if re.search(r"\bv\d+\b", text) and not any(_iter_json_candidates(text)):
        return None

    # Try JSON parse of first [ ... ] block
    # Use the last valid numeric list so we can tolerate explanatory examples
    # earlier in the generation.
    last_arr = None
    for candidate in _iter_json_candidates(text):
        try:
            last_arr = np.array(json.loads(candidate), dtype=np.float32)
        except (json.JSONDecodeError, ValueError):
            pass
    if last_arr is not None:
        return _coerce_expected_len(last_arr, expected_len)

    # 3. bare numbers
    # Avoid extracting accidental digits from non-numeric placeholders.
    if re.search(r"[A-DF-Za-df-z]", text):
        return None
    nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
    if nums:
        arr = np.array([float(n) for n in nums], dtype=np.float32)
        return _coerce_expected_len(arr, expected_len)

    return None


def parse_success_rate(texts: list[str], expected_len: int) -> float:
    """Return fraction of texts that parse successfully."""
    ok = sum(1 for t in texts if parse_output(t, expected_len) is not None)
    return ok / max(len(texts), 1)
