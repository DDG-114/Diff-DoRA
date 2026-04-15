"""
src/utils/node_context.py
-------------------------
Helpers for resolving dataset node ids and extracting per-node static context.
"""
from __future__ import annotations

from typing import Any


STATIC_FIELD_ALIASES = {
    "zone_type": ("zone_type", "type", "area_type", "category"),
    "capacity": ("capacity", "station_capacity", "pile_count"),
    "area": ("area", "station_area"),
    "road_length": ("road_length", "road_len", "road"),
    "poi_summary": ("poi_summary", "poi", "poi_distribution"),
}


def resolve_node_id(
    node_idx: int,
    node_ids: list[str] | tuple[str, ...] | None = None,
    node_meta=None,
) -> str | int:
    """Resolve a positional node index into the dataset node id when possible."""
    if node_ids and 0 <= int(node_idx) < len(node_ids):
        return node_ids[int(node_idx)]
    if node_meta is not None and not getattr(node_meta, "empty", True) and 0 <= int(node_idx) < len(node_meta):
        return node_meta.index[int(node_idx)]
    return int(node_idx)


def _lookup_meta_row(node_meta, node_idx: int, node_id: str | int):
    if node_meta is None or getattr(node_meta, "empty", True):
        return None

    candidates = [node_id, str(node_id)]
    try:
        candidates.append(int(node_id))
    except (TypeError, ValueError):
        pass

    for candidate in candidates:
        if candidate in node_meta.index:
            row = node_meta.loc[candidate]
            if hasattr(row, "iloc") and getattr(row, "ndim", 1) > 1:
                return row.iloc[0]
            return row

    if 0 <= int(node_idx) < len(node_meta):
        return node_meta.iloc[int(node_idx)]
    return None


def _pick_field(data: dict[str, Any], aliases: tuple[str, ...]) -> Any:
    lowered = {str(k).lower(): k for k in data.keys()}
    for alias in aliases:
        key = lowered.get(alias.lower())
        if key is not None:
            return data[key]
    return None


def extract_node_static_context(
    node_idx: int,
    *,
    node_ids: list[str] | tuple[str, ...] | None = None,
    node_meta=None,
) -> dict[str, Any]:
    """
    Return lightweight static context for the node, only when the fields exist.

    The context is intentionally sparse so prompt builders can include it without
    assuming every dataset has every physical attribute.
    """
    node_id = resolve_node_id(node_idx, node_ids=node_ids, node_meta=node_meta)
    context: dict[str, Any] = {"node_id": str(node_id)}

    row = _lookup_meta_row(node_meta, node_idx=node_idx, node_id=node_id)
    if row is None or not hasattr(row, "to_dict"):
        return context

    raw = row.to_dict()
    for logical_name, aliases in STATIC_FIELD_ALIASES.items():
        value = _pick_field(raw, aliases)
        if value is None:
            continue
        if hasattr(value, "item"):
            value = value.item()
        context[logical_name] = value
    return context


def normalise_domain_label(raw_label: Any) -> str | None:
    """Map dataset/domain strings to a stable prompt label."""
    if raw_label is None:
        return None
    label = str(raw_label).strip()
    if not label:
        return None
    lowered = label.lower()
    if any(token in lowered for token in ("cbd", "commercial", "business")):
        return "CBD"
    if any(token in lowered for token in ("res", "residential", "living")):
        return "Residential"
    return label.title()
