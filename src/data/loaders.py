"""Dataset registry used by training, cache-building, and evaluation entry points."""
from __future__ import annotations

from collections.abc import Callable

from src.data.load_st_evcdp import load_st_evcdp
from src.data.load_urbanev import load_urbanev
from src.data.load_gs_market import load_gs_market, load_gs_market_2025
from src.data.load_gs_price import load_gs_price, load_gs_price_2025
from src.data.load_renewable_generation import load_renewable_solar, load_renewable_wind
from src.data.load_wotai_evcdp import load_wotai_evcdp

DATASET_LOADERS: dict[str, Callable[[], dict]] = {
    "st_evcdp": load_st_evcdp,
    "urbanev": load_urbanev,
    "wotai_evcdp": load_wotai_evcdp,
    "renewable_solar": load_renewable_solar,
    "renewable_wind": load_renewable_wind,
    "gs_market": load_gs_market,
    "gs_market_2025": load_gs_market_2025,
    "gs_price": load_gs_price,
    "gs_price_2025": load_gs_price_2025,
}

DATASET_CHOICES = tuple(DATASET_LOADERS.keys())


def load_dataset(dataset: str) -> dict:
    try:
        return DATASET_LOADERS[dataset]()
    except KeyError as exc:
        raise ValueError(
            f"Unknown dataset={dataset!r}; expected one of {sorted(DATASET_LOADERS)}"
        ) from exc
