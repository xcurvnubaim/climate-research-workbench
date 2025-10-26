"""
Core implementation layer for the EDA tool.

This module contains pure implementation utilities with no Streamlit
dependencies. The view layer (Streamlit UI) should import from here.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np
import xarray as xr
import pandas as pd


# =============================
# Configuration & Constants
# =============================


@dataclass
class AppConfig:
    title: str = "📊 Easy EDA for NetCDF / GRIB / Zarr"
    default_chunks: Dict[str, int] = field(default_factory=dict)
    allowed_backends: Tuple[str, ...] = ("auto", "netcdf", "grib", "zarr")
    show_dev_tips: bool = True


CONFIG = AppConfig()


# =============================
# Utilities
# =============================


def detect_backend(path_or_url: str) -> str:
    p = (path_or_url or "").lower()
    if p.endswith((".zarr", ".zarr/")):
        return "zarr"
    if p.endswith((".grib", ".grb", ".grb2")):
        return "grib"
    if p.endswith((".nc", ".nc4", ".cdf")):
        return "netcdf"
    if any(t in p for t in ["dodsc", "dap4", "opendap"]):
        return "netcdf"
    return "auto"


def parse_json_safe(txt: str, fallback: Any) -> Any:
    try:
        return json.loads(txt) if txt and txt.strip() else fallback
    except Exception:
        return fallback


def guess_lat_lon_names(ds: xr.Dataset) -> Tuple[Optional[str], Optional[str]]:
    cand_lat = ["lat", "latitude", "y"]
    cand_lon = ["lon", "longitude", "x"]
    lat = next((c for c in cand_lat if c in ds.coords), None)
    lon = next((c for c in cand_lon if c in ds.coords), None)
    return lat, lon


def pick_default_var(ds: xr.Dataset) -> Optional[str]:
    # Prefer first non-trivial data var
    for v, da in ds.data_vars.items():
        if not set(da.dims).issubset(set(ds.coords)):
            return v
    return next(iter(ds.data_vars.keys()), None)


# =============================
# Data Access
# =============================


def open_dataset(
    path: str,
    backend: str,
    chunks: Optional[Dict[str, int]] = None,
    storage_options: Optional[Dict[str, Any]] = None,
    cfgrib_filter: Optional[Dict[str, Any]] = None,
    consolidated: Optional[bool] = None,
) -> xr.Dataset:
    """Open a dataset from local/remote paths with the requested backend.

    Note: Caching should be handled by the caller (UI layer) if desired.
    """
    storage_options = storage_options or {}

    if backend == "zarr":
        ds = xr.open_zarr(path, storage_options=storage_options, consolidated=consolidated)
        return ds.chunk(chunks) if chunks else ds

    if backend == "grib":
        kwargs: Dict[str, Any] = {"engine": "cfgrib"}
        if cfgrib_filter:
            kwargs["backend_kwargs"] = {"filter_by_keys": cfgrib_filter}
        return xr.open_dataset(path, chunks=chunks, storage_options=storage_options, **kwargs)

    if backend in ("netcdf", "auto"):
        try:
            return xr.open_dataset(path, chunks=chunks, engine=None)
        except Exception:
            for eng in ("netcdf4", "h5netcdf", "scipy"):
                try:
                    return xr.open_dataset(path, chunks=chunks)
                except Exception:
                    continue
            raise

    raise ValueError(f"Unsupported backend: {backend}")


# =============================
# Transforms & Helpers
# =============================


def slice_with_selection(da: xr.DataArray, selection: Dict[str, Any]) -> xr.DataArray:
    """Apply a mixed index/label selection to a DataArray.

    selection: mapping of dim -> int (positional) or value (label)
    """
    sliced = da
    # positional
    isel_map = {k: v for k, v in selection.items() if isinstance(v, int)}
    if isel_map:
        sliced = sliced.isel(isel_map)
    # label-based
    for k, v in selection.items():
        if not isinstance(v, int):
            sliced = sliced.sel({k: v})
    return sliced


def compute_quick_stats(sliced: xr.DataArray) -> Dict[str, float]:
    # Get raw ndarray (this will realize dask-backed arrays too)
    vals = np.asarray(sliced.values)

    # If not numeric, coerce via pandas (handles strings, objects, None, etc.)
    if not np.issubdtype(vals.dtype, np.number):
        # Option 1: wrap in Series so we always get a Series back
        flat = pd.Series(vals.ravel(order="K"))
        flat = pd.to_numeric(flat, errors="coerce")
        vals = flat.to_numpy(dtype=np.float64).reshape(vals.shape)
    else:
        # Ensure float64 (NumPy 2.0-safe: don't pass copy=)
        vals = vals.astype(np.float64, copy=False)

    finite = np.isfinite(vals)
    if not finite.any():
        return {"count": 0, "min": np.nan, "max": np.nan, "mean": np.nan, "std": np.nan}

    return {
        "count": float(finite.sum()),
        "min": float(np.nanmin(vals)),
        "max": float(np.nanmax(vals)),
        "mean": float(np.nanmean(vals)),
        "std": float(np.nanstd(vals)),
    }


# =============================
# Export
# =============================


def export_to_netcdf(sliced: xr.DataArray, filename: str) -> None:
    ds_out = sliced.to_dataset(name=str(sliced.name))
    ds_out.load().to_netcdf(filename)


def export_to_zarr(sliced: xr.DataArray, store_path: str) -> None:
    ds_out = sliced.to_dataset(name=str(sliced.name))
    ds_out.load().to_zarr(store_path, mode="w")
