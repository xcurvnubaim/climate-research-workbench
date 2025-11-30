"""
Core implementation layer for the EDA tool.

This module contains pure implementation utilities with no Streamlit
dependencies. The view layer (Streamlit UI) should import from here.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import fsspec
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


def _is_remote_path(path: str) -> bool:
    """Check if the path is a remote URL (S3, HTTP, etc.)."""
    path_lower = path.lower()
    return path_lower.startswith(("s3://", "http://", "https://", "gs://", "gcs://"))


def _open_remote_netcdf(
    path: str,
    chunks: Optional[Dict[str, int]] = None,
    storage_options: Optional[Dict[str, Any]] = None,
) -> xr.Dataset:
    """Open a remote NetCDF file using fsspec and h5netcdf.
    
    This handles S3, HTTP, and other remote URLs that netCDF4 cannot open directly.
    The file object is opened using fsspec.open() which properly manages the
    lifecycle of the file object when the dataset is closed.
    """
    storage_options = storage_options or {}
    
    # Use fsspec.open which returns a context-managed file object.
    # xarray keeps a reference to the file object and closes it when ds.close() is called.
    file_obj = fsspec.open(path, mode="rb", **storage_options).open()
    
    # Use h5netcdf engine which works with file-like objects
    return xr.open_dataset(file_obj, engine="h5netcdf", chunks=chunks)


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
        return xr.open_dataset(path, chunks=chunks, **kwargs)

    if backend in ("netcdf", "auto"):
        # Handle remote URLs (S3, HTTP, etc.) using fsspec + h5netcdf
        if _is_remote_path(path):
            try:
                return _open_remote_netcdf(path, chunks=chunks, storage_options=storage_options)
            except Exception as remote_err:
                # Fall back to direct xr.open_dataset if fsspec approach fails
                try:
                    return xr.open_dataset(path, chunks=chunks, engine=None)
                except Exception:
                    # Re-raise the original remote error if fallback also fails
                    raise remote_err
        
        # Local file handling
        try:
            return xr.open_dataset(path, chunks=chunks, engine=None)
        except Exception:
            for eng in ("netcdf4", "h5netcdf", "scipy"):
                try:
                    return xr.open_dataset(path, chunks=chunks, engine=eng)
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
