"""
Streamlit view layer for the EDA tool.

This module defines all UI components and relies on functions from core.py
for implementation details (I/O, transforms, stats, export).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import xarray as xr

from core import (
    CONFIG,
    compute_quick_stats,
    detect_backend,
    export_to_netcdf,
    export_to_zarr,
    guess_lat_lon_names,
    parse_json_safe,
    pick_default_var,
    slice_with_selection,
)


# Optional Cartopy for maps
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from cartopy.util import add_cyclic_point
    HAS_CARTOPY = True
except Exception:
    HAS_CARTOPY = False


# =============================
# Sidebar Inputs
# =============================


def sidebar_inputs() -> Dict[str, Any]:
    st.sidebar.title("Data Source")
    # mode = st.sidebar.radio("Open dataset from:", ["Local file", "Remote URL"], index=0)

    path_input: Optional[str] = None
    # uploaded_file = None

    # if mode == "Local file":
    #     uploaded_file = st.sidebar.file_uploader(
    #         "Upload .nc/.grib/.grb2 (single file). For Zarr, use URL mode.",
    #         type=["nc", "grib", "grb", "grb2"],
    #     )
    #     if uploaded_file is not None:
    #         tmp_dir = Path(".st_tmp")
    #         tmp_dir.mkdir(exist_ok=True)
    #         tmp_path = tmp_dir / uploaded_file.name
    #         tmp_path.write_bytes(uploaded_file.getbuffer())
    #         path_input = str(tmp_path.resolve())
    # else:
    path_input = st.sidebar.text_input("URL or fsspec path (https://, s3://, gs://, local path)")

    backend_guess = detect_backend(path_input or "")
    backend = st.sidebar.selectbox("Backend", CONFIG.allowed_backends, index=CONFIG.allowed_backends.index(backend_guess))

    use_dask = st.sidebar.checkbox("Use Dask & set chunks", value=True)
    chunks_text = st.sidebar.text_input("Chunks (JSON)", value=json.dumps(CONFIG.default_chunks))
    chunks = parse_json_safe(chunks_text, None) if use_dask else None

    # Storage options
    storage_options: Dict[str, Any] = {}
    with st.sidebar.expander("Storage options (fsspec)"):
        key = st.text_input("Access key / username", value="")
        secret = st.text_input("Secret / password", value="", type="password")
        token = st.text_input("Session token", value="", type="password")
        if key:
            storage_options["key"] = key
        if secret:
            storage_options["secret"] = secret
        if token:
            storage_options["token"] = token

    cfgrib_filter = None
    if backend == "grib":
        with st.sidebar.expander("GRIB filter (cfgrib)"):
            st.caption('e.g. {"typeOfLevel": "isobaricInhPa", "shortName": "t"}')
            cfgrib_filter = parse_json_safe(st.text_area("filter_by_keys (JSON)", value="{}"), None)

    consolidated = None
    if backend == "zarr":
        z = st.sidebar.selectbox("Zarr consolidated?", ["auto", "True", "False"], index=0)
        consolidated = None if z == "auto" else (z == "True")

    open_trigger = st.sidebar.button("Open Dataset", type="primary", use_container_width=True)

    return {
        "path": path_input,
        "backend": backend,
        "chunks": chunks,
        "storage_options": storage_options,
        "cfgrib_filter": cfgrib_filter,
        "consolidated": consolidated,
        "open_trigger": open_trigger,
    }


# =============================
# Content Sections
# =============================


def section_dataset_summary(ds: xr.Dataset) -> None:
    with st.expander("📦 Dataset summary", expanded=True):
        left, right = st.columns([2, 1])
        with left:
                # Brief headline numbers
                st.markdown(f"**Variables:** {len(ds.data_vars)} · **Coordinates:** {len(ds.coords)} · **Dimensions:** {dict(ds.dims)}")

                # Time coverage (if any datetime-like coordinate exists)
                time_name: Optional[str] = None
                # Prefer a coord literally named 'time' if present
                if "time" in ds.coords:
                    time_name = "time"
                else:
                    # Fallback: first coord with datetime dtype
                    for c in ds.coords:
                        try:
                            if np.issubdtype(ds[c].dtype, np.datetime64):
                                time_name = c
                                break
                        except Exception:
                            pass

                def _fmt_time(v: Any) -> str:
                    try:
                        return np.datetime_as_string(np.datetime64(v), unit="D")
                    except Exception:
                        return str(v)

                if time_name is not None:
                    try:
                        tcoord = ds[time_name]
                        if tcoord.size > 0:
                            tmin = tcoord.min().item() if hasattr(tcoord.min(), "item") else tcoord.min()
                            tmax = tcoord.max().item() if hasattr(tcoord.max(), "item") else tcoord.max()
                            steps = int(tcoord.size)
                            st.markdown(f"**Time:** {_fmt_time(tmin)} → {_fmt_time(tmax)} · {steps} steps")
                        else:
                            st.markdown("**Time:** (empty)")
                    except Exception:
                        st.markdown("**Time:** (unavailable)")
                else:
                    st.markdown("**Time:** n/a")

                # Jupyter-like detailed dataset view (collapsed)
                with st.expander("Full dataset details (Jupyter-like)", expanded=False):
                    def _fmt_bytes(n: Optional[int]) -> str:
                        try:
                            n = int(n or 0)
                        except Exception:
                            return "unknown"
                        for unit in ["B", "KB", "MB", "GB", "TB"]:
                            if n < 1024:
                                return f"{n:.0f} {unit}"
                            n /= 1024
                        return f"{n:.0f} PB"

                    def _fmt_any(v: Any) -> str:
                        try:
                            if np.issubdtype(np.array(v).dtype, np.datetime64):
                                return np.datetime_as_string(np.datetime64(v), unit="s")
                        except Exception:
                            pass
                        return str(v)

                    def _fmt_datetime(v: Any) -> str:
                        """Format various datetime-like scalars to ISO 8601 (seconds precision)."""
                        try:
                            # Prefer pandas for robust handling of ints/ns
                            import pandas as pd  # type: ignore
                            return pd.to_datetime(v).isoformat()
                        except Exception:
                            try:
                                return np.datetime_as_string(np.datetime64(v), unit="s")
                            except Exception:
                                return str(v)

                    def _fmt_timedelta(v: Any) -> str:
                        """Format various timedelta-like scalars to a human string (e.g., '9 days 00:00:00')."""
                        try:
                            import pandas as pd  # type: ignore
                            return str(pd.to_timedelta(v))
                        except Exception:
                            # Fallback: best-effort seconds breakdown assuming ns if numeric
                            try:
                                # Convert to total seconds
                                try:
                                    td_sec = int(np.timedelta64(v, 'ns') / np.timedelta64(1, 's'))
                                except Exception:
                                    td_sec = int(v) // 1_000_000_000
                                days, rem = divmod(td_sec, 86400)
                                hours, rem = divmod(rem, 3600)
                                minutes, seconds = divmod(rem, 60)
                                if days:
                                    if hours or minutes or seconds:
                                        return f"{days} days {hours:02}:{minutes:02}:{seconds:02}"
                                    return f"{days} days"
                                return f"{hours:02}:{minutes:02}:{seconds:02}"
                            except Exception:
                                return str(v)

                    def _summarize_coord(name: str) -> str:
                        try:
                            c = ds[name]
                            size = int(c.size)
                            dtype = str(c.dtype)
                            dtype_kind = getattr(c.dtype, 'kind', '')
                            if size == 0:
                                rng = "(empty)"
                            else:
                                try:
                                    vmin = getattr(c.min(), 'values', c.min())
                                    vmax = getattr(c.max(), 'values', c.max())
                                    if dtype_kind == 'M':  # datetime64
                                        cmin = _fmt_datetime(vmin)
                                        cmax = _fmt_datetime(vmax)
                                    elif dtype_kind == 'm':  # timedelta64
                                        cmin = _fmt_timedelta(vmin)
                                        cmax = _fmt_timedelta(vmax)
                                    else:
                                        cmin = _fmt_any(vmin)
                                        cmax = _fmt_any(vmax)
                                    rng = f"{cmin} → {cmax}"
                                except Exception:
                                    try:
                                        v0 = c.values[0]
                                        v1 = c.values[-1]
                                        if dtype_kind == 'M':
                                            c0 = _fmt_datetime(v0)
                                            c1 = _fmt_datetime(v1)
                                        elif dtype_kind == 'm':
                                            c0 = _fmt_timedelta(v0)
                                            c1 = _fmt_timedelta(v1)
                                        else:
                                            c0 = _fmt_any(v0)
                                            c1 = _fmt_any(v1)
                                        rng = f"{c0} → {c1}"
                                    except Exception:
                                        rng = "(range unavailable)"
                            units = c.attrs.get("units", None)
                            units_str = f", units={units}" if units is not None else ""
                            return f"  - {name}: size={size}, dtype={dtype}{units_str}, range=[{rng}]"
                        except Exception as e:
                            return f"  - {name}: (unavailable: {e})"

                    def _summarize_var(name: str) -> str:
                        try:
                            v = ds[name]
                            dims = tuple(str(d) for d in v.dims)
                            shape = tuple(int(s) for s in v.shape)
                            dtype = str(v.dtype)
                            chunks = None
                            try:
                                if hasattr(v.data, "chunks") and v.data.chunks is not None:
                                    chunks = tuple(int(c[0]) for c in v.data.chunks)
                            except Exception:
                                pass
                            units = v.attrs.get("units", None)
                            units_str = f", units={units}" if units is not None else ""
                            ch_str = f", chunks={chunks}" if chunks is not None else ""
                            return f"  - {name}: dims={dims}, shape={shape}, dtype={dtype}{ch_str}{units_str}"
                        except Exception as e:
                            return f"  - {name}: (unavailable: {e})"

                    lines = []
                    try:
                        try:
                            nbytes = getattr(ds, "nbytes", None)
                            mem = _fmt_bytes(nbytes) if nbytes is not None else "unknown"
                        except Exception:
                            mem = "unknown"
                        lines.append(f"Dimensions: {dict(ds.sizes)}")
                        lines.append(f"Estimated size: {mem}")

                        lines.append("Coordinates:")
                        for cname in list(ds.coords):
                            lines.append(_summarize_coord(str(cname)))

                        lines.append("Data variables:")
                        for vname in list(ds.data_vars):
                            lines.append(_summarize_var(str(vname)))

                        lines.append("Attributes:")
                        if ds.attrs:
                            for k, v in ds.attrs.items():
                                lines.append(f"  - {k}: {v}")
                        else:
                            lines.append("  (none)")

                        st.code("\n".join(lines))
                    except Exception:
                        st.text(ds)
        with right:
            st.markdown("**Global attributes**")
            if ds.attrs:
                for k, v in ds.attrs.items():
                    st.write(f"- **{k}**: {v}")
            else:
                st.write("(none)")


def section_variable_picker(ds: xr.Dataset) -> xr.DataArray:
    var_names = list(ds.data_vars)
    if not var_names:
        st.error("No data variables found in dataset.")
        st.stop()
    default = pick_default_var(ds)
    var = st.selectbox("Choose a variable", var_names, index=var_names.index(default) if default in var_names else 0)

    da = ds[var]
    with st.expander("📐 Dims & Coords (selected variable)", expanded=False):
        st.json({"dims": da.dims, "shape": list(da.shape)})
        coords_details = {c: {"size": int(da[c].size), "attrs": {k: str(v) for k, v in da[c].attrs.items()}} for c in da.coords}
        st.json(coords_details)
    return da


def section_slice_controls(da: xr.DataArray) -> xr.DataArray:
    nonspatial_dims = [d for d in da.dims if d not in ["lat", "latitude", "y", "lon", "longitude", "x"]]
    sel_isel: Dict[str, Any] = {}

    with st.expander("🎚️ Slice controls (non-spatial dims)", expanded=True):
        for d in nonspatial_dims:
            coord = da[d]
            if np.issubdtype(coord.dtype, np.number) or np.issubdtype(coord.dtype, np.datetime64):
                idx = st.slider(f"Index for '{d}' (0..{coord.size-1})", 0, int(coord.size - 1), 0, 1)
                sel_isel[d] = idx
            else:
                options = coord.values.astype(str).tolist()
                choice = st.selectbox(f"Value for '{d}'", options, index=0)
                sel_isel[d] = choice

    sliced = slice_with_selection(da, sel_isel)
    return sliced


def section_quick_stats(sliced: xr.DataArray) -> None:
    with st.expander("📈 Quick stats for current slice", expanded=True):
        stats = compute_quick_stats(sliced)
        if stats["count"] == 0:
            st.info("No finite values in slice.")
            return
        st.json({
            "count": int(stats["count"]),
            "min": stats["min"],
            "max": stats["max"],
            "mean": stats["mean"],
            "std": stats["std"],
        })


def section_plot(sliced: xr.DataArray, lat_name: Optional[str], lon_name: Optional[str]) -> None:
    is_2d = lat_name in sliced.dims and lon_name in sliced.dims if (lat_name and lon_name) else False

    if is_2d:
        st.subheader("🗺️ 2D Map Preview")

        # 1) Squeeze singleton dims to avoid (1, M, N)
        da = sliced.squeeze(drop=True)

        # 2) Ensure we have lat/lon coords alongside the data
        lat = da[lat_name]
        lon = da[lon_name]

        # Optional: add a cyclic point to avoid the dateline seam if lon is 0..360 or -180..180
        def maybe_add_cyclic(da2d, lon_coord):
            try:
                # Only for 1D lon
                if lon_coord.ndim == 1 and da2d.ndim == 2:
                    c_vals, c_lon = add_cyclic_point(da2d.values, coord=lon_coord.values)
                    return c_vals, c_lon
            except Exception:
                pass
            return da2d.values, lon_coord.values

        fig = plt.figure(figsize=(6, 4))

        if HAS_CARTOPY and lat_name and lon_name:
            ax = plt.axes(projection=ccrs.PlateCarree())
            ax.coastlines(linewidth=0.5)
            ax.add_feature(cfeature.BORDERS, linewidth=0.3)

            # 3) Handle 1D vs 2D (curvilinear) grids
            if lat.ndim == 1 and lon.ndim == 1:
                C = da.values
                # add cyclic (optional; helps if your grid is global)
                C, lon_vals = maybe_add_cyclic(da, lon)
                lat_vals = lat.values  # unchanged

                im = ax.pcolormesh(
                    lon_vals, lat_vals, C,
                    transform=ccrs.PlateCarree(),
                    shading="auto"  # <-- key to match full-sized C with center coords
                )
            else:
                # Curvilinear case: lat/lon must be same shape as C
                C = da.values
                LON = lon.values
                LAT = lat.values
                # If a leading singleton still sneaked in, squeeze everything
                C = np.squeeze(C)
                LON = np.squeeze(LON)
                LAT = np.squeeze(LAT)
                im = ax.pcolormesh(
                    LON, LAT, C,
                    transform=ccrs.PlateCarree(),
                    shading="auto"
                )

            plt.colorbar(im, ax=ax, orientation='vertical', shrink=0.8, pad=0.02, label=str(da.name))
            ax.set_title(str(da.name))
            ax.set_global()
        else:
            # No cartopy: just plot the 2D array with default pixel grid
            ax = plt.gca()
            im = ax.pcolormesh(da.squeeze().values, shading='auto')
            plt.colorbar(im, ax=ax, orientation='vertical', shrink=0.8, pad=0.02, label=str(da.name))
            ax.set_title(str(da.name))

        st.pyplot(fig, clear_figure=True, use_container_width=True)

    else:
        st.subheader("📉 1D Preview (e.g., time series)")
        try:
            fig = plt.figure(figsize=(6, 3))
            ax = plt.gca()
            sliced.squeeze(drop=True).plot(ax=ax)
            ax.set_title(str(sliced.name))
            st.pyplot(fig, clear_figure=True, use_container_width=True)
        except Exception as e:
            st.info(f"Plot not available: {e}")

def section_point_timeseries(da: xr.DataArray, ds: xr.Dataset, lat_name: Optional[str], lon_name: Optional[str]) -> None:
    if not (lat_name and lon_name and lat_name in ds.coords and lon_name in ds.coords):
        return
    st.subheader("📍 Point time series (nearest grid)")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        lat_q = st.number_input("Latitude", value=float(-7.281) if ds[lat_name].size > 0 else 0.0, format="%f")
    with c2:
        lon_q = st.number_input("Longitude", value=float(112.797) if ds[lon_name].size > 0 else 0.0, format="%f")
    with c3:
        ts_agg = st.selectbox("Aggregation", ["None", "Daily", "Weekly", "Monthly"], index=0)
    with c4:
        ts_btn = st.button("Extract time series")

    if ts_btn:
        try:
            ts = da.sel({lat_name: lat_q, lon_name: lon_q}, method='nearest')
            ts = ts.squeeze()
            match(ts_agg):
                case "Daily":
                    ts = ts.resample(time="1D").mean()
                case "Weekly":
                    ts = ts.resample(time="1W").mean()
                case "Monthly":
                    ts = ts.resample(time="1M").mean()  
            if 'time' in ts.dims:
                fig = plt.figure(figsize=(7, 3))
                ax = plt.gca()
                ts.plot(ax=ax)
                ax.set_title(f"{da.name} @ ({lat_q:.3f}, {lon_q:.3f})")
                st.pyplot(fig, clear_figure=True, use_container_width=True)
            else:
                st.write(ts.to_dataframe(name=str(da.name)).head())
        except Exception as e:
            st.warning(f"Failed to extract time series: {e}")


def section_histogram(sliced: xr.DataArray) -> None:
    with st.expander("📦 Histogram of current slice", expanded=False):
        try:
            data = np.ravel(np.asarray(sliced.values))
            data = data[np.isfinite(data)]
            if data.size == 0:
                st.info("No finite values to histogram.")
                return
            bins = st.slider("Bins", 10, 200, 50)
            fig = plt.figure(figsize=(5, 3))
            plt.hist(data, bins=bins)
            plt.title(f"Histogram: {sliced.name}")
            st.pyplot(fig, clear_figure=True, use_container_width=True)
        except Exception as e:
            st.info(f"Histogram not available: {e}")


def section_export(sliced: xr.DataArray) -> None:
    st.subheader("💾 Export current variable slice")
    col_a, col_b = st.columns(2)
    with col_a:
        fname_nc = st.text_input("NetCDF file name", value=f"subset_{sliced.name or 'var'}.nc")
        if st.button("Save NetCDF"):
            try:
                export_to_netcdf(sliced, fname_nc)
                with open(fname_nc, 'rb') as f:
                    st.download_button("Download NetCDF", f, file_name=fname_nc)
            except Exception as e:
                st.error(f"Failed to save NetCDF: {e}")
    with col_b:
        fname_zarr = st.text_input("Zarr directory", value=f"subset_{sliced.name or 'var'}.zarr")
        if st.button("Save Zarr"):
            try:
                export_to_zarr(sliced, fname_zarr)
                st.success(f"Saved Zarr directory: {fname_zarr}")
            except Exception as e:
                st.error(f"Failed to save Zarr: {e}")
