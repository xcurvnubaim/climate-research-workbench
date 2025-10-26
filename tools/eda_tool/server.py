# Streamlit EDA for NetCDF / GRIB / Zarr — Layered Version
# ========================================================
"""
Entry point for the EDA tool. This file wires the view layer (Streamlit UI)
with the implementation layer (core functions).

Modules:
- core.py  : implementation (no Streamlit imports)
- views.py : Streamlit UI components calling into core

Run: streamlit run server.py
"""

from __future__ import annotations

from typing import Optional
import streamlit as st
import xarray as xr

from core import CONFIG, detect_backend, open_dataset as open_dataset_impl
import views as ui

st.set_page_config(page_title=CONFIG.title, layout="wide")
@st.cache_resource(show_spinner=True)
def open_dataset_cached(
    path: str,
    backend: str,
    chunks: Optional[dict] = None,
    storage_options: Optional[dict] = None,
    cfgrib_filter: Optional[dict] = None,
    consolidated: Optional[bool] = None,
) -> xr.Dataset:
    """Streamlit-cached wrapper around core.open_dataset."""
    return open_dataset_impl(
        path=path,
        backend=backend,
        chunks=chunks,
        storage_options=storage_options,
        cfgrib_filter=cfgrib_filter,
        consolidated=consolidated,
    )

# =============================
# Main App Flow
# =============================

def main() -> None:
    st.title(CONFIG.title)

    inputs = ui.sidebar_inputs()
    path = inputs["path"]

    if not path:
        st.info("Provide a local file or a remote URL/path to begin.")
        return

    if inputs["open_trigger"] or (path and not st.session_state.get("_opened_once")):
        try:
            ds = open_dataset_cached(
                path=path,
                backend=inputs["backend"] if inputs["backend"] != "auto" else detect_backend(path),
                chunks=inputs["chunks"],
                storage_options=inputs["storage_options"],
                cfgrib_filter=inputs["cfgrib_filter"],
                consolidated=inputs["consolidated"],
            )
            st.session_state["ds"] = ds
            st.session_state["_opened_once"] = True
        except Exception as e:
            st.error(f"Failed to open dataset: {e}")
            return

    ds: Optional[xr.Dataset] = st.session_state.get("ds")
    if ds is None:
        st.warning("Dataset not opened yet.")
        return

    # Sections
    ui.section_dataset_summary(ds)

    da = ui.section_variable_picker(ds)
    from core import guess_lat_lon_names  # local import to keep namespace minimal
    lat_name, lon_name = guess_lat_lon_names(ds)

    sliced = ui.section_slice_controls(da)
    ui.section_quick_stats(sliced)
    ui.section_plot(sliced, lat_name, lon_name)
    ui.section_point_timeseries(da, ds, lat_name, lon_name)
    ui.section_histogram(sliced)
    ui.section_export(sliced)

    if CONFIG.show_dev_tips:
        st.caption("Tip: For large data, set Dask chunks and use cfgrib filters to limit memory.")


if __name__ == "__main__":
    main()
