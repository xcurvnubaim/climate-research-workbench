"""
Script to fix DL4DS and Quantile Mapping model filtering in evaluation_visualization.ipynb

The issue: DL4DS and Quantile Mapping models have preupsample="no-preupsample" in the CSV data,
but cells 2, 3, and 4 of the notebook filter only for preupsample="preupsample", 
which completely excludes these models from the summary table, radar chart, and grouped bar chart.

This fix replaces the entire source of the affected cells with corrected versions.
"""

import json
import os

notebook_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "evaluation_visualization.ipynb"
)

with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

cells = nb["cells"]

def find_code_cell_containing(cells, search_text):
    """Find a code cell whose source contains search_text."""
    for i, cell in enumerate(cells):
        if cell["cell_type"] == "code":
            source = "".join(cell["source"])
            if search_text in source:
                return i
    return None


# ==============================================================================
# Fix Cell 2: Summary table
# ==============================================================================
cell2_idx = find_code_cell_containing(cells, "Inspect summary")
if cell2_idx is not None:
    cells[cell2_idx]["source"] = [
        '# --- Inspect summary: preupsample + 2 dataset (including DL4DS and QM) ---\n',
        '# DL4DS and Quantile Mapping have preupsample="no-preupsample", so include via OR\n',
        '_dl4ds_qm_models = {"DL4DS", "Quantile Mapping"}\n',
        'inspect_df = df[\n',
        '    (\n',
        '        (df["preupsample"] == "preupsample") |\n',
        '        (df["model"].isin(_dl4ds_qm_models))\n',
        '    ) &\n',
        '    (df["dataset"] == "2 dataset") &\n',
        '    (df["variable"].isin(["T2m (K)", "TP 24hr (mm)", "U10 (m/s)", "V10 (m/s)"]))\n',
        ']\n',
        'if inspect_df.empty:\n',
        '    raise ValueError("No rows for preupsample + 2 dataset in parsed_notebook_metrics.csv")\n',
        '\n',
        'summary = (\n',
        '    inspect_df.groupby(["variable", "model"], as_index=False)\n',
        '              .agg(\n',
        '                  rmse_mean=("rmse", "mean"), rmse_std=("rmse", "std"),\n',
        '                  mae_mean=("mae", "mean"),   mae_std=("mae", "std"),\n',
        '                  corr_mean=("corr", "mean"), corr_std=("corr", "std"),\n',
        '              )\n',
        ')\n',
        '\n',
        'def _fmt(mean, std, prec=4):\n',
        '    if np.isnan(std):\n',
        '        std = 0.0\n',
        '    return f"{mean:.{prec}f} \\u00b1 {std:.4g}"\n',
        '\n',
        'summary["RMSE"] = summary.apply(lambda r: _fmt(r.rmse_mean, r.rmse_std, 4), axis=1)\n',
        'summary["MAE"]  = summary.apply(lambda r: _fmt(r.mae_mean, r.mae_std, 4), axis=1)\n',
        'summary["Pearson Corr"] = summary.apply(lambda r: _fmt(r.corr_mean, r.corr_std, 4), axis=1)\n',
        '\n',
        'summary = summary[["variable", "model", "RMSE", "MAE", "Pearson Corr"]]\n',
        '\n',
        'var_order = ["T2m (K)", "TP 24hr (mm)", "U10 (m/s)", "V10 (m/s)"]\n',
        'model_order = ["ConvNeXt", "cGAN", "ResNet18", "UNet", "DL4DS", "Quantile Mapping"]\n',
        'summary["variable"] = pd.Categorical(summary["variable"], categories=var_order, ordered=True)\n',
        'summary["model"] = pd.Categorical(summary["model"], categories=model_order, ordered=True)\n',
        'summary = summary.sort_values(["variable", "model"]).reset_index(drop=True)\n',
        'summary'
    ]
    cells[cell2_idx]["outputs"] = []
    cells[cell2_idx]["execution_count"] = None
    print(f"Fixed Cell 2 (index {cell2_idx}): Summary table filter updated")
else:
    print("WARNING: Could not find Cell 2 (summary table)")


# ==============================================================================
# Fix Cell 3: Radar chart
# ==============================================================================
cell3_idx = find_code_cell_containing(cells, "Radar chart")
if cell3_idx is not None:
    cells[cell3_idx]["source"] = [
        '# --- Radar chart: Scenario 4 (preupsample, 2 dataset) ---\n',
        '# Note: use all rows that match preupsample + 2 dataset.\n',
        '# DL4DS and Quantile Mapping have preupsample="no-preupsample", so include via OR\n',
        '_dl4ds_qm_models_radar = {"DL4DS", "Quantile Mapping"}\n',
        '\n',
        'base_filter = (\n',
        '    (\n',
        '        (df["preupsample"] == "preupsample") |\n',
        '        (df["model"].isin(_dl4ds_qm_models_radar))\n',
        '    ) &\n',
        '    (df["dataset"] == "2 dataset") &\n',
        '    (df["variable"].isin(["T2m (K)", "TP 24hr (mm)", "U10 (m/s)", "V10 (m/s)"]))\n',
        ')\n',
        'radar_df = df[base_filter].copy()\n',
        '\n',
        'if radar_df.empty:\n',
        '    raise ValueError(\n',
        '        "No rows found for preupsample + 2 dataset. "\n',
        '        "Check parsed_notebook_metrics.csv entries."\n',
        '    )\n',
        '\n',
        'radar_df["model"] = radar_df["model"].replace({"Unet": "UNet", "GAN": "cGAN"})\n',
        '\n',
        '# Aggregate across runs if multiple entries exist\n',
        'agg = (\n',
        '    radar_df.groupby(["model", "variable"], as_index=False)\n',
        '            .mean(numeric_only=True)\n',
        ')\n',
        '\n',
        '# Build metrics table\n',
        'metrics = agg.pivot(index="model", columns="variable", values=["rmse", "corr"])\n',
        'model_order = ["UNet", "cGAN", "ResNet18", "ConvNeXt", "DL4DS", "Quantile Mapping"]\n',
        'metrics = metrics.reindex(model_order)\n',
        '\n',
        '# Normalize: lower RMSE is better -> invert; higher Corr is better\n',
        't2m_rmse = metrics[("rmse", "T2m (K)")].values\n',
        'tp_rmse = metrics[("rmse", "TP 24hr (mm)")].values\n',
        'u10_corr = metrics[("corr", "U10 (m/s)")].values\n',
        'v10_corr = metrics[("corr", "V10 (m/s)")].values\n',
        '\n',
        'def _inv_minmax(x):\n',
        '    x = np.asarray(x, dtype=float)\n',
        '    x_min, x_max = np.nanmin(x), np.nanmax(x)\n',
        '    if np.isclose(x_max, x_min):\n',
        '        return np.ones_like(x)\n',
        '    return (x_max - x) / (x_max - x_min)\n',
        '\n',
        'def _minmax(x):\n',
        '    x = np.asarray(x, dtype=float)\n',
        '    x_min, x_max = np.nanmin(x), np.nanmax(x)\n',
        '    if np.isclose(x_max, x_min):\n',
        '        return np.ones_like(x)\n',
        '    return (x - x_min) / (x_max - x_min)\n',
        '\n',
        'scores = np.column_stack([\n',
        '    _inv_minmax(t2m_rmse),\n',
        '    _inv_minmax(tp_rmse),\n',
        '    _minmax(u10_corr),\n',
        '    _minmax(v10_corr),\n',
        '])\n',
        '\n',
        'labels = ["RMSE T2m (inv)", "RMSE TP (inv)", "Corr U10", "Corr V10"]\n',
        'angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()\n',
        'angles += angles[:1]\n',
        '\n',
        'fig = plt.figure(figsize=(7.2, 6.2))\n',
        'ax = plt.subplot(111, polar=True)\n',
        'ax.set_theta_offset(np.pi / 2)\n',
        'ax.set_theta_direction(-1)\n',
        'ax.set_thetagrids(np.degrees(angles[:-1]), labels)\n',
        'ax.set_ylim(0, 1)\n',
        'ax.grid(alpha=0.3)\n',
        '\n',
        'colors = ["#3b82f6", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6", "#ec4899"]\n',
        'for i, model in enumerate(metrics.index):\n',
        '    if model not in metrics.index or np.any(np.isnan(scores[i])):\n',
        '        continue\n',
        '    vals = scores[i].tolist()\n',
        '    vals += vals[:1]\n',
        '    ax.plot(angles, vals, linewidth=2, color=colors[i], label=model)\n',
        '    ax.fill(angles, vals, color=colors[i], alpha=0.12)\n',
        '\n',
        'ax.set_title("Preupsample + 2 Dataset Model Performance (Radar)", pad=16)\n',
        'ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=3, frameon=False)\n',
        'plt.tight_layout()\n',
        'plt.show()'
    ]
    cells[cell3_idx]["outputs"] = []
    cells[cell3_idx]["execution_count"] = None
    print(f"Fixed Cell 3 (index {cell3_idx}): Radar chart filter and model_order updated")
else:
    print("WARNING: Could not find Cell 3 (radar chart)")


# ==============================================================================
# Fix Cell 4: 2x2 bar chart
# ==============================================================================
cell4_idx = find_code_cell_containing(cells, "Alternative plot: 2x2 grouped bars")
if cell4_idx is not None:
    cells[cell4_idx]["source"] = [
        '# --- Alternative plot: 2x2 grouped bars with error bars (MAE only) ---\n',
        '# DL4DS and Quantile Mapping have preupsample="no-preupsample", include via OR\n',
        '_dl4ds_qm_models_bar = {"DL4DS", "Quantile Mapping"}\n',
        'plot_df = df[\n',
        '    (\n',
        '        (df["preupsample"] == "preupsample") |\n',
        '        (df["model"].isin(_dl4ds_qm_models_bar))\n',
        '    ) &\n',
        '    (df["dataset"] == "2 dataset") &\n',
        '    (df["variable"].isin(["T2m (K)", "TP 24hr (mm)", "U10 (m/s)", "V10 (m/s)"]))\n',
        ']\n',
        'if plot_df.empty:\n',
        '    raise ValueError("No rows for preupsample + 2 dataset in parsed_notebook_metrics.csv")\n',
        '\n',
        'plot_df["model"] = plot_df["model"].replace({"Unet": "UNet", "GAN": "cGAN"})\n',
        'model_order = ["ConvNeXt", "cGAN", "ResNet18", "UNet", "DL4DS", "Quantile Mapping"]\n',
        'var_order = ["T2m (K)", "TP 24hr (mm)", "U10 (m/s)", "V10 (m/s)"]\n',
        '\n',
        'agg = (\n',
        '    plot_df.groupby(["variable", "model"], as_index=False)\n',
        '           .agg(\n',
        '               mae_mean=("mae", "mean"), mae_std=("mae", "std"),\n',
        '           )\n',
        ')\n',
        'agg["mae_std"] = agg["mae_std"].fillna(0.0)\n',
        '\n',
        'fig, axes = plt.subplots(2, 2, figsize=(11.8, 8.8))\n',
        'axes = axes.ravel()\n',
        '\n',
        'metrics_cfg = [\n',
        '    ("T2m (K)", "MAE", "mae_mean", "mae_std", "Lower is better"),\n',
        '    ("TP 24hr (mm)", "MAE", "mae_mean", "mae_std", "Lower is better"),\n',
        '    ("U10 (m/s)", "MAE", "mae_mean", "mae_std", "Lower is better"),\n',
        '    ("V10 (m/s)", "MAE", "mae_mean", "mae_std", "Lower is better"),\n',
        ']\n',
        '\n',
        'for ax, (var, label, mean_col, std_col, note) in zip(axes, metrics_cfg):\n',
        '    sub = agg[agg["variable"] == var].set_index("model").reindex(model_order)\n',
        '    x = np.arange(len(model_order))\n',
        '    mean_vals = sub[mean_col].values\n',
        '    std_vals = sub[std_col].values\n',
        '    ax.bar(\n',
        '        x, mean_vals, yerr=std_vals,\n',
        '        capsize=3, color=["#3b82f6", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6", "#ec4899"],\n',
        '    )\n',
        '    ax.set_xticks(x)\n',
        '    ax.set_xticklabels(model_order, rotation=25, ha="right")\n',
        '    ax.set_title(f"{var} | {label}", pad=10)\n',
        '    ax.grid(axis="y", linestyle=":", alpha=0.4)\n',
        '    ax.text(0.98, 0.98, note, transform=ax.transAxes, ha="right", va="top", fontsize=8)\n',
        '\n',
        '    # Zoom y-range to emphasize differences\n',
        '    y_min = np.nanmin(mean_vals - std_vals)\n',
        '    y_max = np.nanmax(mean_vals + std_vals)\n',
        '    pad = (y_max - y_min) * 0.15 if y_max > y_min else max(abs(y_max) * 0.1, 0.1)\n',
        '    ax.set_ylim(y_min - pad, y_max + pad)\n',
        '\n',
        'fig.suptitle("Preupsample + 2 Dataset: MAE Comparison", y=0.98)\n',
        'fig.subplots_adjust(hspace=0.45, wspace=0.28, top=0.9)\n',
        'plt.show()'
    ]
    cells[cell4_idx]["outputs"] = []
    cells[cell4_idx]["execution_count"] = None
    print(f"Fixed Cell 4 (index {cell4_idx}): Bar chart filter updated")
else:
    print("WARNING: Could not find Cell 4 (bar chart)")


# Write the modified notebook
with open(notebook_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("\nNotebook updated successfully!")
print("Please re-run the notebook to see the updated visualizations with DL4DS and Quantile Mapping models included.")
