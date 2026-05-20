#!/usr/bin/env python
"""
Hyperparameter Tuning Script for UNet Preupsample Notebooks
============================================================

This script supports both:
  - SC1: Forecast Correction (sc1_unet_preupsample.ipynb)
  - SC2: Perfect Prognosis / MAE (sc2_unet_mae_preupsample.ipynb)

It performs a grid search over selected hyperparameters by modifying
a *template* notebook with different configs, executing each variant
via nbconvert, and collecting results into a summary CSV.

Usage:
    python hp_tuning_unet_preupsample.py --scenario sc1
    python hp_tuning_unet_preupsample.py --scenario sc2

The script will:
  1. Define a hyperparameter grid (learning rate, batch size,
     base_ch, weight_decay, loss_mode, patience, epochs).
  2. For each combination, patch the notebook source cells,
     write a temporary notebook, and execute it.
  3. Parse the final metrics from the executed notebook.
  4. Collect all results into  hp_tuning_results_<scenario>.csv
     in the same directory.

Requirements (pip):
    nbformat, nbconvert, papermill   (or just nbformat + subprocess)
"""

import argparse
import copy
import itertools
import json
import os
import re
import subprocess
import sys
import time
import traceback
from pathlib import Path

try:
    import nbformat
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "nbformat"])
    import nbformat


# ──────────────────────────────────────────────────────────────────────
#  Hyperparameter grid
# ──────────────────────────────────────────────────────────────────────
HP_GRID = {
    "learning_rate": [1e-3, 5e-4, 1e-4],
    "batch_size":    [32, 64],
    "base_ch":       [32, 64],
    "weight_decay":  [1e-4, 1e-3],
    "loss_mode":     ["mae", "dssim+mae"],
    "patience":      [7, 10],
    "epochs":        [100],          # keep constant, early-stop decides
}

# You can reduce the grid for a quick test:
# HP_GRID = {
#     "learning_rate": [5e-4],
#     "batch_size":    [64],
#     "base_ch":       [64],
#     "weight_decay":  [1e-4],
#     "loss_mode":     ["mae"],
#     "patience":      [7],
#     "epochs":        [100],
# }


SCENARIO_TEMPLATES = {
    "sc1": "sc1_unet_preupsample.ipynb",
    "sc2": "sc2_unet_mae_preupsample.ipynb",
}


# ──────────────────────────────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────────────────────────────
def _patch_source(source_lines: list[str], old: str, new: str) -> list[str]:
    """Replace the first occurrence of *old* with *new* inside the joined
    source lines of a code cell.  Returns a NEW list of source lines."""
    joined = "".join(source_lines)
    if old not in joined:
        return source_lines                       # nothing to patch
    patched = joined.replace(old, new, 1)
    # nbformat stores source as a list of lines (each ending with \n)
    return patched.splitlines(keepends=True)


def _patch_notebook(nb, hp: dict) -> None:
    """Mutate *nb* in-place so that the hyper-parameters in *hp* are
    reflected in the relevant code cells."""

    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        src = cell.source
        if not isinstance(src, str):
            src = "".join(src)

        # ── Patch BATCH_SIZE ────────────────────────────────────────
        m = re.search(r'BATCH_SIZE\s*=\s*\d+', src)
        if m:
            cell.source = src.replace(m.group(0),
                                      f'BATCH_SIZE = {hp["batch_size"]}')
            src = cell.source

        # ── Patch base_ch in UNet instantiation ─────────────────────
        m = re.search(r'UNet\(in_ch=4,\s*out_ch=4,\s*base_ch=\d+\)', src)
        if m:
            cell.source = src.replace(
                m.group(0),
                f'UNet(in_ch=4, out_ch=4, base_ch={hp["base_ch"]})')
            src = cell.source

        # ── Patch lr in train() call ────────────────────────────────
        m = re.search(r'epochs\s*=\s*\d+,\s*lr\s*=\s*[\d.eE\-]+', src)
        if m:
            cell.source = src.replace(
                m.group(0),
                f'epochs={hp["epochs"]}, lr={hp["learning_rate"]}')
            src = cell.source

        # ── Patch weight_decay inside train() body ──────────────────
        m = re.search(r'weight_decay\s*=\s*[\d.eE\-]+', src)
        if m:
            cell.source = src.replace(
                m.group(0),
                f'weight_decay={hp["weight_decay"]}')
            src = cell.source

        # ── Patch patience inside train() definition ────────────────
        # Use negative lookbehind to avoid matching 'patience_counter'
        m = re.search(r'(?<!\w)patience\s*=\s*\d+(?!\w)', src)
        if m:
            cell.source = src.replace(
                m.group(0),
                f'patience   = {hp["patience"]}')
            src = cell.source

        # ── Patch loss mode inside train() body ─────────────────────
        m = re.search(r"CombinedLoss\(mode='[^']+'\)", src)
        if m:
            cell.source = src.replace(
                m.group(0),
                f"CombinedLoss(mode='{hp['loss_mode']}')")
            src = cell.source


def _extract_metrics_from_notebook(nb_path: str) -> dict:
    """Open an *executed* notebook and scrape the last training/val loss
    as well as any test-set CSV metrics from stdout outputs."""
    nb = nbformat.read(nb_path, as_version=4)
    metrics: dict = {}
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        for out in cell.get("outputs", []):
            text = out.get("text", "")
            if isinstance(text, list):
                text = "".join(text)
            # best_val_loss from runtime log
            m = re.search(r'best_val_loss=([\d.]+)', text)
            if m:
                metrics["best_val_loss"] = float(m.group(1))
            # Final Training / Validation Loss
            m = re.search(r'Final Training Loss:\s*([\d.]+)', text)
            if m:
                metrics["final_train_loss"] = float(m.group(1))
            m = re.search(r'Final Validation Loss:\s*([\d.]+)', text)
            if m:
                metrics["final_val_loss"] = float(m.group(1))
            # Test-set RMSE lines  (from metrics.csv echo)
            # Example: "variable,rmse,mae,..."
            for line in text.split("\n"):
                if "rmse" in line.lower() and "," in line:
                    # rudimentary CSV parse
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) >= 2:
                        try:
                            float(parts[1])
                            metrics[f"test_{parts[0]}_rmse"] = float(parts[1])
                        except ValueError:
                            pass
    return metrics


def _combo_id(hp: dict) -> str:
    """Short deterministic id for a hyperparameter combo."""
    parts = []
    for k in sorted(hp):
        v = hp[k]
        if isinstance(v, float):
            parts.append(f"{k}={v:.0e}")
        else:
            parts.append(f"{k}={v}")
    return "__".join(parts)


# ──────────────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description="Hyperparameter tuning for UNet preupsample notebooks")
    ap.add_argument("--scenario", required=True, choices=["sc1", "sc2"],
                    help="Which scenario notebook to tune")
    ap.add_argument("--dry-run", action="store_true",
                    help="Only generate notebooks, do not execute")
    ap.add_argument("--timeout", type=int, default=7200,
                    help="Per-notebook execution timeout in seconds (default 7200)")
    args = ap.parse_args()

    template_name = SCENARIO_TEMPLATES[args.scenario]
    template_path = Path(__file__).parent / template_name

    if not template_path.exists():
        print(f"[ERROR] Template notebook not found: {template_path}")
        sys.exit(1)

    # Read template once
    template_nb = nbformat.read(str(template_path), as_version=4)

    # Build all combos
    keys = sorted(HP_GRID.keys())
    values = [HP_GRID[k] for k in keys]
    combos = [dict(zip(keys, v)) for v in itertools.product(*values)]
    print(f"[INFO] Scenario {args.scenario}: {len(combos)} hyperparameter combinations to evaluate.")

    results = []

    out_dir = Path(__file__).parent / f"hp_tuning_{args.scenario}"
    out_dir.mkdir(exist_ok=True)

    for idx, hp in enumerate(combos, 1):
        combo_tag = _combo_id(hp)
        nb_name   = f"{args.scenario}_hptune_{idx:03d}.ipynb"
        nb_path   = out_dir / nb_name

        print(f"\n{'='*72}")
        print(f"[{idx}/{len(combos)}] {combo_tag}")
        print(f"{'='*72}")

        # Deep-copy & patch
        nb = copy.deepcopy(template_nb)
        _patch_notebook(nb, hp)

        # Clear all outputs before executing
        for cell in nb.cells:
            cell["outputs"] = []
            if "execution_count" in cell:
                cell["execution_count"] = None

        # Write patched notebook
        nbformat.write(nb, str(nb_path))
        print(f"  -> wrote {nb_path}")

        if args.dry_run:
            print("  (dry-run -- skipping execution)")
            continue

        # Execute via jupyter nbconvert
        t0 = time.time()
        cmd = [
            sys.executable, "-m", "jupyter", "nbconvert",
            "--to", "notebook",
            "--execute",
            "--inplace",
            f"--ExecutePreprocessor.timeout={args.timeout}",
            str(nb_path),
        ]
        print(f"  Executing ...  (timeout={args.timeout}s)")
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True,
                                  timeout=args.timeout + 60)
            elapsed = time.time() - t0
            if proc.returncode != 0:
                print(f"  [FAIL] (rc={proc.returncode}, {elapsed:.0f}s)")
                print(f"    stderr: {proc.stderr[:500]}")
                row = {**hp, "status": "FAILED", "elapsed_s": elapsed}
            else:
                print(f"  [OK] ({elapsed:.0f}s)")
                metrics = _extract_metrics_from_notebook(str(nb_path))
                row = {**hp, "status": "OK", "elapsed_s": elapsed, **metrics}
        except subprocess.TimeoutExpired:
            elapsed = time.time() - t0
            print(f"  [TIMEOUT] ({elapsed:.0f}s)")
            row = {**hp, "status": "TIMEOUT", "elapsed_s": elapsed}
        except Exception:
            elapsed = time.time() - t0
            print(f"  [ERROR] ({elapsed:.0f}s)")
            traceback.print_exc()
            row = {**hp, "status": "ERROR", "elapsed_s": elapsed}

        results.append(row)

    # ── Write summary CSV ────────────────────────────────────────────
    if results:
        csv_path = Path(__file__).parent / f"hp_tuning_results_{args.scenario}.csv"
        # Collect all column names
        all_keys = []
        for r in results:
            for k in r:
                if k not in all_keys:
                    all_keys.append(k)
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write(",".join(all_keys) + "\n")
            for r in results:
                vals = [str(r.get(k, "")) for k in all_keys]
                f.write(",".join(vals) + "\n")
        print(f"\n[DONE] Results written to {csv_path}")
    else:
        print("\n[DONE] No results to write (dry-run or empty grid).")


if __name__ == "__main__":
    main()
