"""
Refactor sc2_eval_unet_mae.ipynb to multi-run evaluation.
Adapts from the reference sc2_eval_gan.ipynb pattern for SRUNet MAE.
"""

import json
import os
import copy
import sys

sys.stdout.reconfigure(encoding='utf-8')

BASE = os.path.dirname(os.path.abspath(__file__))
NOTEBOOK_PATH = os.path.join(BASE, 'sc2_eval_unet_mae.ipynb')
REFERENCE_PATH = os.path.join(BASE, 'sc2_eval_gan.ipynb')

SC2_UNET_RUNS = [
    "runs TA/sc2_unet/20260406_162603",
    "runs TA/sc2_unet/20260406_162952",
    "runs TA/sc2_unet/20260406_163455",
    "runs TA/sc2_unet/20260406_163958",
    "runs TA/sc2_unet/20260406_164503",
]


def load_notebook(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_notebook(nb, path):
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            cell['outputs'] = []
            cell['execution_count'] = None
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)


def make_source(text):
    lines = text.split('\n')
    result = []
    for i, line in enumerate(lines):
        if i < len(lines) - 1:
            result.append(line + '\n')
        else:
            if line:
                result.append(line)
    return result


def make_code_cell(source_text, cell_id=None):
    cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": make_source(source_text)
    }
    if cell_id:
        cell["id"] = cell_id
    return cell


def make_markdown_cell(source_text, cell_id=None):
    cell = {
        "cell_type": "markdown",
        "metadata": {},
        "source": make_source(source_text)
    }
    if cell_id:
        cell["id"] = cell_id
    return cell


def get_cell_source(nb, idx):
    return ''.join(nb['cells'][idx].get('source', []))


def adapt_eval_loop(ref_eval_src):
    """Adapt the GAN eval loop for SRUNet MAE."""
    lines = ref_eval_src.split('\n')
    out = []
    i = 0
    while i < len(lines):
        line = lines[i]

        # 1. Replace header comment
        if 'all SC2 GAN runs' in line:
            out.append(line.replace('all SC2 GAN runs', 'all SC2 UNet MAE runs'))
            i += 1
            continue

        # 2. Replace GAN model loading block (lines containing RRDBGenerator)
        if 'RRDBGenerator(' in line:
            # Replace the 3 GAN-specific lines with SRUNet loading
            out.append('    model = SRUNet(in_ch=4, out_ch=4, base_ch=64)')
            out.append('    model.load_state_dict(checkpoint["model_state"])')
            out.append('    model.eval()')
            # Skip next lines until we hit norm_stats (skip 'generator.load_state_dict' and 'model = generator' and 'model.eval()')
            i += 1
            while i < len(lines) and 'norm_stats' not in lines[i]:
                i += 1
            continue

        # 3. Fix plot title
        if 'GAN SR' in line:
            out.append(line.replace('GAN SR', 'UNet MAE'))
            i += 1
            continue

        # 4. Fix column title
        if '(GAN ' in line:
            out.append(line.replace('(GAN ', '(UNet '))
            i += 1
            continue

        # 5. Fix cleanup line
        if 'del model, checkpoint, generator' in line:
            out.append(line.replace('del model, checkpoint, generator', 'del model, checkpoint'))
            i += 1
            continue

        out.append(line)
        i += 1

    return '\n'.join(out)


def main():
    nb = load_notebook(NOTEBOOK_PATH)
    ref = load_notebook(REFERENCE_PATH)

    # Get reference cells
    ref_data_src = get_cell_source(ref, 4)      # SC2 data pipeline
    ref_helpers_src = get_cell_source(ref, 5)    # helpers + baseline
    ref_eval_src = get_cell_source(ref, 6)       # eval loop

    # Keep original SRUNet model definition (cell 3)
    orig_model_cell = copy.deepcopy(nb['cells'][3])
    orig_model_cell['outputs'] = []
    orig_model_cell['execution_count'] = None
    orig_model_cell['id'] = 'model_def'

    # Adapt eval loop for SRUNet MAE
    eval_loop_src = adapt_eval_loop(ref_eval_src)

    # ── Assemble new cells ──
    new_cells = []

    # Cell 0: Markdown header
    new_cells.append(make_markdown_cell(
        "# sc2_unet_mae Evaluation\n"
        "\n"
        "This notebook evaluates trained Scenario-2 (Perfect Prognosis) SRUNet runs and computes metrics using:\n"
        "\n"
        "- `MAE = (1/(N*L*W)) * sum(|S - O|)`\n"
        "- `RMSE = sqrt((1/(N*L*W)) * sum((S - O)^2))`\n"
        "\n"
        "where `N` is number of test samples, `L` is latitude size, and `W` is longitude size.\n",
        cell_id="header"
    ))

    # Cell 1: Inputs
    runs_str = ',\n'.join(f'    "{r}"' for r in SC2_UNET_RUNS)
    new_cells.append(make_code_cell(
        '# ==============================\n'
        '# Inputs\n'
        '# ==============================\n'
        'from pathlib import Path\n'
        '\n'
        '# All 5 runs for this model\n'
        'RUN_DIRS = [\n'
        f'{runs_str},\n'
        ']\n'
        '\n'
        '# Scenario-2 data file (Perfect Prognosis uses ERA5 for both LR and HR)\n'
        'TRUTH_PATH = "data/era5_indonesia_2018-2022.zarr"\n'
        '\n'
        '# Optional override (None = read from run config.json)\n'
        'TEST_START_DATE_OVERRIDE = None\n'
        'TEST_END_DATE_OVERRIDE   = None\n'
        '\n'
        '# Variables in fixed order\n'
        'VARS = [\n'
        '    "10m_u_component_of_wind",\n'
        '    "10m_v_component_of_wind",\n'
        '    "2m_temperature",\n'
        '    "total_precipitation_24hr",\n'
        ']\n'
        'VAR_LABELS = ["U10 (m/s)", "V10 (m/s)", "T2m (K)", "TP 24hr (mm)"]\n'
        'TP_IDX = VARS.index("total_precipitation_24hr")\n'
        '\n'
        '# Validate first run directory exists\n'
        'first_run = Path(RUN_DIRS[0])\n'
        'if not first_run.exists():\n'
        '    raise FileNotFoundError(\n'
        '        f"First run dir not found: {first_run}. "\n'
        '        f"Ensure runs TA/ directory is in the working directory."\n'
        '    )\n'
        '\n'
        'print(f"Configured {len(RUN_DIRS)} runs for evaluation")\n'
        'for rd in RUN_DIRS:\n'
        '    print(f"  {rd}")\n',
        cell_id="inputs"
    ))

    # Cell 2: Imports + config
    new_cells.append(make_code_cell(
        '# ==============================\n'
        '# Imports + run config (from first run)\n'
        '# ==============================\n'
        'import json\n'
        'import numpy as np\n'
        'import pandas as pd\n'
        'import xarray as xr\n'
        'import torch\n'
        'import torch.nn as nn\n'
        'import torch.nn.functional as F\n'
        '\n'
        '# Read config from first run (all runs share same hyperparameters)\n'
        'CFG_PATH = Path(RUN_DIRS[0]) / "config.json"\n'
        '\n'
        'cfg = {}\n'
        'if CFG_PATH.exists():\n'
        '    cfg = json.loads(CFG_PATH.read_text(encoding="utf-8"))\n'
        '\n'
        'lead_days = int(cfg.get("lead_days", 0))\n'
        'scale = int(cfg.get("scale", 6))\n'
        '\n'
        'test_start_date = TEST_START_DATE_OVERRIDE or cfg.get("test_start_date", "2022-07-01")\n'
        'test_end_date   = TEST_END_DATE_OVERRIDE   or cfg.get("test_end_date", "2022-12-31")\n'
        '\n'
        'print("lead_days:", lead_days)\n'
        'print("scale:", scale)\n'
        'print("test range:", test_start_date, "->", test_end_date)\n'
        '\n'
        'from torch.utils.data import TensorDataset, DataLoader\n',
        cell_id="config"
    ))

    # Cell 3: Model definition (unchanged SRUNet)
    new_cells.append(orig_model_cell)

    # Cell 4: SC2 Data pipeline (from reference)
    new_cells.append(make_code_cell(ref_data_src, cell_id="data_pipeline"))

    # Cell 5: Helpers + baseline (from reference)
    new_cells.append(make_code_cell(ref_helpers_src, cell_id="helpers_baseline"))

    # Cell 6: Multi-run eval loop (adapted for SRUNet)
    new_cells.append(make_code_cell(eval_loop_src, cell_id="eval_loop"))

    # Cell 7: Footer
    new_cells.append(make_code_cell(
        '# Notebook refactor complete.\n'
        '# Metrics are saved by the evaluation cell above.\n',
        cell_id="footer"
    ))

    nb['cells'] = new_cells
    save_notebook(nb, NOTEBOOK_PATH)
    print(f"✓ Refactored {os.path.basename(NOTEBOOK_PATH)}")
    print(f"  Cells: {len(new_cells)}")
    print(f"  Runs: {len(SC2_UNET_RUNS)}")


if __name__ == '__main__':
    main()
