"""
Batch refactoring script for all remaining evaluation notebooks.

For each notebook:
1. Takes the reference pattern from sc2_eval_gan.ipynb
2. Replaces model architecture + checkpoint loading
3. Sets correct RUN_DIRS
4. Sets correct SC1/SC2 data pipeline
5. Fixes labels and titles
"""

import json
import os
import copy
import sys

sys.stdout.reconfigure(encoding='utf-8')

BASE = os.path.dirname(os.path.abspath(__file__))
REFERENCE_PATH = os.path.join(BASE, 'sc2_eval_gan.ipynb')


# ── Configuration for ALL notebooks ──────────────────────────────────────────

NOTEBOOKS = {
    # SC2 notebooks (Perfect Prognosis)
    "sc2_eval_gan_preupsample.ipynb": {
        "scenario": 2,
        "model_type": "gan_preupsample",
        "title": "sc2_gan_preupsample Evaluation",
        "description": "Scenario-2 (Perfect Prognosis) GAN Pre-Upsample",
        "runs": [
            "runs TA/sc2_gan_preupsample/20260413_141145",
            "runs TA/sc2_gan_preupsample/20260413_171310",
            "runs TA/sc2_gan_preupsample/20260413_195102",
            "runs TA/sc2_gan_preupsample/20260413_222824",
            "runs TA/sc2_gan_preupsample/20260414_010636",
        ],
        "model_load": (
            '    model = RRDBGenerator(in_ch=4, out_ch=4, base_ch=64, num_rrdb=4)\n'
            '    model.load_state_dict(checkpoint["generator"])\n'
            '    model.eval()'
        ),
        "plot_model_name": "GAN Pre-Up",
        "plot_col_label": "GAN",
        "cleanup_line": "del model, checkpoint",
    },
    "sc2_eval_unet_mae_preupsample.ipynb": {
        "scenario": 2,
        "model_type": "unet_preupsample",
        "title": "sc2_unet_mae_preupsample Evaluation",
        "description": "Scenario-2 (Perfect Prognosis) SRUNet Pre-Upsample",
        "runs": [
            "runs TA/sc2_unet_preupsample/20260408_122046",
            "runs TA/sc2_unet_preupsample/20260408_123647",
            "runs TA/sc2_unet_preupsample/20260408_125627",
            "runs TA/sc2_unet_preupsample/20260408_131414",
            "runs TA/sc2_unet_preupsample/20260408_133018",
        ],
        "model_load": (
            '    model = UNet(in_ch=4, out_ch=4, base_ch=64)\n'
            '    model.load_state_dict(checkpoint["model_state"])\n'
            '    model.eval()'
        ),
        "plot_model_name": "UNet MAE Pre-Up",
        "plot_col_label": "UNet",
        "cleanup_line": "del model, checkpoint",
    },
    "sc2_eval_covnext_mae.ipynb": {
        "scenario": 2,
        "model_type": "convnext",
        "title": "sc2_convnext_mae Evaluation",
        "description": "Scenario-2 (Perfect Prognosis) ConvNeXt MAE",
        "runs": [
            "runs TA/sc2_convnext/20260406_182448",
            "runs TA/sc2_convnext/20260406_183136",
            "runs TA/sc2_convnext/20260406_183841",
            "runs TA/sc2_convnext/20260406_184532",
            "runs TA/sc2_convnext/20260406_185223",
        ],
        "model_load": (
            '    model = ConvNeXtSR(in_ch=4, out_ch=4, base_ch=64)\n'
            '    model.load_state_dict(checkpoint["model_state"])\n'
            '    model.eval()'
        ),
        "plot_model_name": "ConvNeXt MAE",
        "plot_col_label": "ConvNeXt",
        "cleanup_line": "del model, checkpoint",
    },
    "sc2_eval_convnext_mae_preupsample.ipynb": {
        "scenario": 2,
        "model_type": "convnext_preupsample",
        "title": "sc2_convnext_mae_preupsample Evaluation",
        "description": "Scenario-2 (Perfect Prognosis) ConvNeXt MAE Pre-Upsample",
        "runs": [
            "runs TA/sc2_convnext_preupsample/20260408_154457",
            "runs TA/sc2_convnext_preupsample/20260408_162953",
            "runs TA/sc2_convnext_preupsample/20260408_171455",
            "runs TA/sc2_convnext_preupsample/20260408_175957",
            "runs TA/sc2_convnext_preupsample/20260408_184448",
        ],
        "model_load": (
            '    model = ConvNeXtPreUpsample(in_ch=4, out_ch=4, base_ch=64)\n'
            '    model.load_state_dict(checkpoint["model_state"])\n'
            '    model.eval()'
        ),
        "plot_model_name": "ConvNeXt Pre-Up",
        "plot_col_label": "ConvNeXt",
        "cleanup_line": "del model, checkpoint",
    },
    "sc2_eval_resnet18_mae.ipynb": {
        "scenario": 2,
        "model_type": "resnet",
        "title": "sc2_resnet_mae Evaluation",
        "description": "Scenario-2 (Perfect Prognosis) ResNet MAE",
        "runs": [
            "runs TA/sc2_resnet/20260406_185921",
            "runs TA/sc2_resnet/20260406_190322",
            "runs TA/sc2_resnet/20260406_190828",
            "runs TA/sc2_resnet/20260406_191237",
            "runs TA/sc2_resnet/20260406_191718",
        ],
        "model_load": (
            '    model = ResNet18SR(in_ch=4, out_ch=4, base_ch=64)\n'
            '    model.load_state_dict(checkpoint["model_state"])\n'
            '    model.eval()'
        ),
        "plot_model_name": "ResNet MAE",
        "plot_col_label": "ResNet",
        "cleanup_line": "del model, checkpoint",
    },
    "sc2_eval_resnet18_mae_preupsample.ipynb": {
        "scenario": 2,
        "model_type": "resnet_preupsample",
        "title": "sc2_resnet_mae_preupsample Evaluation",
        "description": "Scenario-2 (Perfect Prognosis) ResNet MAE Pre-Upsample",
        "runs": [
            "runs TA/sc2_resnet_preupsample/20260408_192913",
            "runs TA/sc2_resnet_preupsample/20260408_193714",
            "runs TA/sc2_resnet_preupsample/20260408_194847",
            "runs TA/sc2_resnet_preupsample/20260408_195414",
            "runs TA/sc2_resnet_preupsample/20260408_195939",
        ],
        "model_load": (
            '    model = ResNet18PreUpsample(in_ch=4, out_ch=4, base_ch=64)\n'
            '    model.load_state_dict(checkpoint["model_state"])\n'
            '    model.eval()'
        ),
        "plot_model_name": "ResNet Pre-Up",
        "plot_col_label": "ResNet",
        "cleanup_line": "del model, checkpoint",
    },
    # SC1 notebooks (Forecast Correction)
    "sc1_eval_convnext_mae.ipynb": {
        "scenario": 1,
        "model_type": "convnext",
        "title": "sc1_convnext_mae Evaluation",
        "description": "Scenario-1 (Forecast Correction) ConvNeXt MAE",
        "runs": [
            "runs TA/sc1_convnext/20260406_180611_lead1d",
            "runs TA/sc1_convnext/20260406_180855_lead1d",
            "runs TA/sc1_convnext/20260406_181148_lead1d",
            "runs TA/sc1_convnext/20260415_083934_lead1d",
            "runs TA/sc1_convnext/20260415_084723_lead1d",
        ],
        "model_load": (
            '    model = ConvNeXtSR(in_ch=4, out_ch=4, base_ch=64)\n'
            '    model.load_state_dict(checkpoint["model_state"])\n'
            '    model.eval()'
        ),
        "plot_model_name": "ConvNeXt MAE",
        "plot_col_label": "ConvNeXt",
        "cleanup_line": "del model, checkpoint",
    },
    "sc1_eval_convnext_mae_pixel.ipynb": {
        "scenario": 1,
        "model_type": "convnext",
        "title": "sc1_convnext_mae_pixel Evaluation",
        "description": "Scenario-1 (Forecast Correction) ConvNeXt MAE (Pixel Loss)",
        "runs": [
            "runs TA/sc1_convnext/20260406_180611_lead1d",
            "runs TA/sc1_convnext/20260406_180855_lead1d",
            "runs TA/sc1_convnext/20260406_181148_lead1d",
            "runs TA/sc1_convnext/20260415_083934_lead1d",
            "runs TA/sc1_convnext/20260415_084723_lead1d",
        ],
        "model_load": (
            '    model = ConvNeXtSR(in_ch=4, out_ch=4, base_ch=64)\n'
            '    model.load_state_dict(checkpoint["model_state"])\n'
            '    model.eval()'
        ),
        "plot_model_name": "ConvNeXt Pixel",
        "plot_col_label": "ConvNeXt",
        "cleanup_line": "del model, checkpoint",
    },
    "sc1_eval_convnext_mae_preupsample.ipynb": {
        "scenario": 1,
        "model_type": "convnext_preupsample",
        "title": "sc1_convnext_mae_preupsample Evaluation",
        "description": "Scenario-1 (Forecast Correction) ConvNeXt MAE Pre-Upsample",
        "runs": [
            "runs TA/sc1_convnext_preupsample/20260413_114526_lead1d",
            "runs TA/sc1_convnext_preupsample/20260413_115018_lead1d",
            "runs TA/sc1_convnext_preupsample/20260413_115501_lead1d",
            "runs TA/sc1_convnext_preupsample/20260413_115945_lead1d",
            "runs TA/sc1_convnext_preupsample/20260413_120429_lead1d",
        ],
        "model_load": (
            '    model = ConvNeXtPreUpsample(in_ch=4, out_ch=4, base_ch=64)\n'
            '    model.load_state_dict(checkpoint["model_state"])\n'
            '    model.eval()'
        ),
        "plot_model_name": "ConvNeXt Pre-Up",
        "plot_col_label": "ConvNeXt",
        "cleanup_line": "del model, checkpoint",
    },
    "sc1_eval_convnext_mae_preupsample_pixel.ipynb": {
        "scenario": 1,
        "model_type": "convnext_preupsample",
        "title": "sc1_convnext_mae_preupsample_pixel Evaluation",
        "description": "Scenario-1 (Forecast Correction) ConvNeXt MAE Pre-Upsample (Pixel Loss)",
        "runs": [
            "runs TA/sc1_convnext_preupsample/20260413_114526_lead1d",
            "runs TA/sc1_convnext_preupsample/20260413_115018_lead1d",
            "runs TA/sc1_convnext_preupsample/20260413_115501_lead1d",
            "runs TA/sc1_convnext_preupsample/20260413_115945_lead1d",
            "runs TA/sc1_convnext_preupsample/20260413_120429_lead1d",
        ],
        "model_load": (
            '    model = SRUNet(in_ch=4, out_ch=4, base_ch=64)\n'
            '    model.load_state_dict(checkpoint["model_state"])\n'
            '    model.eval()'
        ),
        "plot_model_name": "ConvNeXt Pre-Up Pixel",
        "plot_col_label": "ConvNeXt",
        "cleanup_line": "del model, checkpoint",
    },
    "sc1_eval_gan.ipynb": {
        "scenario": 1,
        "model_type": "gan",
        "title": "sc1_gan Evaluation",
        "description": "Scenario-1 (Forecast Correction) GAN",
        "runs": [
            "runs TA/sc1_gan/20260409_150703_lead1d",
            "runs TA/sc1_gan/20260409_160845_lead1d",
            "runs TA/sc1_gan/20260410_120035_lead1d",
            "runs TA/sc1_gan/20260410_135825_lead1d",
            "runs TA/sc1_gan/20260410_152428_lead1d",
        ],
        "model_load": (
            '    model = RRDBGenerator(in_ch=4, out_ch=4, base_ch=64, num_rrdb=4)\n'
            '    model.load_state_dict(checkpoint["generator"])\n'
            '    model.eval()'
        ),
        "plot_model_name": "GAN SR",
        "plot_col_label": "GAN",
        "cleanup_line": "del model, checkpoint",
    },
    "sc1_eval_gan_preupsample.ipynb": {
        "scenario": 1,
        "model_type": "gan_preupsample",
        "title": "sc1_gan_preupsample Evaluation",
        "description": "Scenario-1 (Forecast Correction) GAN Pre-Upsample",
        "runs": [
            "runs TA/sc1_gan_preupsample/20260413_154641_lead1d",
            "runs TA/sc1_gan_preupsample/20260413_171329_lead1d",
            "runs TA/sc1_gan_preupsample/20260413_184014_lead1d",
            "runs TA/sc1_gan_preupsample/20260413_200656_lead1d",
            "runs TA/sc1_gan_preupsample/20260413_213340_lead1d",
        ],
        "model_load": (
            '    model = RRDBGenerator(in_ch=4, out_ch=4, base_ch=64, num_rrdb=4)\n'
            '    model.load_state_dict(checkpoint["generator"])\n'
            '    model.eval()'
        ),
        "plot_model_name": "GAN Pre-Up",
        "plot_col_label": "GAN",
        "cleanup_line": "del model, checkpoint",
    },
    "sc1_eval_resnet18.ipynb": {
        "scenario": 1,
        "model_type": "resnet",
        "title": "sc1_resnet_mae Evaluation",
        "description": "Scenario-1 (Forecast Correction) ResNet MAE",
        "runs": [
            "runs TA/sc1_resnet/20260406_181435_lead1d",
            "runs TA/sc1_resnet/20260406_181625_lead1d",
            "runs TA/sc1_resnet/20260406_181910_lead1d",
            "runs TA/sc1_resnet/20260406_182113_lead1d",
            "runs TA/sc1_resnet/20260406_182300_lead1d",
        ],
        "model_load": (
            '    model = ResNet18SR(in_ch=4, out_ch=4, base_ch=64)\n'
            '    model.load_state_dict(checkpoint["model_state"])\n'
            '    model.eval()'
        ),
        "plot_model_name": "ResNet MAE",
        "plot_col_label": "ResNet",
        "cleanup_line": "del model, checkpoint",
    },
    "sc1_eval_resnet18_preupsample.ipynb": {
        "scenario": 1,
        "model_type": "resnet_preupsample",
        "title": "sc1_resnet_mae_preupsample Evaluation",
        "description": "Scenario-1 (Forecast Correction) ResNet MAE Pre-Upsample",
        "runs": [
            "runs TA/sc1_resnet_preupsample/20260408_134622_lead1d",
            "runs TA/sc1_resnet_preupsample/20260408_134836_lead1d",
            "runs TA/sc1_resnet_preupsample/20260408_135049_lead1d",
            "runs TA/sc1_resnet_preupsample/20260408_135301_lead1d",
            "runs TA/sc1_resnet_preupsample/20260408_135514_lead1d",
        ],
        "model_load": (
            '    model = ResNet18PreUpsample(in_ch=4, out_ch=4, base_ch=64)\n'
            '    model.load_state_dict(checkpoint["model_state"])\n'
            '    model.eval()'
        ),
        "plot_model_name": "ResNet Pre-Up",
        "plot_col_label": "ResNet",
        "cleanup_line": "del model, checkpoint",
    },
    "sc1_eval_unet.ipynb": {
        "scenario": 1,
        "model_type": "unet",
        "title": "sc1_unet_mae Evaluation",
        "description": "Scenario-1 (Forecast Correction) SRUNet MAE",
        "runs": [
            "runs TA/sc1_unet/20260408_081053_lead1d",
            "runs TA/sc1_unet/20260408_081218_lead1d",
            "runs TA/sc1_unet/20260408_081356_lead1d",
            "runs TA/sc1_unet/20260408_081522_lead1d",
            "runs TA/sc1_unet/20260408_081702_lead1d",
        ],
        "model_load": (
            '    model = SRUNet(in_ch=4, out_ch=4, base_ch=64)\n'
            '    model.load_state_dict(checkpoint["model_state"])\n'
            '    model.eval()'
        ),
        "plot_model_name": "UNet MAE",
        "plot_col_label": "UNet",
        "cleanup_line": "del model, checkpoint",
    },
    "sc1_eval_unet_preupsample.ipynb": {
        "scenario": 1,
        "model_type": "unet_preupsample",
        "title": "sc1_unet_mae_preupsample Evaluation",
        "description": "Scenario-1 (Forecast Correction) SRUNet MAE Pre-Upsample",
        "runs": [
            "runs TA/sc1_unet_preupsample/20260407_185328_lead1d",
            "runs TA/sc1_unet_preupsample/20260408_070930_lead1d",
            "runs TA/sc1_unet_preupsample/20260408_071644_lead1d",
            "runs TA/sc1_unet_preupsample/20260408_072410_lead1d",
            "runs TA/sc1_unet_preupsample/20260408_073126_lead1d",
        ],
        "model_load": (
            '    model = UNet(in_ch=4, out_ch=4, base_ch=64)\n'
            '    model.load_state_dict(checkpoint["model_state"])\n'
            '    model.eval()'
        ),
        "plot_model_name": "UNet Pre-Up",
        "plot_col_label": "UNet",
        "cleanup_line": "del model, checkpoint",
    },
}


# ── Utility functions ────────────────────────────────────────────────────────

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
    cell = {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": make_source(source_text)}
    if cell_id:
        cell["id"] = cell_id
    return cell

def make_markdown_cell(source_text, cell_id=None):
    cell = {"cell_type": "markdown", "metadata": {}, "source": make_source(source_text)}
    if cell_id:
        cell["id"] = cell_id
    return cell

def get_cell_source(nb, idx):
    return ''.join(nb['cells'][idx].get('source', []))


def adapt_eval_loop(ref_eval_src, cfg):
    """Adapt the reference GAN eval loop for this notebook's model."""
    lines = ref_eval_src.split('\n')
    out = []
    i = 0
    sc = cfg['scenario']
    label = f"all SC{sc} {cfg['plot_model_name']} runs"

    while i < len(lines):
        line = lines[i]

        # 1. Replace header comment
        if 'all SC2 GAN runs' in line:
            out.append(line.replace('all SC2 GAN runs', label))
            i += 1; continue

        # 2. Replace GAN model loading block
        if 'RRDBGenerator(' in line:
            # Insert the correct model loading code
            for ml in cfg['model_load'].split('\n'):
                out.append(ml)
            # Skip the original GAN lines until norm_stats
            i += 1
            while i < len(lines) and 'norm_stats' not in lines[i]:
                i += 1
            continue

        # 3. Fix plot title
        if 'GAN SR' in line:
            out.append(line.replace('GAN SR', cfg['plot_model_name']))
            i += 1; continue

        # 4. Fix column title
        if '(GAN ' in line:
            out.append(line.replace('(GAN ', f'({cfg["plot_col_label"]} '))
            i += 1; continue

        # 5. Fix cleanup line
        if 'del model, checkpoint, generator' in line:
            out.append(line.replace('del model, checkpoint, generator', cfg['cleanup_line']))
            i += 1; continue

        out.append(line)
        i += 1

    return '\n'.join(out)


def refactor_notebook(filename, cfg, ref):
    """Refactor a single notebook."""
    nb_path = os.path.join(BASE, filename)
    if not os.path.exists(nb_path):
        print(f"  ⚠ Notebook not found: {filename}, skipping")
        return False

    nb = load_notebook(nb_path)

    # Get reference cells
    ref_data_src = get_cell_source(ref, 4)       # SC2 data pipeline
    ref_helpers_src = get_cell_source(ref, 5)     # helpers + baseline
    ref_eval_src = get_cell_source(ref, 6)        # eval loop

    # Keep original model definition (cell 3)
    orig_model_cell = copy.deepcopy(nb['cells'][3])
    orig_model_cell['outputs'] = []
    orig_model_cell['execution_count'] = None
    orig_model_cell['id'] = 'model_def'

    # For SC1, we need to use SC1 data pipeline (from the original notebook's cell 4)
    # For SC2, we use the reference SC2 data pipeline
    if cfg['scenario'] == 1:
        data_src = get_cell_source(nb, 4)  # Keep original SC1 data pipeline
        # Ensure raw aliases exist for the multi-run eval loop
        if "X_test_raw =" not in data_src:
            data_src += "\n\n# Save raw test arrays for multi-run evaluation\n"
            data_src += "X_test_raw = X_test.copy()\n"
            data_src += "Y_test_raw = Y_test.copy()\n"
            data_src += 'print("Raw test arrays saved for multi-run evaluation.")\n'
    else:
        data_src = ref_data_src  # Use reference SC2 data pipeline

    # Adapt eval loop
    eval_loop_src = adapt_eval_loop(ref_eval_src, cfg)

    # Determine scenario-specific inputs
    sc_desc = "Forecast Correction" if cfg['scenario'] == 1 else "Perfect Prognosis"

    # Build cells
    new_cells = []

    # Cell 0: Markdown header
    new_cells.append(make_markdown_cell(
        f"# {cfg['title']}\n"
        f"\n"
        f"This notebook evaluates trained {cfg['description']} runs and computes metrics using:\n"
        f"\n"
        f"- `MAE = (1/(N*L*W)) * sum(|S - O|)`\n"
        f"- `RMSE = sqrt((1/(N*L*W)) * sum((S - O)^2))`\n"
        f"\n"
        f"where `N` is number of test samples, `L` is latitude size, and `W` is longitude size.\n",
        cell_id="header"
    ))

    # Cell 1: Inputs
    runs_str = ',\n'.join(f'    "{r}"' for r in cfg['runs'])
    if cfg['scenario'] == 1:
        data_paths = (
            '# Scenario-1 data files\n'
            'FORECAST_PATH = "data/ifs_lowres_indonesia_2018-2022.zarr"\n'
            'TRUTH_PATH    = "data/era5_indonesia_2018-2022.zarr"\n'
        )
    else:
        data_paths = (
            '# Scenario-2 data file (Perfect Prognosis uses ERA5 for both LR and HR)\n'
            'TRUTH_PATH = "data/era5_indonesia_2018-2022.zarr"\n'
        )

    new_cells.append(make_code_cell(
        '# ==============================\n'
        '# Inputs\n'
        '# ==============================\n'
        'from pathlib import Path\n'
        '\n'
        f'# All {len(cfg["runs"])} runs for this model\n'
        'RUN_DIRS = [\n'
        f'{runs_str},\n'
        ']\n'
        '\n'
        f'{data_paths}'
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
    lead_default = '1' if cfg['scenario'] == 1 else '0'
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
        f'lead_days = int(cfg.get("lead_days", {lead_default}))\n'
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

    # Cell 3: Model definition (unchanged)
    new_cells.append(orig_model_cell)

    # Cell 4: Data pipeline
    new_cells.append(make_code_cell(data_src, cell_id="data_pipeline"))

    # Cell 5: Helpers + baseline
    new_cells.append(make_code_cell(ref_helpers_src, cell_id="helpers_baseline"))

    # Cell 6: Multi-run eval loop
    new_cells.append(make_code_cell(eval_loop_src, cell_id="eval_loop"))

    # Cell 7: Footer
    new_cells.append(make_code_cell(
        '# Notebook refactor complete.\n'
        '# Metrics are saved by the evaluation cell above.\n',
        cell_id="footer"
    ))

    nb['cells'] = new_cells
    save_notebook(nb, nb_path)
    return True


def main():
    ref = load_notebook(REFERENCE_PATH)

    success = 0
    failed = 0
    skipped = 0

    for filename, cfg in NOTEBOOKS.items():
        print(f"Refactoring {filename}...", end=" ")
        try:
            if refactor_notebook(filename, cfg, ref):
                print(f"✓ ({len(cfg['runs'])} runs)")
                success += 1
            else:
                skipped += 1
        except Exception as e:
            print(f"✗ Error: {e}")
            failed += 1

    print(f"\n{'=' * 60}")
    print(f"Refactoring complete:")
    print(f"  ✓ Success: {success}")
    print(f"  ⚠ Skipped: {skipped}")
    print(f"  ✗ Failed:  {failed}")
    print(f"  Total:     {success + skipped + failed}")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
