import json
import os
import sys

sys.stdout.reconfigure(encoding='utf-8')

base = os.path.dirname(os.path.abspath(__file__))

for name in ['sc2_eval_unet_mae.ipynb']:
    path = os.path.join(base, name)
    with open(path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    print(f'\n=== {name} ({len(nb["cells"])} cells) ===')
    for i, cell in enumerate(nb['cells']):
        if i < 3:  # skip cells we already saw
            continue
        src = ''.join(cell.get('source', []))
        lines = src.split('\n')
        cell_id = cell.get('id', f'cell_{i}')
        print(f'\n--- Cell {i} (id={cell_id}, type={cell["cell_type"]}, lines={len(lines)}) ---')
        for j, line in enumerate(lines[:60]):
            print(f'  {j+1:3d}| {line}')
        if len(lines) > 60:
            print(f'  ... ({len(lines) - 60} more lines)')
