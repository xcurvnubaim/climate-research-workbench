"""Verify refactored notebook structure."""
import json, os, sys
sys.stdout.reconfigure(encoding='utf-8')

path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sc2_eval_unet_mae.ipynb')
with open(path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

print(f"=== sc2_eval_unet_mae.ipynb ({len(nb['cells'])} cells) ===")
for i, cell in enumerate(nb['cells']):
    src = ''.join(cell.get('source', []))
    lines = src.split('\n')
    cell_id = cell.get('id', f'cell_{i}')
    print(f"\n--- Cell {i} (id={cell_id}, type={cell['cell_type']}, lines={len(lines)}) ---")
    for j, line in enumerate(lines[:15]):
        print(f"  {j+1:3d}| {line}")
    if len(lines) > 15:
        print(f"  ... ({len(lines) - 15} more lines)")
