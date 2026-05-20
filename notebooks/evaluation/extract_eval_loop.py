"""Extract eval loop cell from reference notebook for analysis."""
import json, os, sys
sys.stdout.reconfigure(encoding='utf-8')

ref_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sc2_eval_gan.ipynb')
with open(ref_path, 'r', encoding='utf-8') as f:
    ref = json.load(f)

# Cell 6 is the eval loop
cell = ref['cells'][6]
src = ''.join(cell.get('source', []))
lines = src.split('\n')
print(f"Eval loop cell: {len(lines)} lines")
print("=" * 80)
for i, line in enumerate(lines):
    print(f"{i+1:4d}| {line}")
