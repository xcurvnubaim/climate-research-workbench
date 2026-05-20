"""Show lines 28-60 of the eval loop in detail."""
import json, os, sys
sys.stdout.reconfigure(encoding='utf-8')

ref_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sc2_eval_gan.ipynb')
with open(ref_path, 'r', encoding='utf-8') as f:
    ref = json.load(f)

src = ''.join(ref['cells'][6].get('source', []))
lines = src.split('\n')
for i in range(27, 80):
    print(f"{i+1:4d}| {lines[i]}")
