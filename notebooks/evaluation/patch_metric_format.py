"""
patch_metric_format.py
======================
Patches every evaluation notebook so that the per-variable metrics table
uses adaptive-precision formatting instead of the hard-coded {:8.4f} format.

Before:
    print(f"{label:<18} | {rmse:8.4f} | {mae:8.4f} | "
          f"{bias:+8.4f} | {corr:8.4f} | {rbase:14.4f} | {skill:+8.4f}")

After:
    print(f"{label:<18} | {_fv(rmse)} | {_fv(mae)} | "
          f"{_fv(bias,s=True)} | {_fv(corr)} | {_fv(rbase,w=14)} | {_fv(skill,s=True)}")

The helper _fv() uses scientific notation automatically for values whose
magnitude is below 1e-4, so values like 1.2345e-08 are displayed correctly
instead of being rounded to 0.0000.

Run once:
    python patch_metric_format.py
"""

import json
import os
import glob

# ── config ────────────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))

# The old print line (both parts)
OLD_PRINT_LINE1 = (
    '        print(f"{label:<18} | {rmse:8.4f} | {mae:8.4f} | "\n'
    '              f"{bias:+8.4f} | {corr:8.4f} | {rbase:14.4f} | {skill:+8.4f}")\n'
)
# variant without trailing newline on last line (some cells omit it)
OLD_PRINT_LINE1_ALT = (
    '        print(f"{label:<18} | {rmse:8.4f} | {mae:8.4f} | "\n'
    '              f"{bias:+8.4f} | {corr:8.4f} | {rbase:14.4f} | {skill:+8.4f}")'
)

# The new print line
NEW_PRINT_LINE = (
    '        print(f"{label:<18} | {_fv(rmse)} | {_fv(mae)} | "\n'
    '              f"{_fv(bias,s=True)} | {_fv(corr)} | {_fv(rbase,w=14)} | {_fv(skill,s=True)}")\n'
)

# The helper function to inject (inserted right before the metrics loop)
# Anchor: the line that starts the per-variable metrics print section
ANCHOR_LINE = '    # \U0001f4ca Print per-variable metrics'
ANCHOR_LINE_ALT = '    # Print per-variable metrics'  # fallback without emoji

HELPER_CODE = '''\
    # ── adaptive-precision formatter ─────────────────────────────────────────
    def _fv(v, w=12, s=False, sig=9):
        """Format metric value with adaptive precision / scientific notation."""
        import math
        if not math.isfinite(v):
            return f"{v:>{w}}"
        sign = '+' if s and v >= 0 else ''
        abs_v = abs(v)
        if abs_v == 0.0:
            return f"{'0':>{w}}"
        if abs_v < 1e-4:
            # scientific notation with sig-1 decimal places
            raw = f"{v:{sign}.{sig - 1}e}"
        else:
            # fixed notation: enough decimals for sig significant figures
            mag = int(math.floor(math.log10(abs_v)))
            dec = max(sig - mag - 1, sig)
            raw = f"{v:{sign}.{dec}f}"
        return f"{raw:>{w}}"

'''


# ── helpers ───────────────────────────────────────────────────────────────────

def load_nb(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_nb(nb, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

def cell_src(nb, idx):
    return ''.join(nb['cells'][idx].get('source', []))

def set_cell_src(nb, idx, text):
    lines = text.split('\n')
    result = []
    for i, line in enumerate(lines):
        result.append(line + '\n' if i < len(lines) - 1 else line)
    nb['cells'][idx]['source'] = result
    nb['cells'][idx]['outputs'] = []
    nb['cells'][idx]['execution_count'] = None


def patch_source(src: str) -> tuple[str, list[str]]:
    """
    Apply all patches to a single cell source string.
    Returns (patched_src, list_of_changes).
    """
    changes = []
    original = src

    # ── 1. Inject _fv helper (only if not already present) ───────────────────
    if '_fv(' not in src:
        # Try to find the "Print per-variable metrics" anchor
        anchor = None
        if ANCHOR_LINE in src:
            anchor = ANCHOR_LINE
        elif ANCHOR_LINE_ALT in src:
            anchor = ANCHOR_LINE_ALT
        elif '# Print per-variable metrics' in src:
            anchor = '# Print per-variable metrics'
        # generic fallback: inject before "hdr = ("
        elif '    hdr = (' in src:
            anchor = '    hdr = ('

        if anchor:
            idx = src.index(anchor)
            # Find start of the line containing the anchor
            line_start = src.rfind('\n', 0, idx) + 1
            src = src[:line_start] + HELPER_CODE + src[line_start:]
            changes.append('injected _fv() helper')
        else:
            changes.append('WARNING: could not find anchor to inject _fv()')
    else:
        changes.append('_fv() already present, skipped injection')

    # ── 2. Replace the fixed-format print line ────────────────────────────────
    replaced = False
    for old in (OLD_PRINT_LINE1, OLD_PRINT_LINE1_ALT):
        if old in src:
            src = src.replace(old, NEW_PRINT_LINE, 1)
            changes.append('replaced fixed-format print line')
            replaced = True
            break

    if not replaced:
        # Try a looser match on individual components
        if '{rmse:8.4f}' in src or '{mae:8.4f}' in src:
            src = src.replace('{rmse:8.4f}', '{_fv(rmse)}')
            src = src.replace('{mae:8.4f}',  '{_fv(mae)}')
            src = src.replace('{bias:+8.4f}', '{_fv(bias,s=True)}')
            src = src.replace('{corr:8.4f}',  '{_fv(corr)}')
            src = src.replace('{rbase:14.4f}', '{_fv(rbase,w=14)}')
            src = src.replace('{skill:+8.4f}', '{_fv(skill,s=True)}')
            changes.append('replaced individual :8.4f tokens')
            replaced = True

    if not replaced:
        changes.append('WARNING: print line not found – no replacement made')

    return src, changes


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    notebooks = sorted(glob.glob(os.path.join(BASE, '*_eval_*.ipynb')))
    if not notebooks:
        print('[WARN] No evaluation notebooks found.')
        return

    ok, warn, skip = 0, 0, 0

    for nb_path in notebooks:
        fname = os.path.basename(nb_path)
        try:
            nb = load_nb(nb_path)
            # The eval loop is always cell 6 in the refactored notebooks
            eval_cell_idx = None
            for i, cell in enumerate(nb['cells']):
                src = ''.join(cell.get('source', []))
                if '{rmse:8.4f}' in src or '_fv(' in src:
                    eval_cell_idx = i
                    break

            if eval_cell_idx is None:
                print(f'  [SKIP] {fname}: eval loop cell not found')
                skip += 1
                continue

            src = cell_src(nb, eval_cell_idx)
            new_src, changes = patch_source(src)

            has_warn = any('WARNING' in c for c in changes)

            if new_src != src:
                set_cell_src(nb, eval_cell_idx, new_src)
                save_nb(nb, nb_path)
                status = '[WARN]' if has_warn else '[OK]  '
                print(f'  {status} {fname} (cell {eval_cell_idx}): {"; ".join(changes)}')
                if has_warn:
                    warn += 1
                else:
                    ok += 1
            else:
                print(f'  [SKIP] {fname}: no changes needed')
                skip += 1

        except Exception as e:
            print(f'  [ERR]  {fname}: {e}')
            warn += 1

    print(f'\n{"=" * 70}')
    print(f'Patched : {ok}')
    print(f'Warnings: {warn}')
    print(f'Skipped : {skip}')
    print(f'Total   : {len(notebooks)}')
    print(f'{"=" * 70}')


if __name__ == '__main__':
    main()
