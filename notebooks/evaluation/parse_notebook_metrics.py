import argparse
import csv
import json
import re
from pathlib import Path


SCENARIO_RE = re.compile(r"['\"]scenario['\"]\s*:\s*['\"]([^'\"]+)['\"]")
COMBINED_LOSS_RE = re.compile(
    r"criterion\s*=\s*CombinedLoss\(\s*mode=['\"]([^'\"]+)['\"](?:\s*,\s*alpha=([^,\)]+)\s*,\s*beta=([^\)]+))?\s*\)",
    re.IGNORECASE,
)
GENERIC_LOSS_RE = re.compile(r"\bloss\s*=\s*['\"]([^'\"]+)['\"]", re.IGNORECASE)
ROW_RE = re.compile(
    r"^\s*(?P<variable>[^|]+?)\s*\|\s*(?P<rmse>[+-]?\d*\.?\d+)\s*\|\s*(?P<mae>[+-]?\d*\.?\d+)\s*\|\s*(?P<bias>[+-]?\d*\.?\d+)\s*\|\s*(?P<corr>[+-]?\d*\.?\d+)\s*\|\s*(?P<baseline_rmse>[+-]?\d*\.?\d+)\s*\|\s*(?P<skill>[+-]?\d*\.?\d+)\s*$"
)


def infer_model(filename: str) -> str:
    name = filename.lower()
    if "convnext" in name or "covnext" in name:
        return "ConvNeXt"
    if "resnet18" in name:
        return "ResNet18"
    if "unet" in name:
        return "Unet"
    if "gan" in name:
        return "GAN"
    if "dl4ds" in name:
        return "DL4DS"
    if "qm" in name:
        return "Quantile Mapping"
    return "Unknown"


def infer_run_number(filename: str):
    name = filename.lower()
    match = re.search(r"(?:^|[_\-])run[_\-]?(\d+)(?:[_\-\.]|$)", name)
    if match:
        return int(match.group(1))

    # Fallback for filenames like "..._r3.ipynb"
    match = re.search(r"(?:^|[_\-])r(\d+)(?:[_\-\.]|$)", name)
    if match:
        return int(match.group(1))

    return None


def infer_scenario(filename: str, text: str) -> str:
    match = SCENARIO_RE.search(text)
    if match:
        return match.group(1)
    if filename.lower().startswith("sc1_"):
        return "scenario1"
    if filename.lower().startswith("sc2_"):
        return "scenario2"
    return "unknown"


def infer_preupsample(filename: str) -> str:
    return "preupsample" if "preupsample" in filename.lower() else "no-preupsample"


def infer_loss(text: str) -> str:
    m = COMBINED_LOSS_RE.search(text)
    if m:
        mode = m.group(1)
        alpha = m.group(2)
        beta = m.group(3)
        if alpha and beta:
            return f"{mode} (alpha={alpha.strip()}; beta={beta.strip()})"
        return mode

    m = GENERIC_LOSS_RE.search(text)
    if m:
        return m.group(1)
    return "n/a"


def get_output_text(output: dict) -> str:
    output_type = output.get("output_type", "")
    if output_type == "stream":
        text = output.get("text", "")
        if isinstance(text, list):
            return "".join(text)
        return str(text)

    if output_type in {"execute_result", "display_data"}:
        data = output.get("data", {})
        plain = data.get("text/plain", "")
        if isinstance(plain, list):
            return "".join(plain)
        return str(plain)

    return ""


RUN_HEADER_RE = re.compile(r"RUN\s+(\d+)/(\d+):\s*(.*)")


def parse_metric_rows(nb_json: dict):
    rows = []

    # Track current run context across all outputs in the cell
    current_run_number = None
    current_run_path = None

    for cell in nb_json.get("cells", []):
        for output in cell.get("outputs", []):
            text = get_output_text(output)
            if not text:
                continue

            for line in text.splitlines():
                line = line.rstrip()

                # Detect RUN header to track which run we are in
                run_match = RUN_HEADER_RE.search(line)
                if run_match:
                    current_run_number = int(run_match.group(1))
                    current_run_path = run_match.group(3).strip()
                    continue

                match = ROW_RE.match(line)
                if not match:
                    continue

                rows.append(
                    {
                        "variable": match.group("variable").strip(),
                        "rmse": float(match.group("rmse")),
                        "mae": float(match.group("mae")),
                        "bias": float(match.group("bias")),
                        "corr": float(match.group("corr")),
                        "baseline_rmse": float(match.group("baseline_rmse")),
                        "skill": float(match.group("skill")),
                        "run_number": current_run_number,
                        "run_path": current_run_path,
                        "raw": line,
                    }
                )
    return rows


def parse_notebook(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    nb_json = json.loads(text)
    scenario = infer_scenario(path.name, text)
    dataset = "2 dataset" if scenario.lower().startswith("scenario1") else "1 dataset"

    return {
        "filename": path.name,
        "run_number": infer_run_number(path.name),
        "scenario": scenario,
        "preupsample": infer_preupsample(path.name),
        "dataset": dataset,
        "model": infer_model(path.name),
        "loss_function": infer_loss(text),
        "metrics": parse_metric_rows(nb_json),
    }


def write_json(records, out_path: Path):
    out_path.write_text(json.dumps(records, indent=2), encoding="utf-8")


def write_csv(records, out_path: Path):
    fields = [
        "filename",
        "run_number",
        "run_path",
        "scenario",
        "preupsample",
        "dataset",
        "model",
        "loss_function",
        "variable",
        "rmse",
        "mae",
        "bias",
        "corr",
        "baseline_rmse",
        "skill",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for rec in records:
            if not rec["metrics"]:
                writer.writerow(
                    {
                        "filename": rec["filename"],
                        "run_number": rec["run_number"],
                        "run_path": "",
                        "scenario": rec["scenario"],
                        "preupsample": rec["preupsample"],
                        "dataset": rec["dataset"],
                        "model": rec["model"],
                        "loss_function": rec["loss_function"],
                    }
                )
                continue
            for row in rec["metrics"]:
                # Prefer per-metric run_number (from RUN header), fall back to filename
                run_num = row.get("run_number") if row.get("run_number") is not None else rec["run_number"]
                writer.writerow(
                    {
                        "filename": rec["filename"],
                        "run_number": run_num,
                        "run_path": row.get("run_path", ""),
                        "scenario": rec["scenario"],
                        "preupsample": rec["preupsample"],
                        "dataset": rec["dataset"],
                        "model": rec["model"],
                        "loss_function": rec["loss_function"],
                        "variable": row["variable"],
                        "rmse": row["rmse"],
                        "mae": row["mae"],
                        "bias": row["bias"],
                        "corr": row["corr"],
                        "baseline_rmse": row["baseline_rmse"],
                        "skill": row["skill"],
                    }
                )


def write_markdown(records, out_path: Path):
    lines = [
        "# Notebook Metrics Summary",
        "",
        f"Total notebooks parsed: {len(records)}",
        "",
    ]

    for rec in records:
        lines.append(f"Scenario : {rec['scenario']}")
        lines.append(f"Filename : {rec['filename']}")
        lines.append(f"Preupsample : {rec['preupsample']}")
        lines.append(rec["dataset"])
        lines.append(f"Model : {rec['model']}")
        lines.append(f"Loss function : {rec['loss_function']}")

        if rec["metrics"]:
            # Group metrics by run_number for multi-run notebooks
            from itertools import groupby
            def run_key(m):
                return (m.get("run_number"), m.get("run_path"))

            for (run_num, run_path), group in groupby(rec["metrics"], key=run_key):
                run_label = f"RUN {run_num}: {run_path}" if run_num is not None else (
                    f"Run number : {rec['run_number'] if rec['run_number'] is not None else 'n/a'}"
                )
                lines.append("=" * 100)
                lines.append(f"  {run_label}")
                lines.append("=" * 100)
                lines.append("Variable           |     RMSE |      MAE |     Bias |     Corr |  Baseline RMSE |    Skill")
                lines.append("-" * 100)
                for m in group:
                    lines.append(
                        f"{m['variable']:<18} | {m['rmse']:8.4f} | {m['mae']:8.4f} | {m['bias']:+8.4f} | {m['corr']:8.4f} | {m['baseline_rmse']:14.4f} | {m['skill']:+8.4f}"
                    )
                lines.append("-" * 100)
        else:
            lines.append("=" * 100)
            lines.append(f"Run number : {rec['run_number'] if rec['run_number'] is not None else 'n/a'}")
            lines.append("=" * 100)
            lines.append("No saved metric table rows found in notebook outputs.")
            lines.append("=" * 100)
        lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Parse metric tables from notebook outputs.")
    parser.add_argument("--input-dir", default=".", help="Directory to scan recursively for .ipynb files")
    parser.add_argument("--output-prefix", default="parsed_notebook_metrics", help="Prefix for output files")
    args = parser.parse_args()

    input_dir = Path(args.input_dir).resolve()
    notebooks = sorted(input_dir.rglob("*.ipynb"))

    records = [parse_notebook(nb) for nb in notebooks]

    out_json = input_dir / f"{args.output_prefix}.json"
    out_csv = input_dir / f"{args.output_prefix}.csv"
    out_md = input_dir / f"{args.output_prefix}.md"

    write_json(records, out_json)
    write_csv(records, out_csv)
    write_markdown(records, out_md)

    with_rows = sum(1 for r in records if r["metrics"])
    print(f"Parsed notebooks: {len(records)}")
    print(f"Notebooks with metric rows: {with_rows}")
    print(f"JSON: {out_json}")
    print(f"CSV: {out_csv}")
    print(f"Markdown: {out_md}")


if __name__ == "__main__":
    main()
