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


def infer_scenario(filename: str, text: str) -> str:
    match = SCENARIO_RE.search(text)
    if match:
        return match.group(1)
    if filename.lower().startswith("sc1_"):
        return "scenario1"
    if filename.lower().startswith("sc2_"):
        return "scenario2"
    return "unknown"


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


def parse_metric_rows(nb_json: dict):
    rows = []
    seen = set()

    for cell in nb_json.get("cells", []):
        for output in cell.get("outputs", []):
            text = get_output_text(output)
            if not text:
                continue

            for line in text.splitlines():
                line = line.rstrip()
                match = ROW_RE.match(line)
                if not match:
                    continue

                key = line.strip()
                if key in seen:
                    continue
                seen.add(key)

                rows.append(
                    {
                        "variable": match.group("variable").strip(),
                        "rmse": float(match.group("rmse")),
                        "mae": float(match.group("mae")),
                        "bias": float(match.group("bias")),
                        "corr": float(match.group("corr")),
                        "baseline_rmse": float(match.group("baseline_rmse")),
                        "skill": float(match.group("skill")),
                        "raw": line,
                    }
                )
    return rows


def parse_notebook(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    nb_json = json.loads(text)

    return {
        "filename": path.name,
        "scenario": infer_scenario(path.name, text),
        "dataset": "2 dataset",
        "model": infer_model(path.name),
        "loss_function": infer_loss(text),
        "metrics": parse_metric_rows(nb_json),
    }


def write_json(records, out_path: Path):
    out_path.write_text(json.dumps(records, indent=2), encoding="utf-8")


def write_csv(records, out_path: Path):
    fields = [
        "filename",
        "scenario",
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
                        "scenario": rec["scenario"],
                        "dataset": rec["dataset"],
                        "model": rec["model"],
                        "loss_function": rec["loss_function"],
                    }
                )
                continue
            for row in rec["metrics"]:
                writer.writerow(
                    {
                        "filename": rec["filename"],
                        "scenario": rec["scenario"],
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
        lines.append(rec["dataset"])
        lines.append(f"Model : {rec['model']}")
        lines.append(f"Loss function : {rec['loss_function']}")
        lines.append("=" * 100)
        lines.append("Variable           |     RMSE |      MAE |     Bias |     Corr |  Baseline RMSE |    Skill")
        lines.append("=" * 100)

        if rec["metrics"]:
            for m in rec["metrics"]:
                lines.append(
                    f"{m['variable']:<18} | {m['rmse']:8.4f} | {m['mae']:8.4f} | {m['bias']:+8.4f} | {m['corr']:8.4f} | {m['baseline_rmse']:14.4f} | {m['skill']:+8.4f}"
                )
        else:
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
