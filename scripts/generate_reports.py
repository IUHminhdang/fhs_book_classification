from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate benchmark reports from experiment results.")
    parser.add_argument("--results-dir", type=str, default="outputs/results")
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--zip-output", type=str, default="", help="Optional zip archive path to write.")
    return parser.parse_args()


def _safe_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _extract_metrics(result: dict) -> dict:
    metrics = result.get("metrics", {})
    return metrics.get("test", {}) or metrics.get("val", {}) or {}


def _flatten_per_class(results: list[dict]) -> list[dict]:
    rows: list[dict] = []
    for result in results:
        experiment_id = result.get("experiment_id", "unknown")
        metrics = result.get("metrics", {})
        test_metrics = metrics.get("test") or metrics.get("val") or {}
        for item in test_metrics.get("per_class", []):
            rows.append(
                {
                    "experiment_id": experiment_id,
                    "label": item.get("label"),
                    "precision": item.get("precision"),
                    "recall": item.get("recall"),
                    "f1": item.get("f1"),
                    "support": item.get("support"),
                }
            )
    return rows


def _write_confusion_matrices(results: list[dict], output_dir: Path) -> None:
    matrix_dir = output_dir / "confusion_matrices"
    matrix_dir.mkdir(parents=True, exist_ok=True)
    for result in results:
        experiment_id = result.get("experiment_id", "unknown")
        metrics = result.get("metrics", {})
        cm = (metrics.get("test") or metrics.get("val") or {}).get("confusion_matrix")
        labels = result.get("labels") or []
        if not cm:
            continue
        frame = pd.DataFrame(cm, index=labels, columns=labels)
        frame.to_csv(matrix_dir / f"{experiment_id}.csv", index=True)


def _plot_summary(summary: pd.DataFrame, output_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    for metric in ["accuracy", "macro_f1", "weighted_f1", "fps", "training_time", "inference_time"]:
        frame = summary[summary[metric].notna()].copy()
        if frame.empty:
            continue
        plt.figure(figsize=(10, 6))
        plt.bar(frame["experiment_id"].astype(str), frame[metric].astype(float), color="#4C78A8")
        plt.xticks(rotation=25, ha="right")
        plt.ylabel(metric.replace("_", " ").title())
        plt.title(f"{metric.replace('_', ' ').title()} by experiment")
        plt.tight_layout()
        plt.savefig(plots_dir / f"{metric}_comparison.png", dpi=200)
        plt.close()

    per_class = pd.read_csv(output_dir / "per_class_comparison.csv") if (output_dir / "per_class_comparison.csv").exists() else pd.DataFrame()
    if not per_class.empty:
        pivot = per_class.pivot_table(index="label", columns="experiment_id", values="f1", aggfunc="mean")
        plt.figure(figsize=(10, 8))
        import seaborn as sns
        sns.heatmap(pivot, cmap="viridis", annot=True, fmt=".2f")
        plt.title("Per-class F1 by experiment")
        plt.tight_layout()
        plt.savefig(plots_dir / "per_class_f1_heatmap.png", dpi=200)
        plt.close()


def _write_report(summary: pd.DataFrame, output_dir: Path) -> None:
    completed = summary[summary["status"].eq("completed")].copy()
    if completed.empty:
        report = "# Benchmark report\n\nNo completed experiments were found.\n"
    else:
        completed = completed.sort_values(["macro_f1"], ascending=False, na_position="last")
        best = completed.iloc[0]
        winner = best["experiment_id"]
        report = [
            "# Benchmark report",
            "",
            f"Best supervised model by validation Macro-F1: **{winner}**",
            "",
            f"- Validation Macro-F1: **{best.get('macro_f1', 'n/a')}**",
            f"- Test Accuracy: **{best.get('accuracy', 'n/a')}**",
            f"- Test Macro-F1: **{best.get('macro_f1', 'n/a')}**",
            f"- FPS: **{best.get('fps', 'n/a')}**",
            f"- Training time: **{best.get('training_time', 'n/a')}s**",
            f"- Inference time: **{best.get('inference_time', 'n/a')}s**",
            "",
            "## Notes",
            "- Results are reported only for experiments that actually completed.",
            "- Test metrics were not used for checkpoint selection; validation Macro-F1 was used for supervised model ranking.",
            "- Incomplete or dependency-missing runs are documented explicitly and excluded from model ranking.",
        ]
        report = "\n".join(report)
    (output_dir / "report.md").write_text(report + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for result_path in sorted(results_dir.glob("*.json")):
        try:
            with result_path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
            records.append(data)
        except Exception:
            continue

    if not records:
        summary = pd.DataFrame(columns=["experiment_id", "status", "accuracy", "macro_f1", "weighted_f1", "micro_f1", "training_time", "inference_time", "fps", "parameter_count", "best_epoch", "seed", "input_columns"])
        summary.to_csv(output_dir / "summary.csv", index=False)
        pd.DataFrame(columns=["experiment_id", "label", "precision", "recall", "f1", "support"]).to_csv(output_dir / "per_class_comparison.csv", index=False)
        pd.DataFrame(columns=["epoch", "loss", "val_loss", "val_accuracy", "val_macro_f1", "experiment_id"]).to_csv(output_dir / "training_history.csv", index=False)
        _write_report(summary, output_dir)
        print(json.dumps({"status": "ok", "experiments": 0, "summary": str(output_dir / "summary.csv")}, ensure_ascii=False, indent=2))
        return

    summary_rows: list[dict] = []
    for result in records:
        experiment_id = result.get("experiment_id", "unknown")
        metrics = result.get("metrics", {})
        test_metrics = metrics.get("test") or metrics.get("val") or {}
        val_metrics = metrics.get("val") or {}
        summary_rows.append(
            {
                "experiment_id": experiment_id,
                "status": result.get("status", "unknown"),
                "accuracy": _safe_float(test_metrics.get("accuracy", val_metrics.get("accuracy"))),
                "macro_f1": _safe_float(test_metrics.get("macro_f1", val_metrics.get("macro_f1"))),
                "weighted_f1": _safe_float(test_metrics.get("weighted_f1", val_metrics.get("weighted_f1"))),
                "micro_f1": _safe_float(test_metrics.get("micro_f1", val_metrics.get("micro_f1"))),
                "training_time": _safe_float(result.get("training_time")),
                "inference_time": _safe_float(result.get("inference_time")),
                "fps": _safe_float(result.get("fps")),
                "parameter_count": _safe_float(result.get("parameter_count")),
                "best_epoch": result.get("best_epoch"),
                "seed": result.get("seed", 42),
                "input_columns": ",".join(result.get("input_columns", ["title", "description"])),
            }
        )

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_dir / "summary.csv", index=False)

    per_class = pd.DataFrame(_flatten_per_class(records))
    per_class.to_csv(output_dir / "per_class_comparison.csv", index=False)

    history_rows: list[dict] = []
    for result in records:
        history = result.get("history") or result.get("training_history") or []
        for row in history:
            history_rows.append({"experiment_id": result.get("experiment_id", "unknown"), **row})
    pd.DataFrame(history_rows).to_csv(output_dir / "training_history.csv", index=False)

    _write_confusion_matrices(records, output_dir)
    _plot_summary(summary, output_dir)
    _write_report(summary, output_dir)

    if args.zip_output:
        import zipfile
        zip_path = Path(args.zip_output)
        zip_path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for path in [output_dir / "summary.csv", output_dir / "per_class_comparison.csv", output_dir / "training_history.csv", output_dir / "report.md"]:
                if path.exists():
                    zf.write(path, arcname=path.name)
            plots_dir = output_dir / "plots"
            if plots_dir.exists():
                for file in sorted(plots_dir.glob("*.png")):
                    zf.write(file, arcname=f"plots/{file.name}")
            matrices_dir = output_dir / "confusion_matrices"
            if matrices_dir.exists():
                for file in sorted(matrices_dir.glob("*.csv")):
                    zf.write(file, arcname=f"confusion_matrices/{file.name}")

    print(json.dumps({"status": "ok", "experiments": len(records), "summary": str(output_dir / "summary.csv")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
