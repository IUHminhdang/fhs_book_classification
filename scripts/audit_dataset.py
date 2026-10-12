from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.config import TARGET_COLUMN
from src.data_validation import validate_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit Fahasa split files.")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--output-dir", type=str, default="outputs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    splits = validate_dataset(data_dir)
    summary = {}
    for key, df in splits.items():
        summary[key] = {
            "rows": int(len(df)),
            "columns": list(df.columns),
            "target_unique": int(df[TARGET_COLUMN].nunique()),
            "missing_values": df.isna().sum().to_dict(),
            "label_counts": df[TARGET_COLUMN].astype(str).value_counts().to_dict(),
        }

    report = output_dir / "dataset_audit.json"
    report.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": "ok", "report": str(report)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
