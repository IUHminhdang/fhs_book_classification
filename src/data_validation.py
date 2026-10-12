from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

from .config import REQUIRED_COLUMNS, TARGET_COLUMN


def _resolve_split_path(data_dir: str | Path, split_name: str) -> Path:
    data_dir = Path(data_dir)
    canonical = f"{split_name}.csv"
    direct = data_dir / canonical
    if direct.exists():
        return direct
    aliases = {
        "val": ["validation.csv", "val.csv"],
        "validation": ["val.csv", "validation.csv"],
    }
    for alias in aliases.get(split_name, []):
        candidate = data_dir / alias
        if candidate.exists():
            return candidate
    return direct


def load_split_csv(data_dir: str | Path, split_name: str) -> pd.DataFrame:
    path = _resolve_split_path(data_dir, split_name)
    if not path.exists():
        raise FileNotFoundError(f"Missing split file: {path}")
    df = pd.read_csv(path, encoding="utf-8")
    return df


def validate_split_schema(df: pd.DataFrame, split_name: str) -> None:
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"{split_name} is missing required columns: {missing}")
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"{split_name} has no target column '{TARGET_COLUMN}'")


def check_target_distribution(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    train_labels = set(train_df[TARGET_COLUMN].dropna().astype(str))
    val_missing = sorted(set(val_df[TARGET_COLUMN].dropna().astype(str)) - train_labels)
    test_missing = sorted(set(test_df[TARGET_COLUMN].dropna().astype(str)) - train_labels)
    if val_missing or test_missing:
        raise ValueError(
            "Validation/test labels absent from training: "
            f"val_missing={val_missing[:10]}, test_missing={test_missing[:10]}"
        )


def validate_section_quality(df: pd.DataFrame, split_name: str) -> None:
    missing_count = int(df[TARGET_COLUMN].isna().sum())
    if missing_count:
        raise ValueError(f"{split_name} target column has {missing_count} missing labels")

    missing_text = int(df[["title", "description"]].isna().sum().sum())
    if missing_text:
        print(f"Warning: {split_name} text inputs contain missing values; these will be filled with empty strings.")


def validate_dataset(data_dir: str | Path, required_splits: Iterable[str] = ("train", "val", "test")) -> dict:
    data_dir = Path(data_dir)
    result = {}
    for split_name in required_splits:
        path = _resolve_split_path(data_dir, split_name)
        if not path.exists():
            raise FileNotFoundError(f"Expected split file: {path}")
        df = pd.read_csv(path, encoding="utf-8")
        validate_split_schema(df, split_name)
        validate_section_quality(df, split_name)
        result[split_name] = df

    check_target_distribution(result["train"], result["val"], result["test"])
    return result


def summarize_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    return df.isna().sum().rename("missing_count").to_frame()
