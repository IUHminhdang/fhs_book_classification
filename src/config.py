from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

DEFAULT_SEED = 42
TARGET_COLUMN = "category_norm"
INPUT_COLUMNS = ["title", "description"]
REQUIRED_COLUMNS = [
    "title",
    "description",
    "price",
    "author",
    "thumbnail",
    "product_code",
    "supplier",
    "translator",
    "publisher",
    "publish_year",
    "language",
    "weight_grams",
    "num_pages",
    "book_format",
    "category_norm",
    "language_norm",
    "description_word_count",
    "missing_description",
    "series_name",
    "volume_number",
    "price_bucket",
    "pages_bucket",
]

EXPERIMENTS = {
    "tfidf_lr": {"name": "TF-IDF + Logistic Regression", "trainable": True},
    "tfidf_svm": {"name": "TF-IDF + LinearSVC", "trainable": True},
    "fasttext_bilstm": {"name": "FastText + BiLSTM", "trainable": True},
    "fasttext_bigru": {"name": "FastText + BiGRU", "trainable": True},
    "minilm_lr": {"name": "MiniLM + Logistic Regression", "trainable": True},
    "minilm_svm": {"name": "MiniLM + LinearSVC", "trainable": True},
    "distilbert": {"name": "DistilBERT fine-tuning", "trainable": True},
    "mbert": {"name": "mBERT fine-tuning", "trainable": True},
    "phobert": {"name": "PhoBERT fine-tuning", "trainable": True},
    "phobert_nli_vi": {"name": "PhoBERT-NLI-VI zero-shot", "trainable": False},
}


@dataclass
class RunConfig:
    experiment_id: str = "tfidf_lr"
    data_dir: Path = Path("data")
    splits_dir: Path = Path("data/splits")
    output_dir: Path = Path("outputs")
    cache_dir: Path = Path(".cache")
    seed: int = DEFAULT_SEED
    resume: bool = False
    device: str = "auto"
    config_name: str = "default"
    extra: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "data_dir": str(self.data_dir),
            "splits_dir": str(self.splits_dir),
            "output_dir": str(self.output_dir),
            "cache_dir": str(self.cache_dir),
            "seed": self.seed,
            "resume": self.resume,
            "device": self.device,
            "config_name": self.config_name,
            "extra": self.extra,
        }


def resolve_data_dir(path: Optional[str | Path] = None) -> Path:
    if path is None:
        return Path("data")
    return Path(path)


def resolve_output_dir(path: Optional[str | Path] = None) -> Path:
    if path is None:
        return Path("outputs")
    return Path(path)
