from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

from .config import EXPERIMENTS, RunConfig, TARGET_COLUMN
from .data_validation import load_split_csv, validate_dataset
from .text_utils import combine_text_fields, segment_vietnamese_text


def _ensure_split_compatibility(data_dir: str | Path) -> dict:
    splits = validate_dataset(data_dir, required_splits=("train", "val", "test"))
    # the user-provided data contains validation.csv, but the contract expects val.csv. Create compatibility aliases.
    if "val" not in splits and "validation" in splits:
        splits["val"] = splits["validation"]
    return splits


def _build_text_frame(df: pd.DataFrame) -> pd.Series:
    return df.apply(lambda row: combine_text_fields(row.get("title", ""), row.get("description", "")), axis=1).fillna("")


def _prepare_split_folders(data_dir: str | Path) -> None:
    data_dir = Path(data_dir)
    splits_dir = data_dir / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    for candidate in ("train", "validation", "test"):
        src = data_dir / f"{candidate}.csv"
        if src.exists() and not (splits_dir / f"{candidate}.csv").exists():
            pd.read_csv(src, encoding="utf-8").to_csv(splits_dir / f"{candidate}.csv", index=False)
    if not (splits_dir / "val.csv").exists() and (splits_dir / "validation.csv").exists():
        pd.read_csv(splits_dir / "validation.csv", encoding="utf-8").to_csv(splits_dir / "val.csv", index=False)
    if not (splits_dir / "train.csv").exists() and (data_dir / "train.csv").exists():
        pd.read_csv(data_dir / "train.csv", encoding="utf-8").to_csv(splits_dir / "train.csv", index=False)
    if not (splits_dir / "test.csv").exists() and (data_dir / "test.csv").exists():
        pd.read_csv(data_dir / "test.csv", encoding="utf-8").to_csv(splits_dir / "test.csv", index=False)


def _compute_metrics(y_true: pd.Series, y_pred: pd.Series, labels: list[str]) -> Dict[str, Any]:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        average=None,
        zero_division=0,
    )
    per_class = []
    for idx, label in enumerate(labels):
        per_class.append(
            {
                "label": label,
                "precision": float(precision[idx]),
                "recall": float(recall[idx]),
                "f1": float(f1[idx]),
                "support": int(support[idx]),
            }
        )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "per_class": per_class,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def _serialize_result(result: Dict[str, Any]) -> Dict[str, Any]:
    if isinstance(result.get("metrics"), dict):
        result["metrics"] = result["metrics"]
    return result


def _run_tfidf_model(model_name: str, train_texts: pd.Series, val_texts: pd.Series, test_texts: pd.Series, y_train: pd.Series, y_val: pd.Series, y_test: pd.Series, labels: list[str]) -> Dict[str, Any]:
    word_vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=50000, sublinear_tf=True)
    char_vectorizer = TfidfVectorizer(ngram_range=(3, 5), analyzer="char_wb", min_df=2, max_features=30000, sublinear_tf=True)
    feature_union = FeatureUnion([("word", word_vectorizer), ("char", char_vectorizer)])
    if model_name == "tfidf_lr":
        classifier = LogisticRegression(max_iter=2500, class_weight="balanced")
    else:
        classifier = LinearSVC(C=1.0, class_weight="balanced")

    pipeline = Pipeline([("features", feature_union), ("clf", classifier)])
    pipeline.fit(train_texts.astype(str), y_train.astype(str))
    val_pred = pipeline.predict(val_texts.astype(str))
    test_pred = pipeline.predict(test_texts.astype(str))
    val_metrics = _compute_metrics(y_val.astype(str), pd.Series(val_pred), labels)
    test_metrics = _compute_metrics(y_test.astype(str), pd.Series(test_pred), labels)
    return {
        "model_id": model_name,
        "status": "completed",
        "metrics": {"val": val_metrics, "test": test_metrics},
        "labels": labels,
    }


def _run_minilm_model(model_name: str, train_texts: pd.Series, val_texts: pd.Series, test_texts: pd.Series, y_train: pd.Series, y_val: pd.Series, y_test: pd.Series, labels: list[str], cache_dir: str | Path) -> Dict[str, Any]:
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as exc:  # pragma: no cover
        return {"model_id": model_name, "status": "dependency_missing", "notes": f"sentence-transformers unavailable: {exc}"}

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    model_name_ = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    encoder = SentenceTransformer(model_name_)
    cache_path = cache_dir / "minilm_embeddings.joblib"
    model = {"train": train_texts.astype(str).tolist(), "val": val_texts.astype(str).tolist(), "test": test_texts.astype(str).tolist()}
    if cache_path.exists():
        saved = joblib.load(cache_path)
        train_emb = saved.get("train")
        val_emb = saved.get("val")
        test_emb = saved.get("test")
    else:
        train_emb = encoder.encode(train_texts.astype(str).tolist(), convert_to_numpy=True, show_progress_bar=False)
        val_emb = encoder.encode(val_texts.astype(str).tolist(), convert_to_numpy=True, show_progress_bar=False)
        test_emb = encoder.encode(test_texts.astype(str).tolist(), convert_to_numpy=True, show_progress_bar=False)
        joblib.dump({"train": train_emb, "val": val_emb, "test": test_emb}, cache_path)

    classifier = LogisticRegression(max_iter=2000, class_weight="balanced") if model_name == "minilm_lr" else LinearSVC(C=1.0, class_weight="balanced")
    classifier.fit(train_emb, y_train.astype(str))
    val_pred = classifier.predict(val_emb)
    test_pred = classifier.predict(test_emb)
    val_metrics = _compute_metrics(y_val.astype(str), pd.Series(val_pred), labels)
    test_metrics = _compute_metrics(y_test.astype(str), pd.Series(test_pred), labels)
    return {"model_id": model_name, "status": "completed", "metrics": {"val": val_metrics, "test": test_metrics}, "labels": labels}


def _run_fasttext_model(model_name: str, train_texts: pd.Series, val_texts: pd.Series, test_texts: pd.Series, y_train: pd.Series, y_val: pd.Series, y_test: pd.Series, labels: list[str]) -> Dict[str, Any]:
    try:
        from gensim.models import FastText
    except Exception as exc:  # pragma: no cover - dependency missing
        return {"model_id": model_name, "status": "dependency_missing", "notes": f"gensim unavailable: {exc}"}

    all_text = pd.concat([train_texts, val_texts, test_texts], ignore_index=True)
    token_corpus = [segment_vietnamese_text(text).split() for text in all_text.astype(str)]
    model = FastText(vector_size=100, window=5, min_count=2, workers=1, sg=1)
    model.build_vocab(token_corpus)
    model.train(token_corpus, total_examples=len(token_corpus), epochs=10)

    def embed_texts(texts: pd.Series) -> list[list[float]]:
        vectors = []
        for text in texts.astype(str):
            tokens = segment_vietnamese_text(text).split()
            if not tokens:
                vectors.append([0.0] * 100)
                continue
            word_vectors = [model.wv[token] for token in tokens if token in model.wv]
            if not word_vectors:
                vectors.append([0.0] * 100)
            else:
                vectors.append([float(v) for v in sum(word_vectors) / len(word_vectors)])
        return vectors

    X_train = embed_texts(train_texts)
    X_val = embed_texts(val_texts)
    X_test = embed_texts(test_texts)

    import numpy as np
    from sklearn.linear_model import LogisticRegression

    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf.fit(np.asarray(X_train), y_train.astype(str))
    val_pred = clf.predict(np.asarray(X_val))
    test_pred = clf.predict(np.asarray(X_test))
    val_metrics = _compute_metrics(y_val.astype(str), pd.Series(val_pred), labels)
    test_metrics = _compute_metrics(y_test.astype(str), pd.Series(test_pred), labels)
    return {"model_id": model_name, "status": "completed", "metrics": {"val": val_metrics, "test": test_metrics}, "labels": labels}


def _inspect_zero_shot_model(model_id: str) -> Dict[str, Any]:
    try:
        from transformers import AutoConfig
    except Exception as exc:  # pragma: no cover
        return {"status": "dependency_missing", "notes": f"transformers unavailable: {exc}"}

    try:
        config = AutoConfig.from_pretrained(model_id)
    except Exception as exc:
        return {"status": "model_unavailable", "notes": f"Could not load model config for {model_id}: {exc}"}

    arch = getattr(config, "architectures", [])
    number_of_labels = getattr(config, "num_labels", None)
    info = {
        "architectures": arch,
        "num_labels": number_of_labels,
        "model_type": getattr(config, "model_type", None),
    }
    if not arch or not any("SequenceClassification" in str(item) for item in arch):
        return {"status": "unsupported_model", "notes": "This checkpoint is not a sequence-classification model suitable for zero-shot classification.", "model_info": info}
    if number_of_labels is not None and number_of_labels < 3:
        return {"status": "unsupported_model", "notes": "This checkpoint is not configured for multi-class NLI-style zero-shot classification.", "model_info": info}
    return {"status": "compatible", "model_info": info}


def _run_phobert_nli_vi(train_texts: pd.Series, val_texts: pd.Series, test_texts: pd.Series, y_train: pd.Series, y_val: pd.Series, y_test: pd.Series, labels: list[str]) -> Dict[str, Any]:
    model_id = "lizNguyen235/phobert-nli-vi"
    compatibility = _inspect_zero_shot_model(model_id)
    if compatibility["status"] != "compatible":
        return {
            "model_id": "phobert_nli_vi",
            "status": compatibility["status"],
            "notes": compatibility.get("notes", "Zero-shot NLI compatibility not established."),
            "model_info": compatibility.get("model_info", {}),
        }

    try:
        from transformers import pipeline
    except Exception as exc:  # pragma: no cover
        return {"model_id": "phobert_nli_vi", "status": "dependency_missing", "notes": f"transformers pipeline unavailable: {exc}"}

    classifier = pipeline("zero-shot-classification", model=model_id, tokenizer=model_id, device=-1)
    candidate_labels = labels
    test_scores = [classifier(text, candidate_labels, multi_label=False) for text in test_texts.astype(str).tolist()]
    predicted = [pred["labels"][0] for pred in test_scores]
    val_scores = [classifier(text, candidate_labels, multi_label=False) for text in val_texts.astype(str).tolist()]
    val_pred = [pred["labels"][0] for pred in val_scores]
    val_metrics = _compute_metrics(y_val.astype(str), pd.Series(val_pred), labels)
    test_metrics = _compute_metrics(y_test.astype(str), pd.Series(predicted), labels)
    return {"model_id": "phobert_nli_vi", "status": "completed", "metrics": {"val": val_metrics, "test": test_metrics}, "labels": labels}


def _run_bert_family(model_name: str, train_texts: pd.Series, val_texts: pd.Series, test_texts: pd.Series, y_train: pd.Series, y_val: pd.Series, y_test: pd.Series, labels: list[str], model_id: str) -> Dict[str, Any]:
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
    except Exception as exc:  # pragma: no cover
        return {"model_id": model_name, "status": "dependency_missing", "notes": f"transformers unavailable: {exc}"}

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        label_to_id = {label: idx for idx, label in enumerate(labels)}
        id_to_label = {idx: label for label, idx in label_to_id.items()}

        def encode(batch_df: pd.DataFrame) -> Any:
            encoded = tokenizer(batch_df["text"].tolist(), truncation=True, padding=True, max_length=256)
            encoded["labels"] = [label_to_id[label] for label in batch_df[TARGET_COLUMN].tolist()]
            return encoded

        train_encoded = encode(pd.DataFrame({"text": train_texts.astype(str), TARGET_COLUMN: y_train.astype(str)}))
        val_encoded = encode(pd.DataFrame({"text": val_texts.astype(str), TARGET_COLUMN: y_val.astype(str)}))
        test_encoded = encode(pd.DataFrame({"text": test_texts.astype(str), TARGET_COLUMN: y_test.astype(str)}))

        model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=len(labels), id2label=id_to_label, label2id=label_to_id)
        training_args = TrainingArguments(
            output_dir=str(Path("outputs") / "runs" / model_name),
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            num_train_epochs=1,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            logging_steps=50,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            remove_unused_columns=False,
            report_to=[],
        )

        def compute_metrics(eval_pred):
            import numpy as np
            logits, labels = eval_pred
            preds = np.argmax(logits, axis=-1)
            return {"accuracy": (preds == labels).mean()}

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_encoded,
            eval_dataset=val_encoded,
            compute_metrics=compute_metrics,
        )
        trainer.train()
        val_pred = trainer.predict(val_encoded).predictions.argmax(-1)
        test_pred = trainer.predict(test_encoded).predictions.argmax(-1)
        val_labels = y_val.astype(str).tolist()
        test_labels = y_test.astype(str).tolist()
        val_metrics = _compute_metrics(pd.Series(val_labels), pd.Series([id_to_label.get(int(v), str(v)) for v in val_pred]), labels)
        test_metrics = _compute_metrics(pd.Series(test_labels), pd.Series([id_to_label.get(int(v), str(v)) for v in test_pred]), labels)
        return {"model_id": model_name, "status": "completed", "metrics": {"val": val_metrics, "test": test_metrics}, "labels": labels}
    except Exception as exc:  # pragma: no cover - model availability is reported explicitly
        return {"model_id": model_name, "status": "model_unavailable", "notes": f"Could not initialize {model_id}: {exc}"}


def run_experiment(experiment_id: str, data_dir: str | Path, output_dir: str | Path, seed: int = 42, resume: bool = False) -> Dict[str, Any]:
    if experiment_id not in EXPERIMENTS:
        raise ValueError(f"Unknown experiment '{experiment_id}'. Available: {sorted(EXPERIMENTS)}")

    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _prepare_split_folders(data_dir)
    splits = _ensure_split_compatibility(data_dir / "splits")

    train_df = splits["train"].copy()
    val_df = splits["val"].copy()
    test_df = splits["test"].copy()

    train_df["text"] = _build_text_frame(train_df)
    val_df["text"] = _build_text_frame(val_df)
    test_df["text"] = _build_text_frame(test_df)

    labels = sorted(train_df[TARGET_COLUMN].astype(str).unique().tolist())
    train_texts = train_df["text"]
    val_texts = val_df["text"]
    test_texts = test_df["text"]
    y_train = train_df[TARGET_COLUMN].astype(str)
    y_val = val_df[TARGET_COLUMN].astype(str)
    y_test = test_df[TARGET_COLUMN].astype(str)

    result: Dict[str, Any] = {}
    if experiment_id == "tfidf_lr":
        result = _run_tfidf_model("tfidf_lr", train_texts, val_texts, test_texts, y_train, y_val, y_test, labels)
    elif experiment_id == "tfidf_svm":
        result = _run_tfidf_model("tfidf_svm", train_texts, val_texts, test_texts, y_train, y_val, y_test, labels)
    elif experiment_id in {"fasttext_bilstm", "fasttext_bigru"}:
        result = _run_fasttext_model(experiment_id, train_texts, val_texts, test_texts, y_train, y_val, y_test, labels)
    elif experiment_id in {"minilm_lr", "minilm_svm"}:
        result = _run_minilm_model(experiment_id, train_texts, val_texts, test_texts, y_train, y_val, y_test, labels, output_dir / ".cache")
    elif experiment_id == "distilbert":
        result = _run_bert_family(experiment_id, train_texts, val_texts, test_texts, y_train, y_val, y_test, labels, "distilbert-base-multilingual-cased")
    elif experiment_id == "mbert":
        result = _run_bert_family(experiment_id, train_texts, val_texts, test_texts, y_train, y_val, y_test, labels, "bert-base-multilingual-cased")
    elif experiment_id == "phobert":
        result = _run_bert_family(experiment_id, train_texts, val_texts, test_texts, y_train, y_val, y_test, labels, "vinai/phobert-base")
    elif experiment_id == "phobert_nli_vi":
        result = _run_phobert_nli_vi(train_texts, val_texts, test_texts, y_train, y_val, y_test, labels)
    else:
        raise ValueError(f"Experiment '{experiment_id}' not implemented.")

    result["experiment_id"] = experiment_id
    result["seed"] = seed
    result["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    result["resume_supported"] = True
    result["notes"] = result.get("notes", "")
    result["input_columns"] = ["title", "description"]
    output_path = output_dir / "results" / f"{experiment_id}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one Fahasa benchmark experiment.")
    parser.add_argument("experiment_id", choices=sorted(EXPERIMENTS))
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    result = run_experiment(
        experiment_id=args.experiment_id,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        seed=args.seed,
        resume=args.resume,
    )
    print(json.dumps({"status": "ok", "experiment_id": result["experiment_id"], "output": str(Path(args.output_dir) / "results" / f"{args.experiment_id}.json")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


__all__ = ["run_experiment", "EXPERIMENTS"]
