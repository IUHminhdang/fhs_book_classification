# Fahasa Book Genre Classification Benchmark

This project implements a multi-class Vietnamese book genre classification benchmark using the pre-split Fahasa datasets.

## Project layout

- `src/`: reusable Python modules for data validation, preprocessing, training, evaluation, and reporting.
- `configs/`: configuration and label mapping files.
- `scripts/`: dataset audit, smoke tests, and reporting utilities.
- `notebooks/`: runnable Kaggle and Colab entry points.
- `tests/`: unit and smoke tests.
- `data/`: source CSV files and generated split outputs.
- `outputs/`: aggregated experiment outputs, plots, and report artifacts.

## Data contract

The dataset must contain the columns defined in the task description. The project uses only `title` and `description` as model input and keeps the existing `category_norm` taxonomy intact.

## Quick start

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
python scripts/audit_dataset.py --data-dir data --output-dir outputs
python -m src.experiments tfidf_lr --data-dir data --output-dir outputs
python scripts/generate_reports.py --results-dir outputs/results --output-dir outputs --zip-output outputs/benchmark_bundle.zip
```

The repo root must be the working directory when invoking scripts directly so the package resolves correctly.

## Models

The benchmark implements the ten requested experiments:

1. `tfidf_lr`
2. `tfidf_svm`
3. `fasttext_bilstm`
4. `fasttext_bigru`
5. `minilm_lr`
6. `minilm_svm`
7. `distilbert`
8. `mbert`
9. `phobert`
10. `phobert_nli_vi`

## Experiment runner notebooks

Use the notebooks under `notebooks/` to choose one experiment ID, set the data directory, output directory, and seed, then run it in Kaggle or Colab.

## Notes

- No split is created automatically; the project expects the user-provided train/validation/test files.
- For compatibility, `data/val.csv` and `data/splits/val.csv` are supported as aliases for the validation split.
- The project keeps generated artifacts out of version control and supports resuming interrupted runs when a checkpoint is present.
