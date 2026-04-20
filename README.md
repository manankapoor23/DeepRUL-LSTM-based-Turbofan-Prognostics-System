# FD004-Only Turbofan RUL Project (Fresh Reset)

This repository was reset to a clean state and now uses only NASA C-MAPSS FD004 data.

## What is included

- `data/raw/train_FD004.txt`
- `data/raw/test_FD004.txt`
- `data/raw/RUL_FD004.txt`
- `scripts/preprocess_fd004.py`
- `scripts/train_fd004.py`
- `scripts/evaluate_fd004.py`
- `scripts/inference_fd004.py`
- `model/lstm_model.py`
- `api/main.py`
- `frontend/index.html`

Everything unrelated to FD004 was removed.

## Pipeline design

- RUL clipping at 125 cycles is used (standard practice).
- Clipped region imbalance is mitigated by downsampling `RUL=125` sequences to 30%.
- Target is normalized during training: `y_norm = y / 125`.
- Lifecycle awareness is added with `cycle_norm = cycle / max_cycle_per_engine`.
- Weighted MSE emphasizes lower-RUL regions:

```python
weights = 1 / (y_true + 0.1)
loss = mean(weights * (y_pred - y_true)**2)
```

- Mandatory sanity test overfits 50 samples before full training.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run end-to-end

```bash
python scripts/preprocess_fd004.py
python scripts/train_fd004.py
python scripts/evaluate_fd004.py
python scripts/inference_fd004.py
```

## Run API and frontend demo

Start API:

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Serve frontend from project root (example):

```bash
python -m http.server 5500
```

Open:

- `http://localhost:5500/frontend/index.html`

## Notes

- API expects sequence shape: `30 x feature_count`.
- Feature count is read from `data/processed/feature_columns.json` after preprocessing.
