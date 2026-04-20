# Turbofan Engine Remaining Useful Life (RUL) Prediction

This project is a presentation-ready, end-to-end LSTM pipeline for turbofan Remaining Useful Life prediction in a Rolls-Royce Aerospace predictive maintenance context. It focuses on clear preprocessing, explainable model behavior, and uncertainty-aware inference with Monte Carlo Dropout. The implementation uses NASA C-MAPSS FD001 only, runs on CPU, and includes notebooks, a FastAPI backend, and a lightweight frontend monitor. The goal is correctness and readability over MLOps complexity.

## Project Structure

```text
turbofan-rul/
├── data/
│   ├── raw/
│   │   ├── train_FD001.txt
│   │   ├── test_FD001.txt
│   │   └── RUL_FD001.txt
│   └── processed/
├── notebooks/
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_model_training.ipynb
│   ├── 03_evaluation.ipynb
│   └── 04_inference_demo.ipynb
├── model/
│   ├── lstm_model.py
│   └── best_model.pth
├── api/
│   └── main.py
├── frontend/
│   └── index.html
├── requirements.txt
└── README.md
```

## Setup

1. Clone repository and enter it.
```bash
git clone <your-repo-url>
cd <your-repo-folder>
```

2. Create and activate virtual environment.
```bash
python3 -m venv .venv
source .venv/bin/activate
```

3. Install pinned dependencies.
```bash
pip install -r requirements.txt
```

4. Prepare FD001 data.
- If you already have CMAPSS files in `CMAPSSData/`, copy FD001 files into `data/raw/`:
```bash
mkdir -p data/raw
cp CMAPSSData/train_FD001.txt data/raw/
cp CMAPSSData/test_FD001.txt data/raw/
cp CMAPSSData/RUL_FD001.txt data/raw/
```
- Otherwise, download NASA C-MAPSS (or Kaggle mirror), then place FD001 files in `data/raw/`.

## Run Notebooks In Order

1. `notebooks/01_data_preprocessing.ipynb`  
   Loads raw FD001 data, applies preprocessing pipeline, visualizes RUL and correlations, and saves processed artifacts.

2. `notebooks/02_model_training.ipynb`  
   Trains the specified LSTM model with engine-level split and early stopping, saves checkpoint and required training plots.

3. `notebooks/03_evaluation.ipynb`  
   Evaluates on test set with RMSE/MAE/R2 and visualizes predicted-vs-actual plus trajectory-level behavior for 3 engines.

4. `notebooks/04_inference_demo.ipynb`  
   Runs MC Dropout over time for one test engine and visualizes mean prediction with 95% uncertainty band.

## FastAPI Backend

Start the API from repository root:

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

Available endpoints:
- `GET /health` -> `{"status": "ok"}`
- `POST /predict_rul` with payload:
```json
{
  "engine_id": 1,
  "sequence": [[0.52, 0.41, 0.38, 0.22, 0.10, 0.55, 0.62, 0.71, 0.49, 0.33, 0.45, 0.29, 0.51, 0.40]]
}
```

## Frontend

Serve the repository root as static files, then open the dashboard:

```bash
python3 -m http.server 5500
```

Open:
- `http://localhost:5500/frontend/index.html`

The frontend calls `http://localhost:8000/predict_rul`, renders an arc-based RUL gauge, displays uncertainty, and tags health state as HEALTHY / MONITOR / CRITICAL.

## Notes

- CPU-only implementation (no CUDA assumptions).
- Model architecture is fixed to:
  - `LSTM(input_size=14, hidden_size=64, num_layers=2, dropout=0.3, batch_first=True)`
  - `Linear(64 -> 1)`
- Checkpoint path: `model/best_model.pth`.
- Required training plots are saved in `model/`:
  - `training_loss_curve.png`
  - `predicted_vs_actual_rul.png`
