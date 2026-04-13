# DeepRUL-LSTM-based-Turbofan-Prognostics-System

LSTM-based Remaining Useful Life (RUL) prediction system for NASA CMAPSS turbofan engine data, with uncertainty estimation, domain adaptation, and MLOps utilities.

## Overview

This project predicts engine RUL from multivariate sensor time series using a stacked LSTM.
It includes:

- Data preprocessing pipeline for CMAPSS
- Piecewise RUL labeling (capped at 125)
- LSTM regression model with RMSE training objective
- Monte Carlo dropout for prediction uncertainty
- Domain adaptation from FD001 to FD003
- MLflow experiment tracking and model registration helpers
- DVC pipeline generation support

## Dataset

This repository uses the NASA CMAPSS dataset.

- Kaggle mirror: https://www.kaggle.com/datasets/behrad3d/nasa-cmaps
- NASA PCoE source: https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/

After download, place the extracted dataset folder in the project root as:

```text
CMAPSSData/
```

Expected files include:

```text
CMAPSSData/train_FD001.txt
CMAPSSData/test_FD001.txt
CMAPSSData/RUL_FD001.txt
CMAPSSData/train_FD003.txt
CMAPSSData/test_FD003.txt
CMAPSSData/RUL_FD003.txt
```

## Project Structure

```text
dataset.py            # CMAPSS loading and feature definitions
data_engineering.py   # RUL labels, scaling, sequence generation
model.py              # LSTM model, RMSE loss, NASA score, MC dropout
train.py              # Training/evaluation loop + MLflow logging
domain_adaptation.py  # FD001 -> FD003 adaptation (freeze encoder, tune head)
mlops.py              # Experiment comparison, model registry, DVC yaml writer
requirements.txt      # Python dependencies
README.md             # Main documentation
```

## Method Summary

1. RUL labeling
- Training labels are computed per engine as: max_cycle - current_cycle
- Labels are capped at 125 cycles (piecewise degradation assumption)

2. Feature processing
- Uses operational settings (`op1`, `op2`, `op3`) and selected informative sensors
- MinMax scaling is fit on training data and persisted as `artifacts/scaler.pkl`

3. Sequence modeling
- Sliding windows of length 30 are built per engine
- Target is the RUL value at the final timestep of each window

4. Uncertainty estimation
- Dropout remains active at inference
- 50 stochastic forward passes provide mean prediction and standard deviation

5. Domain adaptation
- Source model trained on FD001
- LSTM encoder frozen, regression head fine-tuned on FD003

## Installation

Use Python 3.10+ recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Quick Start

1. Train baseline on FD001

```bash
python train.py
```

2. Launch MLflow UI

```bash
mlflow ui
```

Open: http://127.0.0.1:5000

3. Run domain adaptation (FD001 -> FD003)

```bash
python domain_adaptation.py
```

4. Compare runs and register best model

```bash
python mlops.py --compare
python mlops.py --register
```

5. Generate DVC pipeline file

```bash
python mlops.py --dvc
```

Then initialize and reproduce pipeline:

```bash
dvc init
dvc repro
```

## What Gets Saved

Outputs are written to `artifacts/`:

- `best_model.pt` - best FD001 checkpoint
- `adapted_FD003.pt` - adapted model checkpoint
- `scaler.pkl` - fitted MinMax scaler
- `sequences_FD001.npz` / `sequences_FD003.npz` - processed data
- plots (`loss_curves.png`, `predictions.png`, `scatter.png`, adaptation plot)

MLflow logs:

- Hyperparameters
- Training/validation/test metrics
- Uncertainty metrics
- Model artifacts

## Typical Performance (FD001)

Expected ranges (hardware/data split dependent):

- RMSE: ~13 to 16 cycles
- NASA score: ~200 to 400
- Average uncertainty (MC std): ~3 to 6 cycles

## CLI Reference

Training:

```bash
python train.py
```

Domain adaptation:

```bash
python domain_adaptation.py
```

MLOps utilities:

```bash
python mlops.py --compare
python mlops.py --register
python mlops.py --dvc
```

## Troubleshooting

- File not found for CMAPSS files:
	- Ensure dataset is in `CMAPSSData/` at the project root.
- `no checkpoint at artifacts/best_model.pt` during adaptation:
	- Run `python train.py` first.
- MLflow command not found:
	- Reinstall dependencies with `pip install -r requirements.txt`.

## Citation Note

If you use this project in academic/industrial work, cite the NASA CMAPSS dataset source and mention the benchmark subset(s) used.
