from pathlib import Path
from typing import List

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from model.lstm_model import LSTMRULModel, mc_dropout_predict


class PredictRequest(BaseModel):
    engine_id: int = Field(..., ge=0)
    sequence: List[List[float]]

    @field_validator("sequence")
    @classmethod
    def validate_sequence_shape(cls, value: List[List[float]]):
        if len(value) != 30:
            raise ValueError("sequence must have exactly 30 timesteps")
        for row in value:
            if len(row) != 14:
                raise ValueError("each timestep must contain exactly 14 features")
        return value


class PredictResponse(BaseModel):
    engine_id: int
    mean_rul: float
    std_rul: float
    confidence_interval_95: List[float]


app = FastAPI(title="Turbofan RUL Inference API")

MODEL_PATH = Path("model/best_model.pth")
MODEL = None
DEVICE = torch.device("cpu")


@app.on_event("startup")
def load_model_once() -> None:
    global MODEL
    model = LSTMRULModel(input_size=14, hidden_size=64, num_layers=2, dropout=0.3)

    if not MODEL_PATH.exists():
        MODEL = None
        return

    state = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE)
    MODEL = model


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/predict_rul", response_model=PredictResponse)
def predict_rul(payload: PredictRequest) -> PredictResponse:
    if MODEL is None:
        raise HTTPException(
            status_code=503,
            detail="Model checkpoint not found. Train the model to create model/best_model.pth first.",
        )

    sequence_tensor = torch.tensor([payload.sequence], dtype=torch.float32, device=DEVICE)
    mean, std = mc_dropout_predict(MODEL, sequence_tensor, n_passes=50)

    mean_rul = float(mean.item())
    std_rul = float(std.item())
    ci_low = mean_rul - 1.96 * std_rul
    ci_high = mean_rul + 1.96 * std_rul

    return PredictResponse(
        engine_id=payload.engine_id,
        mean_rul=round(mean_rul, 1),
        std_rul=round(std_rul, 1),
        confidence_interval_95=[round(ci_low, 1), round(ci_high, 1)],
    )
