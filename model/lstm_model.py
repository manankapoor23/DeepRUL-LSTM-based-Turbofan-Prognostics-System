import torch
import torch.nn as nn


class LSTMRULModel(nn.Module):
    """LSTM regressor for turbofan RUL prediction."""

    def __init__(self, input_size: int = 14, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Use the final timestep representation for scalar RUL regression.
        out, _ = self.lstm(x)
        last_hidden = out[:, -1, :]
        return self.fc(last_hidden).squeeze(-1)


def mc_dropout_predict(model: nn.Module, x: torch.Tensor, n_passes: int = 50):
    """Run MC Dropout inference and return mean/std predictions."""
    model.train()  # Keep dropout active at inference time.
    preds = []
    with torch.no_grad():
        for _ in range(n_passes):
            preds.append(model(x).cpu())

    stacked = torch.stack(preds, dim=0)
    mean = stacked.mean(dim=0)
    std = stacked.std(dim=0)
    return mean, std
