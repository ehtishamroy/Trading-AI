"""
LSTM Model — Deep learning model for price direction prediction.
Processes sequences of 60 candles → predicts up/down probability.
Train on vast.ai GPU, infer on local CPU.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from loguru import logger
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config.settings import (
    LSTM_SEQUENCE_LEN, LSTM_HIDDEN_SIZE, LSTM_NUM_LAYERS,
    LSTM_DROPOUT, MODELS_DIR, TRAINING_DEVICE
)


class TradingLSTM(nn.Module):
    """
    LSTM neural network for predicting price direction.

    Input: sequence of [60 candles × N features]
    Output: probability of price going UP in next 12 bars
    """

    def __init__(self, input_size: int, hidden_size: int = LSTM_HIDDEN_SIZE,
                 num_layers: int = LSTM_NUM_LAYERS, dropout: float = LSTM_DROPOUT):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1),
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x shape: (batch, seq_len, features)
        lstm_out, _ = self.lstm(x)
        # lstm_out: (batch, seq_len, hidden_size)

        # Attention mechanism — focus on the most important bars
        attn_weights = self.attention(lstm_out)         # (batch, seq_len, 1)
        attn_weights = torch.softmax(attn_weights, dim=1)
        context = (lstm_out * attn_weights).sum(dim=1)  # (batch, hidden_size)

        # Classify direction
        out = self.classifier(context)                  # (batch, 1)
        return out.squeeze(-1)


def create_sequences(df: pd.DataFrame, feature_cols: list, target_col: str = "target",
                     seq_len: int = LSTM_SEQUENCE_LEN) -> tuple:
    """
    Create input sequences from DataFrame for LSTM training.

    Returns:
        X: numpy array of shape (n_samples, seq_len, n_features)
        y: numpy array of shape (n_samples,) with 0/1 labels
    """
    features = df[feature_cols].values
    targets = df[target_col].values

    X, y = [], []
    for i in range(seq_len, len(features)):
        X.append(features[i - seq_len:i])
        y.append(targets[i])

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def train_lstm(X_train, y_train, X_val, y_val, input_size: int,
               epochs: int = 100, batch_size: int = 64, lr: float = 0.001,
               market: str = "EURUSD") -> TradingLSTM:
    """
    Train the LSTM model.

    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        input_size: Number of features per bar
        epochs: Training epochs
        batch_size: Batch size
        lr: Learning rate
        market: Market name (for saving)

    Returns:
        Trained TradingLSTM model
    """
    device = torch.device(TRAINING_DEVICE if torch.cuda.is_available() else "cpu")
    logger.info(f"Training LSTM on: {device}")

    model = TradingLSTM(input_size=input_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    # Convert to tensors
    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.FloatTensor(y_train).to(device)
    X_val_t = torch.FloatTensor(X_val).to(device)
    y_val_t = torch.FloatTensor(y_val).to(device)

    train_data = torch.utils.data.TensorDataset(X_train_t, y_train_t)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

    best_val_loss = float("inf")
    patience_counter = 0
    max_patience = 15

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # Validation
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t)
            val_loss = criterion(val_pred, y_val_t).item()
            val_acc = ((val_pred > 0.5).float() == y_val_t).float().mean().item()

        scheduler.step(val_loss)

        if (epoch + 1) % 10 == 0:
            logger.info(
                f"Epoch {epoch+1}/{epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val Accuracy: {val_acc:.2%}"
            )

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            save_lstm_model(model, market)
        else:
            patience_counter += 1
            if patience_counter >= max_patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

    logger.success(f"Training complete! Best val loss: {best_val_loss:.4f}")
    return model


def save_lstm_model(model: TradingLSTM, market: str):
    """Save trained model weights."""
    path = MODELS_DIR / f"lstm_{market}.pt"
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    logger.info(f"Model saved → {path}")


def load_lstm_model(market: str, input_size: int) -> TradingLSTM:
    """Load a trained model for inference."""
    path = MODELS_DIR / f"lstm_{market}.pt"
    if not path.exists():
        raise FileNotFoundError(f"No trained LSTM model for {market}")

    model = TradingLSTM(input_size=input_size)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    logger.info(f"Loaded LSTM model for {market}")
    return model


def predict(model: TradingLSTM, X: np.ndarray) -> dict:
    """
    Make a prediction with the LSTM model.

    Args:
        model: Trained model
        X: Feature array of shape (seq_len, n_features)

    Returns:
        {direction: 'up'/'down', confidence: 0.0-1.0}
    """
    model.eval()
    with torch.no_grad():
        x_tensor = torch.FloatTensor(X).unsqueeze(0)  # Add batch dimension
        prob = model(x_tensor).item()

    direction = "up" if prob > 0.5 else "down"
    confidence = prob if prob > 0.5 else (1 - prob)

    return {
        "direction": direction,
        "confidence": round(confidence, 4),
        "raw_probability": round(prob, 4),
    }
