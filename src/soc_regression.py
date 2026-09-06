"""
SOC Regression Models for Battery State-of-Charge Estimation.

Implements and compares multiple regression approaches:
- Support Vector Regression (SVR)
- Random Forest Regression
- Gradient Boosting (XGBoost, LightGBM)
- LSTM Neural Network
"""

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class RegressionResult:
    """Container for model training results."""

    model_name: str
    rmse: float
    mae: float
    r2: float
    mape: float
    predictions: np.ndarray
    model: object
    best_params: dict
    training_time: float


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute regression metrics for SOC estimation."""
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    # MAPE (SOC is 0-1, so multiply by 100 for percentage)
    nonzero = y_true > 0.01
    if nonzero.sum() > 0:
        mape = (
            np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])) * 100
        )
    else:
        mape = np.nan

    return {
        "RMSE": round(rmse, 4),
        "MAE": round(mae, 4),
        "R2": round(r2, 4),
        "MAPE": round(mape, 2),
        "RMSE_pct": round(rmse * 100, 2),  # As percentage of SOC range
        "MAE_pct": round(mae * 100, 2),
    }


# ---------------------------------------------------------------------------
# SVR
# ---------------------------------------------------------------------------


def train_svr(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    tune_hyperparams: bool = True,
) -> RegressionResult:
    """Train Support Vector Regression."""
    import time

    from sklearn.svm import SVR

    logger.info("Training SVR...")
    start = time.time()

    if tune_hyperparams and len(X_train) < 10000:
        param_grid = {
            "C": [1, 10, 100],
            "gamma": ["scale", 0.01, 0.1],
            "epsilon": [0.01, 0.05, 0.1],
        }
        svr = GridSearchCV(
            SVR(kernel="rbf"),
            param_grid,
            cv=3,
            scoring="neg_mean_squared_error",
            n_jobs=-1,
        )
    else:
        svr = SVR(kernel="rbf", C=100, gamma="scale", epsilon=0.01)

    svr.fit(X_train, y_train)
    elapsed = time.time() - start

    predictions = svr.predict(X_test)
    metrics = evaluate_regression(y_test, predictions)

    best_params = svr.best_params_ if hasattr(svr, "best_params_") else {}

    logger.info(
        "SVR: RMSE=%.4f, MAE=%.4f, R²=%.4f (%.1fs)",
        metrics["RMSE"],
        metrics["MAE"],
        metrics["R2"],
        elapsed,
    )

    return RegressionResult(
        model_name="SVR (RBF)",
        rmse=metrics["RMSE"],
        mae=metrics["MAE"],
        r2=metrics["R2"],
        mape=metrics["MAPE"],
        predictions=predictions,
        model=svr,
        best_params=best_params,
        training_time=elapsed,
    )


# ---------------------------------------------------------------------------
# Random Forest
# ---------------------------------------------------------------------------


def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    tune_hyperparams: bool = True,
) -> RegressionResult:
    """Train Random Forest Regression."""
    import time

    from sklearn.ensemble import RandomForestRegressor

    logger.info("Training Random Forest...")
    start = time.time()

    if tune_hyperparams:
        param_grid = {
            "n_estimators": [100, 200],
            "max_depth": [10, 20, None],
            "min_samples_split": [2, 5],
        }
        rf = GridSearchCV(
            RandomForestRegressor(random_state=42, n_jobs=-1),
            param_grid,
            cv=3,
            scoring="neg_mean_squared_error",
            n_jobs=-1,
        )
    else:
        rf = RandomForestRegressor(
            n_estimators=200,
            max_depth=20,
            random_state=42,
            n_jobs=-1,
        )

    rf.fit(X_train, y_train)
    elapsed = time.time() - start

    predictions = rf.predict(X_test)
    metrics = evaluate_regression(y_test, predictions)
    best_params = rf.best_params_ if hasattr(rf, "best_params_") else {}

    logger.info(
        "Random Forest: RMSE=%.4f, MAE=%.4f, R²=%.4f (%.1fs)",
        metrics["RMSE"],
        metrics["MAE"],
        metrics["R2"],
        elapsed,
    )

    return RegressionResult(
        model_name="Random Forest",
        rmse=metrics["RMSE"],
        mae=metrics["MAE"],
        r2=metrics["R2"],
        mape=metrics["MAPE"],
        predictions=predictions,
        model=rf,
        best_params=best_params,
        training_time=elapsed,
    )


# ---------------------------------------------------------------------------
# XGBoost
# ---------------------------------------------------------------------------


def train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> RegressionResult:
    """Train XGBoost Regression."""
    import time

    from xgboost import XGBRegressor

    logger.info("Training XGBoost...")
    start = time.time()

    xgb = XGBRegressor(
        n_estimators=300,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
    )

    xgb.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )
    elapsed = time.time() - start

    predictions = xgb.predict(X_test)
    metrics = evaluate_regression(y_test, predictions)

    logger.info(
        "XGBoost: RMSE=%.4f, MAE=%.4f, R²=%.4f (%.1fs)",
        metrics["RMSE"],
        metrics["MAE"],
        metrics["R2"],
        elapsed,
    )

    return RegressionResult(
        model_name="XGBoost",
        rmse=metrics["RMSE"],
        mae=metrics["MAE"],
        r2=metrics["R2"],
        mape=metrics["MAPE"],
        predictions=predictions,
        model=xgb,
        best_params=xgb.get_params(),
        training_time=elapsed,
    )


# ---------------------------------------------------------------------------
# LightGBM
# ---------------------------------------------------------------------------


def train_lightgbm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> RegressionResult:
    """Train LightGBM Regression."""
    import time

    from lightgbm import LGBMRegressor

    logger.info("Training LightGBM...")
    start = time.time()

    lgbm = LGBMRegressor(
        n_estimators=300,
        max_depth=8,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )

    lgbm.fit(X_train, y_train)
    elapsed = time.time() - start

    predictions = lgbm.predict(X_test)
    metrics = evaluate_regression(y_test, predictions)

    logger.info(
        "LightGBM: RMSE=%.4f, MAE=%.4f, R²=%.4f (%.1fs)",
        metrics["RMSE"],
        metrics["MAE"],
        metrics["R2"],
        elapsed,
    )

    return RegressionResult(
        model_name="LightGBM",
        rmse=metrics["RMSE"],
        mae=metrics["MAE"],
        r2=metrics["R2"],
        mape=metrics["MAPE"],
        predictions=predictions,
        model=lgbm,
        best_params=lgbm.get_params(),
        training_time=elapsed,
    )


# ---------------------------------------------------------------------------
# LSTM (PyTorch)
# ---------------------------------------------------------------------------


def train_lstm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    seq_length: int = 50,
    hidden_size: int = 64,
    n_layers: int = 2,
    epochs: int = 50,
    batch_size: int = 64,
    learning_rate: float = 0.001,
) -> RegressionResult:
    """Train LSTM neural network for SOC estimation."""
    import time

    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    logger.info(
        "Training LSTM (seq_len=%d, hidden=%d, layers=%d)...",
        seq_length,
        hidden_size,
        n_layers,
    )
    start = time.time()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    # Create sequences
    def create_sequences(X, y, seq_len):
        Xs, ys = [], []
        for i in range(len(X) - seq_len):
            Xs.append(X[i : i + seq_len])
            ys.append(y[i + seq_len])
        return np.array(Xs), np.array(ys)

    X_train_seq, y_train_seq = create_sequences(X_train, y_train, seq_length)
    X_test_seq, y_test_seq = create_sequences(X_test, y_test, seq_length)

    # Convert to tensors
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train_seq),
        torch.FloatTensor(y_train_seq),
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    X_test_tensor = torch.FloatTensor(X_test_seq).to(device)

    # LSTM model
    class LSTMRegressor(nn.Module):
        def __init__(self, input_size, hidden_size, n_layers, dropout=0.2):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size,
                hidden_size,
                n_layers,
                batch_first=True,
                dropout=dropout if n_layers > 1 else 0,
            )
            self.fc = nn.Sequential(
                nn.Linear(hidden_size, 32),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(32, 1),
                nn.Sigmoid(),  # SOC is bounded [0, 1]
            )

        def forward(self, x):
            lstm_out, _ = self.lstm(x)
            out = self.fc(lstm_out[:, -1, :])
            return out.squeeze(-1)

    n_features = X_train.shape[1]
    model = LSTMRegressor(n_features, hidden_size, n_layers).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        patience=5,
        factor=0.5,
    )

    # Training loop
    best_loss = float("inf")
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        scheduler.step(avg_loss)

        if avg_loss < best_loss:
            best_loss = avg_loss

        if (epoch + 1) % 10 == 0:
            logger.info("Epoch %d/%d - Loss: %.6f", epoch + 1, epochs, avg_loss)

    elapsed = time.time() - start

    # Evaluate
    model.eval()
    with torch.no_grad():
        predictions = model(X_test_tensor).cpu().numpy()

    metrics = evaluate_regression(y_test_seq, predictions)

    logger.info(
        "LSTM: RMSE=%.4f, MAE=%.4f, R²=%.4f (%.1fs)",
        metrics["RMSE"],
        metrics["MAE"],
        metrics["R2"],
        elapsed,
    )

    return RegressionResult(
        model_name="LSTM",
        rmse=metrics["RMSE"],
        mae=metrics["MAE"],
        r2=metrics["R2"],
        mape=metrics["MAPE"],
        predictions=predictions,
        model=model,
        best_params={
            "seq_length": seq_length,
            "hidden_size": hidden_size,
            "n_layers": n_layers,
            "epochs": epochs,
        },
        training_time=elapsed,
    )


# ---------------------------------------------------------------------------
# Model comparison
# ---------------------------------------------------------------------------


def compare_all_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    include_lstm: bool = True,
) -> pd.DataFrame:
    """Train and compare all regression models."""
    results = []

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 1. SVR (subsample for speed if large)
    max_svr = min(len(X_train_scaled), 5000)
    results.append(
        train_svr(
            X_train_scaled[:max_svr],
            y_train[:max_svr],
            X_test_scaled,
            y_test,
            tune_hyperparams=max_svr < 3000,
        )
    )

    # 2. Random Forest
    results.append(
        train_random_forest(
            X_train_scaled,
            y_train,
            X_test_scaled,
            y_test,
        )
    )

    # 3. XGBoost
    try:
        results.append(train_xgboost(X_train_scaled, y_train, X_test_scaled, y_test))
    except ImportError:
        logger.warning("XGBoost not installed, skipping")

    # 4. LightGBM
    try:
        results.append(train_lightgbm(X_train_scaled, y_train, X_test_scaled, y_test))
    except ImportError:
        logger.warning("LightGBM not installed, skipping")

    # 5. LSTM
    if include_lstm:
        try:
            results.append(train_lstm(X_train_scaled, y_train, X_test_scaled, y_test))
        except Exception as e:
            logger.warning("LSTM failed: %s", e)

    # Build comparison table
    rows = []
    for r in results:
        rows.append(
            {
                "Model": r.model_name,
                "RMSE": r.rmse,
                "RMSE (%)": r.rmse * 100,
                "MAE": r.mae,
                "MAE (%)": r.mae * 100,
                "R²": r.r2,
                "MAPE (%)": r.mape,
                "Time (s)": round(r.training_time, 1),
            }
        )

    comparison = pd.DataFrame(rows).sort_values("RMSE")
    logger.info("\n%s", comparison.to_string(index=False))
    return comparison


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SOC Regression Models")
    parser.add_argument(
        "--input", type=str, default="data/processed/battery_features.csv"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="all",
        choices=["svr", "rf", "xgboost", "lightgbm", "lstm", "all"],
    )
    parser.add_argument("--output", type=str, default="results")
    parser.add_argument("--test-cycles", type=int, default=50)
    args = parser.parse_args()

    from .data_loader import temporal_train_test_split
    from .feature_engineering import get_feature_columns

    df = pd.read_csv(args.input)
    train_df, test_df = temporal_train_test_split(df, test_cycles=args.test_cycles)

    feature_cols = get_feature_columns(df)
    X_train = train_df[feature_cols].values
    y_train = train_df["soc"].values
    X_test = test_df[feature_cols].values
    y_test = test_df["soc"].values

    if args.model == "all":
        comparison = compare_all_models(X_train, y_train, X_test, y_test)
        Path(args.output).mkdir(parents=True, exist_ok=True)
        comparison.to_csv(f"{args.output}/model_comparison.csv", index=False)
    else:
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        model_map = {
            "svr": train_svr,
            "rf": train_random_forest,
            "xgboost": train_xgboost,
            "lightgbm": train_lightgbm,
            "lstm": train_lstm,
        }
        result = model_map[args.model](X_train_s, y_train, X_test_s, y_test)
        print(f"\n{result.model_name}: RMSE={result.rmse:.4f}, R²={result.r2:.4f}")
