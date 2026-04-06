"""
model.py
--------
Retail Sales Forecasting – Feature Engineering & Model Training

Reads train.csv, engineers time-series features, trains a
RandomForestRegressor, evaluates with RMSE, and saves the model
to model.pkl using joblib.

Run:
    python model.py
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib
import os

# ──────────────────────────────────────────────
# 1. Load & clean data
# ──────────────────────────────────────────────

CSV_PATH = "train.csv"
MODEL_PATH = "model.pkl"
RMSE_PATH = "rmse.txt"


def load_and_aggregate(csv_path: str) -> pd.DataFrame:
    """Load raw CSV and aggregate daily sales for a single-store view."""
    df = pd.read_csv(csv_path)

    # Parse date – try multiple common formats automatically
    df["Order Date"] = pd.to_datetime(df["Order Date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["Order Date", "Sales"])

    # Aggregate to one row per day (sum of all orders on that day)
    daily = (
        df.groupby("Order Date", as_index=False)["Sales"]
        .sum()
        .sort_values("Order Date")
        .reset_index(drop=True)
    )
    return daily


# ──────────────────────────────────────────────
# 2. Feature engineering
# ──────────────────────────────────────────────

def build_features(daily: pd.DataFrame) -> pd.DataFrame:
    """Add calendar and lag/rolling features to the daily sales DataFrame."""
    df = daily.copy()

    # Calendar features
    df["day"] = df["Order Date"].dt.day
    df["month"] = df["Order Date"].dt.month
    df["year"] = df["Order Date"].dt.year
    df["day_of_week"] = df["Order Date"].dt.dayofweek  # extra signal

    # Lag features (shift fills leading rows with NaN automatically)
    df["lag_1"] = df["Sales"].shift(1)
    df["lag_7"] = df["Sales"].shift(7)

    # Rolling average (min_periods=1 avoids NaN at the start)
    df["rolling_mean_7"] = (
        df["Sales"].shift(1).rolling(window=7, min_periods=1).mean()
    )

    # Drop rows that still have NaN after rolling (first row has no lag_1)
    df = df.dropna(subset=["lag_1", "lag_7", "rolling_mean_7"]).reset_index(drop=True)

    return df


# ──────────────────────────────────────────────
# 3. Train / evaluate / save
# ──────────────────────────────────────────────

FEATURE_COLS = ["day", "month", "year", "day_of_week", "lag_1", "lag_7", "rolling_mean_7"]
TARGET_COL = "Sales"


def train(csv_path: str = CSV_PATH) -> float:
    """Full pipeline: load → feature-engineer → split → train → evaluate → save."""

    print(f"Loading data from: {csv_path}")
    daily = load_and_aggregate(csv_path)
    print(f"  Daily rows after aggregation: {len(daily)}")

    df = build_features(daily)
    print(f"  Rows after feature engineering: {len(df)}")

    X = df[FEATURE_COLS]
    y = df[TARGET_COL]

    # Chronological split (no shuffle) preserves temporal order
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, shuffle=False
    )
    print(f"  Train size: {len(X_train)} | Test size: {len(X_test)}")

    # Model
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    # Evaluation
    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    print(f"  RMSE on test set: {rmse:.2f}")

    # Persist model
    joblib.dump(model, MODEL_PATH)
    print(f"  Model saved → {MODEL_PATH}")

    # Persist RMSE so the Streamlit app can read it without retraining
    with open(RMSE_PATH, "w") as f:
        f.write(str(round(rmse, 2)))
    print(f"  RMSE saved  → {RMSE_PATH}")

    return rmse


if __name__ == "__main__":
    train()
