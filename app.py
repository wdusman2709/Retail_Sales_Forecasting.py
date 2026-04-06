"""
app.py
------
Retail Sales Forecasting – Streamlit Application

Auto-trains the model from train.csv if model.pkl is missing or
incompatible with the current Python version.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
from datetime import date

# ──────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Retail Sales Forecaster",
    page_icon="🛒",
    layout="centered",
)

# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────
MODEL_PATH = "model.pkl"
RMSE_PATH  = "rmse.txt"
CSV_PATH   = "train.csv"
FEATURE_COLS = ["day", "month", "year", "day_of_week",
                "lag_1", "lag_7", "rolling_mean_7"]


# ──────────────────────────────────────────────
# Auto-train helper
# ──────────────────────────────────────────────

def train_model():
    """Train RandomForest from train.csv and save model.pkl + rmse.txt."""
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error

    df = pd.read_csv(CSV_PATH)
    df["Order Date"] = pd.to_datetime(df["Order Date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["Order Date", "Sales"])

    daily = (
        df.groupby("Order Date", as_index=False)["Sales"]
        .sum()
        .sort_values("Order Date")
        .reset_index(drop=True)
    )

    daily["day"]          = daily["Order Date"].dt.day
    daily["month"]        = daily["Order Date"].dt.month
    daily["year"]         = daily["Order Date"].dt.year
    daily["day_of_week"]  = daily["Order Date"].dt.dayofweek
    daily["lag_1"]        = daily["Sales"].shift(1)
    daily["lag_7"]        = daily["Sales"].shift(7)
    daily["rolling_mean_7"] = (
        daily["Sales"].shift(1).rolling(window=7, min_periods=1).mean()
    )
    daily = daily.dropna(subset=["lag_1", "lag_7", "rolling_mean_7"]).reset_index(drop=True)

    X = daily[FEATURE_COLS]
    y = daily["Sales"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    model = RandomForestRegressor(
        n_estimators=200, max_depth=10, random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)

    rmse = round(float(np.sqrt(mean_squared_error(y_test, model.predict(X_test)))), 2)
    joblib.dump(model, MODEL_PATH)
    with open(RMSE_PATH, "w") as f:
        f.write(str(rmse))

    return model, rmse


# ──────────────────────────────────────────────
# Load / auto-train model
# ──────────────────────────────────────────────

@st.cache_resource(show_spinner="Setting up model…")
def load_model():
    """Load model.pkl if available; otherwise auto-train from train.csv."""
    if os.path.exists(MODEL_PATH):
        try:
            return joblib.load(MODEL_PATH)
        except Exception:
            # pkl built on a different Python version — retrain
            pass

    if not os.path.exists(CSV_PATH):
        st.error(
            "**train.csv** not found. Please add it to your GitHub repo."
        )
        st.stop()

    with st.spinner("Training model — please wait ~30 seconds…"):
        model, _ = train_model()
    return model


def load_rmse() -> str:
    if os.path.exists(RMSE_PATH):
        with open(RMSE_PATH) as f:
            return f.read().strip()
    return "N/A"


@st.cache_data(show_spinner="Loading historical data…")
def load_historical() -> pd.DataFrame:
    if not os.path.exists(CSV_PATH):
        return pd.DataFrame()
    df = pd.read_csv(CSV_PATH)
    df["Order Date"] = pd.to_datetime(df["Order Date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["Order Date", "Sales"])
    daily = (
        df.groupby("Order Date", as_index=False)["Sales"]
        .sum()
        .sort_values("Order Date")
        .reset_index(drop=True)
    )
    return daily


def render_trend_chart(daily: pd.DataFrame, pred_date: date, pred_value: float):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(daily["Order Date"], daily["Sales"],
            color="#4C72B0", linewidth=1.2, alpha=0.8, label="Historical Sales")
    rolling = daily["Sales"].rolling(window=14, min_periods=1).mean()
    ax.plot(daily["Order Date"], rolling,
            color="#DD8452", linewidth=1.8, linestyle="--", label="14-day Rolling Avg")
    ax.scatter(pd.Timestamp(pred_date), pred_value,
               color="#C44E52", s=120, zorder=5,
               label=f"Prediction: ${pred_value:,.2f}")
    ax.set_title("Daily Sales Trend & Prediction", fontsize=14, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Sales ($)")
    ax.legend(loc="upper left")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    fig.tight_layout()
    return fig


# ──────────────────────────────────────────────
# UI
# ──────────────────────────────────────────────

st.title("🛒 Retail Sales Forecaster")
st.markdown("Enter the features below and click **Predict Sales** to get a forecast.")

with st.sidebar:
    st.header("ℹ️ Model Info")
    rmse_val = load_rmse()
    st.metric("Test RMSE", f"${rmse_val}" if rmse_val != "N/A" else "N/A")
    st.markdown("""
        **Algorithm:** Random Forest Regressor  
        **Features:** Calendar + Lag + Rolling  
        **Split:** 80 / 20 (chronological)
    """)

with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        pred_date = st.date_input("📅 Select Date", value=date.today())
        store_id  = st.number_input("🏪 Store ID", min_value=1, max_value=100,
                                     value=1, step=1)

    with col2:
        lag_1          = st.number_input("lag_1 – Yesterday's Sales ($)",
                                          min_value=0.0, value=230.0, step=10.0)
        lag_7          = st.number_input("lag_7 – Sales 7 Days Ago ($)",
                                          min_value=0.0, value=230.0, step=10.0)
        rolling_mean_7 = st.number_input("rolling_mean_7 – 7-Day Avg ($)",
                                          min_value=0.0, value=230.0, step=10.0)

    submitted = st.form_submit_button("🔮 Predict Sales", use_container_width=True)

if submitted:
    model = load_model()

    features = {
        "day":            pred_date.day,
        "month":          pred_date.month,
        "year":           pred_date.year,
        "day_of_week":    pd.Timestamp(pred_date).dayofweek,
        "lag_1":          lag_1,
        "lag_7":          lag_7,
        "rolling_mean_7": rolling_mean_7,
    }
    X_input    = pd.DataFrame([features])[FEATURE_COLS]
    prediction = float(model.predict(X_input)[0])

    st.markdown("---")
    st.subheader("📊 Forecast Result")

    c1, c2, c3 = st.columns(3)
    c1.metric("📅 Date",            str(pred_date))
    c2.metric("🏪 Store ID",        store_id)
    c3.metric("💰 Predicted Sales", f"${prediction:,.2f}")

    rmse_val = load_rmse()
    st.info(f"**Model RMSE:** ${rmse_val}  — predictions may vary by ≈ this amount.")

    st.markdown("### 📈 Sales Trend")
    daily_df = load_historical()
    if not daily_df.empty:
        fig = render_trend_chart(daily_df, pred_date, prediction)
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.warning("`train.csv` not found — trend chart unavailable.")

    with st.expander("🔍 Input features used for prediction"):
        st.dataframe(X_input.T.rename(columns={0: "Value"}), use_container_width=True)
        s=120,
        zorder=5,
        label=f"Prediction: ${pred_value:,.2f}",
    )

    ax.set_title("Daily Sales Trend & Prediction", fontsize=14, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Sales ($)")
    ax.legend(loc="upper left")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    fig.tight_layout()
    return fig


# ──────────────────────────────────────────────
# App layout
# ──────────────────────────────────────────────

st.title("🛒 Retail Sales Forecaster")
st.markdown(
    "Enter the features below and click **Predict Sales** to get a forecast."
)

# ── Sidebar: model info ──
with st.sidebar:
    st.header("ℹ️ Model Info")
    rmse_val = load_rmse()
    st.metric("Test RMSE", f"${rmse_val}" if rmse_val != "N/A" else "N/A")
    st.markdown(
        """
        **Algorithm:** Random Forest Regressor  
        **Features:** Calendar + Lag + Rolling  
        **Split:** 80 / 20 (chronological)
        """
    )
    st.markdown("---")
    st.caption("Run `python model.py` to retrain.")

# ── Main input form ──
with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        pred_date = st.date_input(
            "📅 Select Date",
            value=date.today(),
            help="The date you want to forecast sales for.",
        )
        store_id = st.number_input(
            "🏪 Store ID",
            min_value=1,
            max_value=100,
            value=1,
            step=1,
            help="Identifier for the store (informational; single-store model).",
        )

    with col2:
        lag_1 = st.number_input(
            "lag_1 – Yesterday's Sales ($)",
            min_value=0.0,
            value=230.0,
            step=10.0,
            help="Total sales from the previous day.",
        )
        lag_7 = st.number_input(
            "lag_7 – Sales 7 Days Ago ($)",
            min_value=0.0,
            value=230.0,
            step=10.0,
            help="Total sales from exactly 7 days ago.",
        )
        rolling_mean_7 = st.number_input(
            "rolling_mean_7 – 7-Day Avg ($)",
            min_value=0.0,
            value=230.0,
            step=10.0,
            help="Average daily sales over the past 7 days.",
        )

    submitted = st.form_submit_button("🔮 Predict Sales", use_container_width=True)

# ── Prediction ──
if submitted:
    model = load_model()

    # Build feature row from user inputs
    features = {
        "day": pred_date.day,
        "month": pred_date.month,
        "year": pred_date.year,
        "day_of_week": pd.Timestamp(pred_date).dayofweek,
        "lag_1": lag_1,
        "lag_7": lag_7,
        "rolling_mean_7": rolling_mean_7,
    }
    X_input = pd.DataFrame([features])[FEATURE_COLS]

    prediction = float(model.predict(X_input)[0])

    # ── Result card ──
    st.markdown("---")
    st.subheader("📊 Forecast Result")

    res_col1, res_col2, res_col3 = st.columns(3)
    res_col1.metric("📅 Date", str(pred_date))
    res_col2.metric("🏪 Store ID", store_id)
    res_col3.metric("💰 Predicted Sales", f"${prediction:,.2f}")

    # ── RMSE display ──
    st.info(f"**Model RMSE:** ${rmse_val}  — predictions may vary by ≈ this amount.")

    # ── Trend chart ──
    st.markdown("### 📈 Sales Trend")
    daily_df = load_historical()

    if not daily_df.empty:
        fig = render_trend_chart(daily_df, pred_date, prediction)
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.warning(
            "`train.csv` not found in the working directory — "
            "trend chart unavailable."
        )

    # ── Feature table ──
    with st.expander("🔍 Input features used for prediction"):
        st.dataframe(X_input.T.rename(columns={0: "Value"}), use_container_width=True)
