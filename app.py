"""
app.py
Retail Sales Forecasting – Streamlit Application
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
from datetime import date

# Page config
st.set_page_config(
    page_title="Retail Sales Forecaster",
    page_icon="🛒",
    layout="centered",
)

# Constants
MODEL_PATH = "model.pkl"
RMSE_PATH = "rmse.txt"
CSV_PATH = "train.csv"

FEATURE_COLS = [
    "day", "month", "year", "day_of_week",
    "lag_1", "lag_7", "rolling_mean_7"
]

# -----------------------------
# TRAIN MODEL
# -----------------------------
def train_model():
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error

    df = pd.read_csv(CSV_PATH)
    df["Order Date"] = pd.to_datetime(df["Order Date"], errors="coerce")
    df = df.dropna(subset=["Order Date", "Sales"])

    daily = df.groupby("Order Date", as_index=False)["Sales"].sum()
    daily = daily.sort_values("Order Date")

    # Features
    daily["day"] = daily["Order Date"].dt.day
    daily["month"] = daily["Order Date"].dt.month
    daily["year"] = daily["Order Date"].dt.year
    daily["day_of_week"] = daily["Order Date"].dt.dayofweek

    # Lag features
    daily["lag_1"] = daily["Sales"].shift(1)
    daily["lag_7"] = daily["Sales"].shift(7)

    # Rolling mean
    daily["rolling_mean_7"] = daily["Sales"].shift(1).rolling(7).mean()

    daily = daily.dropna()

    X = daily[FEATURE_COLS]
    y = daily["Sales"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    rmse = np.sqrt(mean_squared_error(y_test, model.predict(X_test)))

    joblib.dump(model, MODEL_PATH)
    with open(RMSE_PATH, "w") as f:
        f.write(str(round(rmse, 2)))

    return model


# -----------------------------
# LOAD MODEL
# -----------------------------
@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        try:
            return joblib.load(MODEL_PATH)
        except:
            pass

    if not os.path.exists(CSV_PATH):
        st.error("train.csv not found!")
        st.stop()

    with st.spinner("Training model..."):
        return train_model()


def load_rmse():
    if os.path.exists(RMSE_PATH):
        return open(RMSE_PATH).read()
    return "N/A"


# -----------------------------
# LOAD DATA
# -----------------------------
@st.cache_data
def load_data():
    if not os.path.exists(CSV_PATH):
        return pd.DataFrame()

    df = pd.read_csv(CSV_PATH)
    df["Order Date"] = pd.to_datetime(df["Order Date"], errors="coerce")
    df = df.dropna(subset=["Order Date", "Sales"])

    daily = df.groupby("Order Date", as_index=False)["Sales"].sum()
    return daily.sort_values("Order Date")


# -----------------------------
# CHART
# -----------------------------
def plot_chart(df, pred_date, pred):
    fig, ax = plt.subplots()

    ax.plot(df["Order Date"], df["Sales"], label="Sales")
    ax.scatter(pd.Timestamp(pred_date), pred, label="Prediction")

    ax.legend()
    return fig


# -----------------------------
# UI
# -----------------------------
st.title("🛒 Retail Sales Forecaster")

with st.sidebar:
    st.header("Model Info")
    st.metric("RMSE", load_rmse())

    st.markdown(
        "**Algorithm:** Random Forest\n"
        "**Features:** Lag + Rolling\n"
        "**Split:** 80/20"
    )

with st.form("form"):
    col1, col2 = st.columns(2)

    with col1:
        pred_date = st.date_input("Select Date", value=date.today())
        store_id = st.number_input("Store ID", value=1)

    with col2:
        lag_1 = st.number_input("lag_1", value=200.0)
        lag_7 = st.number_input("lag_7", value=200.0)
        rolling_mean_7 = st.number_input("rolling_mean_7", value=200.0)

    submit = st.form_submit_button("Predict")

if submit:
    model = load_model()

    data = pd.DataFrame([{
        "day": pred_date.day,
        "month": pred_date.month,
        "year": pred_date.year,
        "day_of_week": pd.Timestamp(pred_date).dayofweek,
        "lag_1": lag_1,
        "lag_7": lag_7,
        "rolling_mean_7": rolling_mean_7
    }])

    pred = model.predict(data)[0]

    st.success(f"Predicted Sales: {pred:.2f}")

    df = load_data()
    if not df.empty:
        fig = plot_chart(df, pred_date, pred)
        st.pyplot(fig)
