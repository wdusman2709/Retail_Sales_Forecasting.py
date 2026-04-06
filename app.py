import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Load model
model = joblib.load("model.pkl")

st.title("🛒 Retail Sales Prediction App")

# Inputs
date = st.date_input("Select Date")
store_id = st.number_input("Enter Store ID", min_value=1)

# Dummy values for lag (since real-time not available)
lag_1 = st.number_input("Previous Day Sales (lag_1)")
lag_7 = st.number_input("Last Week Sales (lag_7)")
rolling_mean = st.number_input("7-day Avg Sales")

# Extract features
day = date.day
month = date.month
year = date.year

# Prediction
if st.button("Predict Sales"):
    input_data = np.array([[store_id, day, month, year, lag_1, lag_7, rolling_mean]])
    prediction = model.predict(input_data)

    st.success(f"Predicted Sales: {prediction[0]:.2f}")

    # Simple trend chart
    values = [lag_7, lag_1, prediction[0]]
    labels = ['Last Week', 'Yesterday', 'Predicted']

    plt.figure()
    plt.plot(labels, values, marker='o')
    plt.title("Sales Trend")
    plt.xlabel("Time")
    plt.ylabel("Sales")

    st.pyplot(plt)
