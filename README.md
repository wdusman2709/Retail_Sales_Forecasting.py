🛒 Retail Sales Forecaster

A production-ready, end-to-end Machine Learning project that predicts daily retail sales using historical data, time-series feature engineering, and a Random Forest model — deployed with an interactive Streamlit dashboard.

---

🚀 Project Overview

This project builds a sales forecasting system that:

- 📊 Aggregates raw transaction data into daily sales
- 🧠 Applies feature engineering (date, lag, rolling features)
- 🤖 Trains a Random Forest Regressor
- 📈 Evaluates performance using RMSE
- 🌐 Deploys predictions through a Streamlit web app

---

📂 Project Structure

Retail_Sales_Forecasting/
│
├── app.py              # Streamlit application
├── model.py            # Model training script
├── model.pkl           # Trained ML model
├── train.csv           # Dataset
├── rmse.txt            # Model performance (RMSE)
├── requirements.txt    # Dependencies
├── runtime.txt         # Python version (for deployment)
└── README.md           # Project documentation

---

📊 Dataset

- Source: Retail transaction dataset ("train.csv")
- Key columns:
  - "Order Date"
  - "Sales"

📌 Data is aggregated to daily level before modeling.

---

⚙️ Feature Engineering

The model uses the following features:

- 📅 Date-based:
  - Day, Month, Year
  - Day of Week
- ⏮️ Lag Features:
  - "lag_1" → Previous day sales
  - "lag_7" → Sales 7 days ago
- 📉 Rolling Feature:
  - "rolling_mean_7" → 7-day moving average

---

🤖 Model

- Algorithm: Random Forest Regressor
- Train/Test Split: 80 / 20 (chronological)
- Evaluation Metric: RMSE (Root Mean Squared Error)

---

📈 Streamlit App Features

🔹 Input

- Date selection
- Store ID (default = 1)
- Lag values (lag_1, lag_7)
- Rolling mean

🔹 Output

- 💰 Predicted Sales
- 📊 Sales trend chart
- 📉 Model RMSE

---

🌐 Deployment

This project is deployed using Streamlit Cloud.

Steps:

1. Push code to GitHub
2. Go to https://share.streamlit.io
3. Select repository
4. Set main file: "app.py"
5. Deploy 🚀

---

📌 Notes

- "model.pkl" is used for fast predictions
- If missing or incompatible, the model auto-trains from "train.csv"
- Lag values are user inputs in UI (basic implementation)

---

🔮 Future Improvements

- Auto-generate lag features from historical data
- Use advanced models (XGBoost, LSTM)
- Add multi-store support
- Improve UI/UX dashboard
- Add real-time data integration

---

🎯 Conclusion

This project demonstrates a complete ML pipeline:

- Data preprocessing
- Feature engineering
- Model training & evaluation
- Deployment with Streamlit

It is suitable for:

- 🎓 Academic submission
- 💼 Portfolio projects
- 🧠 Interview demonstrations

---

## 📝 Notes

- The dataset is treated as a **single store** — all orders are aggregated by date.
- Default lag/rolling values in the UI are set to the dataset's mean sales (~$230) so the app is immediately usable without prior knowledge.
- `@st.cache_resource` is used for the model and `@st.cache_data` for the CSV to keep the app fast on repeated interactions.

