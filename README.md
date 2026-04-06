# 🛒 Retail Sales Forecaster

A production-ready, end-to-end machine learning project that predicts **daily retail store sales** using historical order data, engineered time-series features, and a Random Forest model — served through an interactive Streamlit dashboard.

---

## 📌 Project Description

This project takes a raw retail transactions CSV (`train.csv`) and:

1. **Aggregates** individual orders into daily sales totals.
2. **Engineers** calendar, lag, and rolling-window features to capture sales patterns.
3. **Trains** a `RandomForestRegressor` on 80 % of the data and evaluates on the remaining 20 % using RMSE.
4. **Serves** predictions through a Streamlit web app where users can input lag features and a target date to get a forecast.

---

## 📂 Project Structure

```
retail-sales-forecaster/
│
├── train.csv              # Raw dataset (Superstore-style orders)
├── model.py               # Feature engineering + model training script
├── app.py                 # Streamlit application
├── requirements.txt       # Python dependencies
├── README.md              # This file
│
# Generated after running model.py:
├── model.pkl              # Trained RandomForest model (joblib)
└── rmse.txt               # Test RMSE score (read by app.py)
```

---

## 🧰 Feature Engineering

| Feature | Description |
|---|---|
| `day` | Day of the month (1–31) |
| `month` | Month of the year (1–12) |
| `year` | Calendar year |
| `day_of_week` | Day of week (0 = Monday) |
| `lag_1` | Previous day's total sales |
| `lag_7` | Total sales from 7 days prior |
| `rolling_mean_7` | 7-day rolling average of past sales |

---

## 🚀 How to Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/your-username/retail-sales-forecaster.git
cd retail-sales-forecaster
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Train the model

Place `train.csv` in the project root, then run:

```bash
python model.py
```

This generates `model.pkl` and `rmse.txt`.

### 4. Launch the Streamlit app

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## ☁️ Deploy on Streamlit Cloud

1. Push the repo (including `train.csv`, `model.pkl`, `rmse.txt`) to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io) and click **New app**.
3. Select your repository, branch (`main`), and set **Main file path** to `app.py`.
4. Click **Deploy** — Streamlit Cloud installs dependencies from `requirements.txt` automatically.

> **Tip:** Commit `model.pkl` and `rmse.txt` to the repo so the app works immediately without re-running `model.py` in the cloud.

---

## 🖥️ Streamlit App Features

| Component | Details |
|---|---|
| Date Picker | Select the date to forecast |
| Store ID | Informational input (single-store model) |
| lag_1 | Enter yesterday's sales |
| lag_7 | Enter sales from 7 days ago |
| rolling_mean_7 | Enter the 7-day average |
| Predicted Sales | Displayed as a metric card |
| RMSE Badge | Test RMSE shown in sidebar and result area |
| Trend Chart | Historical daily sales + 14-day rolling avg + predicted point |
| Feature Table | Expandable view of the exact input sent to the model |

---

## 📊 Model Performance

The model is evaluated on the most recent 20 % of dates (chronological split).
RMSE is stored in `rmse.txt` and displayed live in the app sidebar.

---

## 📝 Notes

- The dataset is treated as a **single store** — all orders are aggregated by date.
- Default lag/rolling values in the UI are set to the dataset's mean sales (~$230) so the app is immediately usable without prior knowledge.
- `@st.cache_resource` is used for the model and `@st.cache_data` for the CSV to keep the app fast on repeated interactions.

