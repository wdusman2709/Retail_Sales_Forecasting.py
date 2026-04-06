import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import joblib

# Load dataset
df = pd.read_csv("train.csv")

# Convert date column
df['Date'] = pd.to_datetime(df['Date'])

# Feature Engineering
df['day'] = df['Date'].dt.day
df['month'] = df['Date'].dt.month
df['year'] = df['Date'].dt.year

# Sort data
df = df.sort_values(by=['Store', 'Date'])

# Lag features
df['lag_1'] = df.groupby('Store')['Sales'].shift(1)
df['lag_7'] = df.groupby('Store')['Sales'].shift(7)

# Rolling mean
df['rolling_mean_7'] = df.groupby('Store')['Sales'].transform(lambda x: x.rolling(7).mean())

# Drop NA
df = df.dropna()

# Features & Target
features = ['Store', 'day', 'month', 'year', 'lag_1', 'lag_7', 'rolling_mean_7']
X = df[features]
y = df['Sales']

# Model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# Save model
joblib.dump(model, "model.pkl")

print("Model trained and saved successfully!")
