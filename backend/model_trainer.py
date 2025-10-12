# MiniProjectML/backend/model_trainer.py

import yfinance as yf
import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

print("Starting model training...")

# 1. Get and Prepare Data
ticker = 'AAPL'
data = yf.download(ticker, start='2020-01-01', end='2025-12-31')
df = data.copy()
X = df[['Open', 'High', 'Low', 'Volume']]
df['Target'] = df['Close'].shift(-1)
y = df['Target']

# Remove last row with no target
X = X[:-1]
y = y[:-1]

# 2. Train and Save Model
model = LinearRegression()
model.fit(X, y)
joblib.dump(model, 'stock_predictor.pkl')

print("Model trained and saved as 'stock_predictor.pkl'")