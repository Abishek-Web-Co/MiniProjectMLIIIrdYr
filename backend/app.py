# MiniProjectML/backend/app.py

from flask import Flask, jsonify
from flask_cors import CORS
import yfinance as yf
import joblib

# Initialize App
app = Flask(__name__)
CORS(app) # Enable cross-origin requests

# Load Model
model = joblib.load('stock_predictor.pkl')

@app.route('/predict', methods=['GET'])
def predict():
    try:
        # Get latest data
        latest_data = yf.download('AAPL', period='1d', interval='1d')
        features = [[
            latest_data['Open'].iloc[0],
            latest_data['High'].iloc[0],
            latest_data['Low'].iloc[0],
            latest_data['Volume'].iloc[0]
        ]]
        
        # Predict
        prediction = model.predict(features)
        return jsonify({'predicted_price': round(prediction[0], 2)})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=5000)