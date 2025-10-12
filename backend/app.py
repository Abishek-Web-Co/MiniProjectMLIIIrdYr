# MiniProjectML/backend/app.py (Corrected Version)

from flask import Flask, jsonify
from flask_cors import CORS
import yfinance as yf
import joblib
import pandas as pd  # Make sure to import pandas

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
        
        # --- THIS IS THE FIX ---
        # Instead of a list of lists, we create a pandas DataFrame
        # that exactly matches the format of our training data.
        features_df = pd.DataFrame([{
            'Open': latest_data['Open'].iloc[0],
            'High': latest_data['High'].iloc[0],
            'Low': latest_data['Low'].iloc[0],
            'Volume': latest_data['Volume'].iloc[0]
        }])
        
        # Predict using the DataFrame
        prediction = model.predict(features_df)
        
        return jsonify({'predicted_price': round(prediction[0], 2)})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=5000)