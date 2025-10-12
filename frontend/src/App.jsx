// src/App.jsx

import React, { useState } from 'react';
import './App.css'; // This file can be used for styling

function App() {
    // State variables to store our data
    const [prediction, setPrediction] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState('');

    // Function to call our Python backend
    const getPrediction = async () => {
        setLoading(true);
        setError('');
        setPrediction(null);
        
        try {
            // Send a request to the Flask API endpoint
            const response = await fetch('http://127.0.0.1:5000/predict');
            
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            
            const data = await response.json();

            if (data.predicted_price) {
                setPrediction(`$${data.predicted_price}`);
            } else {
                setError(data.error || 'Failed to get a valid prediction.');
            }
        } catch (err) {
            setError('An error occurred while fetching the prediction.');
            console.error(err);
        }
        
        setLoading(false);
    };

    return (
        <div className="App">
            <header className="App-header">
                <h1>Stock Price Predictor 📈</h1>
                <p>Get the next day's predicted closing price for Apple (AAPL)</p>
                
                <button onClick={getPrediction} disabled={loading}>
                    {loading ? 'Thinking...' : 'Get Prediction'}
                </button>
                
                {prediction && (
                    <div className="result">
                        Predicted Price: <strong>{prediction}</strong>
                    </div>
                )}
                
                {error && (
                    <div className="error">
                        {error}
                    </div>
                )}
            </header>
        </div>
    );
}

export default App;