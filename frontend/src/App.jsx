import React, { useState } from 'react';
import axios from 'axios';
// No need to import App.css anymore!

function App() {
  const [inputText, setInputText] = useState('');
  const [predictionResult, setPredictionResult] = useState(null);
  const [isLoading, setIsLoading] = useState(false);

  const handlePredictClick = () => {
    if (!inputText) {
      alert('Please enter some text!');
      return;
    }

    setIsLoading(true);
    setPredictionResult(null);

    const apiUrl = 'http://127.0.0.1:5000/predict';

    axios.post(apiUrl, { text: inputText })
      .then(response => {
        setPredictionResult(response.data);
        setIsLoading(false);
      })
      .catch(error => {
        console.error('Error:', error);
        alert('Sorry, something went wrong with the server.');
        setIsLoading(false);
      });
  };

  return (
    // Main container with a light gray background
    <div className="bg-gray-100 min-h-screen flex items-center justify-center p-4">
      
      {/* Card container */}
      <div className="max-w-2xl w-full bg-white p-8 rounded-xl shadow-lg text-center">
        
        {/* Header */}
        <h1 className="text-4xl font-bold text-gray-800 mb-6">
          Emotion Detection from Text 😊😡😭
        </h1>
        
        {/* Textarea for input */}
        <textarea
          className="w-full h-40 p-3 text-base text-gray-700 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-blue-500 mb-6 resize-none"
          placeholder="Type your text here..."
          value={inputText}
          onChange={e => setInputText(e.target.value)}
        />
        
        {/* Predict Button */}
        <button
          className="py-3 px-6 text-lg font-semibold text-white bg-blue-600 rounded-md hover:bg-blue-700 transition-colors disabled:bg-gray-400 disabled:cursor-not-allowed"
          onClick={handlePredictClick}
          disabled={isLoading}
        >
          {isLoading ? 'Detecting...' : 'Detect Emotion'}
        </button>

        {/* Result Display Area */}
        <div className="mt-6 text-xl text-gray-700 min-h-[3rem] flex items-center justify-center">
          {predictionResult ? (
            <p>
              <strong>Emotion:</strong> {predictionResult.emotion}{' '}
              <span className="text-gray-500">
                ({(predictionResult.score * 100).toFixed(2)}%)
              </span>
            </p>
          ) : (
            <p className="text-gray-400">Emotion will be shown here...</p>
          )}
        </div>
        
      </div>
    </div>
  );
}

export default App;