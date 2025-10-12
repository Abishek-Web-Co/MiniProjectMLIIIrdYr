# backend/app.py

from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import pipeline

# Initialize the Flask app
app = Flask(__name__)
CORS(app)  # This allows your frontend to talk to your backend

# Load your fine-tuned model from the saved directory
# Make sure the path points to where you saved your model after training
emotion_classifier = pipeline(
    'text-classification', 
    model='./emotion-detection-model', 
    tokenizer='./emotion-detection-model'
)

# Define the prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data.get('text')
    
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    
    # Get prediction from the model
    result = emotion_classifier(text)[0]
    
    # Return the result as JSON
    return jsonify({
        'emotion': result['label'],
        'score': result['score']
    })

# Run the app
if __name__ == '__main__':
    app.run(debug=True, port=5000)