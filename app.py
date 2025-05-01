from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np

app = Flask(__name__)
CORS(app)

# Load the saved ML model
model = joblib.load('model_battery_rul_gb.pkl')

@app.route('/')
def home():
    return jsonify({"message": "Battery RUL Prediction API is running."})

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    
    if not data or 'features' not in data:
        return jsonify({'error': 'Missing input features'}), 400

    try:
        features = np.array(data['features']).reshape(1, -1)
        prediction = model.predict(features)
        return jsonify({'predicted_RUL': float(prediction[0])})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
