#!/usr/bin/env python3
"""
Flask Web App for Predictive Maintenance
Provides a web interface to test the conveyor belt prediction model
"""

from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
import joblib
import os
from datetime import datetime

app = Flask(__name__)

# Load the trained model and scaler
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'conveyor_model.pkl')
SCALER_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'scaler.pkl')
FEATURES_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'feature_names.txt')

# Check if model exists, if not use a simple placeholder
try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    with open(FEATURES_PATH, 'r') as f:
        feature_names = [line.strip() for line in f.readlines()]
    print(f"✅ Model loaded successfully with {len(feature_names)} features")
    model_loaded = True
except Exception as e:
    print(f"⚠️ Warning: Could not load model: {e}")
    print("Using simple rule-based prediction instead")
    model_loaded = False

# Condition names for display
CONDITION_NAMES = {
    0: "✅ NORMAL",
    1: "⚠️ MINOR TEAR",
    2: "🔴 MAJOR TEAR",
    3: "🚨 DISLODGEMENT IMMINENT"
}

COLORS = {
    0: "green",
    1: "yellow",
    2: "red",
    3: "darkred"
}

@app.route('/')
def home():
    """Render the main page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    try:
        # Get data from form
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        vibration = int(request.form['vibration'])
        motor_current = float(request.form['motor_current'])
        
        # Create feature vector
        if model_loaded:
            # Use ML model
            features = create_features(temperature, humidity, vibration, motor_current)
            prediction, confidence = predict_with_model(features)
        else:
            # Use simple rules
            prediction, confidence = simple_prediction(temperature, humidity, vibration, motor_current)
        
        # Get condition name and color
        condition = CONDITION_NAMES[prediction]
        color = COLORS[prediction]
        
        return render_template('result.html',
                             temperature=temperature,
                             humidity=humidity,
                             vibration=vibration,
                             motor_current=motor_current,
                             condition=condition,
                             confidence=f"{confidence:.1%}",
                             color=color,
                             prediction=prediction)
    
    except Exception as e:
        return f"Error: {str(e)}"

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for predictions"""
    try:
        data = request.get_json()
        temperature = float(data['temperature'])
        humidity = float(data['humidity'])
        vibration = int(data['vibration'])
        motor_current = float(data['motor_current'])
        
        if model_loaded:
            features = create_features(temperature, humidity, vibration, motor_current)
            prediction, confidence = predict_with_model(features)
        else:
            prediction, confidence = simple_prediction(temperature, humidity, vibration, motor_current)
        
        return jsonify({
            'prediction': int(prediction),
            'condition': CONDITION_NAMES[prediction],
            'confidence': float(confidence),
            'color': COLORS[prediction]
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

def create_features(temp, hum, vib, current):
    """Create feature vector from input data"""
    # Simple features for demo
    features = {
        'vibration': vib,
        'temperature': temp,
        'humidity': hum,
        'motor_current': current,
        'temp_humidity_ratio': temp / (hum + 1),
        'vibration_lag1': 0,  # Simplified for demo
        'vibration_lag2': 0,
        'vibration_rolling_mean': vib,
        'vibration_rolling_std': 0
    }
    
    # Create DataFrame with correct feature order
    df = pd.DataFrame([features])
    
    # Ensure all expected features exist
    for feature in feature_names:
        if feature not in df.columns:
            df[feature] = 0
    
    return df[feature_names]

def predict_with_model(features):
    """Make prediction using ML model"""
    # Scale features
    features_scaled = scaler.transform(features)
    
    # Predict
    prediction = model.predict(features_scaled)[0]
    probabilities = model.predict_proba(features_scaled)[0]
    confidence = probabilities[prediction]
    
    return prediction, confidence

def simple_prediction(temp, hum, vib, current):
    """Simple rule-based prediction when model isn't available"""
    # Simple rules for demo
    if vib == 1 and current > 1.5:
        return 3, 0.85  # Dislodgement
    elif current > 1.2:
        return 2, 0.75  # Major tear
    elif vib == 1:
        return 1, 0.65  # Minor tear
    else:
        return 0, 0.90  # Normal

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
