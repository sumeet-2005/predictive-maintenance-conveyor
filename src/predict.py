#!/usr/bin/env python3
"""
Real-time Prediction Module
Runs on Raspberry Pi to predict belt failures in real-time
"""

import cv2
import numpy as np
import RPi.GPIO as GPIO
import Adafruit_DHT
import time
import joblib
import os
from datetime import datetime
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

class ConveyorPredictor:
    def __init__(self, model_dir='../models'):
        # Load model and scaler
        self.model = joblib.load(os.path.join(model_dir, 'conveyor_model.pkl'))
        self.scaler = joblib.load(os.path.join(model_dir, 'scaler.pkl'))
        
        # Load feature names
        with open(os.path.join(model_dir, 'feature_names.txt'), 'r') as f:
            self.feature_names = [line.strip() for line in f.readlines()]
        
        # Setup GPIO
        GPIO.setmode(GPIO.BCM)
        self.vibration_pin = 17
        self.current_pin = 18
        GPIO.setup(self.vibration_pin, GPIO.IN)
        
        # Alert pins
        self.green_led = 22
        self.yellow_led = 23
        self.red_led = 24
        self.buzzer = 25
        
        GPIO.setup(self.green_led, GPIO.OUT)
        GPIO.setup(self.yellow_led, GPIO.OUT)
        GPIO.setup(self.red_led, GPIO.OUT)
        GPIO.setup(self.buzzer, GPIO.OUT)
        
        # DHT sensor
        self.dht_pin = 4
        
        # History buffer for derived features
        self.vibration_history = []
        self.history_length = 5
        
        print("="*50)
        print("Conveyor Belt Predictive Maintenance System")
        print("="*50)
        print("System initialized. Monitoring conveyor belt...")
        
    def read_sensors(self):
        """Read all sensor values"""
        # Read vibration
        vibration = GPIO.input(self.vibration_pin)
        
        # Read temperature and humidity
        humidity, temperature = Adafruit_DHT.read_retry(Adafruit_DHT.DHT11, self.dht_pin)
        
        # Read motor current (simulated ADC)
        motor_current = np.random.uniform(0.5, 2.0)
        
        return {
            'vibration': vibration,
            'temperature': temperature,
            'humidity': humidity,
            'motor_current': motor_current
        }
    
    def prepare_features(self, sensor_data):
        """Prepare features for prediction"""
        # Update history
        self.vibration_history.append(sensor_data['vibration'])
        if len(self.vibration_history) > self.history_length:
            self.vibration_history.pop(0)
        
        # Create feature vector
        features = {
            'vibration': sensor_data['vibration'],
            'temperature': sensor_data['temperature'],
            'humidity': sensor_data['humidity'],
            'motor_current': sensor_data['motor_current'],
            'temp_humidity_ratio': sensor_data['temperature'] / (sensor_data['humidity'] + 1),
            'vibration_lag1': self.vibration_history[-2] if len(self.vibration_history) > 1 else 0,
            'vibration_lag2': self.vibration_history[-3] if len(self.vibration_history) > 2 else 0,
            'vibration_rolling_mean': np.mean(self.vibration_history) if self.vibration_history else 0,
            'vibration_rolling_std': np.std(self.vibration_history) if len(self.vibration_history) > 1 else 0
        }
        
        # Create DataFrame with correct feature order
        df = pd.DataFrame([features])[self.feature_names]
        return df
    
    def set_alerts(self, prediction, confidence):
        """Set visual/audio alerts based on prediction"""
        # Turn off all alerts first
        GPIO.output(self.green_led, GPIO.LOW)
        GPIO.output(self.yellow_led, GPIO.LOW)
        GPIO.output(self.red_led, GPIO.LOW)
        GPIO.output(self.buzzer, GPIO.LOW)
        
        if prediction == 0:  # Normal
            GPIO.output(self.green_led, GPIO.HIGH)
            status = "✅ NORMAL"
            color = '\033[92m'  # Green
        elif prediction == 1:  # Minor tear
            GPIO.output(self.yellow_led, GPIO.HIGH)
            status = "⚠️ MINOR TEAR DETECTED"
            color = '\033[93m'  # Yellow
        elif prediction == 2:  # Major tear
            GPIO.output(self.red_led, GPIO.HIGH)
            GPIO.output(self.buzzer, GPIO.HIGH)
            time.sleep(0.5)
            GPIO.output(self.buzzer, GPIO.LOW)
            status = "🔴 MAJOR TEAR DETECTED"
            color = '\033[91m'  # Red
        else:  # Dislodgement
            GPIO.output(self.red_led, GPIO.HIGH)
            for _ in range(3):
                GPIO.output(self.buzzer, GPIO.HIGH)
                time.sleep(0.2)
                GPIO.output(self.buzzer, GPIO.LOW)
                time.sleep(0.2)
            status = "🚨 DISLODGEMENT IMMINENT"
            color = '\033[91;1m'  # Bold Red
        
        reset = '\033[0m'
        print(f"{color}{status} (confidence: {confidence:.2%}){reset}")
    
    def run(self, interval_seconds=2):
        """Main prediction loop"""
        try:
            while True:
                # Read sensors
                sensor_data = self.read_sensors()
                
                if None not in sensor_data.values():
                    # Prepare features
                    X = self.prepare_features(sensor_data)
                    
                    # Scale features
                    X_scaled = self.scaler.transform(X)
                    
                    # Make prediction
                    prediction = self.model.predict(X_scaled)[0]
                    probabilities = self.model.predict_proba(X_scaled)[0]
                    confidence = probabilities[prediction]
                    
                    # Show timestamp and sensor values
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    print(f"\n[{timestamp}] T:{sensor_data['temperature']:.1f}°C "
                          f"H:{sensor_data['humidity']:.1f}% "
                          f"V:{sensor_data['vibration']} "
                          f"I:{sensor_data['motor_current']:.2f}A")
                    
                    # Set alerts
                    self.set_alerts(prediction, confidence)
                
                time.sleep(interval_seconds)
                
        except KeyboardInterrupt:
            print("\nShutting down...")
        finally:
            GPIO.cleanup()

if __name__ == "__main__":
    predictor = ConveyorPredictor()
    predictor.run()
