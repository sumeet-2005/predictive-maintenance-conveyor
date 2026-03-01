#!/usr/bin/env python3
"""
Train Machine Learning Model for Conveyor Belt Predictive Maintenance
Uses collected sensor data to predict belt failures
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os
import glob

class PredictiveMaintenanceModel:
    def __init__(self, model_dir='../models'):
        self.model_dir = model_dir
        self.model = None
        self.scaler = StandardScaler()
        
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
    
    def load_data(self, data_files):
        """
        Load and combine multiple CSV data files
        Args:
            data_files: List of CSV file paths or glob pattern
        """
        if isinstance(data_files, str):
            data_files = glob.glob(data_files)
        
        dfs = []
        for file in data_files:
            df = pd.read_csv(file)
            dfs.append(df)
        
        data = pd.concat(dfs, ignore_index=True)
        print(f"Loaded {len(data)} samples from {len(data_files)} files")
        return data
    
    def prepare_features(self, data):
        """
        Prepare features for training
        """
        # Select features (excluding timestamp, image_path, and label)
        feature_columns = ['vibration', 'temperature', 'humidity', 'motor_current']
        X = data[feature_columns].copy()
        
        # Handle missing values
        X = X.fillna(X.mean())
        
        # Add derived features
        X['temp_humidity_ratio'] = X['temperature'] / (X['humidity'] + 1)
        X['vibration_lag1'] = X['vibration'].shift(1).fillna(0)
        X['vibration_lag2'] = X['vibration'].shift(2).fillna(0)
        
        # Rolling statistics
        X['vibration_rolling_mean'] = X['vibration'].rolling(window=5, min_periods=1).mean()
        X['vibration_rolling_std'] = X['vibration'].rolling(window=5, min_periods=1).std().fillna(0)
        
        # Target variable
        y = data['belt_condition']
        
        return X, y
    
    def train(self, X, y, test_size=0.2):
        """
        Train Random Forest model
        """
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test_scaled)
        print("\n=== Model Performance ===")
        print(classification_report(y_test, y_pred, 
              target_names=['Normal', 'Minor Tear', 'Major Tear', 'Dislodgement']))
        
        # Feature importance
        feature_names = X.columns.tolist()
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        print("\n=== Top 5 Important Features ===")
        for i in range(min(5, len(feature_names))):
            print(f"{i+1}. {feature_names[indices[i]]}: {importances[indices[i]]:.3f}")
        
        return self.model
    
    def save_model(self, filename='conveyor_model.pkl'):
        """Save trained model and scaler"""
        model_path = os.path.join(self.model_dir, filename)
        scaler_path = os.path.join(self.model_dir, 'scaler.pkl')
        
        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler, scaler_path)
        
        print(f"Model saved to {model_path}")
        print(f"Scaler saved to {scaler_path}")
        
        # Save feature names
        feature_names_path = os.path.join(self.model_dir, 'feature_names.txt')
        with open(feature_names_path, 'w') as f:
            f.write('\n'.join(self.feature_names_))
    
    def load_model(self, filename='conveyor_model.pkl'):
        """Load trained model"""
        model_path = os.path.join(self.model_dir, filename)
        scaler_path = os.path.join(self.model_dir, 'scaler.pkl')
        
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        print(f"Model loaded from {model_path}")
        return self.model

if __name__ == "__main__":
    # Initialize model trainer
    trainer = PredictiveMaintenanceModel()
    
    # Load all CSV data files
    data_files = '../data/sensor_data_*.csv'
    data = trainer.load_data(data_files)
    
    # Prepare features
    X, y = trainer.prepare_features(data)
    trainer.feature_names_ = X.columns.tolist()
    
    # Train model
    trainer.train(X, y)
    
    # Save model
    trainer.save_model()
