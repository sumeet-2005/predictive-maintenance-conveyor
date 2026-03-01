#!/usr/bin/env python3
"""
Data Collection Module for Conveyor Belt Predictive Maintenance
Collects data from:
- Camera for belt visual inspection
- Vibration sensor
- Temperature sensor
- Motor current sensor
"""

import cv2
import numpy as np
import RPi.GPIO as GPIO
import Adafruit_DHT
import time
import csv
import os
from datetime import datetime
from picamera import PiCamera
from picamera.array import PiRGBArray

class ConveyorDataCollector:
    def __init__(self, data_dir='../data'):
        self.data_dir = data_dir
        self.camera = PiCamera()
        self.camera.resolution = (640, 480)
        self.camera.framerate = 30
        self.raw_capture = PiRGBArray(self.camera, size=(640, 480))
        
        # Setup GPIO pins
        GPIO.setmode(GPIO.BCM)
        self.vibration_pin = 17  # GPIO17 for vibration sensor
        self.current_pin = 18     # GPIO18 for current sensor (ADC)
        GPIO.setup(self.vibration_pin, GPIO.IN)
        
        # DHT11 temperature/humidity sensor
        self.dht_pin = 4  # GPIO4
        
        # Create data directory if not exists
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
            
        # CSV file for logging
        self.csv_file = os.path.join(data_dir, f'sensor_data_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
        self.init_csv()
        
    def init_csv(self):
        """Initialize CSV with headers"""
        with open(self.csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'vibration', 'temperature', 'humidity', 
                           'motor_current', 'belt_condition', 'image_path'])
    
    def read_vibration(self):
        """Read vibration sensor (digital)"""
        return GPIO.input(self.vibration_pin)
    
    def read_temperature_humidity(self):
        """Read DHT11 sensor"""
        humidity, temperature = Adafruit_DHT.read_retry(Adafruit_DHT.DHT11, self.dht_pin)
        return temperature, humidity
    
    def read_motor_current(self):
        """Read motor current sensor (simulated ADC reading)"""
        # In real setup, you'd use an ADC like MCP3008
        # This is a placeholder - implement actual ADC reading
        return np.random.uniform(0.5, 2.0)  # Simulated current in Amps
    
    def capture_image(self):
        """Capture image from camera"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        image_path = os.path.join(self.data_dir, f'images/belt_{timestamp}.jpg')
        
        # Ensure images directory exists
        os.makedirs(os.path.dirname(image_path), exist_ok=True)
        
        # Capture image
        self.camera.capture(image_path)
        return image_path
    
    def analyze_belt_condition(self, image_path):
        """
        Analyze belt condition from image
        Returns: 0 = Normal, 1 = Minor tear, 2 = Major tear, 3 = Dislodgement
        """
        # Load and preprocess image
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Simple edge detection for tears
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        # Check for dislodgements using contour detection
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Rule-based classification (in real project, use ML model)
        if len(contours) > 10:
            return 3  # Dislodgement
        elif edge_density > 0.15:
            return 2  # Major tear
        elif edge_density > 0.05:
            return 1  # Minor tear
        else:
            return 0  # Normal
    
    def collect_data(self, duration_minutes=60, sample_interval_seconds=5):
        """
        Collect data for specified duration
        Args:
            duration_minutes: Total collection time in minutes
            sample_interval_seconds: Time between samples in seconds
        """
        total_samples = int(duration_minutes * 60 / sample_interval_seconds)
        print(f"Starting data collection for {duration_minutes} minutes...")
        print(f"Will collect {total_samples} samples every {sample_interval_seconds} seconds")
        
        for i in range(total_samples):
            try:
                # Collect sensor data
                timestamp = datetime.now().isoformat()
                vibration = self.read_vibration()
                temperature, humidity = self.read_temperature_humidity()
                motor_current = self.read_motor_current()
                
                # Capture and analyze image
                image_path = self.capture_image()
                belt_condition = self.analyze_belt_condition(image_path)
                
                # Save to CSV
                with open(self.csv_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([timestamp, vibration, temperature, humidity, 
                                   motor_current, belt_condition, image_path])
                
                print(f"Sample {i+1}/{total_samples} - Temp: {temperature:.1f}°C, "
                      f"Belt condition: {belt_condition}")
                
                # Wait for next sample
                time.sleep(sample_interval_seconds)
                
            except KeyboardInterrupt:
                print("\nData collection stopped by user")
                break
            except Exception as e:
                print(f"Error collecting data: {e}")
                continue
        
        self.cleanup()
    
    def cleanup(self):
        """Clean up GPIO and camera"""
        self.camera.close()
        GPIO.cleanup()
        print("Cleanup complete")

if __name__ == "__main__":
    collector = ConveyorDataCollector()
    # Collect data for 10 minutes with 5-second intervals
    collector.collect_data(duration_minutes=10, sample_interval_seconds=5)
