#!/usr/bin/env python3
"""
Utility functions for the predictive maintenance system
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_sensor_data(csv_file, output_dir='../outputs'):
    """Plot sensor data from CSV file"""
    df = pd.read_csv(csv_file)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Temperature
    axes[0, 0].plot(pd.to_datetime(df['timestamp']), df['temperature'])
    axes[0, 0].set_title('Temperature Over Time')
    axes[0, 0].set_ylabel('Temperature (°C)')
    
    # Humidity
    axes[0, 1].plot(pd.to_datetime(df['timestamp']), df['humidity'])
    axes[0, 1].set_title('Humidity Over Time')
    axes[0, 1].set_ylabel('Humidity (%)')
    
    # Vibration
    axes[1, 0].plot(pd.to_datetime(df['timestamp']), df['vibration'])
    axes[1, 0].set_title('Vibration Over Time')
    axes[1, 0].set_ylabel('Vibration (digital)')
    
    # Motor Current
    axes[1, 1].plot(pd.to_datetime(df['timestamp']), df['motor_current'])
    axes[1, 1].set_title('Motor Current Over Time')
    axes[1, 1].set_ylabel('Current (A)')
    
    plt.tight_layout()
    
    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'sensor_data_plot.png')
    plt.savefig(output_file)
    print(f"Plot saved to {output_file}")
    plt.show()

def generate_report(data_file, output_dir='../outputs'):
    """Generate analysis report from collected data"""
    df = pd.read_csv(data_file)
    
    report = []
    report.append("="*50)
    report.append("CONVEYOR BELT PREDICTIVE MAINTENANCE REPORT")
    report.append("="*50)
    report.append(f"Data file: {data_file}")
    report.append(f"Total samples: {len(df)}")
    report.append(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    report.append("\n=== BELT CONDITION SUMMARY ===")
    
    condition_counts = df['belt_condition'].value_counts().sort_index()
    condition_names = ['Normal', 'Minor Tear', 'Major Tear', 'Dislodgement']
    
    for i, count in condition_counts.items():
        percentage = (count / len(df)) * 100
        report.append(f"{condition_names[int(i)]}: {count} samples ({percentage:.1f}%)")
    
    report.append("\n=== SENSOR STATISTICS ===")
    for col in ['temperature', 'humidity', 'motor_current']:
        report.append(f"\n{col.upper()}:")
        report.append(f"  Mean: {df[col].mean():.2f}")
        report.append(f"  Std: {df[col].std():.2f}")
        report.append(f"  Min: {df[col].min():.2f}")
        report.append(f"  Max: {df[col].max():.2f}")
    
    # Save report
    os.makedirs(output_dir, exist_ok=True)
    report_file = os.path.join(output_dir, 'analysis_report.txt')
    with open(report_file, 'w') as f:
        f.write('\n'.join(report))
    
    print(f"Report saved to {report_file}")
    return report

if __name__ == "__main__":
    # Example usage
    import glob
    csv_files = glob.glob('../data/sensor_data_*.csv')
    if csv_files:
        latest_file = max(csv_files, key=os.path.getctime)
        print(f"Analyzing latest file: {latest_file}")
        plot_sensor_data(latest_file)
        generate_report(latest_file)
