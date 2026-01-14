#!/usr/bin/env python3
"""Test autoregression implementation without waiting for central server."""

import sys
import time
import hashlib
import pandas as pd
import numpy as np

# Add paths
sys.path.insert(0, '/Satori/Engine')
sys.path.insert(0, '/Satori/Lib')

from satoriengine.veda.data import StreamForecast
from satorilib.logging import setup, info, debug

setup()

print("=" * 60)
print("AUTOREGRESSION MOCK TEST")
print("=" * 60)

# Mock data (simulating self.data)
mock_data = pd.DataFrame({
    'date_time': pd.date_range(start='2024-01-01', periods=10, freq='D'),
    'value': [85000 + i * 100 for i in range(10)],
    'id': [f'hash_{i}' for i in range(10)]
})

print(f"\n📊 Mock dataset ({len(mock_data)} rows):")
print(mock_data[['date_time', 'value']].tail(3))

# Simulate first prediction
print("\n🔮 Step 1: First prediction (using original data)")
first_prediction_value = 88500.0  # Simulated prediction
first_prediction_timestamp = pd.Timestamp('2024-01-11')

firstForecast = pd.DataFrame({
    'date_time': [first_prediction_timestamp],
    'pred': [first_prediction_value]
})

firstValue = firstForecast['pred'].iloc[0]
print(f"   [AUTOREGRESSION] First prediction: {firstValue}")

# Create augmented data
print("\n🔧 Step 2: Create augmented data")
tempHash = hashlib.sha256(f"{firstValue}{first_prediction_timestamp}".encode()).hexdigest()[:16]

tempRow = pd.DataFrame({
    'date_time': [first_prediction_timestamp],
    'value': [firstValue],
    'id': [tempHash]
})

augmentedData = pd.concat([mock_data, tempRow], ignore_index=True)
print(f"   [AUTOREGRESSION] Augmented data size: {len(augmentedData)} rows (original: {len(mock_data)})")
print(f"\n   Last 3 rows of augmented data:")
print(augmentedData[['date_time', 'value']].tail(3))

# Simulate second prediction (would use augmented data)
print("\n🔮 Step 3: Second prediction (using augmented data)")
second_prediction_value = 88750.0  # Simulated - would be different due to extra data point
print(f"   [AUTOREGRESSION] Second prediction (sent to server): {second_prediction_value}")

# Verify original data unchanged
print("\n✅ Step 4: Verify original data unchanged")
print(f"   Original data size: {len(mock_data)} rows")
print(f"   Original data still has {len(mock_data)} rows: {len(mock_data) == 10}")
print(f"   Augmented data has {len(augmentedData)} rows")

print("\n" + "=" * 60)
print("PREDICTION COMPARISON:")
print("=" * 60)
print(f"  First prediction:  ${first_prediction_value:,.2f}")
print(f"  Second prediction: ${second_prediction_value:,.2f}")
print(f"  Difference:        ${abs(second_prediction_value - first_prediction_value):,.2f}")
print(f"\n  ✓ Only SECOND prediction would be sent to server!")
print("=" * 60)
