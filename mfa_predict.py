"""
predict.py
──────────
Loads the trained model and predicts failure probability
for new machine sensor readings.

Usage (command line):
    python predict.py --temp 320 --vibration 0.85 --pressure 142

Usage (as module):
    from predict import predict_failure
    result = predict_failure({'temperature': 320, 'vibration': 0.85})
"""

import pickle
import argparse
import os
import numpy as np
import pandas as pd

from config import MODEL_PATH, SCALER_PATH, FEATURE_COLS, FAILURE_THRESHOLD

# Cache to avoid reloading on every call
_cache = {}


def load_artifacts() -> tuple:
    """Load model and scaler from disk with caching."""
    if _cache:
        return _cache['model'], _cache['scaler']

    for path in [MODEL_PATH, SCALER_PATH]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Artifact not found: {path}\n"
                "Please run `python src/train.py` first."
            )

    with open(MODEL_PATH,  'rb') as f: _cache['model']  = pickle.load(f)
    with open(SCALER_PATH, 'rb') as f: _cache['scaler'] = pickle.load(f)

    return _cache['model'], _cache['scaler']


def predict_failure(sensor_data: dict) -> dict:
    """
    Predict whether a machine will fail based on sensor readings.

    Args:
        sensor_data (dict): Dictionary of sensor readings.
                            Keys should match FEATURE_COLS in config.py
                            e.g. {'temperature': 320, 'vibration': 0.85,
                                  'pressure': 142, 'tool_wear': 200}

    Returns:
        dict: Prediction result with keys:
              - 'prediction': 'FAILURE' or 'NORMAL'
              - 'failure_probability': float (0.0 to 1.0)
              - 'risk_level': 'HIGH', 'MEDIUM', or 'LOW'
              - 'message': Human-readable result string
    """
    model, scaler = load_artifacts()

    # Build input DataFrame using only known feature columns
    existing_cols = [c for c in FEATURE_COLS if c in sensor_data]
    input_df = pd.DataFrame([sensor_data])[existing_cols]

    # Scale the input using the saved scaler
    input_scaled = scaler.transform(input_df)

    # Get failure probability
    prob = model.predict_proba(input_scaled)[0][1]
    prediction = 'FAILURE' if prob >= FAILURE_THRESHOLD else 'NORMAL'

    # Determine risk level
    if prob >= 0.75:
        risk = 'HIGH'
    elif prob >= 0.40:
        risk = 'MEDIUM'
    else:
        risk = 'LOW'

    # Human-readable message
    if prediction == 'FAILURE':
        message = f"⚠️  HIGH RISK — Machine is likely to fail ({prob:.1%} confidence). Immediate inspection recommended."
    elif risk == 'MEDIUM':
        message = f"🔶 MEDIUM RISK — Monitor closely ({prob:.1%} failure probability)."
    else:
        message = f"✅ LOW RISK — Machine is operating normally ({prob:.1%} failure probability)."

    return {
        'prediction': prediction,
        'failure_probability': round(float(prob), 4),
        'risk_level': risk,
        'message': message
    }


# ── Command-line interface ─────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict manufacturing equipment failure')
    parser.add_argument('--temp',       type=float, help='Temperature (°C)')
    parser.add_argument('--vibration',  type=float, help='Vibration level')
    parser.add_argument('--pressure',   type=float, help='Pressure (PSI)')
    parser.add_argument('--rpm',        type=float, help='Rotational speed (RPM)')
    parser.add_argument('--torque',     type=float, help='Torque (Nm)')
    parser.add_argument('--tool_wear',  type=float, help='Tool wear (minutes)')
    args = parser.parse_args()

    sensor_input = {
        'temperature':      args.temp,
        'vibration':        args.vibration,
        'pressure':         args.pressure,
        'rotational_speed': args.rpm,
        'torque':           args.torque,
        'tool_wear':        args.tool_wear,
    }
    # Remove None values
    sensor_input = {k: v for k, v in sensor_input.items() if v is not None}

    print(f"\nSensor Input: {sensor_input}")
    print("─" * 50)
    result = predict_failure(sensor_input)
    print(result['message'])
    print(f"Risk Level        : {result['risk_level']}")
    print(f"Failure Probability: {result['failure_probability']:.1%}")
