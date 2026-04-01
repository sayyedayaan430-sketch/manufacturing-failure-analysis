"""
config.py
─────────
Central configuration for Manufacturing Failure Analysis.
All paths, column names, and model settings are defined here.

Usage:
    from config import DATA_PATH, TARGET_COL, MODEL_PATH
"""

import os

# ─── Base Directories ──────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR      = os.path.join(BASE_DIR, 'data')
RAW_DIR       = os.path.join(DATA_DIR, 'raw')
PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')
MODEL_DIR     = os.path.join(BASE_DIR, 'models')
CHARTS_DIR    = os.path.join(BASE_DIR, 'outputs', 'charts')
ASSETS_DIR    = os.path.join(BASE_DIR, 'assets')

# ─── Data File Paths ───────────────────────────────────────────────────────────
DATA_PATH      = os.path.join(RAW_DIR, 'manufacturing_data.csv')  # ← Update filename
PROCESSED_PATH = os.path.join(PROCESSED_DIR, 'clean_data.csv')

# ─── Model Artifact Paths ──────────────────────────────────────────────────────
MODEL_PATH  = os.path.join(MODEL_DIR, 'failure_model.pkl')
SCALER_PATH = os.path.join(MODEL_DIR, 'scaler.pkl')

# ─── Dataset Column Names ──────────────────────────────────────────────────────
# Update these to match your actual CSV column names
TARGET_COL    = 'failure'          # 1 = failure, 0 = no failure
MACHINE_COL   = 'machine_id'       # Machine identifier
TIME_COL      = 'timestamp'        # Date/time column
FAILURE_TYPE  = 'failure_type'     # Type of failure (if available)

# Feature columns used for training
FEATURE_COLS  = [
    'temperature',      # Machine temperature (°C)
    'vibration',        # Vibration level
    'pressure',         # Pressure (PSI)
    'rotational_speed', # RPM
    'torque',           # Torque (Nm)
    'tool_wear',        # Tool wear (minutes)
]

# ─── Preprocessing Settings ────────────────────────────────────────────────────
TEST_SIZE    = 0.2     # 20% of data used for testing
RANDOM_STATE = 42      # Seed for reproducibility

# ─── Model Hyperparameters (Random Forest) ────────────────────────────────────
N_ESTIMATORS = 100     # Number of trees in the forest
MAX_DEPTH    = 10      # Maximum depth of each tree
MIN_SAMPLES  = 5       # Minimum samples required to split a node

# ─── Prediction Settings ───────────────────────────────────────────────────────
FAILURE_THRESHOLD = 0.5   # Probability above this = predicted failure

# ─── Chart Settings ────────────────────────────────────────────────────────────
CHART_STYLE  = 'dark_background'   # Matplotlib style
CHART_DPI    = 150                 # Chart image resolution
PRIMARY_COLOR   = '#2ea043'        # Green (GitHub style)
SECONDARY_COLOR = '#388bfd'        # Blue
DANGER_COLOR    = '#da3633'        # Red (failure color)


if __name__ == '__main__':
    print("─── Manufacturing Failure Analysis — Config ───")
    print(f"  Data Path    : {DATA_PATH}")
    print(f"  Model Path   : {MODEL_PATH}")
    print(f"  Charts Dir   : {CHARTS_DIR}")
    print(f"  Features     : {FEATURE_COLS}")
    print(f"  Target Col   : {TARGET_COL}")
