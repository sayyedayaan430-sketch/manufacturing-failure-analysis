"""
train.py
────────
Trains a Random Forest classifier to predict manufacturing equipment failures.

Pipeline:
  1. Load preprocessed data
  2. Split into train / test sets
  3. Train Random Forest model
  4. Evaluate with accuracy, precision, recall, F1, ROC-AUC
  5. Save trained model to models/
"""

import os
import pickle
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score,
    roc_auc_score, classification_report,
    confusion_matrix
)

from config import (
    PROCESSED_PATH, MODEL_PATH, MODEL_DIR,
    FEATURE_COLS, TARGET_COL,
    N_ESTIMATORS, MAX_DEPTH, MIN_SAMPLES,
    TEST_SIZE, RANDOM_STATE
)


def load_processed_data() -> pd.DataFrame:
    """Load the cleaned and preprocessed dataset."""
    if not os.path.exists(PROCESSED_PATH):
        raise FileNotFoundError(
            f"Processed data not found: {PROCESSED_PATH}\n"
            "Please run `python src/preprocess.py` first."
        )
    df = pd.read_csv(PROCESSED_PATH)
    print(f"  ✅ Loaded {len(df)} rows from {PROCESSED_PATH}")
    return df


def get_features_and_target(df: pd.DataFrame):
    """
    Split DataFrame into feature matrix X and target vector y.

    Returns:
        X (pd.DataFrame): Feature columns
        y (pd.Series): Target column (0 = no failure, 1 = failure)
    """
    existing_features = [c for c in FEATURE_COLS if c in df.columns]

    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found in dataset.")

    X = df[existing_features]
    y = df[TARGET_COL]

    print(f"  ✅ Features used: {existing_features}")
    print(f"  ✅ Target: '{TARGET_COL}'")
    print(f"  ✅ Class balance — No Failure: {(y==0).sum()}  |  Failure: {(y==1).sum()}")
    return X, y


def train():
    """Run the full model training pipeline."""

    print("\n" + "═" * 55)
    print("   MANUFACTURING FAILURE ANALYSIS — TRAINING")
    print("═" * 55)

    # ── Step 1: Load Data ──────────────────────────────────────────────────────
    print("\n[1/4] Loading preprocessed data...")
    df = load_processed_data()

    # ── Step 2: Prepare Features ──────────────────────────────────────────────
    print("\n[2/4] Preparing features and target...")
    X, y = get_features_and_target(df)

    # Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"  ✅ Train: {len(X_train)} rows  |  Test: {len(X_test)} rows")

    # ── Step 3: Train Model ───────────────────────────────────────────────────
    print("\n[3/4] Training Random Forest model...")
    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,   # 100 decision trees
        max_depth=MAX_DEPTH,          # Limit tree depth to avoid overfitting
        min_samples_split=MIN_SAMPLES,
        class_weight='balanced',      # Handle imbalanced failure data
        random_state=RANDOM_STATE,
        n_jobs=-1                     # Use all CPU cores
    )
    model.fit(X_train, y_train)
    print("  ✅ Training complete!")

    # ── Step 4: Evaluate ──────────────────────────────────────────────────────
    print("\n[4/4] Evaluating model...")
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print("\n" + "─" * 55)
    print("   EVALUATION RESULTS")
    print("─" * 55)
    print(f"  Accuracy  : {accuracy_score(y_test, y_pred):.4f}")
    print(f"  Precision : {precision_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"  Recall    : {recall_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"  F1 Score  : {f1_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"  ROC-AUC   : {roc_auc_score(y_test, y_prob):.4f}")

    print("\n  Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"              Predicted")
    print(f"              No Fail  Fail")
    print(f"  Actual No Fail  {cm[0][0]:>5}  {cm[0][1]:>5}")
    print(f"  Actual Fail     {cm[1][0]:>5}  {cm[1][1]:>5}")

    # ── Save Model ────────────────────────────────────────────────────────────
    print("\n" + "─" * 55)
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
    print(f"  ✅ Model saved → {MODEL_PATH}")
    print("\n✅ Training complete!\n")


if __name__ == '__main__':
    train()
