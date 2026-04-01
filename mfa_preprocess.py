"""
preprocess.py
─────────────
Data cleaning and preprocessing pipeline for manufacturing sensor data.

Steps:
  1. Load raw CSV data
  2. Handle missing values
  3. Remove outliers using IQR method
  4. Encode categorical columns
  5. Scale numerical features
  6. Save cleaned data to processed/
"""

import os
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

from config import (
    DATA_PATH, PROCESSED_PATH, SCALER_PATH,
    FEATURE_COLS, TARGET_COL, PROCESSED_DIR
)


def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    """
    Load raw CSV dataset from disk.

    Args:
        path (str): Path to the CSV file

    Returns:
        pd.DataFrame: Loaded raw DataFrame
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found: {path}\n"
            "Please place your CSV file in data/raw/ and update config.py"
        )
    df = pd.read_csv(path)
    print(f"  ✅ Loaded {len(df)} rows, {len(df.columns)} columns from {path}")
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values in the dataset.
    - Numerical columns: fill with median
    - Categorical columns: fill with mode

    Args:
        df (pd.DataFrame): Raw DataFrame

    Returns:
        pd.DataFrame: DataFrame with no missing values
    """
    missing = df.isnull().sum()
    if missing.sum() == 0:
        print("  ✅ No missing values found")
        return df

    print(f"  ⚠️  Found missing values:\n{missing[missing > 0]}")

    # Fill numerical columns with median
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())

    # Fill categorical columns with mode
    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    print(f"  ✅ Missing values filled")
    return df


def remove_outliers(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """
    Remove outliers using the IQR (Interquartile Range) method.
    Rows where any feature value falls outside 1.5x IQR are removed.

    Args:
        df (pd.DataFrame): Input DataFrame
        cols (list): Columns to check for outliers

    Returns:
        pd.DataFrame: DataFrame with outliers removed
    """
    original_len = len(df)

    for col in cols:
        if col not in df.columns:
            continue
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower) & (df[col] <= upper)]

    removed = original_len - len(df)
    print(f"  ✅ Removed {removed} outlier rows ({removed/original_len*100:.1f}%)")
    return df.reset_index(drop=True)


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode categorical columns using one-hot encoding.

    Args:
        df (pd.DataFrame): Input DataFrame

    Returns:
        pd.DataFrame: DataFrame with encoded categorical columns
    """
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()

    # Don't encode the target column if it's categorical
    if TARGET_COL in cat_cols:
        cat_cols.remove(TARGET_COL)

    if not cat_cols:
        print("  ✅ No categorical columns to encode")
        return df

    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    print(f"  ✅ Encoded columns: {cat_cols}")
    return df


def scale_features(df: pd.DataFrame,
                   cols: list,
                   save_scaler: bool = True) -> pd.DataFrame:
    """
    Standardize numerical feature columns (mean=0, std=1).

    Args:
        df (pd.DataFrame): Input DataFrame
        cols (list): Columns to scale
        save_scaler (bool): Whether to save the scaler to disk

    Returns:
        pd.DataFrame: DataFrame with scaled features
    """
    # Only scale columns that exist in the DataFrame
    existing_cols = [c for c in cols if c in df.columns]

    scaler = StandardScaler()
    df[existing_cols] = scaler.fit_transform(df[existing_cols])

    if save_scaler:
        os.makedirs(os.path.dirname(SCALER_PATH), exist_ok=True)
        with open(SCALER_PATH, 'wb') as f:
            pickle.dump(scaler, f)
        print(f"  ✅ Scaler saved to {SCALER_PATH}")

    return df


def preprocess(save: bool = True) -> pd.DataFrame:
    """
    Run the full preprocessing pipeline end-to-end.

    Args:
        save (bool): Whether to save the cleaned data to disk

    Returns:
        pd.DataFrame: Fully cleaned and preprocessed DataFrame
    """
    print("\n" + "═" * 50)
    print("   DATA PREPROCESSING PIPELINE")
    print("═" * 50)

    print("\n[1/5] Loading data...")
    df = load_data()

    print("\n[2/5] Handling missing values...")
    df = handle_missing_values(df)

    print("\n[3/5] Removing outliers...")
    df = remove_outliers(df, FEATURE_COLS)

    print("\n[4/5] Encoding categorical columns...")
    df = encode_categoricals(df)

    print("\n[5/5] Scaling features...")
    df = scale_features(df, FEATURE_COLS)

    if save:
        os.makedirs(PROCESSED_DIR, exist_ok=True)
        df.to_csv(PROCESSED_PATH, index=False)
        print(f"\n  ✅ Saved cleaned data → {PROCESSED_PATH}")

    print(f"\n  Final dataset: {len(df)} rows × {len(df.columns)} columns")
    print("\n✅ Preprocessing complete!\n")
    return df


if __name__ == '__main__':
    preprocess()
