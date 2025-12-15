import numpy as np
import pandas as pd
from src.train import load_data

# Define the target column
TARGET_COLUMN = "is_high_risk"

# Define numeric feature columns
NUMERIC_FEATURES = ["recency_days", "frequency", "monetary"]


def test_training_data_loads():
    df = load_data()
    
    # Check that target column exists
    assert TARGET_COLUMN in df.columns
    
    # Separate features and target
    X = df.drop(TARGET_COLUMN, axis=1)
    y = df[TARGET_COLUMN]
    
    # Shape consistency
    assert len(X) == len(y)
    assert X.shape[0] > 0
    assert X.shape[1] > 0

    # No missing values in target
    assert not y.isnull().any()
    
    # No missing values in numeric features
    assert not df[NUMERIC_FEATURES].isnull().any().any()
    
    # Numeric features are actually numeric
    assert all(np.issubdtype(df[col].dtype, np.number) for col in NUMERIC_FEATURES)

    # Target sanity: should only contain 0 or 1
    assert set(y.unique()).issubset({0, 1})


def test_training_data_finite_values():
    df = load_data()
    
    # Only numeric features
    X_numeric = df[NUMERIC_FEATURES]
    
    # Check that all values are finite
    assert np.isfinite(X_numeric.values).all()