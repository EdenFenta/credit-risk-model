import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import logging

# Configuration
RAW_DATA_PATH = Path("data/raw/data.csv")
PROCESSED_DATA_PATH = Path("data/processed/features.csv")

NUMERIC_FEATURES = [
    "recency_days",
    "transaction_count",
    "total_transaction_value",
    "avg_transaction_value",
]

CATEGORICAL_FEATURES = [
    "ProductCategory",
    "ChannelId",
    "PricingStrategy",
]

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Feature Engineering Helpers
def load_raw_data() -> pd.DataFrame:
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(f"Missing raw data at {RAW_DATA_PATH}")
    df = pd.read_csv(RAW_DATA_PATH)
    df["TransactionStartTime"] = pd.to_datetime(df["TransactionStartTime"])
    return df


def build_customer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate transaction-level data to customer-level features.
    """
    snapshot_date = df["TransactionStartTime"].max() + pd.Timedelta(days=1)

    rfm = df.groupby("CustomerId").agg(
        recency_days=("TransactionStartTime", lambda x: (snapshot_date - x.max()).days),
        transaction_count=("TransactionId", "count"),
        total_transaction_value=("Value", "sum"),
        avg_transaction_value=("Value", "mean"),
    ).reset_index()

    return rfm


# Pipeline Construction
def build_preprocessing_pipeline() -> ColumnTransformer:
    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer(transformers=[
        ("num", numeric_pipeline, NUMERIC_FEATURES),
        ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
    ])


# Main Processing Logic
def main() -> None:
    logger.info("Loading raw data")
    df = load_raw_data()

    logger.info("Building customer-level features")
    customer_df = build_customer_features(df)

    # Merge categorical attributes (mode per customer)
    cat_df = df.groupby("CustomerId")[CATEGORICAL_FEATURES].agg(
        lambda x: x.mode().iloc[0]
    ).reset_index()

    full_df = customer_df.merge(cat_df, on="CustomerId", how="left")

    logger.info("Applying preprocessing pipeline")
    preprocessor = build_preprocessing_pipeline()
    transformed = preprocessor.fit_transform(full_df)

    feature_names = (
        NUMERIC_FEATURES +
        list(preprocessor.named_transformers_["cat"]
             .named_steps["encoder"]
             .get_feature_names_out(CATEGORICAL_FEATURES))
    )

    processed_df = pd.DataFrame(transformed, columns=feature_names)
    processed_df["CustomerId"] = full_df["CustomerId"].values

    PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    processed_df.to_csv(PROCESSED_DATA_PATH, index=False)

    logger.info("Saved processed features to %s", PROCESSED_DATA_PATH)


if __name__ == "__main__":
    main()
