import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import logging

# Configuration
RAW_DATA_PATH = Path("data/raw/data.csv")
OUTPUT_PATH = Path("data/processed/customer_with_target.csv")

RFM_COLUMNS = ["recency_days", "frequency", "monetary"]

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Data Loading
def load_data() -> pd.DataFrame:
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(f"Missing raw data at {RAW_DATA_PATH}")

    df = pd.read_csv(RAW_DATA_PATH)
    df["TransactionStartTime"] = pd.to_datetime(df["TransactionStartTime"])
    return df

# RFM Calculation
def calculate_rfm(df: pd.DataFrame) -> pd.DataFrame:
    snapshot_date = df["TransactionStartTime"].max() + pd.Timedelta(days=1)

    rfm = df.groupby("CustomerId").agg(
        recency_days=("TransactionStartTime", lambda x: (snapshot_date - x.max()).days),
        frequency=("TransactionId", "count"),
        monetary=("Value", "sum"),
    ).reset_index()

    return rfm

# Clustering and Target Assignment
def assign_high_risk_label(rfm_df: pd.DataFrame) -> pd.DataFrame:
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm_df[RFM_COLUMNS])

    kmeans = KMeans(n_clusters=3, random_state=42)
    rfm_df["cluster"] = kmeans.fit_predict(rfm_scaled)

    # Identify high-risk cluster:
    # High recency, low frequency, low monetary
    cluster_summary = rfm_df.groupby("cluster")[RFM_COLUMNS].mean()

    high_risk_cluster = cluster_summary[
        (cluster_summary["recency_days"] == cluster_summary["recency_days"].max())
    ].index[0]

    rfm_df["is_high_risk"] = (rfm_df["cluster"] == high_risk_cluster).astype(int)

    logger.info("High-risk cluster identified as cluster %s", high_risk_cluster)
    logger.info(
        "High-risk customers: %.2f%%",
        rfm_df["is_high_risk"].mean() * 100
    )

    return rfm_df

# Main
def main() -> None:
    logger.info("Loading data")
    df = load_data()

    logger.info("Calculating RFM metrics")
    rfm_df = calculate_rfm(df)

    logger.info("Assigning proxy risk label")
    rfm_with_target = assign_high_risk_label(rfm_df)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    rfm_with_target.to_csv(OUTPUT_PATH, index=False)

    logger.info("Saved dataset with target to %s", OUTPUT_PATH)

if __name__ == "__main__":
    main()