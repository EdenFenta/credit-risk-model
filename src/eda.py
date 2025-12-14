import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import logging

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
DATA_PATH = Path("data/raw/data.csv")
FIGURES_PATH = Path("reports/figures")
FIGURES_PATH.mkdir(parents=True, exist_ok=True)

REQUIRED_COLUMNS = {
    "TransactionId",
    "CustomerId",
    "TransactionStartTime",
    "Value",
    "ProductCategory",
}

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Data Loading & Validation
# -----------------------------------------------------------------------------
def load_data() -> pd.DataFrame:
    """
    Load raw transaction data and perform basic validation.

    Returns:
        pd.DataFrame: Cleaned transaction dataset
    """
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Data file not found at {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    missing_cols = REQUIRED_COLUMNS - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    df["TransactionStartTime"] = pd.to_datetime(
        df["TransactionStartTime"], errors="coerce"
    )

    logger.info("Data loaded successfully with shape %s", df.shape)
    return df

# -----------------------------------------------------------------------------
# EDA Visualization Functions
# -----------------------------------------------------------------------------
def transaction_value_distribution(df: pd.DataFrame) -> None:
    """
    Plot log-scaled distribution of transaction values
    to analyze skewness and outliers.
    """
    plt.figure(figsize=(8, 5))
    sns.histplot(np.log1p(df["Value"]), bins=50)
    plt.title("Log-Scaled Distribution of Transaction Values")
    plt.xlabel("log(Value + 1)")
    plt.tight_layout()
    plt.savefig(FIGURES_PATH / "log_transaction_value_distribution.png")
    plt.close()

    logger.info("Saved transaction value distribution plot")


def transactions_per_customer(df: pd.DataFrame) -> None:
    """
    Analyze customer transaction frequency behavior.
    """
    tx_per_customer = df.groupby("CustomerId")["TransactionId"].count()

    plt.figure(figsize=(8, 5))
    sns.histplot(np.log1p(tx_per_customer), bins=50)
    plt.title("Log-Scaled Transactions per Customer")
    plt.xlabel("log(Transaction Count + 1)")
    plt.tight_layout()
    plt.savefig(FIGURES_PATH / "transactions_per_customer.png")
    plt.close()

    logger.info("Saved transactions per customer plot")


def monetary_value_per_customer(df: pd.DataFrame) -> None:
    """
    Analyze monetary value aggregated at customer level.
    """
    monetary = df.groupby("CustomerId")["Value"].sum()

    plt.figure(figsize=(8, 5))
    sns.histplot(np.log1p(monetary), bins=50)
    plt.title("Log-Scaled Monetary Value per Customer")
    plt.xlabel("log(Total Value + 1)")
    plt.tight_layout()
    plt.savefig(FIGURES_PATH / "monetary_value_per_customer.png")
    plt.close()

    logger.info("Saved monetary value per customer plot")


def recency_distribution(df: pd.DataFrame) -> None:
    """
    Analyze customer recency distribution (RFM component).
    """
    snapshot_date = df["TransactionStartTime"].max() + pd.Timedelta(days=1)
    recency = df.groupby("CustomerId")["TransactionStartTime"].max()
    recency_days = (snapshot_date - recency).dt.days

    plt.figure(figsize=(8, 5))
    sns.histplot(recency_days, bins=50)
    plt.title("Customer Recency Distribution (Days)")
    plt.xlabel("Days Since Last Transaction")
    plt.tight_layout()
    plt.savefig(FIGURES_PATH / "recency_distribution.png")
    plt.close()

    logger.info("Saved recency distribution plot")


def category_concentration(df: pd.DataFrame) -> None:
    """
    Visualize concentration of transactions across product categories.
    """
    top_categories = df["ProductCategory"].value_counts().head(10)

    plt.figure(figsize=(10, 5))
    top_categories.plot(kind="bar")
    plt.title("Top 10 Product Categories by Transaction Count")
    plt.ylabel("Transaction Count")
    plt.tight_layout()
    plt.savefig(FIGURES_PATH / "top_product_categories.png")
    plt.close()

    logger.info("Saved top product categories plot")

# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------
def main() -> None:
    df = load_data()

    logger.info("Missing values summary:\n%s", df.isnull().sum())

    transaction_value_distribution(df)
    transactions_per_customer(df)
    monetary_value_per_customer(df)
    recency_distribution(df)
    category_concentration(df)

    logger.info("EDA completed successfully")

if __name__ == "__main__":
    main()