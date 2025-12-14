import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

DATA_PATH = "data/raw/data.csv"
FIGURES_PATH = Path("reports/figures")
FIGURES_PATH.mkdir(parents=True, exist_ok=True)


def load_data():
    df = pd.read_csv(DATA_PATH)
    df["TransactionStartTime"] = pd.to_datetime(df["TransactionStartTime"])
    return df


def transaction_value_distribution(df):
    """
    Use absolute transaction value and log scale
    """
    plt.figure(figsize=(8, 5))
    sns.histplot(np.log1p(df["Value"]), bins=50)
    plt.title("Log-Scaled Distribution of Transaction Values")
    plt.xlabel("log(Value + 1)")
    plt.tight_layout()
    plt.savefig(FIGURES_PATH / "log_transaction_value_distribution.png")
    plt.close()


def transactions_per_customer(df):
    """
    Frequency behavior at customer level
    """
    tx_per_customer = df.groupby("CustomerId")["TransactionId"].count()

    plt.figure(figsize=(8, 5))
    sns.histplot(np.log1p(tx_per_customer), bins=50)
    plt.title("Log-Scaled Transactions per Customer")
    plt.xlabel("log(Transaction Count + 1)")
    plt.tight_layout()
    plt.savefig(FIGURES_PATH / "transactions_per_customer.png")
    plt.close()


def monetary_value_per_customer(df):
    """
    Monetary behavior per customer
    """
    monetary = df.groupby("CustomerId")["Value"].sum()

    plt.figure(figsize=(8, 5))
    sns.histplot(np.log1p(monetary), bins=50)
    plt.title("Log-Scaled Monetary Value per Customer")
    plt.xlabel("log(Total Value + 1)")
    plt.tight_layout()
    plt.savefig(FIGURES_PATH / "monetary_value_per_customer.png")
    plt.close()


def recency_distribution(df):
    """
    Recency supports RFM logic
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


def category_concentration(df):
    """
    Validate dominance of product categories
    """
    top_categories = df["ProductCategory"].value_counts().head(10)

    plt.figure(figsize=(10, 5))
    top_categories.plot(kind="bar")
    plt.title("Top 10 Product Categories by Transaction Count")
    plt.ylabel("Transaction Count")
    plt.tight_layout()
    plt.savefig(FIGURES_PATH / "top_product_categories.png")
    plt.close()


def main():
    df = load_data()

    print("Dataset shape:", df.shape)
    print("\nMissing values:\n", df.isnull().sum())

    transaction_value_distribution(df)
    transactions_per_customer(df)
    monetary_value_per_customer(df)
    recency_distribution(df)
    category_concentration(df)


if __name__ == "__main__":
    main()