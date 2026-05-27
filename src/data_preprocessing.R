"""
data_preprocessing.py
---------------------
Handles loading, cleaning, and feature engineering for the
Customer Segmentation project.

Author: Shalmalee Sharma
"""

import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)


def load_data(filepath: str) -> pd.DataFrame:
    """Load raw retail transaction data from CSV."""
    logger.info(f"Loading data from {filepath}")
    df = pd.read_csv(filepath, encoding="ISO-8859-1")
    logger.info(f"Loaded {df.shape[0]:,} rows and {df.shape[1]} columns")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean raw transaction data:
    - Remove cancellations and returns
    - Drop nulls in key columns
    - Remove invalid quantities and prices
    """
    logger.info("Cleaning data...")
    initial_rows = len(df)

    # Drop rows with missing CustomerID
    df = df.dropna(subset=["CustomerID"])

    # Remove cancellations (InvoiceNo starting with 'C')
    df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]

    # Remove negative or zero quantities and prices
    df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]

    # Parse invoice date
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

    # Compute total price per line item
    df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]

    # Cast CustomerID to integer
    df["CustomerID"] = df["CustomerID"].astype(int)

    cleaned_rows = len(df)
    logger.info(f"Removed {initial_rows - cleaned_rows:,} invalid rows. {cleaned_rows:,} rows remaining.")
    return df


def compute_rfm(df: pd.DataFrame, snapshot_date: datetime = None) -> pd.DataFrame:
    """
    Compute RFM (Recency, Frequency, Monetary) metrics per customer.

    Args:
        df: Cleaned transaction DataFrame
        snapshot_date: Reference date for recency (defaults to day after last transaction)

    Returns:
        DataFrame with CustomerID, Recency, Frequency, Monetary columns
    """
    if snapshot_date is None:
        snapshot_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)

    logger.info(f"Computing RFM metrics with snapshot date: {snapshot_date.date()}")

    rfm = df.groupby("CustomerID").agg(
        Recency   =("InvoiceDate",  lambda x: (snapshot_date - x.max()).days),
        Frequency =("InvoiceNo",    "nunique"),
        Monetary  =("TotalPrice",   "sum")
    ).reset_index()

    logger.info(f"RFM computed for {len(rfm):,} customers")
    logger.info(f"\n{rfm.describe().round(2)}")
    return rfm


def scale_features(rfm: pd.DataFrame) -> tuple[pd.DataFrame, StandardScaler]:
    """
    Normalise RFM features using StandardScaler.

    Returns:
        Tuple of (scaled DataFrame, fitted scaler)
    """
    scaler = StandardScaler()
    features = ["Recency", "Frequency", "Monetary"]
    rfm_scaled = rfm.copy()
    rfm_scaled[features] = scaler.fit_transform(rfm[features])
    logger.info("Features scaled using StandardScaler")
    return rfm_scaled, scaler


if __name__ == "__main__":
    # Quick smoke test
    import os
    data_path = os.path.join(os.path.dirname(__file__), "../data/online_retail.csv")
    df = load_data(data_path)
    df = clean_data(df)
    rfm = compute_rfm(df)
    rfm_scaled, scaler = scale_features(rfm)
    print(rfm.head())