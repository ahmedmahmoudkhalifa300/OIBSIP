"""Data loading and verification module for Credit Card Fraud Detection."""

import logging
from pathlib import Path
from typing import Optional, Union
import numpy as np
import pandas as pd

from src.config import (
    DEFAULT_DATA_PATH,
    FALLBACK_DATA_PATH,
    TARGET_COLUMN,
    TIME_COLUMN,
    AMOUNT_COLUMN,
    PCA_FEATURES,
)

logger = logging.getLogger(__name__)


def resolve_data_path(custom_path: Optional[Union[str, Path]] = None) -> Path:
    """Resolve the dataset path from custom argument, default path, or fallback path.

    Raises:
        FileNotFoundError: If no valid dataset file is found, with Kaggle download instructions.
    """
    candidates = []
    if custom_path:
        candidates.append(Path(custom_path))
    candidates.extend([DEFAULT_DATA_PATH, FALLBACK_DATA_PATH])

    for p in candidates:
        if p.exists() and p.is_file():
            return p.resolve()

    download_instructions = (
        f"\nDataset file not found! Checked locations:\n"
        + "\n".join(f" - {c.resolve()}" for c in candidates)
        + "\n\nPlease download the dataset from Kaggle:"
        + "\nURL: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud"
        + "\nCLI Command:"
        + "\n  kaggle datasets download -d mlg-ulb/creditcardfraud -p data/raw --unzip"
        + "\nOr manually extract 'creditcard.csv' into 'data/raw/creditcard.csv'."
    )
    raise FileNotFoundError(download_instructions)


def load_data(
    path: Optional[Union[str, Path]] = None,
    nrows: Optional[int] = None,
) -> pd.DataFrame:
    """Load credit card transactions dataset into a pandas DataFrame.

    Args:
        path: Optional file path. If None, resolves default/fallback paths.
        nrows: Optional number of rows to read (useful for quick testing/debugging).

    Returns:
        pd.DataFrame containing transaction records.
    """
    resolved = resolve_data_path(path)
    logger.info("Loading dataset from: %s", resolved)
    df = pd.read_csv(resolved, nrows=nrows)
    logger.info("Loaded %d rows and %d columns successfully.", df.shape[0], df.shape[1])
    return df


def validate_dataset(df: pd.DataFrame) -> dict:
    """Validate dataset columns, class distribution, missing values, duplicates, and ranges.

    Returns:
        dict: Summary statistics and data health metrics.
    """
    required_cols = [TIME_COLUMN, AMOUNT_COLUMN, TARGET_COLUMN] + PCA_FEATURES
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Dataset is missing required columns: {missing_cols}")

    total_rows = len(df)
    class_counts = df[TARGET_COLUMN].value_counts().to_dict()
    legit_count = int(class_counts.get(0, 0))
    fraud_count = int(class_counts.get(1, 0))
    fraud_rate = (fraud_count / total_rows) * 100 if total_rows > 0 else 0.0

    missing_vals = int(df.isnull().sum().sum())
    # Check infinite values on numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    inf_vals = int(np.isinf(df[numeric_cols].to_numpy()).sum())
    duplicate_rows = int(df.duplicated().sum())

    stats = {
        "total_rows": total_rows,
        "total_columns": df.shape[1],
        "legit_count": legit_count,
        "fraud_count": fraud_count,
        "fraud_rate_pct": fraud_rate,
        "fraud_ratio": f"1:{legit_count // fraud_count}" if fraud_count > 0 else "N/A",
        "missing_values": missing_vals,
        "infinite_values": inf_vals,
        "duplicate_rows": duplicate_rows,
        "time_min": float(df[TIME_COLUMN].min()),
        "time_max": float(df[TIME_COLUMN].max()),
        "amount_min": float(df[AMOUNT_COLUMN].min()),
        "amount_max": float(df[AMOUNT_COLUMN].max()),
        "amount_mean": float(df[AMOUNT_COLUMN].mean()),
        "amount_median": float(df[AMOUNT_COLUMN].median()),
    }
    return stats


def print_dataset_summary(stats: dict) -> None:
    """Print a clean, formatted report of dataset validation metrics."""
    print("=" * 60)
    print(" CREDIT CARD FRAUD DETECTION - DATASET VALIDATION REPORT")
    print("=" * 60)
    print(f"Total Transactions:         {stats['total_rows']:,}")
    print(f"Total Features:             {stats['total_columns']}")
    print(f"Legitimate Transactions:    {stats['legit_count']:,} ({100 - stats['fraud_rate_pct']:.4f}%)")
    print(f"Fraudulent Transactions:    {stats['fraud_count']:,} ({stats['fraud_rate_pct']:.4f}%)")
    print(f"Fraud Ratio:                {stats['fraud_ratio']}")
    print(f"Missing Values:             {stats['missing_values']}")
    print(f"Infinite Values:            {stats['infinite_values']}")
    print(f"Duplicate Rows:             {stats['duplicate_rows']:,}")
    print(f"Time Range (Seconds):       {stats['time_min']:.1f} to {stats['time_max']:.1f} (~{stats['time_max']/3600:.1f} hours)")
    print(f"Amount Range:               ${stats['amount_min']:.2f} to ${stats['amount_max']:.2f} (Median: ${stats['amount_median']:.2f})")
    print("=" * 60)
