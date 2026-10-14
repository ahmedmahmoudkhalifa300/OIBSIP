"""Preprocessing and feature engineering module for Fraud Detection."""

import logging
from typing import Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.config import (
    RANDOM_STATE,
    TARGET_COLUMN,
    TEST_SIZE,
    TIME_COLUMN,
    AMOUNT_COLUMN,
    PCA_FEATURES,
)

logger = logging.getLogger(__name__)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create engineered features from Time and Amount without modifying the original DataFrame.

    Engineered features:
      - `time_hours`: Elapsed time in hours (Time / 3600).
      - `time_bin`: Integer bucket of elapsed hours (Time // 3600), representing
                    elapsed time slice since recording began (NOT clock hour of day).
      - `log_amount`: Natural log of (Amount + 1) to tame extreme right-skewness.

    Returns:
        pd.DataFrame with original + engineered features.
    """
    df_feat = df.copy()
    if TIME_COLUMN in df_feat.columns:
        df_feat["time_hours"] = df_feat[TIME_COLUMN] / 3600.0
        df_feat["time_bin"] = (df_feat[TIME_COLUMN] // 3600).astype(int)
    if AMOUNT_COLUMN in df_feat.columns:
        df_feat["log_amount"] = np.log1p(df_feat[AMOUNT_COLUMN])

    return df_feat


def get_model_features(use_engineered: bool = True) -> list[str]:
    """Return the list of feature column names used for training models.

    Args:
        use_engineered: If True, uses log_amount and time_hours alongside V1-V28.
                       If False, uses original Amount, Time, and V1-V28.
    """
    if use_engineered:
        # We exclude time_bin from raw linear models to avoid ordinal/categorical misuse
        return PCA_FEATURES + ["log_amount", "time_hours"]
    return PCA_FEATURES + [AMOUNT_COLUMN, TIME_COLUMN]


def split_features_target(
    df: pd.DataFrame,
    feature_cols: Optional[list[str]] = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Separate feature matrix X and target vector y.

    Args:
        df: DataFrame containing features and target column.
        feature_cols: Optional subset of feature columns to retain.

    Returns:
        Tuple of (X, y)
    """
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found in dataframe.")

    y = df[TARGET_COLUMN].copy()
    if feature_cols:
        X = df[feature_cols].copy()
    else:
        drop_cols = [TARGET_COLUMN]
        # Drop time_bin if present from direct feature matrix
        if "time_bin" in df.columns:
            drop_cols.append("time_bin")
        X = df.drop(columns=drop_cols).copy()

    return X, y


def perform_train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split data into train and test sets using stratification on y to preserve fraud ratio.

    Returns:
        X_train, X_test, y_train, y_test
    """
    logger.info(
        "Performing stratified train/test split (test_size=%.2f, random_state=%d)...",
        test_size,
        random_state,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    # Verification
    n_train_fraud = int((y_train == 1).sum())
    n_test_fraud = int((y_test == 1).sum())
    logger.info(
        "Train set: %d samples (%d fraud, %.4f%%)",
        len(y_train),
        n_train_fraud,
        (n_train_fraud / len(y_train)) * 100,
    )
    logger.info(
        "Test set:  %d samples (%d fraud, %.4f%%)",
        len(y_test),
        n_test_fraud,
        (n_test_fraud / len(y_test)) * 100,
    )

    if n_train_fraud == 0 or n_test_fraud == 0:
        raise RuntimeError(
            "Stratified split failed to allocate fraud cases to both train and test partitions!"
        )

    return X_train, X_test, y_train, y_test


def build_column_preprocessor(feature_cols: list[str]) -> ColumnTransformer:
    """Build a ColumnTransformer that scales continuous features while passing PCA components.

    PCA features (V1-V28) are already zero-centered orthogonal components.
    Non-PCA features (Time/time_hours, Amount/log_amount) are scaled via StandardScaler.
    """
    scale_cols = [c for c in feature_cols if c in [TIME_COLUMN, AMOUNT_COLUMN, "time_hours", "log_amount"]]
    passthrough_cols = [c for c in feature_cols if c not in scale_cols]

    transformers = []
    if scale_cols:
        transformers.append(("scaler", StandardScaler(), scale_cols))
    if passthrough_cols:
        transformers.append(("passthrough", "passthrough", passthrough_cols))

    return ColumnTransformer(transformers=transformers, remainder="drop")
