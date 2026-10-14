"""Unit and integration tests for Credit Card Fraud Detection pipeline."""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from src.config import PCA_FEATURES, TARGET_COLUMN
from src.data_loader import load_data, validate_dataset, resolve_data_path
from src.preprocessing import (
    engineer_features,
    get_model_features,
    perform_train_test_split,
    split_features_target,
)
from src.models import (
    build_baseline_logistic_pipeline,
    build_logistic_smote_pipeline,
    build_random_forest_pipeline,
)
from src.evaluation import evaluate_model, evaluate_thresholds


@pytest.fixture
def synthetic_sample_df():
    """Create a small, self-contained synthetic dataset matching the schema."""
    np.random.seed(42)
    n_samples = 300
    data = {f"V{i}": np.random.randn(n_samples) for i in range(1, 29)}
    data["Time"] = np.sort(np.random.uniform(0, 100000, n_samples))
    data["Amount"] = np.random.exponential(50, n_samples)
    # 10 fraud cases out of 300
    classes = np.zeros(n_samples, dtype=int)
    fraud_indices = np.random.choice(n_samples, size=15, replace=False)
    classes[fraud_indices] = 1
    data["Class"] = classes
    return pd.DataFrame(data)


def test_schema_and_feature_engineering(synthetic_sample_df):
    """Test feature engineering creates time_hours, time_bin, and log_amount."""
    df_feat = engineer_features(synthetic_sample_df)
    assert "time_hours" in df_feat.columns
    assert "time_bin" in df_feat.columns
    assert "log_amount" in df_feat.columns
    assert len(df_feat) == len(synthetic_sample_df)
    assert np.all(df_feat["log_amount"] >= 0)


def test_validate_dataset(synthetic_sample_df):
    """Test validation reports accurate counts and checks."""
    stats = validate_dataset(synthetic_sample_df)
    assert stats["total_rows"] == 300
    assert stats["fraud_count"] == 15
    assert stats["legit_count"] == 285
    assert stats["missing_values"] == 0
    assert stats["infinite_values"] == 0


def test_stratified_train_test_split(synthetic_sample_df):
    """Test stratified split maintains positive fraud cases in both train and test partitions."""
    df_feat = engineer_features(synthetic_sample_df)
    features = get_model_features(use_engineered=True)
    X, y = split_features_target(df_feat, feature_cols=features)

    X_train, X_test, y_train, y_test = perform_train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)
    assert (y_train == 1).sum() > 0, "Train partition must contain fraud instances"
    assert (y_test == 1).sum() > 0, "Test partition must contain fraud instances"

    # Ratio verification: train and test fraud proportions should be very close
    train_ratio = (y_train == 1).mean()
    test_ratio = (y_test == 1).mean()
    assert abs(train_ratio - test_ratio) < 0.05


def test_pipelines_fit_and_evaluate(synthetic_sample_df):
    """Test fitting and evaluation of all pipelines on a small partition."""
    df_feat = engineer_features(synthetic_sample_df)
    features = get_model_features(use_engineered=True)
    X, y = split_features_target(df_feat, feature_cols=features)
    X_train, X_test, y_train, y_test = perform_train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    # Test Logistic + SMOTE Pipeline
    lr_pipe = build_logistic_smote_pipeline()
    lr_pipe.fit(X_train, y_train)
    eval_res = evaluate_model(lr_pipe, X_test, y_test, model_name="TestLR")

    required_keys = [
        "model_name", "threshold", "true_positives", "true_negatives",
        "false_positives", "false_negatives", "precision", "recall",
        "f1_score", "roc_auc", "pr_auc", "confusion_matrix", "y_proba"
    ]
    for key in required_keys:
        assert key in eval_res, f"Missing key '{key}' in evaluation results"

    assert 0.0 <= eval_res["roc_auc"] <= 1.0
    assert 0.0 <= eval_res["pr_auc"] <= 1.0

    # Test Random Forest Pipeline
    rf_pipe = build_random_forest_pipeline(use_smote=False, n_estimators=10)
    rf_pipe.fit(X_train, y_train)
    rf_eval = evaluate_model(rf_pipe, X_test, y_test, model_name="TestRF")
    assert rf_eval["true_positives"] + rf_eval["false_negatives"] == (y_test == 1).sum()


def test_threshold_analysis_structure(synthetic_sample_df):
    """Test evaluate_thresholds outputs a valid dataframe with proper columns."""
    y_true = pd.Series([0, 0, 0, 1, 1])
    y_proba = np.array([0.1, 0.2, 0.4, 0.7, 0.9])
    df_th = evaluate_thresholds(y_true, y_proba, thresholds=[0.2, 0.5, 0.8])

    assert len(df_th) == 3
    assert "threshold" in df_th.columns
    assert "precision" in df_th.columns
    assert "recall" in df_th.columns
    assert "f1_score" in df_th.columns
    assert "flagged_transactions" in df_th.columns


def test_real_dataset_loader_if_present():
    """Integration test: if real creditcard.csv exists, verify loading and target column."""
    try:
        path = resolve_data_path()
        df = load_data(path, nrows=50)
        assert TARGET_COLUMN in df.columns
        assert "Time" in df.columns
        assert "Amount" in df.columns
        assert len(df) == 50
    except FileNotFoundError:
        pytest.skip("Dataset file not found; skipping real data loader test.")
