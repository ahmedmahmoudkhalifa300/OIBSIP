"""Complete Command-Line Pipeline for Credit Card Fraud Detection.

Usage:
    python run_pipeline.py --data-path data/raw/creditcard.csv
"""

import argparse
import logging
import sys
from pathlib import Path

# Force UTF-8 encoding on standard output/error to safely handle multilingual paths on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from src.config import (
    DEFAULT_DATA_PATH,
    FIGURES_DIR,
    METRICS_DIR,
    MODELS_DIR,
    RANDOM_STATE,
    TEST_SIZE,
    EVALUATION_THRESHOLDS,
)
from src.data_loader import load_data, print_dataset_summary, validate_dataset
from src.evaluation import (
    evaluate_model,
    evaluate_thresholds,
    plot_confusion_matrix,
    plot_pr_curves_comparison,
    plot_roc_curves_comparison,
    save_metrics_summary,
)
from src.models import (
    build_baseline_logistic_pipeline,
    build_logistic_smote_pipeline,
    build_random_forest_pipeline,
    save_model,
)
from src.preprocessing import (
    engineer_features,
    get_model_features,
    perform_train_test_split,
    split_features_target,
)
from src.visualization import (
    plot_amount_distribution,
    plot_boxplot_amount,
    plot_class_distribution,
    plot_correlation_heatmap,
    plot_fraud_rate_by_time_bin,
    plot_log_amount_distribution,
    plot_logistic_coefficients,
    plot_random_forest_importance,
    plot_time_distribution,
    plot_top_pca_distributions,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("fraud_pipeline")


def run_pipeline(
    data_path: Path,
    n_estimators: int = 150,
    quick_run: bool = False,
) -> None:
    """Execute the end-to-end ML pipeline."""
    print("=" * 80)
    print(" STARTING CREDIT CARD FRAUD DETECTION PIPELINE")
    print("=" * 80)

    # 1. Load Data
    nrows = 20000 if quick_run else None
    if quick_run:
        logger.warning("Quick run mode enabled: reading first %d rows only.", nrows)

    df_raw = load_data(path=data_path, nrows=nrows)

    # 2. Validation
    stats = validate_dataset(df_raw)
    print_dataset_summary(stats)

    # 3. Feature Engineering
    logger.info("Executing feature engineering (time_hours, time_bin, log_amount)...")
    df_feat = engineer_features(df_raw)

    # 4. Generate and Save EDA Figures
    logger.info("Generating and saving exploratory figures to %s...", FIGURES_DIR)
    plot_class_distribution(df_feat, FIGURES_DIR / "01_class_distribution.png")
    plot_amount_distribution(df_feat, FIGURES_DIR / "02_amount_distribution.png")
    plot_log_amount_distribution(df_feat, FIGURES_DIR / "03_log_amount_distribution.png")
    plot_boxplot_amount(df_feat, FIGURES_DIR / "04_boxplot_amount.png")
    plot_fraud_rate_by_time_bin(df_feat, FIGURES_DIR / "05_fraud_rate_by_time_bin.png")
    plot_time_distribution(df_feat, FIGURES_DIR / "06_time_distribution.png")
    plot_correlation_heatmap(df_feat, FIGURES_DIR / "07_correlation_heatmap.png")
    plot_top_pca_distributions(df_feat, None, FIGURES_DIR / "08_top_pca_distributions.png")
    logger.info("Saved 8 EDA figures successfully.")

    # 5. Prepare Features and Stratified Train/Test Split
    feature_cols = get_model_features(use_engineered=True)
    X, y = split_features_target(df_feat, feature_cols=feature_cols)

    X_train, X_test, y_train, y_test = perform_train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # Dictionary to collect evaluation models
    results = []
    models_dict_for_curves = {}

    # 6. Model 0: Baseline Logistic Regression (NO SMOTE)
    print("\n--- Training Model 0: Baseline Logistic Regression (No SMOTE) ---")
    base_pipeline = build_baseline_logistic_pipeline()
    base_pipeline.fit(X_train, y_train)
    save_model(base_pipeline, "baseline_logistic_regression.joblib")

    res_base = evaluate_model(base_pipeline, X_test, y_test, model_name="Baseline (No SMOTE)")
    results.append(res_base)
    models_dict_for_curves["Baseline Logistic"] = (y_test, res_base["y_proba"])
    plot_confusion_matrix(res_base["confusion_matrix"], "Baseline Logistic", FIGURES_DIR / "09_cm_baseline.png")

    # 7. Model 1: Logistic Regression + SMOTE
    print("\n--- Training Model 1: Logistic Regression + SMOTE ---")
    lr_smote_pipeline = build_logistic_smote_pipeline()
    lr_smote_pipeline.fit(X_train, y_train)
    save_model(lr_smote_pipeline, "logistic_regression_smote.joblib")

    res_lr = evaluate_model(lr_smote_pipeline, X_test, y_test, model_name="Logistic Regression + SMOTE")
    results.append(res_lr)
    models_dict_for_curves["Logistic + SMOTE"] = (y_test, res_lr["y_proba"])
    plot_confusion_matrix(res_lr["confusion_matrix"], "Logistic Regression + SMOTE", FIGURES_DIR / "10_cm_logistic_smote.png")

    # 8. Model 2: Random Forest (Balanced Class Weights)
    print("\n--- Training Model 2: Random Forest Classifier ---")
    rf_pipeline = build_random_forest_pipeline(use_smote=False, n_estimators=n_estimators)
    rf_pipeline.fit(X_train, y_train)
    save_model(rf_pipeline, "random_forest_balanced.joblib")

    res_rf = evaluate_model(rf_pipeline, X_test, y_test, model_name="Random Forest (Balanced)")
    results.append(res_rf)
    models_dict_for_curves["Random Forest"] = (y_test, res_rf["y_proba"])
    plot_confusion_matrix(res_rf["confusion_matrix"], "Random Forest (Balanced)", FIGURES_DIR / "11_cm_random_forest.png")

    # 9. Comparative Curve Visualizations
    logger.info("Generating comparative ROC and Precision-Recall curves...")
    plot_roc_curves_comparison(models_dict_for_curves, FIGURES_DIR / "12_roc_curve_comparison.png")
    plot_pr_curves_comparison(models_dict_for_curves, FIGURES_DIR / "13_pr_curve_comparison.png")

    # 10. Threshold Analysis
    logger.info("Executing decision threshold tuning...")
    df_th_lr = evaluate_thresholds(y_test, res_lr["y_proba"])
    df_th_lr.to_csv(METRICS_DIR / "threshold_analysis_logistic_smote.csv", index=False)

    df_th_rf = evaluate_thresholds(y_test, res_rf["y_proba"])
    df_th_rf.to_csv(METRICS_DIR / "threshold_analysis_random_forest.csv", index=False)

    # 11. Interpretability: Feature Importance and Coefficients
    logger.info("Extracting feature importances and coefficients...")
    lr_model = lr_smote_pipeline.named_steps["model"]
    plot_logistic_coefficients(
        lr_model.coef_,
        feature_cols,
        top_n=15,
        save_path=FIGURES_DIR / "14_logistic_coefficients.png",
    )

    rf_model = rf_pipeline.named_steps["model"]
    plot_random_forest_importance(
        rf_model.feature_importances_,
        feature_cols,
        top_n=15,
        save_path=FIGURES_DIR / "15_random_forest_importance.png",
    )

    # 12. Save Summary Metrics
    df_summary = save_metrics_summary(results, METRICS_DIR / "metrics.csv")

    print("\n" + "=" * 80)
    print(" FINAL MODEL PERFORMANCE COMPARISON TABLE")
    print("=" * 80)
    print(df_summary.to_string(index=False))
    print("=" * 80)
    print("Metrics saved to: outputs/metrics/metrics.csv")
    print("Figures saved to: outputs/figures/")
    print("Models saved to:  outputs/models/")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Credit Card Fraud Detection ML Pipeline")
    parser.add_argument(
        "--data-path",
        type=str,
        default=str(DEFAULT_DATA_PATH),
        help="Path to creditcard.csv (defaults to data/raw/creditcard.csv or ./creditcard.csv)",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=150,
        help="Number of trees in Random Forest model",
    )
    parser.add_argument(
        "--quick-run",
        action="store_true",
        help="Run quickly on a 20k subset for testing pipeline functionality",
    )

    args = parser.parse_args()
    run_pipeline(
        data_path=Path(args.data_path),
        n_estimators=args.n_estimators,
        quick_run=args.quick_run,
    )
