"""Evaluation module for credit card fraud detection models."""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.config import FIGURES_DIR, METRICS_DIR, EVALUATION_THRESHOLDS

logger = logging.getLogger(__name__)


def evaluate_model(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str = "Model",
    threshold: float = 0.5,
) -> dict:
    """Evaluate a trained model pipeline on un-resampled test data.

    Returns:
        dict: Detailed metrics including TP, TN, FP, FN, precision, recall, f1, roc_auc, pr_auc.
    """
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        scores = model.decision_function(X_test)
        y_proba = 1 / (1 + np.exp(-scores))
    else:
        y_proba = model.predict(X_test).astype(float)

    y_pred = (y_proba >= threshold).astype(int)

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_proba)
    pr_auc = average_precision_score(y_test, y_proba)
    report = classification_report(
        y_test, y_pred, target_names=["Legitimate", "Fraud"], digits=4, zero_division=0
    )

    results = {
        "model_name": model_name,
        "threshold": threshold,
        "true_positives": int(tp),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "confusion_matrix": cm,
        "y_proba": y_proba,
        "classification_report": report,
    }
    return results


def evaluate_thresholds(
    y_true: pd.Series,
    y_proba: np.ndarray,
    thresholds: Optional[List[float]] = None,
) -> pd.DataFrame:
    """Analyze precision, recall, F1, and flagged count across different probability thresholds."""
    thresholds = thresholds or EVALUATION_THRESHOLDS
    records = []

    for th in thresholds:
        y_pred = (y_proba >= th).astype(int)
        flagged = int(y_pred.sum())
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()

        records.append({
            "threshold": th,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "flagged_transactions": flagged,
            "true_positives": int(tp),
            "false_positives": int(fp),
            "false_negatives": int(fn),
        })

    return pd.DataFrame(records)


def save_metrics_summary(results_list: List[dict], output_path: Optional[Path] = None) -> pd.DataFrame:
    """Save a comparative metrics table to CSV."""
    records = []
    for r in results_list:
        records.append({
            "model": r["model_name"],
            "precision": round(r["precision"], 4),
            "recall": round(r["recall"], 4),
            "f1": round(r["f1_score"], 4),
            "roc_auc": round(r["roc_auc"], 4),
            "pr_auc": round(r["pr_auc"], 4),
            "tp": r["true_positives"],
            "fp": r["false_positives"],
            "fn": r["false_negatives"],
            "tn": r["true_negatives"],
        })
    df_metrics = pd.DataFrame(records)
    save_file = output_path or (METRICS_DIR / "metrics.csv")
    save_file.parent.mkdir(parents=True, exist_ok=True)
    df_metrics.to_csv(save_file, index=False)
    logger.info("Saved metrics summary to %s", save_file)
    return df_metrics


def plot_confusion_matrix(
    cm: np.ndarray,
    model_name: str,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Plot and save a styled confusion matrix."""
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt=",d",
        cmap="Blues",
        cbar=False,
        xticklabels=["Legitimate (0)", "Fraud (1)"],
        yticklabels=["Legitimate (0)", "Fraud (1)"],
        ax=ax,
    )
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Predicted Label", fontsize=11, fontweight="semibold")
    ax.set_ylabel("True Label", fontsize=11, fontweight="semibold")
    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    return fig


def plot_roc_curves_comparison(
    models_dict: Dict[str, Tuple[pd.Series, np.ndarray]],
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Plot overlaid ROC curves for multiple models with random baseline."""
    fig, ax = plt.subplots(figsize=(8, 6))

    for name, (y_test, y_proba) in models_dict.items():
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc_score = roc_auc_score(y_test, y_proba)
        ax.plot(fpr, tpr, lw=2, label=f"{name} (ROC-AUC = {auc_score:.4f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Random Chance (AUC = 0.5000)")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11, fontweight="semibold")
    ax.set_ylabel("True Positive Rate (Recall / Sensitivity)", fontsize=11, fontweight="semibold")
    ax.set_title("Receiver Operating Characteristic (ROC) Curve Comparison", fontsize=13, fontweight="bold", pad=12)
    ax.legend(loc="lower right", frameon=True, fontsize=10)
    ax.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    return fig


def plot_pr_curves_comparison(
    models_dict: Dict[str, Tuple[pd.Series, np.ndarray]],
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Plot Precision-Recall curves with positive class prevalence baseline."""
    fig, ax = plt.subplots(figsize=(8, 6))

    # Calculate baseline (fraud prevalence in test set)
    first_y_test = next(iter(models_dict.values()))[0]
    baseline_val = float(first_y_test.mean())

    for name, (y_test, y_proba) in models_dict.items():
        precision, recall, _ = precision_recall_curve(y_test, y_proba)
        pr_auc = average_precision_score(y_test, y_proba)
        ax.plot(recall, precision, lw=2, label=f"{name} (PR-AUC = {pr_auc:.4f})")

    ax.axhline(
        y=baseline_val,
        color="crimson",
        linestyle="--",
        lw=1.5,
        label=f"No-Skill Baseline (Prevalence = {baseline_val*100:.3f}%)",
    )
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Recall (Fraud Coverage)", fontsize=11, fontweight="semibold")
    ax.set_ylabel("Precision (Positive Predictive Value)", fontsize=11, fontweight="semibold")
    ax.set_title("Precision-Recall (PR) Curve Comparison\n(Preferred Metric for Severe Imbalance)", fontsize=13, fontweight="bold", pad=12)
    ax.legend(loc="upper right", frameon=True, fontsize=10)
    ax.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    return fig
