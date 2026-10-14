"""Visualization module for Exploratory Data Analysis (EDA) and Model Interpretability."""

import logging
from pathlib import Path
from typing import Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.config import FIGURES_DIR, TARGET_COLUMN, TIME_COLUMN, AMOUNT_COLUMN

logger = logging.getLogger(__name__)

# Apply modern clean visualization theme
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.titlesize": 13,
})


def plot_class_distribution(df: pd.DataFrame, save_path: Optional[Path] = None) -> plt.Figure:
    """Plot bar chart showing extreme class imbalance with counts and percentages."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    counts = df[TARGET_COLUMN].value_counts().sort_index()
    total = len(df)
    labels = ["Legitimate (0)", "Fraud (1)"]
    palette = ["#2b5c8f", "#d9534f"]

    # Linear Scale
    bars1 = ax1.bar(labels, counts.values, color=palette, edgecolor="black", width=0.5)
    ax1.set_title("Class Frequency (Linear Scale)", fontweight="bold")
    ax1.set_ylabel("Count")
    for bar in bars1:
        h = bar.get_height()
        ax1.annotate(
            f"{int(h):,}\n({h/total*100:.3f}%)",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontweight="semibold",
        )

    # Log Scale
    bars2 = ax2.bar(labels, counts.values, color=palette, edgecolor="black", width=0.5)
    ax2.set_yscale("log")
    ax2.set_title("Class Frequency (Logarithmic Scale)", fontweight="bold")
    ax2.set_ylabel("Count (log10)")
    for bar in bars2:
        h = bar.get_height()
        ax2.annotate(
            f"{int(h):,}",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontweight="semibold",
        )

    plt.suptitle("Credit Card Transactions — Class Imbalance Overview", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    return fig


def plot_amount_distribution(df: pd.DataFrame, save_path: Optional[Path] = None) -> plt.Figure:
    """Plot transaction amount distribution for legitimate vs fraudulent transactions."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    legit = df[df[TARGET_COLUMN] == 0][AMOUNT_COLUMN]
    fraud = df[df[TARGET_COLUMN] == 1][AMOUNT_COLUMN]

    # Linear Histograms up to 99th percentile of legitimate
    p99 = legit.quantile(0.99)
    ax1.hist(legit, bins=50, range=(0, p99), density=True, alpha=0.6, color="#2b5c8f", label=f"Legitimate (<=99th %: ${p99:.0f})")
    ax1.hist(fraud, bins=50, range=(0, p99), density=True, alpha=0.7, color="#d9534f", label="Fraud")
    ax1.set_title("Amount Distribution (Capped at 99th percentile)", fontweight="bold")
    ax1.set_xlabel("Transaction Amount ($)")
    ax1.set_ylabel("Density")
    ax1.legend()

    # Log Boxplot
    sns.boxplot(
        data=df,
        x=TARGET_COLUMN,
        y=AMOUNT_COLUMN,
        hue=TARGET_COLUMN,
        palette=["#2b5c8f", "#d9534f"],
        ax=ax2,
        showfliers=False,
        legend=False,
    )
    ax2.set_yscale("log")
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(["Legitimate (0)", "Fraud (1)"])
    ax2.set_title("Amount Distribution by Class (Log Scale, Whiskers Only)", fontweight="bold")
    ax2.set_ylabel("Amount ($ - log scale)")
    ax2.set_xlabel("Transaction Class")

    plt.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    return fig


def plot_log_amount_distribution(df: pd.DataFrame, save_path: Optional[Path] = None) -> plt.Figure:
    """Plot KDE and histogram of log1p(Amount) by class."""
    fig, ax = plt.subplots(figsize=(8, 5))
    df_temp = df.copy()
    if "log_amount" not in df_temp.columns:
        df_temp["log_amount"] = np.log1p(df_temp[AMOUNT_COLUMN])

    sns.kdeplot(
        data=df_temp,
        x="log_amount",
        hue=TARGET_COLUMN,
        common_norm=False,
        fill=True,
        palette=["#2b5c8f", "#d9534f"],
        alpha=0.4,
        ax=ax,
    )
    ax.set_title("Log-Transformed Amount Distribution: log1p(Amount)", fontweight="bold")
    ax.set_xlabel("log1p(Amount)")
    ax.set_ylabel("Density")
    ax.legend(title="Class", labels=["Fraud (1)", "Legitimate (0)"])
    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    return fig


def plot_boxplot_amount(df: pd.DataFrame, save_path: Optional[Path] = None) -> plt.Figure:
    """Plot detailed boxplot of Amount by Class."""
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.boxplot(
        data=df,
        x=TARGET_COLUMN,
        y=AMOUNT_COLUMN,
        hue=TARGET_COLUMN,
        palette=["#2b5c8f", "#d9534f"],
        showmeans=True,
        meanprops={"marker": "o", "markerfacecolor": "yellow", "markeredgecolor": "black"},
        ax=ax,
        legend=False,
    )
    ax.set_yscale("log")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Legitimate (0)", "Fraud (1)"])
    ax.set_title("Transaction Amount Boxplot by Class (Log Scale)\n(Yellow dot indicates mean)", fontweight="bold")
    ax.set_xlabel("Class")
    ax.set_ylabel("Transaction Amount ($ log scale)")
    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    return fig


def plot_fraud_rate_by_time_bin(df: pd.DataFrame, save_path: Optional[Path] = None) -> plt.Figure:
    """Plot fraud rate across elapsed-time bins (hours elapsed since recording start)."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    df_temp = df.copy()
    if "time_bin" not in df_temp.columns:
        df_temp["time_bin"] = (df_temp[TIME_COLUMN] // 3600).astype(int)

    # Bin-level counts
    bin_stats = df_temp.groupby("time_bin")[TARGET_COLUMN].agg(
        total="count",
        fraud_count="sum",
    ).reset_index()
    bin_stats["fraud_rate_pct"] = (bin_stats["fraud_count"] / bin_stats["total"]) * 100

    # Total transactions per hour
    ax1.bar(bin_stats["time_bin"], bin_stats["total"], color="#2b5c8f", alpha=0.8, edgecolor="none")
    ax1.set_ylabel("Total Transactions")
    ax1.set_title("Transaction Volume and Fraud Rate Across Elapsed Time (Hours 0 to 48)", fontweight="bold")
    ax1.grid(True, linestyle=":", alpha=0.6)

    # Fraud rate per hour
    ax2.plot(bin_stats["time_bin"], bin_stats["fraud_rate_pct"], color="#d9534f", marker="o", lw=2)
    overall_rate = (df[TARGET_COLUMN].sum() / len(df)) * 100
    ax2.axhline(overall_rate, color="navy", linestyle="--", alpha=0.8, label=f"Average Fraud Rate ({overall_rate:.3f}%)")
    ax2.set_xlabel("Elapsed Time Since Dataset Inception (Hours)")
    ax2.set_ylabel("Fraud Rate (%)")
    ax2.legend()
    ax2.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    return fig


def plot_time_distribution(df: pd.DataFrame, save_path: Optional[Path] = None) -> plt.Figure:
    """Plot distribution of transaction timestamps (in elapsed hours) for each class."""
    fig, ax = plt.subplots(figsize=(10, 5))
    df_temp = df.copy()
    if "time_hours" not in df_temp.columns:
        df_temp["time_hours"] = df_temp[TIME_COLUMN] / 3600.0

    sns.kdeplot(
        data=df_temp,
        x="time_hours",
        hue=TARGET_COLUMN,
        common_norm=False,
        fill=True,
        palette=["#2b5c8f", "#d9534f"],
        alpha=0.4,
        ax=ax,
    )
    ax.set_title("Density of Transactions Over Elapsed Time (Hours)", fontweight="bold")
    ax.set_xlabel("Elapsed Time from Dataset Start (Hours)")
    ax.set_ylabel("Density")
    ax.legend(title="Class", labels=["Fraud (1)", "Legitimate (0)"])
    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    return fig


def plot_correlation_heatmap(df: pd.DataFrame, save_path: Optional[Path] = None) -> plt.Figure:
    """Plot correlation matrix between Class, Amount, Time, and top PCA components."""
    # Find PCA features with highest correlation with Class
    corr = df.corr()
    top_corr_features = corr[TARGET_COLUMN].abs().sort_values(ascending=False).head(15).index.tolist()
    # Ensure Time and Amount are also included
    for col in [TIME_COLUMN, AMOUNT_COLUMN]:
        if col in df.columns and col not in top_corr_features:
            top_corr_features.append(col)

    subset_corr = df[top_corr_features].corr()

    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(
        subset_corr,
        cmap="vlag",
        center=0,
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        ax=ax,
        cbar_kws={"shrink": 0.8},
    )
    ax.set_title("Correlation Heatmap — Top Features Associated with Fraud", fontweight="bold", pad=12)
    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    return fig


def plot_top_pca_distributions(
    df: pd.DataFrame,
    top_features: Optional[list[str]] = None,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Plot distribution of key PCA features that strongly separate fraud from legitimate."""
    features = top_features or ["V14", "V17", "V12", "V10", "V4", "V11"]
    n_features = len(features)
    cols = 3
    rows = (n_features + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 4))
    axes = np.array(axes).flatten()

    for idx, feat in enumerate(features):
        ax = axes[idx]
        sns.kdeplot(
            data=df,
            x=feat,
            hue=TARGET_COLUMN,
            common_norm=False,
            fill=True,
            palette=["#2b5c8f", "#d9534f"],
            alpha=0.4,
            ax=ax,
        )
        ax.set_title(f"Distribution of {feat} by Class", fontweight="bold")
        ax.set_xlabel(feat)

    # Hide extra axes if any
    for idx in range(len(features), len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle("Discriminative Power of Top PCA Components", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    return fig


def plot_logistic_coefficients(
    coefficients: np.ndarray,
    feature_names: list[str],
    top_n: int = 15,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Plot top positive and negative coefficients from Logistic Regression."""
    coef_df = pd.DataFrame({
        "feature": feature_names,
        "coefficient": coefficients.flatten(),
    })
    coef_df["abs_coef"] = coef_df["coefficient"].abs()
    top_df = coef_df.sort_values(by="abs_coef", ascending=False).head(top_n).sort_values(by="coefficient")

    fig, ax = plt.subplots(figsize=(9, 6))
    colors = ["#d9534f" if c > 0 else "#2b5c8f" for c in top_df["coefficient"]]
    ax.barh(top_df["feature"], top_df["coefficient"], color=colors, edgecolor="black")
    ax.axvline(0, color="black", linestyle="--", lw=1)
    ax.set_title(f"Logistic Regression: Top {top_n} Coefficients\n(Red: Increases Fraud Odds | Blue: Decreases)", fontweight="bold")
    ax.set_xlabel("Coefficient Value (Log-Odds Impact)")
    ax.set_ylabel("Feature")
    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    return fig


def plot_random_forest_importance(
    importances: np.ndarray,
    feature_names: list[str],
    top_n: int = 15,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Plot top feature importances from Random Forest."""
    imp_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances,
    }).sort_values(by="importance", ascending=False).head(top_n).sort_values(by="importance", ascending=True)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(imp_df["feature"], imp_df["importance"], color="#2ca02c", edgecolor="black")
    ax.set_title(f"Random Forest: Top {top_n} Feature Importances (Gini Impurity)", fontweight="bold")
    ax.set_xlabel("Mean Impurity Reduction (Feature Importance)")
    ax.set_ylabel("Feature")
    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    return fig
