"""Model architecture and pipeline construction module for Fraud Detection."""

import logging
from pathlib import Path
from typing import Optional, Union
import joblib
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from src.config import RANDOM_STATE, MODELS_DIR

logger = logging.getLogger(__name__)


def build_baseline_logistic_pipeline() -> ImbPipeline:
    """Build an unweighted Baseline Logistic Regression pipeline (NO SMOTE).

    Used as an unadjusted benchmark to prove why raw accuracy without imbalance handling fails.
    """
    return ImbPipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE))
    ])


def build_logistic_smote_pipeline() -> ImbPipeline:
    """Build Logistic Regression pipeline with StandardScaler and SMOTE.

    Steps:
      1. StandardScaler: Standardizes features to zero mean and unit variance.
      2. SMOTE: Synthesizes minority fraud instances ONLY during fit (training time).
      3. LogisticRegression: Linear classification model with L2 regularization.
    """
    return ImbPipeline([
        ("scaler", StandardScaler()),
        ("smote", SMOTE(random_state=RANDOM_STATE)),
        ("model", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE))
    ])


def build_random_forest_pipeline(
    use_smote: bool = False,
    n_estimators: int = 150,
    min_samples_leaf: int = 2,
    max_depth: Optional[int] = 16,
) -> ImbPipeline:
    """Build Random Forest pipeline, either utilizing SMOTE or class_weight='balanced'.

    Cost-sensitive learning via class_weight='balanced' inversely weights classes by their frequency
    without generating synthetic points in high-dimensional space, preventing synthetic noise.
    Comparing it with SMOTE allows empirical verification of the optimal approach.

    Args:
        use_smote: If True, inserts SMOTE step. If False, sets class_weight='balanced'.
        n_estimators: Number of decision trees.
        min_samples_leaf: Minimum samples per leaf node to reduce overfitting.
        max_depth: Maximum tree depth.
    """
    steps = [("scaler", StandardScaler())]

    if use_smote:
        steps.append(("smote", SMOTE(random_state=RANDOM_STATE)))
        rf_model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    else:
        rf_model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

    steps.append(("model", rf_model))
    return ImbPipeline(steps)


def save_model(pipeline: ImbPipeline, filename: str, save_dir: Optional[Path] = None) -> Path:
    """Save trained pipeline artifact to disk using joblib."""
    target_dir = save_dir or MODELS_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    filepath = target_dir / filename
    joblib.dump(pipeline, filepath)
    logger.info("Saved trained model artifact to: %s", filepath)
    return filepath


def load_model(filepath: Union[str, Path]) -> ImbPipeline:
    """Load a trained model pipeline artifact from disk."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Model artifact not found at {path}")
    model = joblib.load(path)
    logger.info("Loaded model from %s", path)
    return model
