"""Configuration module for the Credit Card Fraud Detection pipeline."""

from pathlib import Path

# Random seed for reproducibility
RANDOM_STATE: int = 42

# Data Paths
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DEFAULT_DATA_PATH: Path = PROJECT_ROOT / "data" / "raw" / "creditcard.csv"
FALLBACK_DATA_PATH: Path = PROJECT_ROOT / "creditcard.csv"

# Target & Feature Constants
TARGET_COLUMN: str = "Class"
TIME_COLUMN: str = "Time"
AMOUNT_COLUMN: str = "Amount"
PCA_FEATURES: list[str] = [f"V{i}" for i in range(1, 29)]

# Train / Test Splitting
TEST_SIZE: float = 0.20
STRATIFY: bool = True

# Outputs
OUTPUT_DIR: Path = PROJECT_ROOT / "outputs"
FIGURES_DIR: Path = OUTPUT_DIR / "figures"
METRICS_DIR: Path = OUTPUT_DIR / "metrics"
MODELS_DIR: Path = OUTPUT_DIR / "models"

# Decision Thresholds to evaluate
EVALUATION_THRESHOLDS: list[float] = [
    0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90
]

# Ensure output directories exist
for directory in [OUTPUT_DIR, FIGURES_DIR, METRICS_DIR, MODELS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)
