# Dataset Directory

## Dataset Overview
- **Name**: Credit Card Fraud Detection Dataset
- **Source**: Machine Learning Group (MLG - ULB) / Worldline & Université Libre de Bruxelles
- **Kaggle URL**: [https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Local Path**: `data/raw/creditcard.csv`

## Dataset Specifications
- **Total Transactions**: 284,807
- **Fraudulent Transactions (Class = 1)**: 492
- **Legitimate Transactions (Class = 0)**: 284,315
- **Fraud Rate**: ~0.1727% (Extreme class imbalance)
- **Features**:
  - `Time`: Number of seconds elapsed between this transaction and the first transaction in the dataset (covers roughly 48 hours / 2 days).
  - `V1` to `V28`: Anonymized features obtained via Principal Component Analysis (PCA) due to confidentiality.
  - `Amount`: Transaction amount in currency units.
  - `Class`: Response variable (1 for fraudulent transaction, 0 for legitimate).

## How to Download

### Option 1: Using Kaggle CLI
1. Install kaggle CLI:
   ```bash
   pip install kaggle
   ```
2. Place your `kaggle.json` API token in `~/.kaggle/` (Linux/macOS) or `%USERPROFILE%\.kaggle\` (Windows).
3. Run the download command into `data/raw/`:
   ```bash
   kaggle datasets download -d mlg-ulb/creditcardfraud -p data/raw --unzip
   ```

### Option 2: Manual Download
1. Visit [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).
2. Click **Download** (requires Kaggle login).
3. Extract `creditcard.csv` and place it in `data/raw/creditcard.csv`.

> **Note**: Raw data files (`*.csv`) are excluded from Git version control via `.gitignore`.
