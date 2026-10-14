# Credit Card Fraud Detection — End-to-End Machine Learning Pipeline

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Tests: Pytest](https://img.shields.io/badge/tests-pytest-green.svg)](https://pytest.org)

An industrial-grade, reproducible machine learning pipeline designed to detect fraudulent credit card transactions in streams characterized by **extreme class imbalance** (~0.1727% positive instances). Built with strict data leakage prevention, synthetic oversampling (`SMOTE`), cost-sensitive learning (`class_weight='balanced'`), decision threshold optimization, and an enterprise scalability architecture.

---

## Table of Contents
1. [Project Overview](#1-project-overview)
2. [Dataset Description & Verification](#2-dataset-description--verification)
3. [Methodology & Leakage Prevention](#3-methodology--leakage-prevention)
4. [The Fallacy of Accuracy & Metrics Justification](#4-the-fallacy-of-accuracy--metrics-justification)
5. [Project Architecture](#5-project-architecture)
6. [Installation & Setup](#6-installation--setup)
7. [How to Download Kaggle Data](#7-how-to-download-kaggle-data)
8. [Running the Pipeline & Notebook](#8-running-the-pipeline--notebook)
9. [Running Tests](#9-running-tests)
10. [Empirical Experimental Results](#10-empirical-experimental-results)
11. [Decision Threshold Optimization](#11-decision-threshold-optimization)
12. [Feature Importance & PCA Interpretability](#12-feature-importance--pca-interpretability)
13. [Scalability Architecture: 1 Million Transactions / Hour](#13-scalability-architecture-1-million-transactions--hour)
14. [Real-World Limitations](#14-real-world-limitations)
15. [Reproducibility & Safety Disclaimers](#15-reproducibility--safety-disclaimers)
16. [References & Documentation](#16-references--documentation)

---

## 1. Project Overview

Credit card fraud inflicts tens of billions of dollars in annual damages on financial institutions, merchants, and cardholders. Automated machine learning defense systems must satisfy two stringent criteria:
1. **High Recall**: Maximize the identification of fraudulent transactions to curtail financial liability.
2. **High Precision**: Minimize false positive alerts to prevent frustrating legitimate cardholders and wasting fraud investigation team bandwidth.

In real-world data, legitimate transactions overwhelmingly dominate the distribution. Standard machine learning algorithms optimize for global loss (accuracy), leading to classifiers that predict every transaction as legitimate. This repository implements an end-to-end Python pipeline and pedagogical Jupyter Notebook demonstrating how to properly handle extreme class imbalance with zero data leakage.

---

## 2. Dataset Description & Verification

- **Name**: Credit Card Fraud Detection
- **Host**: [Kaggle Dataset Link](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Curated By**: Machine Learning Group (MLG - ULB) / Worldline & Université Libre de Bruxelles
- **Local Target Path**: `data/raw/creditcard.csv`

### Verified Specifications
| Metric | Verified Value | Description |
| :--- | :--- | :--- |
| **Total Transactions** | `284,807` | Recorded over a 48-hour observation window (Sept 2013) |
| **Legitimate Transactions (0)** | `284,315` | `99.8273%` of the dataset |
| **Fraudulent Transactions (1)** | `492` | `0.1727%` of the dataset (extreme minority) |
| **Imbalance Ratio** | `~ 1 : 577` | 1 fraud case for every 577 legitimate transactions |
| **Features** | `31` total | `Time`, `Amount`, `Class`, and `V1`..`V28` |
| **Missing Values** | `0` | Clean tabular records without nulls or NaNs |
| **Duplicate Rows** | `1,081` | Preserved as valid repeated payment authorizations |

### Features Description
- `Time`: Elapsed seconds between the given transaction and the first transaction in the dataset. Covers approximately 172,792 seconds (~48 hours).
- `Amount`: Monetary charge value (Mean: `$88.35`, Median: `$22.00`, Max: `$25,691.16`).
- `V1` to `V28`: Anonymized continuous numerical variables obtained through Principal Component Analysis (PCA) to preserve cardholder privacy.
- `Class`: Response target label (`0` = Legitimate, `1` = Fraudulent).

---

## 3. Methodology & Leakage Prevention

### 1. Stratified Partitioning Before Any Transformation
To guarantee that the tiny minority class appears proportionally in both training and evaluation, we apply `train_test_split` with `stratify=y` (`test_size=0.20`, `random_state=42`):
- **Train partition**: 227,845 records (394 frauds, `0.1729%`)
- **Test partition**: 56,962 records (98 frauds, `0.1720%`)

### 2. Feature Engineering
We extract non-linear transformations from the raw signals:
- `time_hours = Time / 3600.0`: Continuous elapsed hours.
- `time_bin = (Time // 3600).astype(int)`: Integer bucket indicating the chronological time window.
  > **Note**: `time_bin` represents elapsed time slices since dataset inception, **NOT** the wall-clock hour of the day (e.g., 3 PM). It is not fed directly into linear models without cyclical encoding.
- `log_amount = log1p(Amount)`: Compresses extreme right-skewed monetary amounts to stabilize model convergence.

### 3. Leakage-Free Preprocessing with `imbalanced-learn` Pipelines
Data leakage is a fatal flaw in fraud models. Applying oversampling or scaling on the full dataset before splitting leaks test distribution information into the training phase.
We utilize `imblearn.pipeline.Pipeline` where:
- `StandardScaler` fits parameters ($\mu, \sigma$) strictly on `X_train`.
- `SMOTE` generates synthetic minority examples strictly during the `fit()` phase of `X_train`.
- The hold-out `X_test` remains completely untouched and retains its raw, real-world distribution.

---

## 4. The Fallacy of Accuracy & Metrics Justification

> **The Accuracy Paradox in Fraud Detection:**  
> Accuracy is misleading for this dataset because fraudulent transactions represent only about 0.1727% of all transactions. A classifier that predicts every transaction as legitimate can achieve approximately 99.83% accuracy while detecting zero fraudulent transactions. Therefore, this project focuses on Recall, Precision, F1-score, ROC-AUC, and preferably Precision-Recall AUC.

### Why Precision-Recall AUC (PR-AUC) Dominates ROC-AUC
In extreme class imbalance:
- False Positive Rate ($FPR = \frac{FP}{FP + TN}$) has a massive denominator ($FP + 56,844$). Even if a model makes 1,000 false alarms, $FPR \approx 0.017$, causing the ROC curve to look outstanding ($ROC\text{-}AUC > 0.95$).
- Precision ($\frac{TP}{TP + FP}$) is directly destroyed by false alarms.
- The **Precision-Recall Curve** isolates Precision against Recall against a no-skill baseline equal to the positive class prevalence ($0.172\%$). Hence, **PR-AUC (Average Precision) is the gold standard**.

---

## 5. Project Architecture

```text
fraud-detection-project/
├── data/
│   ├── raw/
│   │   └── creditcard.csv                 # Kaggle dataset (git-ignored)
│   └── README.md                          # Data schema & download instructions
├── notebooks/
│   └── 01_credit_card_fraud_detection.ipynb # Complete pedagogical tutorial
├── src/
│   ├── __init__.py                        # Package initialization
│   ├── config.py                          # Paths, seeds, constants, thresholds
│   ├── data_loader.py                     # Safe loading, path resolver, validator
│   ├── preprocessing.py                   # Feature engineering & stratified split
│   ├── models.py                          # ImbPipelines (Baseline, SMOTE, RF)
│   ├── evaluation.py                      # Metrics, threshold audit, curves
│   └── visualization.py                   # High-DPI EDA & interpretability plots
├── outputs/
│   ├── figures/                           # 15 High-resolution diagnostic charts
│   ├── metrics/                           # metrics.csv, threshold CSV tables
│   └── models/                            # joblib serialized pipelines
├── tests/
│   ├── conftest.py                        # Path configuration for test runner
│   └── test_pipeline.py                   # Automated pytest suite (6 tests)
├── .gitignore                             # Ignores .csv, .joblib, virtualenvs
├── pytest.ini                             # Pytest configuration
├── requirements.txt                       # Locked dependencies
├── run_pipeline.py                        # CLI entrypoint for full training
└── README.md                              # Comprehensive documentation
```

---

## 6. Installation & Setup

### Prerequisites
- Python 3.10+ (Tested on Python 3.11)
- Virtual environment recommended (`venv` or `conda`)

```bash
# Clone the repository
git clone https://github.com/<your-username>/fraud-detection-project.git
cd fraud-detection-project

# Create and activate virtual environment
python -m venv .venv

# On Windows:
.venv\Scripts\activate
# On Linux / macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## 7. How to Download Kaggle Data

The dataset file `creditcard.csv` must be placed in `data/raw/creditcard.csv`.

### Option A: Using Kaggle CLI (Recommended)
1. Install Kaggle package: `pip install kaggle`
2. Obtain your `kaggle.json` API token from Kaggle -> Account Settings -> Create New Token.
3. Place `kaggle.json` into:
   - Windows: `%USERPROFILE%\.kaggle\kaggle.json`
   - Linux/macOS: `~/.kaggle/kaggle.json`
4. Run:
```bash
kaggle datasets download -d mlg-ulb/creditcardfraud -p data/raw --unzip
```

### Option B: Manual Download
1. Visit the Kaggle competition page: [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).
2. Click **Download** (150 MB zip).
3. Unzip and copy `creditcard.csv` into `data/raw/creditcard.csv`.

---

## 8. Running the Pipeline & Notebook

### Execute the Full Automated Pipeline
To run data validation, feature engineering, EDA generation, training, threshold tuning, and model persistence:

```bash
python run_pipeline.py --data-path data/raw/creditcard.csv
```

*For rapid testing on a 20,000-row subset:*
```bash
python run_pipeline.py --quick-run
```

### Launch the Interactive Jupyter Notebook
```bash
jupyter notebook notebooks/01_credit_card_fraud_detection.ipynb
```

---

## 9. Running Tests

Unit tests verify schema integrity, feature transformations, stratified partitioning, pipeline training, threshold tables, and dataset resolution:

```bash
pytest -v
```

All 6 test cases run self-contained and execute in <10 seconds without requiring an internet connection.

---

## 10. Empirical Experimental Results

Evaluated on the un-resampled test set containing **56,962 transactions** (56,864 legitimate, **98 fraudulent**):

| Model | Imbalance Strategy | Precision | Recall | F1-Score | ROC-AUC | PR-AUC | Caught Fraud (TP) | False Alarms (FP) | Missed Fraud (FN) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline Logistic** | None (Unweighted) | `0.8312` | `0.6531` | `0.7314` | `0.9595` | `0.7386` | 64 | **13** | 34 |
| **Logistic Regression** | **SMOTE Pipeline** | `0.0581` | **`0.9184`** | `0.1093` | **`0.9710`** | `0.7215` | **90** | 1,459 | **8** |
| **Random Forest** | **Balanced Weights** | **`0.8020`** | `0.8265` | **`0.8141`** | `0.9653` | **`0.8546`** | 81 | **20** | 17 |

### Comparative Analysis
1. **The Baseline Model Fails Real-World Fraud Detection**:
   Despite attaining **99.91% raw accuracy**, the baseline model fails to catch **34 out of 98 fraud cases** (Recall: 65.31%). In banking, a 35% missed fraud rate represents catastrophic financial leakage.
2. **SMOTE Dramatically Boosts Linear Recall**:
   Adding SMOTE to Logistic Regression increases Recall from **65.31% to 91.84%** (catching 90 of 98 frauds, only 8 missed). However, because linear models cannot partition complex non-linear manifolds, it generates 1,459 false positives.
3. **Random Forest (Balanced) Delivers Superior Enterprise Balance**:
   The Random Forest model achieves the optimal trade-off: **82.65% Recall** while generating only **20 false alarms** across nearly 57,000 transactions, leading to the highest overall **F1-Score (0.8141)** and dominant **PR-AUC (0.8546)**.

---

## 11. Decision Threshold Optimization

Never deploy a fraud model using an arbitrary default probability threshold of `0.50`. Below is the empirical threshold audit for **Random Forest (Balanced)**:

| Threshold | Flagged Volume | Caught Fraud (TP) | Missed Fraud (FN) | False Alarms (FP) | Precision | Recall | F1-Score |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0.10** | 289 | 88 | 10 | 201 | 30.45% | **89.80%** | 0.4548 |
| **0.20** | 169 | 87 | 11 | 82 | 51.48% | **88.78%** | 0.6517 |
| **0.30** | 129 | 84 | 14 | 45 | 65.12% | 85.71% | 0.7401 |
| **0.40** | 111 | 83 | 15 | 28 | 74.77% | 84.69% | 0.7943 |
| **0.50** | 101 | 81 | 17 | 20 | 80.20% | 82.65% | 0.8141 |
| **0.60** | 89 | 79 | 19 | 10 | 88.76% | 80.61% | 0.8449 |
| **0.70** | 81 | 76 | 22 | 5 | **93.83%** | 77.55% | **0.8492** |
| **0.80** | 75 | 73 | 25 | 2 | **97.33%** | 74.49% | 0.8439 |
| **0.90** | 65 | 63 | 35 | 2 | 96.92% | 64.29% | 0.7730 |

### Business Takeaways
- **Aggressive Fraud Prevention (e.g., Threshold = 0.20)**: Catches **88.78%** of fraud (87/98), flagging only 169 transactions in total (~0.3% of traffic) with 82 false alarms.
- **Low-Friction Automated Blocking (e.g., Threshold = 0.70)**: Delivers **93.83% Precision** (only 5 false alarms) while still intercepting **77.55%** of all fraud attempts.

---

## 12. Feature Importance & PCA Interpretability

- **Logistic Regression (SMOTE)**: The top positive predictors of fraud risk are `V4`, `V11`, and `V21`. The strongest negative protectors are `V14`, `V12`, `V10`, and `V17`.
- **Random Forest**: The most influential split features measured by Gini impurity reduction are `V17`, `V12`, `V14`, `V10`, and `V11`.
- **PCA Disclaimer**: Because `V1` through `V28` are anonymized linear projections, domain analysts cannot translate them directly into merchant category codes or IP locations without access to the original projection matrix.

---

## 13. Scalability Architecture: 1 Million Transactions / Hour

### Sizing Calculation
$$\text{Required Throughput} = \frac{1,000,000 \text{ transactions}}{3,600 \text{ seconds}} \approx 278 \text{ transactions/second (TPS)}$$
At peak surge hours (e.g., flash sales), systems must accommodate **1,000 to 1,500 TPS** with a strict service-level agreement of **p99 latency < 15ms**.

### Production Blueprint
1. **Decoupled Lifecycle**:
   - **Offline Asynchronous Training**: Model retraining, historical backtesting, and SMOTE resampling are performed offline on distributed compute clusters (Apache Spark, Databricks, Ray).
   - **Online Low-Latency Inference**: Pre-trained frozen pipelines are served through dedicated, stateless microservice pods.
2. **Streaming Ingestion**:
   - High-throughput transaction events publish to partitioned **Apache Kafka** / **AWS Kinesis** topics.
   - **Apache Flink** calculates real-time streaming stateful features (e.g., sliding window card swipe count in last 5 minutes).
3. **Low-Latency Serving**:
   - Models are exported to **ONNX** or **TensorRT** and hosted in **Triton Inference Server** or asynchronous **FastAPI/gRPC** microservices running on Kubernetes with Horizontal Pod Autoscaling (HPA).
4. **SMOTE in Production Rule**:
   - **SMOTE is NEVER executed during live inference.** SMOTE is strictly a training-phase data synthesizer. Real-time transactions are scored directly against the learned model weights.
5. **Continuous Model Observability**:
   - Monitor feature distribution drift via **Population Stability Index (PSI)** and **Kolmogorov-Smirnov (KS)** tests.
   - Log prediction scores, latencies, and investigation outcomes into ClickHouse / Elasticsearch to monitor concept drift and schedule automated retraining.
6. **Tiered Decision Flow**:
   - `Probability >= 0.80`: Immediate automated decline.
   - `0.25 <= Probability < 0.80`: Step-Up challenge (SMS OTP or 3D-Secure biometric prompt).
   - `Probability < 0.25`: Transparent approval.

---

## 14. Real-World Limitations

1. **Short Observation Horizon**: Dataset covers only 48 hours; seasonal patterns, monthly salary deposits, and holiday shopping surges cannot be modeled.
2. **Anonymized Variables**: Lack of raw variables (merchant name, cardholder age, terminal type, merchant category code) limits contextual domain rules.
3. **Relative Elapsed Time**: `Time` is a relative duration from record zero, not an absolute UTC timestamp with day-of-week or time-zone attributes.
4. **Adversarial Concept Drift**: Fraud syndicates adapt evasive strategies rapidly; static models degrade without continuous human-in-the-loop retraining.
5. **SMOTE Boundary Risks**: In complex non-linear spaces, SMOTE can interpolate synthetic minority points in marginal clean regions, potentially elevating false positive rates if uncalibrated.

---

## 15. Reproducibility & Safety Disclaimers

- `random_state = 42` is fixed across data splitting, SMOTE synthesis, and model seeds.
- No confidential credentials, API keys, or raw dataset files are committed to Git (`.gitignore` enforced).
- Data files are stored locally in `data/raw/` and excluded from version control.

---

## 16. References & Documentation

- **Kaggle Dataset**: [Credit Card Fraud Detection on Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Imbalanced-Learn SMOTE Reference**: [SMOTE Documentation — imbalanced-learn](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html)
- **Scikit-Learn Classification Metrics**: [Scikit-Learn Evaluation Metrics](https://scikit-learn.org/stable/modules/model_evaluation.html)
- **Foundational Paper**: Andrea Dal Pozzolo, Olivier Caelen, Reid A. Johnson, and Gianluca Bontempi. *Calibrating Probability with Undersampling for Unbalanced Classification*. In IEEE SSCI, 2015.
