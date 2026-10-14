# OIBSIP — OASIS INFOBYTE Internship Portfolio

[![GitHub](https://img.shields.io/badge/GitHub-ahmedmahmoudkhalifa300-181717?logo=github)](https://github.com/ahmedmahmoudkhalifa300)
[![Repository](https://img.shields.io/badge/Repository-OIBSIP-blue?logo=github)](https://github.com/ahmedmahmoudkhalifa300/OIBSIP)
[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Data Analytics](https://img.shields.io/badge/Track-Data%20Analytics-2E8B57)](https://github.com/ahmedmahmoudkhalifa300/OIBSIP)

## About This Repository

This repository contains my project submissions and learning outcomes from the **OASIS INFOBYTE Internship Program**. It serves as a structured portfolio of practical work completed across data analytics, exploratory data analysis, and machine learning tasks.

The projects in this repository are organized according to the internship submission format and are designed to demonstrate the complete workflow of solving real-world data problems: understanding a business or analytical objective, preparing data, building a solution, evaluating results, documenting decisions, and communicating findings clearly.

## About the Internship

The OASIS INFOBYTE internship is a project-based learning experience focused on applying technical knowledge to practical tasks. The program encourages interns to build portfolio-ready projects, document their work, publish their implementations on GitHub, and present completed work through a project demonstration.

The internship workflow includes:

- Reviewing the assigned task and feature checklist.
- Building the project using the required technology stack.
- Documenting the implementation and results.
- Publishing the completed work in the official `OIBSIP` repository.
- Creating a demonstration of the completed project.
- Sharing the project professionally and participating in peer evaluation.

## Internship Track

**Track:** Data Analytics  
**Repository:** [OIBSIP](https://github.com/ahmedmahmoudkhalifa300/OIBSIP)  
**Author:** Ahmed Mahmoud Khalifa  
**Role:** AI Engineer / Data Analytics Intern

The Data Analytics track includes practical tasks covering exploratory analysis, data cleaning, customer segmentation, sentiment analysis, regression, classification, and fraud detection. The tasks are intended to develop end-to-end analytical thinking and machine learning implementation skills.

## Repository Structure

```text
OIBSIP/
├── DataAnalytics-L1-EDARetailSales/
├── Data Analytics-L1-Task4-Sentiment Analysis/
├── Data Analytics-L2-Task3-Fraud Detection/
└── README.md
```

Each task folder is designed to contain its own implementation, documentation, notebooks, results, and supporting resources where applicable.

## Completed and Featured Projects

### 1. Exploratory Data Analysis on Retail Sales Data

**Folder:** [`DataAnalytics-L1-EDARetailSales/`](./DataAnalytics-L1-EDARetailSales/)

This project focuses on exploring retail sales data to identify patterns, customer behavior trends, product performance, and business insights through descriptive analysis and visualizations.

Typical analytical activities include:

- Dataset inspection and data-quality checks.
- Descriptive statistics.
- Time-series analysis.
- Customer demographic analysis.
- Product and revenue analysis.
- Correlation analysis.
- Business recommendations based on the findings.

### 2. Sentiment Analysis

**Folder:** [`Data Analytics-L1-Task4-Sentiment Analysis/`](./Data%20Analytics-L1-Task4-Sentiment%20Analysis/)

This project applies natural language processing and machine learning techniques to classify text sentiment. It demonstrates text preprocessing, feature extraction, model training, and evaluation.

Key areas include:

- Text cleaning and preprocessing.
- Sentiment-label analysis.
- TF-IDF feature extraction.
- Classification model training.
- Precision, recall, F1-score, and confusion-matrix evaluation.
- Error analysis and visual reporting.

### 3. Credit Card Fraud Detection

**Folder:** [`Data Analytics-L2-Task3-Fraud Detection/`](./Data%20Analytics-L2-Task3-Fraud%20Detection/)

This is the featured machine learning project in the repository. It builds an end-to-end fraud-detection pipeline using the Kaggle **Credit Card Fraud Detection** dataset provided by the Machine Learning Group of Université Libre de Bruxelles.

Dataset source: [Credit Card Fraud Detection — Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

The dataset contains 284,807 transactions, including 492 fraudulent transactions. Because fraudulent transactions represent only a very small percentage of the data, the project treats class imbalance as a central machine learning challenge.

The project covers:

- Class-imbalance analysis.
- Exploratory analysis of transaction amounts.
- Elapsed-time analysis.
- Stratified train/test splitting.
- SMOTE oversampling applied only to the training data.
- Logistic Regression.
- Random Forest classification.
- Precision, Recall, F1-score, ROC-AUC, and Precision-Recall analysis.
- Confusion matrices and model comparison.
- Decision-threshold analysis.
- Logistic Regression coefficient analysis.
- Random Forest feature importance.
- Scalability considerations for high-volume transaction processing.

## Technology Stack

The projects in this repository use technologies from the Python data and machine learning ecosystem, including:

- Python
- pandas
- NumPy
- scikit-learn
- imbalanced-learn
- matplotlib
- seaborn
- Jupyter Notebook
- Natural Language Processing tools where required
- Git and GitHub

## Machine Learning Methodology

The fraud-detection project follows a reproducible workflow:

1. Load and validate the dataset.
2. Inspect data types, missing values, duplicates, and class distribution.
3. Perform exploratory data analysis.
4. Split the data using stratification so fraud cases appear in both training and test sets.
5. Apply imbalance handling only to the training data.
6. Train multiple classification models.
7. Evaluate on the original, untouched test set.
8. Compare models using fraud-sensitive metrics.
9. Analyze feature influence and model behavior.
10. Discuss deployment and scalability considerations.

### Why Accuracy Is Not the Main Metric

Fraud detection datasets are highly imbalanced. A model can achieve very high accuracy by predicting almost every transaction as legitimate while failing to detect fraudulent transactions.

For this reason, the fraud-detection project emphasizes:

- **Recall:** How many actual fraud cases were detected.
- **Precision:** How many flagged transactions were actually fraudulent.
- **F1-score:** The balance between precision and recall.
- **ROC-AUC:** The model's ability to rank positive cases across thresholds.
- **PR-AUC:** A particularly useful summary for rare positive classes.

The preferred operating threshold depends on the business cost of missed fraud, false alerts, manual investigations, and customer disruption.

## Data and Privacy Notes

The fraud-detection dataset contains anonymized numerical features. The original transaction attributes are not available, and the PCA-transformed variables cannot be interpreted as direct business fields such as merchant name, country, or customer identity.

The raw dataset should not be committed to GitHub. It is large and should be downloaded separately from Kaggle and placed in the relevant project data directory according to that project's README instructions.

No API keys, credentials, private data, or sensitive configuration files should be stored in this repository.

## How to Use This Repository

Clone the repository:

```bash
git clone https://github.com/ahmedmahmoudkhalifa300/OIBSIP.git
cd OIBSIP
```

Open the project folder you want to explore and follow its local README instructions.

For the fraud-detection project, the expected workflow is similar to:

```bash
cd "Data Analytics-L2-Task3-Fraud Detection"
pip install -r requirements.txt
python run_pipeline.py --data-path data/raw/creditcard.csv
pytest -q
jupyter notebook notebooks/01_credit_card_fraud_detection.ipynb
```

The Kaggle dataset can be downloaded from:

```text
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
```

Do not place Kaggle credentials directly in source code. If the Kaggle CLI is configured, the dataset can be downloaded using the command documented in the project-level README.

## Reproducibility

The projects aim to be reproducible by:

- Documenting installation and execution steps.
- Using fixed random seeds where applicable.
- Separating raw data from generated outputs.
- Keeping preprocessing and model evaluation steps explicit.
- Saving relevant figures, metrics, and model artifacts when appropriate.
- Avoiding data leakage between training and test data.

## Learning Outcomes

Through these projects, I am developing practical experience in:

- Data cleaning and validation.
- Exploratory data analysis.
- Statistical summaries and visualization.
- Feature engineering.
- Supervised and unsupervised machine learning.
- Natural language processing.
- Imbalanced classification.
- Model evaluation and interpretation.
- Reproducible Python project organization.
- Git-based version control.
- Communicating technical findings to both technical and non-technical audiences.

## Acknowledgements

I would like to thank **OASIS INFOBYTE** for providing the internship structure, practical task roadmap, and opportunity to build portfolio-oriented projects using real-world datasets and industry-relevant tools.

The Credit Card Fraud Detection project uses the dataset published on Kaggle by the ULB Machine Learning Group:

- [Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- [imbalanced-learn SMOTE Documentation](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html)

## Author

**Ahmed Mahmoud Khalifa**  
AI Engineer | Data Analytics and Machine Learning Enthusiast

- GitHub: [ahmedmahmoudkhalifa300](https://github.com/ahmedmahmoudkhalifa300)
- Repository: [OIBSIP](https://github.com/ahmedmahmoudkhalifa300/OIBSIP)

---

This repository is maintained as part of my OASIS INFOBYTE internship portfolio and will be updated as additional tasks and improvements are completed.
