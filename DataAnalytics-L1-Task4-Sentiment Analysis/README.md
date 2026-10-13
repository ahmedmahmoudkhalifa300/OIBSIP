# 📊 End-to-End Sentiment Analysis Pipeline
### **OASIS INFOBYTE — Data Science Internship (Task 4)**

---

## 🎯 Project Overview
This project builds a robust, production-grade **Natural Language Processing (NLP)** and **Sentiment Analysis** system using social media posts and tweets collected across Twitter, Facebook, and Instagram.

The project emphasizes industry-standard software engineering and machine learning practices to tackle real-world data science challenges:
1. **High Label Cardinality (191 Granular Emotion Classes)**: The raw dataset contains 191 fine-grained emotion labels (such as *Joy, Despair, Serenity, Melancholy, etc.*) burdened with trailing whitespace anomalies and severe data sparsity (many labels appear only 1–3 times). We resolved this by engineering a domain-informed taxonomic mapping that consolidates these into three canonical sentiment classes: **Positive**, **Negative**, and **Neutral**.
2. **Severe Class Imbalance**: Following consolidation, Positive sentiment represents the majority class. This was resolved via **Balanced Class Weighting** in the loss functions and **Stratified K-Fold Cross-Validation** to guarantee fair evaluation across all sentiment polarities.
3. **Social Media Noise**: Social texts contain emojis, informal contractions, web links (`http://`), and username mentions (`@user`). We implemented an advanced cleaning pipeline that translates emojis into descriptive text, expands contractions, cleans noise, and applies sentiment-preserving lemmatization.

---

## 📂 Project Architecture

```text
OIBSIP_Task4_Sentiment/
├── data/
│   └── 3) Sentiment dataset.csv              # Raw social media sentiment dataset
├── notebooks/
│   └── 01_eda_and_sentiment_pipeline.ipynb   # Comprehensive EDA, visualization & modeling notebook
├── src/
│   ├── __init__.py                          # Package initialization
│   ├── preprocess.py                        # Text cleaning, emoji translation, contraction expansion, lemmatization
│   ├── label_engineering.py                 # 191-class consolidation & balanced class weights calculation
│   ├── train.py                             # Training orchestration, 5-Fold Stratified CV, GridSearchCV
│   └── predict.py                           # Standalone inference engine & interactive CLI
├── outputs/
│   ├── models/
│   │   └── best_sentiment_model.joblib      # Production-ready trained model pipeline
│   ├── figures/                             # Generated analytical visualizations
│   │   ├── confusion_matrix.png             # Test set confusion matrix heatmap
│   │   ├── model_comparison.png             # Candidate models benchmark comparison
│   │   ├── sentiment_distribution.png       # Consolidated sentiment distribution
│   │   └── wordclouds_by_sentiment.png      # Word clouds for each sentiment class
│   └── metrics/
│       ├── evaluation_summary.json          # Complete evaluation metrics and classification report
│       └── model_comparison.csv             # Benchmark comparison table across candidate models
├── requirements.txt                         # Python dependencies
└── README.md                                # Project documentation
```

---

## ⚙️ Environment Setup & Installation

### 1. Install Required Dependencies
```bash
pip install -r requirements.txt
```

**Key Libraries**:
- `scikit-learn`: Model training, hyperparameter tuning, pipelines, and evaluation metrics.
- `pandas` & `numpy`: Data manipulation, indexing, and tabular operations.
- `nltk`: Lexical resources, tokenization, lemmatization, and sentiment-aware stopword removal.
- `emoji`: Demojizing Unicode emojis into descriptive English tokens.
- `matplotlib` & `seaborn`: Data visualization and performance reporting.
- `wordcloud`: Visual word cloud generation for distinct sentiment classes.
- `joblib`: Model serialization and persistence.

---

## 🧠 Technical Pipeline Stages

### 1. Advanced Text Preprocessing (`src/preprocess.py`)
- **Emoji Demojization**: Translates Unicode emojis into descriptive text tokens (e.g., `💪` $\rightarrow$ `muscle`, `😊` $\rightarrow$ `smile face`) so models capture sentiment signals embedded in emojis.
- **Contraction Expansion**: Normalizes informal English abbreviations (e.g., `can't` $\rightarrow$ `cannot`, `i'm` $\rightarrow$ `i am`, `don't` $\rightarrow$ `do not`).
- **Sanitization**: Strips web links (`https?://`), user tags (`@username`), and punctuation while retaining hashtag text.
- **Sentiment-Preserving Stopword Filtering**: Retains critical negation and intensifier words (`not`, `no`, `never`, `very`, `bad`, `good`, etc.) to prevent polarity inversion.
- **Lemmatization**: Uses NLTK's `WordNetLemmatizer` with verb and noun fallback to reduce inflected forms to their dictionary root.

---

### 2. Label Engineering & Imbalance Mitigation (`src/label_engineering.py`)
- **191-Class Harmonization**: Maps every granular emotion to a standardized 3-class schema:
  - **Positive** (61.5%): *Joy, Excitement, Contentment, Gratitude, Serenity, Love, Pride, etc.*
  - **Negative** (25.5%): *Despair, Sad, Hate, Fear, Frustration, Melancholy, Anger, etc.*
  - **Neutral** (13.0%): *Neutral, Nostalgia, Curiosity, Reflection, Surprise, etc.*
- **Class Weighting Calculation**:
  Dynamically computes balanced class weights using `compute_class_weight`:
  - `Negative`: ~1.3048
  - `Neutral`: ~2.5684
  - `Positive`: ~0.5422
  These weights penalize misclassifications on minority classes proportionally, counteracting bias toward the majority class.

---

### 3. Model Training & Cross-Validation (`src/train.py`)
Features are extracted using **TF-IDF Vectorization** with unigram and bigram ranges `(1, 2)` and logarithmic sublinear term frequency scaling (`sublinear_tf=True`).

We benchmarked 4 distinct machine learning classifiers using **5-Fold Stratified Cross-Validation**:
1. **Logistic Regression (Class-Weighted)**
2. **Linear Support Vector Classifier (Calibrated for Probabilities)**
3. **Random Forest Classifier (Balanced)**
4. **Multinomial Naive Bayes**

Hyperparameter tuning was conducted on the top-performing architecture via **GridSearchCV**, optimizing regularizer strength ($C$), solver algorithms, and TF-IDF feature constraints.

#### Model Comparison Benchmark Results:

| Model Architecture | 5-Fold CV F1-Weighted | Test Accuracy | F1-Score (Weighted) | F1-Score (Macro) |
|---|:---:|:---:|:---:|:---:|
| 🏆 **Logistic Regression (Tuned)** | **0.8628** | **86.39%** | **0.8530** | **0.7790** |
| **Linear SVC (Calibrated)** | 0.8636 | 85.71% | 0.8467 | 0.7740 |
| **Random Forest (Balanced)** | 0.7236 | 78.23% | 0.7572 | 0.6558 |
| **Multinomial Naive Bayes** | 0.5963 | 73.47% | 0.6759 | 0.5134 |

---

### 4. Final Model Evaluation on Holdout Test Set

```text
              precision    recall  f1-score   support

    Negative       0.89      0.87      0.88        38
     Neutral       0.80      0.42      0.55        19
    Positive       0.86      0.96      0.91        90

    accuracy                           0.86       147
   macro avg       0.85      0.75      0.78       147
weighted avg       0.86      0.86      0.85       147
```

- **Overall Accuracy**: **86.39%**
- **Positive Sentiment F1-Score**: **0.91**
- **Negative Sentiment F1-Score**: **0.88**

---

## 🚀 How to Run the Project

### 1. Train the Pipeline & Generate Reports
```bash
python src/train.py
```
This script will:
- Preprocess data and perform label consolidation.
- Execute 5-Fold Stratified Cross-Validation across candidate models.
- Run hyperparameter optimization using GridSearchCV.
- Export the best pipeline to `outputs/models/best_sentiment_model.joblib`.
- Save visualizations to `outputs/figures/` and metrics to `outputs/metrics/`.

---

### 2. Predict a Single Text via Command Line
```bash
python src/predict.py --text "Just finished an amazing workout! Super energized! 💪"
```
**Expected Output:**
```text
Text      : Just finished an amazing workout! Super energized! 💪
Cleaned   : finish amaze workout super energize flex biceps
Sentiment : 🟢 [POSITIVE] Positive
Confidence: 87.15%
Probabilities: {'Negative': 0.0833, 'Neutral': 0.0452, 'Positive': 0.8715}
```

---

### 3. Real-Time Interactive CLI Mode
```bash
python src/predict.py
```
Launches an interactive console interface where you can enter tweets or sentences continuously, receiving immediate sentiment classifications, confidence scores, and visual probability bars.

---

### 4. Explore the Jupyter Notebook
Open the interactive demonstration notebook:
```bash
jupyter notebook notebooks/01_eda_and_sentiment_pipeline.ipynb
```

---

## 📈 Generated Artifacts & Visualizations

- **Confusion Matrix**: `outputs/figures/confusion_matrix.png`
- **Model Comparison Benchmark Chart**: `outputs/figures/model_comparison.png`
- **Consolidated Sentiment Distribution**: `outputs/figures/sentiment_distribution.png`
- **Word Clouds by Sentiment Polarity**: `outputs/figures/wordclouds_by_sentiment.png`
- **Full Evaluation Summary (JSON)**: `outputs/metrics/evaluation_summary.json`
- **Benchmark Comparison Table (CSV)**: `outputs/metrics/model_comparison.csv`
- **Trained Production Pipeline**: `outputs/models/best_sentiment_model.joblib`
