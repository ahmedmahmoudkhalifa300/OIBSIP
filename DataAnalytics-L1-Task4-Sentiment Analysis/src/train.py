"""
Train Module
------------
End-to-End training pipeline:
1. Loads raw dataset
2. Cleans text using TextPreprocessor
3. Maps granular 191 labels to canonical sentiments using LabelEngineering
4. Handles class imbalance with class weights
5. Performs Stratified Cross-Validation and Model Comparison
6. Executes Hyperparameter Optimization using GridSearchCV
7. Evaluates on holdout test set (Confusion Matrix, F1, Accuracy, Precision, Recall)
8. Saves best pipeline, figures, and metrics to outputs directory.
"""

import sys
import os

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

import argparse
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    classification_report, confusion_matrix
)

# Relative / local imports
from src.preprocess import clean_text
from src.label_engineering import engineer_labels


def generate_eda_figures(df: pd.DataFrame, output_dir: str = "outputs/figures"):
    """Generate and save EDA plots: sentiment distribution and word clouds."""
    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")

    # 1. Sentiment Distribution Plot
    plt.figure(figsize=(8, 5))
    palette = {'Positive': '#2ecc71', 'Negative': '#e74c3c', 'Neutral': '#3498db'}
    counts = df['Clean_Sentiment'].value_counts()
    ax = sns.barplot(x=counts.index, y=counts.values, hue=counts.index, palette=palette, legend=False)
    plt.title('Cleaned Sentiment Distribution (3 Standard Classes)', fontsize=14, fontweight='bold', pad=12)
    plt.xlabel('Sentiment Class', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    for p in ax.patches:
        height = p.get_height()
        ax.annotate(f'{int(height)} ({height/len(df):.1%})',
                    (p.get_x() + p.get_width() / 2., height),
                    ha='center', va='bottom', fontsize=10, fontweight='bold',
                    xytext=(0, 3), textcoords='offset points')
    plt.tight_layout()
    dist_path = os.path.join(output_dir, 'sentiment_distribution.png')
    plt.savefig(dist_path, dpi=300)
    plt.close()
    print(f"[EDA] Saved distribution plot to: {dist_path}")

    # 2. Word Clouds for each sentiment class
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    class_colors = {'Positive': 'Greens', 'Negative': 'Reds', 'Neutral': 'Blues'}
    for idx, (cls, cmap) in enumerate(class_colors.items()):
        subset = df[df['Clean_Sentiment'] == cls]['Clean_Text'].dropna().str.cat(sep=' ')
        if not subset.strip():
            subset = "no content available"
        wc = WordCloud(
            width=600, height=400,
            background_color='white',
            colormap=cmap,
            max_words=100,
            random_state=42
        ).generate(subset)
        axes[idx].imshow(wc, interpolation='bilinear')
        axes[idx].set_title(f'Word Cloud: {cls}', fontsize=14, fontweight='bold')
        axes[idx].axis('off')
    plt.tight_layout()
    wc_path = os.path.join(output_dir, 'wordclouds_by_sentiment.png')
    plt.savefig(wc_path, dpi=300)
    plt.close()
    print(f"[EDA] Saved word clouds to: {wc_path}")


def plot_confusion_matrix(cm: np.ndarray, classes: list, output_path: str):
    """Plot and save a styled confusion matrix heatmap."""
    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=classes, yticklabels=classes,
        cbar=True, annot_kws={'size': 14, 'weight': 'bold'}
    )
    plt.title('Confusion Matrix - Best Sentiment Model', fontsize=14, fontweight='bold', pad=12)
    plt.xlabel('Predicted Sentiment', fontsize=12, fontweight='bold')
    plt.ylabel('Actual Sentiment', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[Evaluation] Saved confusion matrix to: {output_path}")


def plot_model_comparison(comparison_df: pd.DataFrame, output_path: str):
    """Plot bar chart comparing model performances across metrics."""
    plt.figure(figsize=(10, 6))
    melted = comparison_df.melt(id_vars=['Model'], value_vars=['CV_F1_Weighted', 'Test_Accuracy', 'Test_F1_Weighted'],
                                var_name='Metric', value_name='Score')
    palette = {'CV_F1_Weighted': '#34495e', 'Test_Accuracy': '#3498db', 'Test_F1_Weighted': '#2ecc71'}
    ax = sns.barplot(data=melted, x='Model', y='Score', hue='Metric', palette=palette)
    plt.title('Sentiment Analysis Models Performance Comparison', fontsize=14, fontweight='bold', pad=12)
    plt.ylim(0.0, 1.05)
    plt.ylabel('Score (0 - 1.0)', fontsize=12)
    plt.xlabel('Model Architecture', fontsize=12)
    plt.legend(title='Metric', frameon=True)
    for p in ax.patches:
        val = p.get_height()
        if val > 0:
            ax.annotate(f'{val:.2f}', (p.get_x() + p.get_width() / 2., val),
                        ha='center', va='bottom', fontsize=9, fontweight='bold',
                        xytext=(0, 2), textcoords='offset points')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[Evaluation] Saved model comparison chart to: {output_path}")


def train_pipeline(data_path: str = "data/3) Sentiment dataset.csv", tune_hyperparameters: bool = True):
    """Main training orchestration routine."""
    print("=" * 70)
    print("🚀 STARTING SENTIMENT ANALYSIS PIPELINE TRAINING")
    print("=" * 70)

    # 1. Load Data
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset file not found at: {data_path}")

    print(f"📥 Loading dataset from {data_path}...")
    df = pd.read_csv(data_path)
    print(f"   Loaded {len(df)} samples with columns: {list(df.columns)}")

    # 2. Text Preprocessing
    print("🧹 Cleaning and normalizing text...")
    df['Clean_Text'] = df['Text'].astype(str).apply(clean_text)

    # Remove empty texts if any
    df = df[df['Clean_Text'].str.strip() != ""].reset_index(drop=True)

    # 3. Label Engineering (Mapping 191 classes -> 3 classes)
    print("🏷️  Mapping 191 granular categories to 3 standard sentiments...")
    df, label_metadata = engineer_labels(df, sentiment_column='Sentiment')
    print(f"   Class distribution: {label_metadata['class_distribution']}")
    print(f"   Balanced class weights: {label_metadata['balanced_class_weights']}")

    # 4. Generate EDA Visualizations
    generate_eda_figures(df, output_dir="outputs/figures")

    # 5. Train/Test Split (Stratified)
    X = np.array(df['Clean_Text'].astype(str).tolist())
    y = np.array(df['Clean_Sentiment'].astype(str).tolist())
    classes = sorted(list(np.unique(y)))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"✂️  Data Split: Train set = {len(X_train)} samples, Test set = {len(X_test)} samples")

    # 6. Candidate Classifiers Setup
    # Note: We use CalibratedClassifierCV on LinearSVC to enable predict_proba
    candidate_models = {
        'Logistic Regression (Balanced)': LogisticRegression(
            class_weight='balanced', max_iter=1000, random_state=42
        ),
        'Linear SVC (Calibrated)': CalibratedClassifierCV(
            LinearSVC(class_weight='balanced', max_iter=2000, random_state=42)
        ),
        'Multinomial Naive Bayes': MultinomialNB(),
        'Random Forest (Balanced)': RandomForestClassifier(
            n_estimators=100, class_weight='balanced', random_state=42
        )
    }

    cv_folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    comparison_records = []

    print("\n🔍 Evaluating candidate models with 5-Fold Stratified Cross-Validation...")
    for model_name, clf in candidate_models.items():
        pipe = Pipeline([
            ('tfidf', TfidfVectorizer(ngram_range=(1, 2), max_features=5000, sublinear_tf=True)),
            ('clf', clf)
        ])
        cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv_folds, scoring='f1_weighted')
        mean_cv_f1 = float(np.mean(cv_scores))
        std_cv_f1 = float(np.std(cv_scores))

        # Fit on train and evaluate on hold-out test
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        test_acc = float(accuracy_score(y_test, y_pred))
        p_w, r_w, f1_w, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted', zero_division=0)
        p_m, r_m, f1_m, _ = precision_recall_fscore_support(y_test, y_pred, average='macro', zero_division=0)

        comparison_records.append({
            'Model': model_name,
            'CV_F1_Weighted': round(mean_cv_f1, 4),
            'CV_F1_Std': round(std_cv_f1, 4),
            'Test_Accuracy': round(test_acc, 4),
            'Test_F1_Weighted': round(float(f1_w), 4),
            'Test_Precision_Weighted': round(float(p_w), 4),
            'Test_Recall_Weighted': round(float(r_w), 4),
            'Test_F1_Macro': round(float(f1_m), 4),
            'Test_Precision_Macro': round(float(p_m), 4),
            'Test_Recall_Macro': round(float(r_m), 4)
        })
        print(f"   ▶ {model_name:30s} | CV F1-W: {mean_cv_f1:.4f} (±{std_cv_f1:.4f}) | Test Acc: {test_acc:.4f} | Test F1-W: {f1_w:.4f}")

    comp_df = pd.DataFrame(comparison_records)
    comp_df = comp_df.sort_values(by='CV_F1_Weighted', ascending=False).reset_index(drop=True)

    os.makedirs("outputs/metrics", exist_ok=True)
    comp_csv_path = "outputs/metrics/model_comparison.csv"
    comp_df.to_csv(comp_csv_path, index=False)
    print(f"\n📊 Saved model comparison metrics table to: {comp_csv_path}")

    # Plot comparison chart
    plot_model_comparison(comp_df, "outputs/figures/model_comparison.png")

    # 7. Hyperparameter Tuning using GridSearchCV on LogisticRegression / LinearSVC
    print("\n⚙️  Running Hyperparameter Tuning (GridSearchCV)...")
    base_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(sublinear_tf=True)),
        ('clf', LogisticRegression(class_weight='balanced', max_iter=2000, random_state=42))
    ])

    param_grid = {
        'tfidf__ngram_range': [(1, 1), (1, 2)],
        'tfidf__max_features': [3000, 5000, None],
        'tfidf__min_df': [1, 2],
        'clf__C': [0.1, 1.0, 5.0, 10.0],
        'clf__solver': ['lbfgs']
    }

    grid_search = GridSearchCV(
        estimator=base_pipeline,
        param_grid=param_grid,
        cv=cv_folds,
        scoring='f1_weighted',
        n_jobs=-1,
        verbose=0
    )
    grid_search.fit(X_train, y_train)

    best_pipeline = grid_search.best_estimator_
    print(f"   🌟 Best Hyperparameters: {grid_search.best_params_}")
    print(f"   🌟 Best CV F1-Weighted Score: {grid_search.best_score_:.4f}")

    # 8. Final Evaluation on Holdout Test Set
    y_test_pred = best_pipeline.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(y_test, y_test_pred, average='macro', zero_division=0)
    prec_weighted, rec_weighted, f1_weighted, _ = precision_recall_fscore_support(y_test, y_test_pred, average='weighted', zero_division=0)

    cm = confusion_matrix(y_test, y_test_pred, labels=classes)
    report_dict = classification_report(y_test, y_test_pred, labels=classes, output_dict=True, zero_division=0)
    report_text = classification_report(y_test, y_test_pred, labels=classes, zero_division=0)

    print("\n" + "=" * 70)
    print("🏆 FINAL MODEL PERFORMANCE EVALUATION (TEST SET)")
    print("=" * 70)
    print(report_text)
    print(f"Overall Accuracy: {test_accuracy:.4f}")

    # Save Confusion Matrix figure
    plot_confusion_matrix(cm, classes, "outputs/figures/confusion_matrix.png")

    # 9. Save Best Model
    os.makedirs("outputs/models", exist_ok=True)
    model_save_path = "outputs/models/best_sentiment_model.joblib"
    joblib.dump(best_pipeline, model_save_path)
    print(f"\n💾 Saved best sentiment pipeline to: {model_save_path}")

    # 10. Save Detailed Summary JSON
    summary_data = {
        'best_model': 'LogisticRegression (Tuned + TF-IDF)',
        'best_params': {str(k): (str(v) if not isinstance(v, (int, float, bool)) else v) for k, v in grid_search.best_params_.items()},
        'cv_best_f1_weighted': round(float(grid_search.best_score_), 4),
        'test_metrics': {
            'accuracy': round(float(test_accuracy), 4),
            'f1_weighted': round(float(f1_weighted), 4),
            'precision_weighted': round(float(prec_weighted), 4),
            'recall_weighted': round(float(rec_weighted), 4),
            'f1_macro': round(float(f1_macro), 4),
            'precision_macro': round(float(prec_macro), 4),
            'recall_macro': round(float(rec_macro), 4),
        },
        'per_class_metrics': {
            cls: {
                'precision': round(float(report_dict[cls]['precision']), 4),
                'recall': round(float(report_dict[cls]['recall']), 4),
                'f1_score': round(float(report_dict[cls]['f1-score']), 4),
                'support': int(report_dict[cls]['support'])
            } for cls in classes
        },
        'confusion_matrix': {
            'classes': classes,
            'matrix': cm.tolist()
        },
        'dataset_info': label_metadata
    }

    metrics_json_path = "outputs/metrics/evaluation_summary.json"
    with open(metrics_json_path, 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, indent=4, ensure_ascii=False)
    print(f"📝 Saved comprehensive evaluation summary to: {metrics_json_path}")

    print("\n✅ Pipeline execution completed successfully!")
    return best_pipeline, summary_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Sentiment Analysis Pipeline")
    parser.add_argument("--data_path", type=str, default="data/3) Sentiment dataset.csv")
    args = parser.parse_args()
    train_pipeline(data_path=args.data_path)
