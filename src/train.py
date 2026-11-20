"""
train.py
--------
Trains, tunes, and compares machine learning models for financial fraud detection.

Models Evaluated:
  1. Baseline: Logistic Regression (class_weight='balanced')
  2. Candidate 1: Random Forest Classifier (class_weight='balanced_subsample')
  3. Candidate 2: HistGradientBoosting Classifier (class_weight='balanced')

Outputs:
  - Best full pipeline saved to models/fraud_model.joblib
  - All candidate models saved to models/ for comprehensive evaluation.
  - Training metrics summary logged to console and saved to disk.
"""

import os
import sys

# Ensure repository root is in sys.path
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    classification_report,
    f1_score,
    recall_score,
    precision_score
)
from src.preprocessing import create_preprocessing_pipeline, load_and_prepare_data


def train_models(
    sample_size: int = 50000,
    test_size: float = 0.2,
    random_state: int = 42,
    output_dir: str = "models"
):
    """Train baseline and ensemble models, compare on validation set, and persist best model."""
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)

    print("=" * 70)
    print("STEP 1: INGESTING DATA & GENERATING STRATIFIED TRAIN/TEST SPLIT")
    print("=" * 70)
    X_train, X_test, y_train, y_test = load_and_prepare_data(
        data_path="data/raw/fraudTest.csv",
        sample_size=sample_size,
        test_size=test_size,
        random_state=random_state
    )

    # Persist the held-out test set for evaluation in evaluate.py
    test_data = X_test.copy()
    test_data['is_fraud'] = y_test
    test_data.to_csv("data/processed/test_set.csv", index=False)
    print("Saved held-out test data to: data/processed/test_set.csv")

    # Fit preprocessing pipeline strictly on training data
    print("\n" + "=" * 70)
    print("STEP 2: FITTING PREPROCESSING PIPELINE STRICTLY ON X_TRAIN")
    print("=" * 70)
    preprocessor = create_preprocessing_pipeline()
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)
    print(f"Feature matrix dimensions: Train {X_train_transformed.shape}, Test {X_test_transformed.shape}")

    # Model definitions with hyperparameter configurations
    models = {
        "Logistic_Regression_Baseline": LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            C=1.0,
            solver="lbfgs",
            random_state=random_state
        ),
        "Random_Forest": RandomForestClassifier(
            class_weight="balanced_subsample",
            n_estimators=100,
            max_depth=12,
            min_samples_leaf=5,
            n_jobs=-1,
            random_state=random_state
        ),
        "Hist_Gradient_Boosting": HistGradientBoostingClassifier(
            class_weight="balanced",
            max_iter=120,
            learning_rate=0.08,
            max_depth=8,
            min_samples_leaf=20,
            random_state=random_state
        )
    }

    results = []
    trained_pipelines = {}

    print("\n" + "=" * 70)
    print("STEP 3: MODEL TRAINING & VALIDATION COMPARISON")
    print("=" * 70)

    for name, clf in models.items():
        print(f"\n--- Training {name} ---")
        clf.fit(X_train_transformed, y_train)

        # Probabilities for positive class (fraud)
        y_train_prob = clf.predict_proba(X_train_transformed)[:, 1]
        y_test_prob = clf.predict_proba(X_test_transformed)[:, 1]

        # Default 0.5 binary predictions for initial check
        y_test_pred = (y_test_prob >= 0.5).astype(int)

        # Evaluation metrics
        train_pr_auc = average_precision_score(y_train, y_train_prob)
        test_pr_auc = average_precision_score(y_test, y_test_prob)
        test_roc_auc = roc_auc_score(y_test, y_test_prob)
        test_recall = recall_score(y_test, y_test_pred, zero_division=0)
        test_precision = precision_score(y_test, y_test_pred, zero_division=0)
        test_f1 = f1_score(y_test, y_test_pred, zero_division=0)

        print(f"  Test PR-AUC (Average Precision): {test_pr_auc:.4f}")
        print(f"  Test ROC-AUC:                    {test_roc_auc:.4f}")
        print(f"  Test Recall (threshold 0.5):     {test_recall:.4f}")
        print(f"  Test Precision (threshold 0.5):  {test_precision:.4f}")
        print(f"  Test F1-Score:                   {test_f1:.4f}")

        results.append({
            "model": name,
            "train_pr_auc": train_pr_auc,
            "test_pr_auc": test_pr_auc,
            "test_roc_auc": test_roc_auc,
            "recall_at_0.5": test_recall,
            "precision_at_0.5": test_precision,
            "f1_at_0.5": test_f1
        })

        # Build full standalone pipeline (Raw Input -> Features -> Scaling -> Classifier)
        full_pipe = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", clf)
        ])
        pipeline_path = os.path.join(output_dir, f"{name.lower()}_pipeline.joblib")
        joblib.dump(full_pipe, pipeline_path)
        trained_pipelines[name] = full_pipe

    results_df = pd.DataFrame(results).sort_values(by="test_pr_auc", ascending=False)
    results_path = os.path.join(output_dir, "model_comparison_results.csv")
    results_df.to_csv(results_path, index=False)

    print("\n" + "=" * 70)
    print("STEP 4: MODEL COMPARISON SUMMARY TABLE")
    print("=" * 70)
    print(results_df.to_string(index=False))

    # Best model selection based on highest PR-AUC
    best_model_name = results_df.iloc[0]["model"]
    best_pipeline = trained_pipelines[best_model_name]
    best_model_path = os.path.join(output_dir, "fraud_model.joblib")
    joblib.dump(best_pipeline, best_model_path)

    print("\n" + "=" * 70)
    print(f"SELECTION DECISION: BEST MODEL IS [{best_model_name}]")
    print(f"Selected based on highest PR-AUC: {results_df.iloc[0]['test_pr_auc']:.4f}")
    print(f"Production pipeline successfully serialized to: {best_model_path}")
    print("=" * 70)

    return results_df, best_model_name


if __name__ == "__main__":
    train_models(sample_size=50000)
