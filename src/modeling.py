# scripts/modeling.py
# -----------------------------
# Model training utilities for cardiovascular risk prediction.
# Includes Logistic Regression, Random Forest, HistGradientBoosting,
# cross-validation, model comparison, calibration and test evaluation.

import pandas as pd
import numpy as np
from typing import Dict, Tuple

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss
)

# ---------------------------------------
# 1. Logistic Regression
# ---------------------------------------
def train_logistic_regression(
    preprocessor,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int,
) -> Tuple[Pipeline, Dict]:
    """
    Train a Logistic Regression model with light hyperparameter tuning.
    """
    pipe = Pipeline([
        ("prep", preprocessor),
        ("clf", LogisticRegression(max_iter=1000, penalty="l2", random_state=random_state)),
    ])

    param_grid = {"clf__C": [0.1, 1, 10]}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    grid = GridSearchCV(pipe, param_grid, cv=cv, scoring="roc_auc")
    grid.fit(X_train, y_train)

    return grid.best_estimator_, grid.best_params_

# ---------------------------------------
# 2. Random Forest
# ---------------------------------------
def train_random_forest(
    preprocessor,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int,
) -> Tuple[Pipeline, Dict]:
    """
    Train a Random Forest classifier with fixed parsimonious hyperparameters.
    """
    model = RandomForestClassifier(
        n_estimators=400,
        max_depth=10,
        min_samples_leaf=5,
        random_state=random_state,
    )

    pipe = Pipeline([
        ("prep", preprocessor),
        ("clf", model),
    ])

    pipe.fit(X_train, y_train)

    return pipe, {
        "n_estimators": 400,
        "max_depth": 10,
        "min_samples_leaf": 5,
    }

# ---------------------------------------
# 3. HistGradientBoosting
# ---------------------------------------
def train_histgradientboosting(
    preprocessor,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int,
) -> Tuple[Pipeline, Dict]:
    """
    Train a HistGradientBoosting model with early stopping and regularization.
    """
    model = HistGradientBoostingClassifier(
        max_depth=3,
        learning_rate=0.1,
        max_iter=300,
        early_stopping=True,
        validation_fraction=0.15,
        min_samples_leaf=20,
        l2_regularization=0.0,
        random_state=random_state,
    )

    pipe = Pipeline([
        ("prep", preprocessor),
        ("clf", model),
    ])

    pipe.fit(X_train, y_train)

    return pipe, {
        "max_depth": 3,
        "learning_rate": 0.1,
        "max_iter": 300,
        "min_samples_leaf": 20,
    }

# ---------------------------------------
# 4. Cross-validation
# ---------------------------------------
def cross_validate_model(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int,
    n_splits: int,
) -> Dict:
    """
    Perform cross-validation for ROC-AUC and PR-AUC.
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    roc_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="roc_auc")
    pr_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="average_precision")

    return {
        "roc_auc_mean": roc_scores.mean(),
        "roc_auc_std": roc_scores.std(),
        "pr_auc_mean": pr_scores.mean(),
        "pr_auc_std": pr_scores.std(),
    }

# ---------------------------------------
# 5. Model comparison
# ---------------------------------------
def compare_models(
    pipelines: Dict[str, Pipeline],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int,
    n_splits: int,
) -> pd.DataFrame:
    """
    Compare multiple models using cross-validation.
    """
    results = []

    for name, pipe in pipelines.items():
        metrics = cross_validate_model(
            pipe,
            X_train,
            y_train,
            random_state=random_state,
            n_splits=n_splits,
        )
        model_name, imput_name = name.split("+")
        results.append({
            "model": model_name,
            "imputation": imput_name,
            **metrics,
        })

    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values(by="roc_auc_mean", ascending=False)

    return df_results

# ---------------------------------------
# 6. Calibration
# ---------------------------------------
def calibrate_model(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    method: str = "isotonic",
) -> Pipeline:
    """
    Apply probability calibration to a trained model.
    """
    model = pipeline.named_steps["clf"]
    preprocessor = pipeline.named_steps["prep"]

    calibrated = CalibratedClassifierCV(model, method=method, cv=5)

    pipe = Pipeline([
        ("prep", preprocessor),
        ("clf", calibrated),
    ])

    pipe.fit(X_train, y_train)

    return pipe

# ---------------------------------------
# 7. Test evaluation
# ---------------------------------------
def evaluate_on_test(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Dict:
    """
    Evaluate a trained model on the test set.
    """
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    return {
        "roc_auc": roc_auc_score(y_test, y_prob),
        "pr_auc": average_precision_score(y_test, y_prob),
    }

# ---------------------------------------
# 8. Brier score
# ---------------------------------------
def compute_brier_score(y_true, y_prob):
    return brier_score_loss(y_true, y_prob)

# ---------------------------------------
# 9. Expected Calibration Error (ECE)
# ---------------------------------------
def compute_ece(y_true, y_prob, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    bin_ids = np.digitize(y_prob, bins) - 1

    ece = 0.0
    for i in range(n_bins):
        mask = bin_ids == i
        if np.sum(mask) == 0:
            continue
        bin_conf = np.mean(y_prob[mask])
        bin_acc = np.mean(y_true[mask])
        ece += (np.sum(mask) / len(y_prob)) * np.abs(bin_conf - bin_acc)

    return ece

# ---------------------------------------
# 10. Reliability curve
# ---------------------------------------
def compute_reliability_curve(y_true, y_prob, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    bin_ids = np.digitize(y_prob, bins) - 1

    accuracies = []
    confidences = []

    for i in range(n_bins):
        mask = bin_ids == i
        if np.sum(mask) == 0:
            accuracies.append(np.nan)
            confidences.append(np.nan)
            continue
        accuracies.append(np.mean(y_true[mask]))
        confidences.append(np.mean(y_prob[mask]))

    return np.array(confidences), np.array(accuracies), bins
