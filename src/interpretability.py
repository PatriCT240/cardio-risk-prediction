"""
Interpretability module for cardiovascular risk prediction.

This module implements permutation importance, partial dependence
functions, univariate ALE, and bivariate ALE interaction effects.
"""

from typing import List, Tuple, Dict
import numpy as np
import pandas as pd
import shap
from sklearn.inspection import permutation_importance

def compute_permutation_importance(
    pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    n_repeats: int = 10,
    seed: int = 42
) -> pd.DataFrame:
    """
    Compute permutation importance for a trained model.

    Parameters
    ----------
    pipeline : Pipeline
        Trained model pipeline.
    X : pd.DataFrame
        Input features.
    y : pd.Series
        True labels.
    n_repeats : int
        Number of permutation repeats.
    seed : int
        Random seed.

    Returns
    -------
    pd.DataFrame
        Sorted permutation importance table.
    """
    model = pipeline.named_steps["clf"]
    X_transformed = pipeline.named_steps["prep"].transform(X)

    pi = permutation_importance(
        model,
        X_transformed,
        y,
        n_repeats=n_repeats,
        random_state=seed,
        scoring="roc_auc"
    )

    features = pipeline.named_steps["prep"].get_feature_names_out()

    df_pi = pd.DataFrame({
        "feature": features,
        "importance_mean": pi.importances_mean,
        "importance_std": pi.importances_std
    }).sort_values(by="importance_mean", ascending=False).reset_index(drop=True)

    return df_pi


def compute_pdp(
    pipeline,
    X: pd.DataFrame,
    feature: str,
    grid_size: int = 50
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Partial Dependence for a single feature.

    Parameters
    ----------
    pipeline : Pipeline
        Trained model pipeline.
    X : pd.DataFrame
        Input features.
    feature : str
        Feature name.
    grid_size : int
        Number of grid points.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Grid values and mean predicted probabilities.
    """
    Xc = X.copy()
    grid = np.linspace(Xc[feature].min(), Xc[feature].max(), grid_size)
    means = []

    for val in grid:
        Xc[feature] = val
        prob = pipeline.predict_proba(Xc)[:, 1].mean()
        means.append(prob)

    return grid, np.array(means)

def compute_shap_values(pipeline, X):
    """
    Compute SHAP values using the underlying tree model.
    CalibratedClassifierCV is not supported by SHAP, so we use the base estimator.
    """
    preprocessor = pipeline.named_steps["prep"]
    calibrated = pipeline.named_steps["clf"]

    model = calibrated.estimator

    X_trans = preprocessor.transform(X)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_trans)

    feature_names = preprocessor.get_feature_names_out()

    return shap_values, feature_names

def compute_shap_interactions(pipeline, X):
    """
    Compute SHAP interaction values using the underlying tree model.
    """
    preprocessor = pipeline.named_steps["prep"]
    calibrated = pipeline.named_steps["clf"]

    model = calibrated.estimator

    X_trans = preprocessor.transform(X)

    explainer = shap.TreeExplainer(model)
    interaction_values = explainer.shap_interaction_values(X_trans)

    feature_names = preprocessor.get_feature_names_out()

    return interaction_values, feature_names
