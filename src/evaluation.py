# scripts/evaluation.py
# -----------------------------
# Evaluation utilities for cardiovascular risk prediction.
# Includes ROC/PR metrics, bootstrap confidence intervals,
# reliability curves, threshold selection and subgroup fairness.

from typing import Dict, Tuple, List
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
    brier_score_loss,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)
from sklearn.calibration import calibration_curve

# ---------------------------------------
# 1. Basic metrics
# ---------------------------------------
def compute_basic_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray
) -> Dict:
    """
    Compute ROC-AUC and PR-AUC.

    Parameters
    ----------
    y_true : np.ndarray
        True labels.
    y_prob : np.ndarray
        Predicted probabilities.

    Returns
    -------
    Dict
        ROC-AUC and PR-AUC scores.
    """
    return {
        "roc_auc": roc_auc_score(y_true, y_prob),
        "pr_auc": average_precision_score(y_true, y_prob)
    }

# ---------------------------------------
# 2. Bootstrap confidence intervals
# ---------------------------------------
def compute_bootstrap_intervals(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_iterations: int = 1000,
    seed: int = 42
) -> Dict:
    """
    Compute bootstrap confidence intervals for ROC-AUC and PR-AUC.

    Parameters
    ----------
    y_true : np.ndarray
        True labels.
    y_prob : np.ndarray
        Predicted probabilities.
    n_iterations : int
        Number of bootstrap samples.
    seed : int
        Random seed.

    Returns
    -------
    Dict
        Mean and 95% CI for ROC-AUC and PR-AUC.
    """
    rng = np.random.default_rng(seed)

    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    roc_scores = []
    pr_scores = []

    for _ in range(n_iterations):
        idx = rng.choice(len(y_true), size=len(y_true), replace=True)
        y_t = y_true[idx]
        y_p = y_prob[idx]
        roc_scores.append(roc_auc_score(y_t, y_p))
        pr_scores.append(average_precision_score(y_t, y_p))

    return {
        "roc_auc_mean": float(np.mean(roc_scores)),
        "roc_auc_ci": (
            float(np.percentile(roc_scores, 2.5)),
            float(np.percentile(roc_scores, 97.5))
        ),
        "pr_auc_mean": float(np.mean(pr_scores)),
        "pr_auc_ci": (
            float(np.percentile(pr_scores, 2.5)),
            float(np.percentile(pr_scores, 97.5))
        )
    }

# ---------------------------------------
# 3. Reliability and Brier score
# ---------------------------------------
def compute_reliability(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10
) -> Dict:
    """
    Compute reliability curve and Brier score.

    Parameters
    ----------
    y_true : np.ndarray
        True labels.
    y_prob : np.ndarray
        Predicted probabilities.
    n_bins : int
        Number of bins for calibration curve.

    Returns
    -------
    Dict
        Reliability curve data and Brier score.
    """
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins)
    brier = brier_score_loss(y_true, y_prob)

    return {
        "prob_true": prob_true,
        "prob_pred": prob_pred,
        "brier_score": brier
    }

# ---------------------------------------
# 4. Threshold evaluation
# ---------------------------------------
def evaluate_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float
) -> Dict:
    """
    Evaluate classification metrics at a given threshold.

    Parameters
    ----------
    y_true : np.ndarray
        True labels.
    y_prob : np.ndarray
        Predicted probabilities.
    threshold : float
        Decision threshold.

    Returns
    -------
    Dict
        Confusion matrix metrics and derived statistics.
    """
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    sens = recall_score(y_true, y_pred)
    spec = tn / (tn + fp)
    ppv = precision_score(y_true, y_pred)
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    f1 = f1_score(y_true, y_pred)

    return {
        "threshold": threshold,
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
        "sensitivity": sens,
        "specificity": spec,
        "PPV": ppv,
        "NPV": npv,
        "F1": f1
    }

# ---------------------------------------
# 5. Threshold selection (Youden, cost, top-k)
# ---------------------------------------
def select_thresholds(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    cost_ratio: float = 5.0,
    top_k_fraction: float = 0.2
) -> Dict:
    """
    Compute optimal thresholds using Youden J, cost-based, and top-k criteria.

    Parameters
    ----------
    y_true : np.ndarray
        True labels.
    y_prob : np.ndarray
        Predicted probabilities.
    cost_ratio : float
        Cost ratio FN:FP for cost-based threshold.
    top_k_fraction : float
        Fraction of highest-risk patients for top-k threshold.

    Returns
    -------
    Dict
        Thresholds for Youden, cost-based, and top-k.
    """
    thresholds = np.linspace(0.01, 0.99, 100)

    youden_scores = []
    cost_scores = []

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        sens = recall_score(y_true, y_pred)
        spec = tn / (tn + fp)
        youden_scores.append((t, sens + spec - 1))

        cost = cost_ratio * fn + fp
        cost_scores.append((t, cost))

    youden_thr = max(youden_scores, key=lambda x: x[1])[0]
    cost_thr = min(cost_scores, key=lambda x: x[1])[0]

    k = int(top_k_fraction * len(y_prob))
    topk_thr = float(np.sort(y_prob)[-k])

    return {
        "youden": youden_thr,
        "cost_based": cost_thr,
        "top_k": topk_thr
    }

# ---------------------------------------
# 6. Subgroup fairness evaluation
# ---------------------------------------
def evaluate_subgroups(
    df: pd.DataFrame,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    group_cols: List[str]
) -> pd.DataFrame:
    """
    Evaluate classification metrics across demographic or clinical subgroups.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing subgroup columns.
    y_true : np.ndarray
        True labels.
    y_prob : np.ndarray
        Predicted probabilities.
    threshold : float
        Decision threshold.
    group_cols : List[str]
        Columns defining subgroups.

    Returns
    -------
    pd.DataFrame
        Metrics per subgroup.
    """
    df_eval = df.copy()
    df_eval["y_true"] = y_true
    df_eval["y_pred"] = (y_prob >= threshold).astype(int)

    results = []

    for col in group_cols:
        for val in df_eval[col].unique():
            mask = df_eval[col] == val
            if mask.sum() == 0:
                continue

            y_t = df_eval.loc[mask, "y_true"]
            y_p = df_eval.loc[mask, "y_pred"]

            tn, fp, fn, tp = confusion_matrix(y_t, y_p).ravel()

            sens = tp / (tp + fn) if (tp + fn) > 0 else 0
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0
            ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
            npv = tn / (tn + fn) if (tn + fn) > 0 else 0

            results.append({
                "subgroup": col,
                "value": val,
                "n": int(mask.sum()),
                "TP": int(tp),
                "FP": int(fp),
                "FN": int(fn),
                "TN": int(tn),
                "sensitivity": float(sens),
                "specificity": float(spec),
                "PPV": float(ppv),
                "NPV": float(npv)
            })

    return pd.DataFrame(results)
