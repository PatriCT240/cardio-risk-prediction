"""
Data preparation module for cardiovascular risk prediction.

This module loads the raw dataset, applies strict and standard clinical
cleaning rules, performs feature engineering, injects missingness and
Gaussian noise for robustness experiments, applies winsorization, and
produces train/test splits along with traceability metadata.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict


def load_raw_data(path: str) -> pd.DataFrame:
    """
    Load the raw cardiovascular dataset.

    Parameters
    ----------
    path : str
        Path to the raw CSV file.

    Returns
    -------
    pd.DataFrame
        Loaded dataset.
    """
    df = pd.read_csv(path, sep=";")
    return df.copy()


def apply_strict_clinical_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply strict clinical rules to remove physiologically impossible records.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataset.

    Returns
    -------
    pd.DataFrame
        Dataset after strict clinical cleaning.
    """
    df = df.copy()
    df["flag_ap_incoherent"] = (df["ap_lo"] >= df["ap_hi"]).astype(int)
    df = df[df["flag_ap_incoherent"] == 0].copy()
    return df


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create engineered features including BMI, age in years, age bands,
    hypertension flags, lifestyle risk flags, and clinical risk flags.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset after strict cleaning.

    Returns
    -------
    pd.DataFrame
        Dataset with engineered features.
    """
    df = df.copy()
    df["age_years"] = (df["age"] / 365.25).round(1)
    df["BMI"] = (df["weight"] / ((df["height"] / 100) ** 2)).round(2)

    age_bins = [0, 40, 50, 60, 70, np.inf]
    age_labels = ["[0,40)", "[40,50)", "[50,60)", "[60,70)", "[70,+)"]
    df["age_band"] = pd.cut(df["age_years"], bins=age_bins, labels=age_labels, right=False)

    df["ap_hi_cat"] = df["ap_hi"].apply(lambda x: "normal" if x < 120 else ("elevated" if x < 130 else "HTA"))
    df["ap_lo_cat"] = df["ap_lo"].apply(lambda x: "normal" if x < 80 else "HTA")
    df["hypertension_flag"] = ((df["ap_hi_cat"] == "HTA") | (df["ap_lo_cat"] == "HTA")).astype(int)

    df["clinical_risk_flag"] = (
        (df["ap_hi"] >= 140) |
        (df["ap_lo"] >= 90) |
        (df["BMI"] >= 30) |
        (df["age_years"] >= 60)
    ).astype(int)

    df["lifestyle_risk_flag"] = (
        (df["smoke"] == 1) |
        (df["alco"] == 1) |
        (df["active"] == 0)
    ).astype(int)

    return df


def apply_standard_winsorization(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply clinical winsorization to physiological variables and create
    was_capped flags for traceability.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset with engineered features.

    Returns
    -------
    pd.DataFrame
        Winsorized dataset with was_capped flags.
    """
    df = df.copy()
    limits = {
        "height": (140, 200),
        "weight": (40, 180),
        "ap_hi": (90, 200),
        "ap_lo": (60, 120),
        "BMI": (10, 80)
    }

    for col, (low, high) in limits.items():
        flag_col = f"was_capped_{col}"
        df[flag_col] = (~df[col].between(low, high)).astype(int)
        df[col] = df[col].clip(lower=low, upper=high)

    return df


def inject_missingness_and_noise(df: pd.DataFrame, seed: int = 42) -> Tuple[pd.DataFrame, Dict]:
    """
    Inject missing values and Gaussian noise into selected variables for
    robustness evaluation.

    Parameters
    ----------
    df : pd.DataFrame
        Winsorized dataset.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    Tuple[pd.DataFrame, Dict]
        Augmented dataset and traceability dictionary.
    """
    rng = np.random.default_rng(seed)
    df_aug = df.copy()
    masks = {}

    missing_cols = [c for c in ["ap_hi", "ap_lo", "BMI", "age_years"] if c in df_aug.columns]
    for col in missing_cols:
        mask = rng.random(len(df_aug)) < 0.10
        df_aug.loc[mask, col] = np.nan
        masks[col] = mask

    noise_cols = [c for c in ["ap_hi", "ap_lo", "weight"] if c in df_aug.columns]
    for col in noise_cols:
        sigma = 0.05 * float(np.nanstd(df_aug[col].values))
        df_aug[col] = df_aug[col] + rng.normal(0, sigma, size=len(df_aug))

    limits = {
        "ap_hi": (90, 200),
        "ap_lo": (60, 120),
        "BMI": (15, 60),
        "weight": (40, 180)
    }

    for col, (low, high) in limits.items():
        if col in df_aug.columns:
            df_aug[col] = df_aug[col].clip(lower=low, upper=high)

    trace = {
        "missingness_masks": {col: int(mask.sum()) for col, mask in masks.items()},
        "noise_sigma": {col: round(0.05 * df_aug[col].std(skipna=True), 3) for col in noise_cols}
    }

    return df_aug, trace
