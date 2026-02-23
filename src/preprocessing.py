# scripts/preprocessing.py
# -----------------------------
# Preprocessing utilities for cardiovascular risk prediction.
# Includes train/test split, feature grouping, and multiple
# preprocessing pipelines (standard, median-based, KNN-based).

import pandas as pd
from typing import List, Dict
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer, KNNImputer

# ---------------------------------------
# 1. Train/test split
# ---------------------------------------
def build_train_test_split(df: pd.DataFrame, random_state: int):
    """
    Split dataset into train and test sets (stratified).
    """
    X = df.drop("cardio", axis=1)
    y = df["cardio"]

    return train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=random_state
    )

# ---------------------------------------
# 2. Feature groups
# ---------------------------------------
def get_feature_groups() -> Dict[str, List[str]]:
    """
    Return the feature groups used across preprocessing pipelines.
    """
    return {
        "numerical": [
            "age_years", "height", "weight", "BMI",
            "ap_hi", "ap_lo"
        ],
        "ordinal": ["cholesterol", "gluc"],
        "binary": ["gender", "smoke", "alco", "active"],
        "flags": [
            "flag_ap_incoherent",
            "hypertension_flag",
            "clinical_risk_flag",
            "lifestyle_risk_flag",
            "was_capped_height",
            "was_capped_weight",
            "was_capped_ap_hi",
            "was_capped_ap_lo",
            "was_capped_BMI"
        ]
    }

# ---------------------------------------
# 3. Standard preprocessing pipeline
# ---------------------------------------
def build_standard_preprocessor() -> ColumnTransformer:
    """
    Standard preprocessing:
    - Median imputation + scaling for numerical features
    - Ordinal encoding for cholesterol/glucose
    - One-hot encoding for binary + flag features
    """
    groups = get_feature_groups()

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    ord_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(categories=[[1, 2, 3], [1, 2, 3]]))
    ])

    bin_flag_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num", num_pipe, groups["numerical"]),
        ("ord", ord_pipe, groups["ordinal"]),
        ("bin_flag", bin_flag_pipe, groups["binary"] + groups["flags"])
    ])

    return preprocessor

# ---------------------------------------
# 4. Advanced preprocessing (median imputation)
# ---------------------------------------
def build_advanced_preprocessor_median() -> ColumnTransformer:
    """
    Advanced preprocessing using:
    - Median imputation + sparse scaling for numerical features
    - One-hot encoding for all categorical + flag features
    """
    groups = get_feature_groups()

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler(with_mean=False))
    ])

    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num", num_pipe, groups["numerical"]),
        ("cat", cat_pipe, groups["ordinal"] + groups["binary"] + groups["flags"])
    ])

    return preprocessor

# ---------------------------------------
# 5. Advanced preprocessing (KNN imputation)
# ---------------------------------------
def build_advanced_preprocessor_knn() -> ColumnTransformer:
    """
    Advanced preprocessing using:
    - KNN imputation + sparse scaling for numerical features
    - One-hot encoding for all categorical + flag features
    """
    groups = get_feature_groups()

    num_pipe = Pipeline([
        ("imputer", KNNImputer(n_neighbors=5, weights="distance")),
        ("scaler", StandardScaler(with_mean=False))
    ])

    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num", num_pipe, groups["numerical"]),
        ("cat", cat_pipe, groups["ordinal"] + groups["binary"] + groups["flags"])
    ])

    return preprocessor
