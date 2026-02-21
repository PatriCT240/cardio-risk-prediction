"""
Preprocessing module for cardiovascular risk prediction.

This module builds preprocessing pipelines for numerical, ordinal,
binary, and engineered categorical features. It supports both the
standard preprocessing pipeline and the advanced pipelines using
median imputation or KNN imputation.
"""
import pandas as pd
from typing import List
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer, KNNImputer

def build_train_test_split(df, random_state):
    X = df.drop("cardio", axis=1)
    y = df["cardio"]
    return train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=random_state
    )


def get_feature_groups() -> dict:
    """
    Return the feature groups used across preprocessing pipelines.

    Returns
    -------
    dict
        Dictionary containing lists of numerical, ordinal, binary,
        and engineered flag features.
    """
    return {
        "numerical": ["age_years", "height", "weight", "BMI", "ap_hi", "ap_lo"],
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


def build_standard_preprocessor() -> ColumnTransformer:
    """
    Build the standard preprocessing pipeline using median imputation
    for numerical features, ordinal encoding for ordinal features,
    and one-hot encoding for binary and flag features.

    Returns
    -------
    ColumnTransformer
        Preprocessing transformer for the standard pipeline.
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


def build_advanced_preprocessor_median() -> ColumnTransformer:
    """
    Build the advanced preprocessing pipeline using median imputation
    for numerical features and one-hot encoding for all categorical
    and flag features.

    Returns
    -------
    ColumnTransformer
        Preprocessing transformer using median imputation.
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


def build_advanced_preprocessor_knn() -> ColumnTransformer:
    """
    Build the advanced preprocessing pipeline using KNN imputation
    for numerical features and one-hot encoding for all categorical
    and flag features.

    Returns
    -------
    ColumnTransformer
        Preprocessing transformer using KNN imputation.
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
