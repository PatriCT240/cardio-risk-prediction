"""
Config for cardiovascular risk prediction.

Variables configuration for EDA and modelling
"""

NUMERIC_FOR_HIST = ["age", "ap_hi", "ap_lo"]
NUMERIC_FOR_BOXPLOT = ["age", "ap_hi", "ap_lo"]

CATEGORICAL_VARS = ["gender", "cholesterol", "gluc", "smoke", "alco", "active"]

TARGET = "cardio"

CATEGORY_LABELS = {
    "gender": {1: "male", 2: "female"},
    "cholesterol": {1: "normal", 2: "above normal", 3: "well above normal"},
    "gluc": {1: "normal", 2: "above normal", 3: "well above normal"},
    "smoke": {0: "no", 1: "yes"},
    "alco": {0: "no", 1: "yes"},
    "active": {0: "no", 1: "yes"}
}

RANDOM_STATE = 42

N_SPLITS = 5
