# scripts/config.py
# -----------------------------
# Configuration for cardiovascular risk prediction project.
# Variable groups for EDA, preprocessing and modelling.

# ---------------------------------------
# 1. Numeric variables
# ---------------------------------------
NUMERIC_FOR_HIST = ["age", "ap_hi", "ap_lo"]
NUMERIC_FOR_BOXPLOT = ["age", "ap_hi", "ap_lo"]

# ---------------------------------------
# 2. Categorical variables
# ---------------------------------------
CATEGORICAL_VARS = [
    "gender",
    "cholesterol",
    "gluc",
    "smoke",
    "alco",
    "active"
]

# ---------------------------------------
# 3. Target variable
# ---------------------------------------
TARGET = "cardio"

# ---------------------------------------
# 4. Human‑readable labels for categories
# ---------------------------------------
CATEGORY_LABELS = {
    "gender": {
        1: "male",
        2: "female"
    },
    "cholesterol": {
        1: "normal",
        2: "above normal",
        3: "well above normal"
    },
    "gluc": {
        1: "normal",
        2: "above normal",
        3: "well above normal"
    },
    "smoke": {
        0: "no",
        1: "yes"
    },
    "alco": {
        0: "no",
        1: "yes"
    },
    "active": {
        0: "no",
        1: "yes"
    }
}

# ---------------------------------------
# 5. Reproducibility and CV settings
# ---------------------------------------
RANDOM_STATE = 42
N_SPLITS = 5
