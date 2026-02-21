# Cardiovascular Risk Prediction – End‑to‑End Machine Learning Pipeline

This repository contains a full, production‑grade machine learning system for predicting **10‑year cardiovascular disease (CVD) risk** using structured clinical data.  
The project integrates **clinical data cleaning**, **feature engineering**, **robust preprocessing**, **multiple model families**, **probability calibration**, **threshold optimization**, **interpretability**, and **deployment utilities**.

The codebase is written in **Python**, follows a **modular senior‑level architecture**, and includes **enterprise‑style docstrings** with no inline comments.

---

## 🚀 Quick Start

Train, evaluate, and generate predictions with a single command sequence:

# 1. Prepare data
python -m src.data_prep

# 2. Train models (standard + advanced)
python -m src.modeling

# 3. Evaluate performance + calibration
python -m src.evaluation

# 4. Generate predictions for new patients
python -m src.predict --input sample.json

---

## 🔍 Project Overview

The goal is to build a clinically meaningful and statistically robust model capable of estimating the probability of cardiovascular disease based on:

- Demographics  
- Anthropometrics  
- Blood pressure  
- Laboratory markers  
- Lifestyle factors  
- Derived clinical flags  

The project includes:
- A **standard pipeline**
- An **advanced robustness pipeline** with injected missingness and Gaussian noise
- A **calibrated final model** ready for deployment

---

## 📁 Repository Structure

cardio-risk-prediction/
│
├── data/
│   ├── raw/                     # Original dataset (cardio_train.csv)
│   └── processed/               # Cleaned, engineered, and split datasets
│
├── notebooks/
│   ├── 01_exploratory_analysis.ipynb       # EDA, distributions, clinical cleaning rules
│   ├── 02_standard_pipeline.ipynb          # Baseline preprocessing + LR/RF training
│   ├── 03_advanced_pipeline.ipynb          # Robust pipeline + HGB training + calibration
│   ├── 04_thresholds_calibration.ipynb     # Threshold optimization (Youden, cost-based, top‑k)
│   └── 05_model_interpretability.ipynb     # PI, PDP, ALE, interactions, SHAP-style analysis
│
├── src/
│   ├── data_prep.py              # Clinical cleaning, feature engineering, noise/missingness injection
│   ├── preprocessing.py          # Preprocessing pipelines (standard + advanced)
│   ├── modeling.py               # Model training, CV, calibration, model selection
│   ├── evaluation.py             # Metrics, bootstrapping, calibration, thresholds, fairness
│   ├── interpretability.py       # Permutation importance, subgroup analyses, PDP, SHAP, interactions
│   ├── visualization.py          # Plotting utilities (ROC, PR, calibration, SHAP, etc.)
│   └── config.py                 # Global configuration (paths, seeds, feature groups)
│
├── models/
│   └── final_pipeline.joblib     # Final calibrated production model (HGB + robustness)
│
├── model_card/
│   └── model_card.md             # Full clinical + technical documentation of the model
│
├── reports/
│   ├── tables/                   # Exported evaluation tables
│   ├── figures/                  # Generated plots (ROC, PR, calibration, ALE, etc.)
│   └── executive_summary.pdf     # High-level summary for stakeholders
│
├── README.md                     # Project overview, installation, usage, structure
└── requirements.txt              # Python dependencies

---

## ⚙️ Installation

git clone https://github.com/PatriCT240/cardio-risk-prediction.git
cd cardio-risk-prediction

python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows

pip install -r requirements.txt

---

## 📦 Data

The project uses the **CardioVascular Disease dataset** (70,000 patients).
Place the raw file here:
data/raw/cardio_train.csv

---

## 🧩 Configuration

config.py defines:

- Numerical variables for histograms and boxplots
- Categorical variables for EDA and modeling
- Target variable (cardio)
- Human‑readable category labels
- Global random seed
- Number of CV splits

----

## 🧼 Data Preparation

data_prep.py performs:

- Strict clinical cleaning
- Winsorization
- Feature engineering (BMI, age bands, hypertension flags, lifestyle flags)
- Missingness injection (10%)
- Gaussian noise injection (5% of std)
- Post‑noise clipping
- Train/test split
- Traceability dictionary

---

## 🔧 Preprocessing

preprocessing.py builds:

- Train/test split with stratification
- Feature group definitions (numerical, ordinal, binary, flags)
- Standard preprocessing pipeline (median imputation, scaling, ordinal encoding, one‑hot encoding)
- Advanced preprocessing (median) with sparse‑safe scaling
- Advanced preprocessing (KNN) for robustness experiments
- Consistent ColumnTransformer outputs for all models

---

## 🧪 Modeling

modeling.py trains:

- Logistic Regression
- Random Forest
- HistGradientBoosting (advanced model)

It also performs:

- 5‑fold stratified cross‑validation
- ROC‑AUC and PR‑AUC evaluation
- Model comparison
- Probability calibration (isotonic)
- Final pipeline assembly

**Final production model:  
HistGradientBoosting + robustness pipeline + isotonic calibration**  
Stored at: models/final_pipeline.joblib

---

## 📊 Evaluation

evaluation.py includes:

- ROC‑AUC, PR‑AUC
- Bootstrapped confidence intervals
- Reliability curves + Brier score + ECE score
- Threshold selection:
    - Youden J
    - Cost‑based (FN:FP = 5:1)
    - Top‑k (20%)
- Subgroup fairness analysis (age × gender)

---

## 🔍 Interpretability

interpretability.py provides:

- Permutation Importance
- Partial Dependence (PDP)
- SHAP (TreeExplainer)
- SHAP interaction 

---

## 📊 Visualization

visualization.py generates:

- Histograms with clinical visualization limits
- Categorical barplots with human‑readable labels
- Boxplots by target
- Correlation matrix
- Category × target heatmaps
- ROC and PR curves
- Calibration plots (reliability + per‑bin ECE)
- Confusion matrix at custom thresholds
- Metrics barplots (sensitivity, specificity, PPV, NPV, F1)
- Permutation Importance
- Partial Dependence (PDP)
- SHAP summary plots
- SHAP dependence plots with automatic feature mapping
- SHAP interaction plots

---

## 📘 Model Card

A full clinical and technical description is available in:
model_card/model_card.md

---

## 🧪 Reproducibility

All modules use:

- Fixed random seeds
- Deterministic preprocessing
- Explicit feature groups
- Traceability for missingness and noise

---

## ⚙️ Requirements
Python 3.10+  
pandas  
numpy  
matplotlib  
seaborn  
scikit‑learn  

---

## 📈 Key Findings
- HistGradientBoosting + median imputation is the best model.  
- Calibrated probabilities improve clinical reliability.  
- Threshold optimization balances sensitivity and specificity.  
- Interpretability confirms known risk factors (hypertension, age, cholesterol).  
- Fairness analysis reveals subgroup disparities requiring attention.

---

## 📄 License

This project is released under the MIT License.

---

## 👤 Author

Patricia C. Torrell
Clinical Data Analyst transitioning into Data Analytics & Medical Writing  
Focused on clinical modeling, reproducible pipelines, and interpretable ML.  

**LinkedIn**: [linkedin.com/in/patricia-c-torrell](https://www.linkedin.com/in/patricia-c-torrell)  
**GitHub**: [github.com/PatriCT240.github.io](https://github.com/PatriCT240.github.io)  

---

## 🔑 Key Takeaways for Recruiters
- **Industry‑grade project architecture** with strict modular separation (`src/` modules, notebooks, reports).  
- **Reproducible and transparent workflow**, with clear saving logic and reporting.  
- **Predictive modeling proficiency**: Logistic Regression, Random Forest, HistGradientBoosting.  
- **Clinical domain expertise**: hypertension, cholesterol, BMI, age bands, lifestyle risk factors.  
- **Professional visualization and reporting layer** with modular plots and consolidated outputs.  
- **Fairness and interpretability focus**, ensuring transparency and equity in predictions.  
- **Clear communication and documentation**, including executive summary and recruiter‑friendly README.

