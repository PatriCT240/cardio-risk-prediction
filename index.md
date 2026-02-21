# Cardiovascular Risk Prediction

This project implements a full end‑to‑end machine learning pipeline for **cardiovascular risk prediction** using structured clinical data.  
It includes rigorous data cleaning, feature engineering, robust preprocessing, model benchmarking, probability calibration, interpretability, fairness analysis, and a fully documented final model.

---

## 🔍 Project Overview

The goal of this project is to build a reliable and interpretable machine learning system capable of predicting cardiovascular disease (CVD) using clinical variables.  
The pipeline is designed to be **modular, reproducible, and deployment‑ready**, following best practices in clinical ML.

---

## 🧠 Key Features

- Clinical‑style data cleaning and validation  
- Feature engineering and preprocessing pipelines  
- Model selection and evaluation across multiple algorithms  
- Final calibrated **HistGradientBoostingClassifier**  
- Threshold optimization using **Youden’s J statistic**  
- Performance evaluation with ROC‑AUC, PR‑AUC and bootstrap confidence intervals  
- Interpretability with feature importance, PDP and SHAP  
- Fairness analysis across age × gender subgroups  
- Fully reproducible codebase with clear structure  

---

## 📊 Final Model Performance

- **ROC‑AUC:** 0.789  
- **PR‑AUC:** 0.773  
- **Bootstrap 95% CI** for both metrics  
- Balanced sensitivity and specificity at the optimized threshold  
- Well‑calibrated probability estimates (isotonic calibration)

---

## 📁 Repository Structure

