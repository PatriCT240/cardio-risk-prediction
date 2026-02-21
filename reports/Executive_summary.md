# EXECUTIVE SUMMARY  
### *Cardiovascular Risk Prediction – Machine Learning Pipeline*

---

## 1. Project Objective
Develop a clinically reliable and interpretable machine learning model to predict **cardiovascular disease (CVD)** using structured clinical data from 70,000 patients.

The system supports **risk stratification**, **preventive decision‑making**, and **transparent probability‑based predictions**.

---

## 2. Dataset Overview
- **Population:** 70,000 adult patients  
- **Features:** demographics, anthropometrics, blood pressure, laboratory markers, lifestyle factors, engineered clinical flags  
- **Target:** `cardio` (binary CVD outcome)  
- **Source:** CardioVascular Disease dataset  

---

## 3. Final Model
### **HistGradientBoostingClassifier**  
**+ Advanced robustness preprocessing**  
**+ Median imputation**  
**+ Isotonic calibration**

Stored at: `models/final_pipeline.joblib`

This configuration achieved the best balance of discrimination, calibration, and interpretability.

---

## 4. Model Performance

### **Discrimination**
| Metric | Value |
|--------|--------|
| **ROC‑AUC** | **0.789** |
| **PR‑AUC** | **0.773** |

### **Bootstrap 95% Confidence Intervals**
- **ROC‑AUC:** 0.789 **[0.781 – 0.797]**  
- **PR‑AUC:** 0.773 **[0.761 – 0.783]**

These intervals demonstrate **stable performance** across resampled populations.

---

## 5. Calibration
Isotonic calibration improved probability reliability:

- Predicted risks closely match observed event rates  
- Reduced overconfidence in high‑risk predictions  
- Lower Expected Calibration Error (ECE)  

This makes the model suitable for **probability‑based decision support**.

---

## 6. Threshold Selection

### **Selected Threshold:** **0.4951**  
(Method: **Youden’s J statistic**)

### **Confusion Matrix at Final Threshold**
| Metric | Value |
|--------|--------|
| **TP** | 4699 |
| **FP** | 1696 |
| **FN** | 2105 |
| **TN** | 5253 |

### **Derived Metrics**
| Metric | Value |
|--------|--------|
| **Sensitivity** | 0.6906 |
| **Specificity** | 0.7559 |
| **PPV** | 0.7348 |
| **NPV** | 0.7139 |
| **F1‑score** | 0.712 |

This threshold provides a **balanced trade‑off** between missed cases and false positives.

---

## 7. Interpretability Findings

### **Global Importance**
- Age  
- Systolic blood pressure (ap_hi)  
- Cholesterol  
- BMI  
- Hypertension and lifestyle flags  

### **SHAP Insights**
- Non‑linear effects of age and blood pressure  
- Interactions between hypertension flags and cholesterol  
- Lifestyle factors contribute modest but consistent risk increments  

These findings align with known cardiovascular risk patterns.

---

## 8. Fairness Analysis
Subgroup evaluation across **age × gender** showed:

- Stable ROC‑AUC across groups  
- Slight sensitivity reduction in older female patients  
- No extreme disparities, but continued monitoring recommended  

---

## 9. Limitations
- Dataset not population‑representative  
- No external validation  
- No ECG, imaging, or advanced biomarkers  
- Observational data → no causal inference  
- Intended for **decision support**, not standalone diagnosis  

---

## 10. Conclusion
The final calibrated HGB model delivers:

- **Strong discrimination**  
- **Reliable probability estimates**  
- **Transparent interpretability**  
- **Clinically meaningful thresholds**  
- **Reproducible, modular architecture**

This system is ready for **risk stratification**, **preventive screening**, and **clinical decision‑support workflows**, with clear pathways for future external validation and deployment.

---
