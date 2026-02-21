# Model Card – Cardiovascular Risk Prediction Model

## 1. Overview

This model estimates the probability of **10‑year cardiovascular disease (CVD)** using structured clinical data.  
It is designed to support **risk stratification**, **preventive interventions**, and **clinical decision‑making** in primary care settings.

The model is not intended to replace clinical judgment.  
It is a decision‑support tool that highlights patients who may benefit from closer monitoring or preventive strategies.

---

## 2. Intended Use

### **Primary use case**
- Assist clinicians in identifying patients at elevated cardiovascular risk.
- Prioritize follow‑up, lifestyle interventions, and diagnostic evaluation.
- Support population‑level screening programs.

### **Intended users**
- Physicians  
- Nurses  
- Clinical researchers  
- Public health analysts  

### **Clinical setting**
- Primary care  
- Preventive medicine  
- Cardiovascular risk screening  

### **Not intended for**
- Emergency decision‑making  
- Diagnosis of acute cardiovascular events  
- Use without clinician oversight  

---

## 3. Data

### **Dataset**
- Source: Cardiovascular Disease dataset (70,000 patients)
- Format: Structured tabular data
- Population: Adults aged 29–65 years old
- Target variable: `cardio` (binary: 0 = no CVD, 1 = CVD)

### **Features**
- **Demographics:** age, gender  
- **Anthropometrics:** height, weight, BMI  
- **Blood pressure:** systolic (ap_hi), diastolic (ap_lo)  
- **Laboratory markers:** cholesterol, glucose  
- **Lifestyle:** smoking, alcohol, physical activity  
- **Engineered features:**
  - age_years  
  - age_band  
  - hypertension_flag  
  - clinical_risk_flag  
  - lifestyle_risk_flag  
  - winsorization flags (was_capped_*)  

### **Data preprocessing**
- Strict clinical cleaning (removal of physiologically impossible values)
- Winsorization of outliers
- Feature engineering
- Injection of:
  - 10% missingness (robustness testing)
  - Gaussian noise (5% of std)
- Train/test split (80/20, stratified)

---

## 4. Model Details

### **Algorithms evaluated**
- Logistic Regression (baseline)
- Random Forest
- HistGradientBoosting (HGB)

### **Final selected model**
**HistGradientBoostingClassifier + Median Imputation + One‑Hot Encoding**

Chosen based on:
- Highest cross‑validated ROC‑AUC
- Best PR‑AUC
- Stable performance under noise and missingness
- Good calibration after isotonic adjustment

### **Calibration**
- Method: **Isotonic regression**
- Reliability curve included in evaluation

---

## 5. Performance

### **Cross‑validation (train set)**
| Metric | Logistic Regression | Random Forest | HGB |
|--------|---------------------|---------------|-----|
| ROC‑AUC | 0.7933 | 0.7980 | **0.7955** |
| PR‑AUC  | 0.7723 | 0.7788 | **0.7758** |

### **Test set (after calibration)**
- ROC‑AUC: **0.7890**
- PR‑AUC: **0.7726**
- Brier score: **0.1865**
- ECE: **0.0097**

### **Bootstrap 95% CI**
- ROC-AUC: 0.789 [0.781 – 0.797]
- PR-AUC: 0.773 [0.761 – 0.783]

---

## 6. Threshold Selection

Three operational thresholds were evaluated:

### **1. Youden J (balanced sensitivity/specificity)**
- Threshold: **0.4951**
- Sensitivity: 0.6906 
- Specificity: 0.7559 
- F1: 0.712 

### **2. Cost‑based (FN:FP = 5:1)**
- Threshold: **0.1486**

### **3. Top‑k (20%)**
- Threshold: **0.8245**

**Final chosen threshold: Youden J (0.4951)**  
Rationale: balanced performance and clinical interpretability.

---

## 7. Interpretability

### **Permutation Importance (Top Features)**
1. Systolic blood pressure (ap_hi)  
2. Age  
3. High cholesterol (category 3)  
4. Clinical risk flag  
5. Diastolic blood pressure (ap_lo)  
6. BMI  
7. Physical inactivity  
8. High glucose  

### **Subgroup Analysis**
- Lower sensitivity in young women and men aged 40–50
- Slightly lower specificity in older women
- Monitoring subgroup performance during deployment is recommended

### **PDP Findings**
- Risk increases monotonically with age  
- Risk increases sharply with systolic blood pressure (ap_hi)

### **SHAP Dependence Plot**
- The contribution of **age_years** increases steadily across the age range
- The effect of age is modulated by systolic blood pressure (ap_hi), suggesting a clinical interaction between aging and hypertension

### **SHAP Interaction Values (ap_hi × BMI)**
- The interaction between **ap_hi** and **BMI** shows that hypertension has a stronger impact in patients with obesity
- This combined effect is consistent with cardiovascular pathophysiology (hemodynamic load + peripheral resistance + metabolic inflammation)
- SHAP interaction values isolate the joint contribution beyond the individual effects

These patterns are clinically consistent.

---

## 8. Fairness & Subgroup Analysis

Subgroups evaluated:
- Gender (male, female)
- Age bands: [0–40), [40–50), [50–60), [60–70)

### **Findings**
- Sensitivity is lower in:
  - Younger women  
  - Men aged 40–50
- Specificity is slightly lower in older women  

### **Mitigation strategies (future work)**
- Reweighting  
- Group‑specific thresholds  
- Fairness‑aware calibration  

---

## 9. Ethical Considerations

### **Potential risks**
- Misclassification may lead to under‑ or over‑treatment  
- Bias across demographic groups  
- Overreliance on automated predictions  

### **Mitigations**
- Model must be used **with clinician oversight**  
- Thresholds can be adjusted per clinical context  
- Subgroup performance is transparently reported  

### **Not for autonomous decision‑making**

---

## 10. Limitations

- Dataset limited to a specific population (29–65 years old)
- No longitudinal data
- No medication history
- No imaging or ECG data
- Lifestyle variables are self‑reported
- Noise injection simulates but does not fully replicate real‑world variability

---

## 11. Recommendations for Deployment

- Use calibrated probabilities  
- Monitor model drift over time  
- Re‑train periodically with updated clinical data  
- Evaluate fairness metrics regularly  
- Integrate into clinical workflows with clear interpretability tools  

---

## 12. Contact

For questions, improvements, or clinical validation discussions, contact the project owner.

