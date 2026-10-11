# Model Card: Chronic Disease Predictor Model

**Model Identifier:** `chronic_disease_predictor`  
**Package Version:** `1.0.0`  
**Release Date:** October 2026  
**License:** Educational & Open Research  
**Primary Intended Task:** Calibrated Clinical Risk Stratification & Patient-Level SHAP Feature Attribution for Type 2 Diabetes and Coronary Heart Disease

---

## 1. Model Overview & Purpose

The **Chronic Disease Predictor Model** is a clinical decision support system designed to evaluate an individual patient's risk profile for Type 2 Diabetes Mellitus and Coronary Heart Disease. Rather than acting as a static binary classifier, the model provides:
1. **Calibrated Event Probabilities:** Platt-scaled probabilities aligned with true empirical incidence.
2. **Three-Tier Risk Stratification:** Actionable classification into `Low`, `Medium`, and `High` risk tiers.
3. **Patient-Level Feature Attribution:** Local Shapley Additive Explanations (SHAP) displaying the exact biometric readings pushing risk upward or downward.
4. **Dual Operational Modes:** Population Survey Screening (`--mode survey`, Track A) and Continuous Diagnostic Laboratory Tests (`--mode clinical`, Track B), with a combined dual evaluation mode (`--mode both`).

---

## 2. Intended Use & Target Users

- **Intended Use:** Preventative risk triage, routine population health screening, and patient lifestyle counseling. Assisting clinicians in prioritizing high-risk individuals for confirmatory diagnostic testing and lifestyle intervention.
- **Target Users:** Primary care physicians, nurse practitioners, public health coordinators, and informed patients exploring disease risk factors.
- **Out of Scope Use:** Acute emergency room diagnosis (e.g., acute myocardial infarction triage), definitive medical diagnosis, or automated treatment prescribing.

---

## 3. Training & Validation Datasets

| Track | Primary Cohort | Records ($N$) | Features | Target Outcome | Baseline Prevalence |
| :--- | :--- | :---: | :---: | :--- | :---: |
| **Track A: Survey** | CDC BRFSS 2015 Health Indicators | 253,680 total (202,944 train / 50,736 test) | 20 non-invasive survey features | `Diabetes_binary` / `HeartDiseaseorAttack` | Diab: 13.93% / Heart: 9.42% |
| **Track B: Clinical Diabetes** | UCI Pima Indians Diabetes | 768 total (614 train / 154 test) | 8 clinical laboratory features | `Outcome` (Diabetes) | 34.85% |
| **Track B: Clinical Heart** | UCI Cleveland Heart Disease | 297 total (237 train / 60 test) | 13 cardiovascular features | `condition` (Heart Disease) | 45.99% |
| **External Validation** | CDC BRFSS 2022 Longitudinal Cohort | 60,000 hold-out sample | Harmonized survey features | `HadHeartAttack` / `HadAngina` | 7.91% |

---

## 4. Champion Model Specifications & Real Test Performance

All champion models are regularized, calibrated pipelines (`CalibratedClassifierCV` with Platt sigmoid scaling on `LogisticRegression`):

| Track & Disease | Champion Algorithm | Operating Threshold | Test ROC-AUC (95% CI) | Test PR-AUC (95% CI) | Test Recall (Sensitivity) | Test Precision | Test F1-Score |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Track A: Diabetes** | `LogisticRegression(C=0.1, L2)` | `0.1473` | **0.8190** `[0.8142, 0.8238]` | **0.3913** `[0.3807, 0.4023]` | **74.79%** | 31.48% | **0.4431** |
| **Track A: Heart Disease** | `LogisticRegression(C=0.01, L2)` | `0.1101` | **0.8458** `[0.8409, 0.8510]` | **0.3594** `[0.3463, 0.3743]` | **75.52%** | 26.29% | **0.3900** |
| **Track B: Diabetes** | `LogisticRegression(C=1.0, L2)` | `0.3318` | **0.8081** `[0.7342, 0.8689]` | **0.6639** `[0.5348, 0.7929]` | **74.07%** | 57.14% | **0.6452** |
| **Track B: Heart Disease** | `LogisticRegression(C=0.1, L2)` | `0.5672` | **0.9275** `[0.8495, 0.9822]` | **0.9423** `[0.8755, 0.9859]` | **71.43%** | 100.00% | **0.8333** |

*All metrics are verified from outputs/tables/champion_models.csv and outputs/tables/validation_confidence_intervals.csv.*

---

## 5. Risk Tier Definitions & Clinical Actionability

Risk categories are computed from calibrated probabilities using structured population cut-offs:

| Track & Condition | Low Risk Tier | Medium Risk Tier | High Risk Tier | Observed High-Tier Prevalence | High-Tier Lift vs. Baseline | Combined Med+High Positive Capture |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Track A: Diabetes** | $p \le 0.076$ | $0.076 < p \le 0.279$ | $p > 0.279$ | 40.83% | **2.93x** | **90.64%** |
| **Track A: Heart Disease** | $p \le 0.051$ | $0.051 < p \le 0.188$ | $p > 0.188$ | 33.60% | **3.57x** | **91.32%** |
| **Track B: Diabetes** | $p \le 0.223$ | $0.223 < p \le 0.697$ | $p > 0.697$ | 67.86% | **1.95x** | **88.89%** |
| **Track B: Heart Disease** | $p \le 0.258$ | $0.258 < p \le 0.920$ | $p > 0.920$ | 100.00% | **2.17x** | **85.71%** |

- **Low Risk:** Safe to defer invasive diagnostic follow-ups; maintain routine preventative lifestyle guidance.
- **Medium Risk:** Prime candidate for targeted lifestyle counseling, nutritional guidance, and annual re-evaluation.
- **High Risk:** Recommended for urgent medical consultation and confirmatory diagnostic workups (e.g., HbA1c, coronary CTA).

---

## 6. Robustness & Validation Summary

1. **Statistical Generalization (PASS):** 1,000-resample bootstrap 95% confidence intervals confirm narrow error bounds on population test cohorts ($\pm 0.005$ ROC-AUC).
2. **Clinical Sensitivity Standard (PASS):** Calibrated thresholds achieve $\ge 74.8\%$ sensitivity on survey data, minimizing dangerous false negatives.
3. **Cross-Validation Stability (PASS):** Repeated 5-fold CV across 15 folds yields standard deviations $\le 0.0062$.
4. **Demographic Equity (PASS):** Sensitivity disparity between females ($73.8\%$) and males ($75.9\%$) is $<2.1\%$; sensitivity across middle-aged and elderly brackets remains stable ($73.5\%$ to $84.0\%$).
5. **Longitudinal Generalization (PASS):** Out-of-cohort evaluation on CDC BRFSS 2022 ($N=60,000$) retains high discrimination (ROC-AUC $= 0.8274$).
6. **Feature Attribution Stability (PASS):** Mean Top-3 SHAP feature overlap between random seeds 42 and 101 is $87.3\%$ (Jaccard similarity $= 0.781$).
7. **Small Sample Discreteness (WEAK):** Cleveland Heart Disease test set ($N=60$) exhibits wide confidence intervals ($[53.6\%, 87.5\%]$ recall) due to small cohort size.

---

## 7. Model Limitations & Ethical Considerations

- **Self-Reported Data:** Track A relies on telephone survey responses subject to recall and social desirability biases.
- **Statistical Association vs. Causality:** SHAP attributions reflect mathematical feature contributions within a predictive model, not proven biological causes.
- **Retrospective Validation:** Models were evaluated on historical public health and clinical research cohorts. Real-world clinical deployment requires formal prospective validation under institutional governance.

---

## 8. Mandatory Clinical Safety Notice

> **IMPORTANT WARNING:**  
> **These are key contributing factors from a statistical model, not proven causes, and this is not a medical diagnosis.**  
> The Chronic Disease Predictor Model is an informational decision support tool. It does not replace professional medical judgment, laboratory testing, or physician consultation.
