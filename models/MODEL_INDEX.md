# Complete Machine Learning Model Index

**Project:** Predictive Risk Identification for Diabetes and Heart Disease  
**Master Model Package:** Chronic Disease Predictor Model  
**Directory:** `models/`  
**Last Audited:** October 10, 2026

---

## 1. Inventory of Serialized Model Artifacts

| Model File Name | Location Folder | Track | Condition | Algorithm | Champion Model | Test ROC-AUC | Test F1-Score | Test Recall (Sensitivity) | Operating Threshold |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `track_a_diabetes_logistic_regression.joblib` | `all_models/track_a_survey/` | Track A (Survey) | Type 2 Diabetes | Logistic Regression | **YES (Champion)** | **0.8190** | **0.4431** | **74.79%** | `0.1473` |
| `track_a_diabetes_decision_tree.joblib` | `all_models/track_a_survey/` | Track A (Survey) | Type 2 Diabetes | Decision Tree | No | 0.8176 | 0.4415 | 74.17% | `0.1611` |
| `track_a_diabetes_random_forest.joblib` | `all_models/track_a_survey/` | Track A (Survey) | Type 2 Diabetes | Random Forest | No | 0.8235 | 0.4556 | 71.92% | `0.1709` |
| `track_a_heart_logistic_regression.joblib` | `all_models/track_a_survey/` | Track A (Survey) | Heart Disease | Logistic Regression | **YES (Champion)** | **0.8458** | **0.3900** | **75.52%** | `0.1101` |
| `track_a_heart_decision_tree.joblib` | `all_models/track_a_survey/` | Track A (Survey) | Heart Disease | Decision Tree | No | 0.8424 | 0.3875 | 73.49% | `0.1356` |
| `track_a_heart_random_forest.joblib` | `all_models/track_a_survey/` | Track A (Survey) | Heart Disease | Random Forest | No | 0.8467 | 0.3996 | 71.52% | `0.1375` |
| `track_b_diabetes_logistic_regression.joblib` | `all_models/track_b_clinical/` | Track B (Clinical) | Type 2 Diabetes | Logistic Regression | **YES (Champion)** | **0.8081** | **0.6452** | **74.07%** | `0.3318` |
| `track_b_diabetes_decision_tree.joblib` | `all_models/track_b_clinical/` | Track B (Clinical) | Type 2 Diabetes | Decision Tree | No | 0.7841 | 0.6457 | 75.93% | `0.4171` |
| `track_b_diabetes_random_forest.joblib` | `all_models/track_b_clinical/` | Track B (Clinical) | Type 2 Diabetes | Random Forest | No | 0.8156 | 0.6667 | 70.37% | `0.4112` |
| `track_b_heart_logistic_regression.joblib` | `all_models/track_b_clinical/` | Track B (Clinical) | Heart Disease | Logistic Regression | **YES (Champion)** | **0.9275** | **0.8333** | **71.43%** | `0.5672` |
| `track_b_heart_decision_tree.joblib` | `all_models/track_b_clinical/` | Track B (Clinical) | Heart Disease | Decision Tree | No | 0.8661 | 0.6531 | 57.14% | `0.5309` |
| `track_b_heart_random_forest.joblib` | `all_models/track_b_clinical/` | Track B (Clinical) | Heart Disease | Random Forest | No | 0.9297 | 0.7826 | 64.29% | `0.6045` |

*Table Caption: Comprehensive Inventory of all 12 Supervised Machine Learning Pipelines trained across dual disease tracks and evaluated on held-out test sets. Performance metrics (ROC-AUC, F1, Recall) and operating triage thresholds are extracted directly from outputs/tables/model_comparison_metrics.csv.*

---

## 2. Directory Structure Overview

```
models/
├── chronic_disease_predictor/         <- Primary production package (Chronic Disease Predictor Model)
│   ├── predictor.py                   <- Main inference engine (ChronicDiseasePredictor class)
│   ├── model_card.md                  <- Official model governance and performance card
│   ├── feature_schema.yaml            <- Permissible biometric bounds and survey encodings
│   ├── risk_tiers.yaml                <- Calibrated cut-off thresholds for Low/Medium/High triage
│   ├── track_a_diabetes_logistic_regression.joblib -> symlink to champion
│   ├── track_a_heart_logistic_regression.joblib    -> symlink to champion
│   ├── track_b_diabetes_logistic_regression.joblib -> symlink to champion
│   └── track_b_heart_logistic_regression.joblib    -> symlink to champion
├── all_models/
│   ├── track_a_survey/                <- CDC BRFSS population screening models (6 files)
│   ├── track_b_clinical/              <- UCI Pima & Cleveland diagnostic lab models (6 files)
│   └── stage2_unsupervised/           <- Stage 2 unsupervised health profiles and clustering README
└── MODEL_INDEX.md                     <- This document
```

---

## 3. Champion Model Architecture & Selection Rationale

The four designated champion models are unified under the **Chronic Disease Predictor Model** package (`models/chronic_disease_predictor/`):

1. **Track A Diabetes:** `CalibratedClassifierCV(LogisticRegression, method='sigmoid')`
   - *Rationale:* Test ROC-AUC of 0.8190 matches Random Forest within 0.0045, satisfying the ~75% recall requirement with smooth linear calibration and log-odds interpretability.
2. **Track A Heart Disease:** `CalibratedClassifierCV(LogisticRegression, method='sigmoid')`
   - *Rationale:* Test ROC-AUC of 0.8458 matches Random Forest within 0.0009 while achieving 75.52% recall at threshold 0.1101.
3. **Track B Diabetes:** `CalibratedClassifierCV(LogisticRegression, method='sigmoid')`
   - *Rationale:* Test ROC-AUC of 0.8081 with 74.07% recall on Pima Indians test set, outperforming decision trees (0.7841) and providing stable coefficients without small-sample overfitting.
4. **Track B Heart Disease:** `CalibratedClassifierCV(LogisticRegression, method='sigmoid')`
   - *Rationale:* Test ROC-AUC of 0.9275, PR-AUC of 0.9423, and 100% precision on positives at threshold 0.5672 on Cleveland Heart Disease test set.
