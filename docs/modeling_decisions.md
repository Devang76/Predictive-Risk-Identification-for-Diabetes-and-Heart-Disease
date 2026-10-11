# Modeling Decisions & Safeguards Manifesto
**Project:** Predictive Risk Identification for Diabetes and Heart Disease  
**Milestone:** Stage 3 Preparation (Pre-Modeling Architecture, Rules, and Data Contracts)  
**Author:** Senior Data Science Team  
**Status:** Approved & Implemented  

---

## 1. Executive Summary & Objective

The objective of Stage 3 is to build trustworthy, clinically actionable machine learning models that:
1. Predict a patient's **Risk Tier** (`Low`, `Medium`, or `High`) separately for **Diabetes** and **Heart Disease**.
2. Identify and explain the **exact variables pushing any individual patient toward high risk** (using SHAP feature attribution), alongside global feature importances.
3. Power an interactive, safe digital health interface where new patient biometrics can be evaluated immediately with bounded validation.

To prevent common clinical data science failures—such as data leakage, metric illusions, population mismatch, and uncalibrated risk predictions—this document establishes the explicit methodological decisions, data architectures, and rules governing all Stage 3 modeling.

---

## 2. Modeling Tracks Architecture

### Decision: Separate Disease Models across Two Parallel Tracks (Never a Merged Model)
In earlier stages, an exploratory unified file was created ([data/clean/integrated_dataset.csv](file:///Users/devangmestry/Predictive%20Risk%20Identification%20for%20Diabetes%20and%20Heart%20Disease%20/data/clean/integrated_dataset.csv)).
> [!IMPORTANT]
> **Strict Rule:** `integrated_dataset.csv` is **PROHIBITED** from use in model training.
> 
> **Rationale:** In `integrated_dataset.csv`, continuous laboratory biometrics (glucose, insulin, continuous blood pressure, serum cholesterol) are **96.0% to 98.8% structurally missing** because the CDC BRFSS survey ($N=25,000$ in sample) only records binary self-reported items (`HighBP`, `HighChol`), while Pima and Cleveland only record lab tests. Naive imputation across this gap would invent artificial data for 97% of the cohort, completely corrupting predictive relationships.

Instead, Stage 3 establishes two dedicated modeling tracks:

```
┌──────────────────────────────────────────────────────────────────────────┐
│                           STAGE 3 MODELING TRACKS                        │
├─────────────────────────────────────┬────────────────────────────────────┤
│ TRACK A: Population Survey Track    │ TRACK B: Clinical Lab Benchmark    │
│ (Primary Production Track)          │ (Validation & Continuous Assays)   │
├─────────────────────────────────────┼────────────────────────────────────┤
│ • Dataset: CDC BRFSS 2015           │ • Datasets: UCI Pima & Cleveland   │
│ • Sample Size: N = 253,680          │ • Sample Sizes: N=768 & N=297      │
│ • Predictors: 20 Survey Features    │ • Predictors: Continuous Lab Tests │
│ • Model A1: Diabetes_binary Model   │ • Model B1: Pima Diabetes Model    │
│ • Model A2: Heart Disease Model     │ • Model B2: Cleveland Heart Model  │
└─────────────────────────────────────┴────────────────────────────────────┘
```

1. **Track A (Primary Production Tool):**
   - **Source:** [data/clean/brfss_clean.csv](file:///Users/devangmestry/Predictive%20Risk%20Identification%20for%20Diabetes%20and%20Heart%20Disease%20/data/clean/brfss_clean.csv) ($N=253,680$, 20 clean predictors).
   - This track directly powers the end-user patient tool because any user can readily enter their lifestyle, demographic, and basic health metrics without requiring invasive venous blood draws or hospital catheterization.
   - Evaluated as two separate models:
     * **Model A1 (Diabetes):** Predicts `Diabetes_binary`.
     * **Model A2 (Heart Disease):** Predicts `HeartDiseaseorAttack`.

2. **Track B (Clinical Validation Track):**
   - **Sources:** [data/clean/pima_clean.csv](file:///Users/devangmestry/Predictive%20Risk%20Identification%20for%20Diabetes%20and%20Heart%20Disease%20/data/clean/pima_clean.csv) ($N=768$, 8 lab features) and [data/clean/cleveland_clean.csv](file:///Users/devangmestry/Predictive%20Risk%20Identification%20for%20Diabetes%20and%20Heart%20Disease%20/data/clean/cleveland_clean.csv) ($N=297$, 13 cardiac features).
   - Serves as a clinical benchmark to evaluate how tree-based models and regularized linear models perform when high-precision continuous biometrics (fasting glucose, 2-hr serum insulin, ST-segment depression, fluoroscopy vessel counts) are available.

---

## 3. Data Leakage Prevention Rules

To guarantee that model accuracy is authentic and will generalize to new, unseen patients, three strict leakage rules are enforced:

### 3.1 Total Ban on Target Proxies (`high_glucose`)
- During Stage 1 EDA, an operational binary proxy (`high_glucose`) was derived directly from the survey target `Diabetes_binary`, producing an artificial Pearson correlation of $r = 0.97$.
- **Rule:** `high_glucose` is permanently banned and purged from every feature matrix across all tracks.
- **Verification:** Tested programmatically via [tests/test_stage3_prep.py](file:///Users/devangmestry/Predictive%20Risk%20Identification%20for%20Diabetes%20and%20Heart%20Disease%20/tests/test_stage3_prep.py); guaranteed absent from all files in [data/model_ready/](file:///Users/devangmestry/Predictive%20Risk%20Identification%20for%20Diabetes%20and%20Heart%20Disease%20/data/model_ready/).

### 3.2 Programmatic Target Correlation Audit ($|r| > 0.90$)
- In [src/03_model_preprocessing.py](file:///Users/devangmestry/Predictive%20Risk%20Identification%20for%20Diabetes%20and%20Heart%20Disease%20/src/03_model_preprocessing.py), an automated check calculates the correlation between every feature and the respective target.
- If any feature has $|r| > 0.90$, the pipeline automatically aborts with a `ValueError`.
- **Audit Findings:** The maximum feature correlation observed in BRFSS was $r = 0.2936$ (`GenHlth` with Diabetes) and $r = 0.2584$ (`GenHlth` with Heart Disease). In Pima, the maximum correlation was $r = 0.4946$ (`Glucose`), and in Cleveland, $r = 0.5205$ (`thal`). No leaky features exist.

### 3.3 Default Cross-Disease Target Exclusion (`include_cross_disease_target: false`)
- BRFSS records both `Diabetes_binary` and `HeartDiseaseorAttack` on the same patient.
- **Decision:** The diabetes label is strictly excluded from the heart disease feature set, and the heart disease label is strictly excluded from the diabetes feature set.
- **Plain-English Rationale:** In real-world clinical triage, when a patient presents for disease screening, whether they have the *other* confirmed condition is frequently unknown or unconfirmed. If the model relies heavily on the other disease's label as a predictive crutch, it:
  1. Masks the underlying lifestyle, biometric, and demographic risk drivers.
  2. Creates operational deadlock (you cannot predict diabetes without knowing heart disease status, and vice versa).
- **Configuration:** Governed by `include_cross_disease_target: false` in [config/model_config.yaml](file:///Users/devangmestry/Predictive%20Risk%20Identification%20for%20Diabetes%20and%20Heart%20Disease%20/config/model_config.yaml).

---

## 4. Feature Matrices & Reproducible Splits

### 4.1 Stratified 80/20 Splitting
- All datasets were partitioned into **80% Training** and **20% Testing** splits using `train_test_split` with:
  - `test_size = 0.20`
  - `random_state = 42`
  - `stratify = y` (ensuring exact preservation of positive disease prevalence across both splits).
- **Split Indices Manifest:** The exact row index arrays for train and test are serialized to [data/model_ready/split_indices.json](file:///Users/devangmestry/Predictive%20Risk%20Identification%20for%20Diabetes%20and%20Heart%20Disease%20/data/model_ready/split_indices.json). Any notebook, script, or team member reloading this file is guaranteed 100% deterministic row-level alignment.

### 4.2 Biological Zero Auditing in Pima
- In the Pima dataset, recorded zeros in `Glucose` (5), `BloodPressure` (35), `SkinThickness` (227), `Insulin` (374), and `BMI` (11) represent uncollected tests (a living human cannot have zero blood pressure or zero glucose). These are strictly treated as `NaN`.
- In contrast, `Pregnancies == 0` (111 women, 14.45%) represents valid nulliparous biology and is preserved intact as integer `0`.

### 4.3 Leakage-Free Scikit-Learn Pipelines
- **Rule:** Data scaling and imputation must **NEVER** be applied to the dataset prior to splitting.
- All imputation (e.g. median substitution for Pima lab tests) and feature scaling (`StandardScaler` or `RobustScaler`) are encapsulated in scikit-learn `Pipeline` objects instantiated via `src.risk_utils.get_preprocessing_pipeline()`.
- Imputation statistics and scaling factors are learned strictly on training folds (`pipe.fit(X_train)`), and applied downstream to test data (`pipe.transform(X_test)`).
- Model-specific pipeline logic:
  - **Tree Models (Decision Tree, Random Forest):** Require only imputation; scaling is omitted because tree split points are invariant to monotonic transformations.
  - **Linear Models (Logistic Regression):** Require median imputation followed by `StandardScaler` to ensure coefficients and L2 regularization penalties are on comparable scales.

---

## 5. Class Imbalance & Small-Data Safeguards

### 5.1 The Imbalance Challenge
- In Track A (CDC BRFSS):
  - **Diabetes:** 13.93% positive (~6.2 to 1 majority-to-minority ratio).
  - **Heart Disease:** 9.42% positive (~9.6 to 1 ratio).
- **Why Accuracy is Prohibited:** A trivial dummy classifier that unconditionally predicts "Healthy" achieves **86.07% accuracy on diabetes** and **90.58% accuracy on heart disease**, yet it fails to detect 100% of sick patients.
- **Evaluation Metric Hierarchy:**
  1. **Primary Metric 1: Recall / Sensitivity:** Minimizes false negatives (the clinical cost of sending a sick patient home without triage).
  2. **Primary Metric 2: F1-Score & PR-AUC (Precision-Recall Area):** Measures the balance between positive capture and false alarms under heavy skew.
  3. **Primary Metric 3: ROC-AUC:** Assesses overall discrimination across all potential classification thresholds.
  4. **Prohibited Metric:** Overall Accuracy (forbidden from model selection criteria).

### 5.2 Training Mitigation Strategies
1. **Cost-Sensitive Weighting (`class_weight='balanced'`):** Penalizes false negatives inversely proportional to class frequencies. This is computationally fast and does not synthesize artificial data.
2. **SMOTE Protocol:** If Synthetic Minority Over-sampling Technique (SMOTE) is benchmarked, it must be executed strictly within `imblearn.pipeline.Pipeline`. This ensures synthetic samples are generated **exclusively within training folds** and never leak into validation or test folds.

### 5.3 Small Clinical Data Protocols (Track B)
- UCI Pima ($N=614$ train / $154$ test) and UCI Cleveland ($N=237$ train / $60$ test) have limited sample sizes.
- **Overfitting Safeguards:**
  - Standard 5-fold CV is augmented to **Repeated Stratified 5-Fold Cross-Validation** (3 repeats = 15 evaluation folds).
  - Logistic Regression will use strict L2 penalty tuning (`C` regularization grid).
  - Decision Trees and Random Forests will be constrained by shallow depth (`max_depth <= 4` or `5`) and `min_samples_split >= 10`.

---

## 6. Clinical Risk Tier Policy

In medical screening, a raw continuous probability (e.g. $P(\text{Heart Disease}) = 0.23$) is difficult for patients and primary-care nurses to act upon without context. Stage 3 introduces a formalized **Tri-Level Risk Categorization Policy** defined in [config/risk_tiers.yaml](file:///Users/devangmestry/Predictive%20Risk%20Identification%20for%20Diabetes%20and%20Heart%20Disease%20/config/risk_tiers.yaml).

```
   0.0                                                     1.0
    ├─────────────┬───────────────────────────┬─────────────┤
    │  LOW RISK   │        MEDIUM RISK        │  HIGH RISK  │
    └─────────────┴───────────────────────────┴─────────────┘
               τ_low                       τ_high
```

### 6.1 Calibration Mathematical Rules
1. **Low Risk Cutoff ($\tau_{\text{low}}$) — Triage Safety Rule:**
   - $\tau_{\text{low}}$ is calibrated so that **at least 90% of all true positive cases** fall into the Medium or High risk tiers.
   - *Clinical Meaning:* The Low Risk tier acts as a safe "rule-out" zone. A patient told they are "Low Risk" has a $\le 10\%$ probability of being a missed true positive.
2. **High Risk Cutoff ($\tau_{\text{high}}$) — Actionable Concentration Rule:**
   - $\tau_{\text{high}}$ is set at the probability threshold where the observed empirical disease rate in that tier is **at least $2.0\times$ baseline prevalence** (Lift $\ge 2.0$).
   - *Clinical Meaning:* Patients designated as "High Risk" have at least double the average population risk, providing an evidence-based trigger for diagnostic lab workups.
3. **Monotonicity Requirement:**
   - The observed positive rate must strictly satisfy:
     $$\text{Prevalence}(\text{Low}) < \text{Prevalence}(\text{Medium}) < \text{Prevalence}(\text{High})$$

### 6.2 Initial Baseline Thresholds

| Track & Disease | Baseline Prevalence | Low Threshold ($\tau_{\text{low}}$) | High Threshold ($\tau_{\text{high}}$) | Min Target High Prevalence (2x) |
| :--- | :---: | :---: | :---: | :---: |
| **Track A: Diabetes** | 13.93% | 0.100 (10%) | 0.300 (30%) | $\ge 27.86\%$ |
| **Track A: Heart Disease** | 9.42% | 0.080 (8%) | 0.200 (20%) | $\ge 18.84\%$ |
| **Track B: Pima Diabetes** | 34.90% | 0.250 (25%) | 0.600 (60%) | $\ge 69.80\%$ |
| **Track B: Cleveland Heart** | 46.13% | 0.250 (25%) | 0.700 (70%) | $\ge 85.00\%$ |

*Note:* During model training, these cutoffs will be fine-tuned and verified on out-of-fold cross-validation probabilities using `src.risk_utils.validate_risk_tier_monotonicity()`.

---

## 7. Patient Input Schema & Validation

The final tool will accept inputs from users via a web form or CLI. To prevent invalid predictions caused by out-of-range biometrics or malformed strings, a rigorous schema was codified in [config/feature_schema.yaml](file:///Users/devangmestry/Predictive%20Risk%20Identification%20for%20Diabetes%20and%20Heart%20Disease%20/config/feature_schema.yaml).

### 7.1 Schema Enforcement Logic
Handled by `validate_patient_input()` in [src/risk_utils.py](file:///Users/devangmestry/Predictive%20Risk%20Identification%20for%20Diabetes%20and%20Heart%20Disease%20/src/risk_utils.py):
- **Range Verification:** Numeric entries outside biological feasibility are rejected with clear clinical error messages:
  - BMI must be within $[12.0, 90.0]\text{ kg/m²}$ (rejects values like $5.0$ or $150.0$).
  - Physical and Mental health days must be within $[0, 30]\text{ days/month}$.
  - Blood Pressure (Pima) must be within $[30.0, 160.0]\text{ mmHg}$.
  - Fasting Glucose (Pima) must be within $[40.0, 300.0]\text{ mg/dL}$.
- **Category Verification:** Ordinal survey codes (General Health: $1\dots 5$, Age brackets: $1\dots 13$, Income: $1\dots 8$) must be exact integer tiers.
- **Default Population:** If an optional question is skipped, the schema safely populates the population median or mode (e.g. `BMI = 27.0`, `PhysActivity = 1`, `Smoker = 0`).

---

## 8. Summary of Model-Ready Datasets

The preprocessing script [src/03_model_preprocessing.py](file:///Users/devangmestry/Predictive%20Risk%20Identification%20for%20Diabetes%20and%20Heart%20Disease%20/src/03_model_preprocessing.py) was executed end-to-end with zero errors. All datasets were written to [data/model_ready/](file:///Users/devangmestry/Predictive%20Risk%20Identification%20for%20Diabetes%20and%20Heart%20Disease%20/data/model_ready/).

Below are the verified, exact dataset dimensions:

| Dataset / Model Track | Total Rows | Features ($X$) | Train Rows (80%) | Train Positive Rate | Test Rows (20%) | Test Positive Rate | Stratification Delta |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Track A: Diabetes (BRFSS)** | **253,680** | **20** | 202,944 | **13.93%** (28,277) | 50,736 | **13.93%** (7,069) | 0.0005% |
| **Track A: Heart Disease (BRFSS)** | **253,680** | **20** | 202,944 | **9.42%** (19,114) | 50,736 | **9.42%** (4,779) | 0.0010% |
| **Track B: Diabetes (Pima)** | **768** | **8** | 614 | **34.85%** (214) | 154 | **35.06%** (54) | 0.2115% |
| **Track B: Heart Disease (Cleveland)** | **297** | **13** | 237 | **45.99%** (109) | 60 | **46.67%** (28) | 0.6751% |

### Verification Suite Results
The test suite [tests/test_stage3_prep.py](file:///Users/devangmestry/Predictive%20Risk%20Identification%20for%20Diabetes%20and%20Heart%20Disease%20/tests/test_stage3_prep.py) was executed via Pytest. All **8 out of 8 tests passed** with 100% compliance:
- `test_no_target_or_cross_target_in_features`: PASSED
- `test_no_high_glucose_in_any_dataset`: PASSED
- `test_train_test_split_no_overlap`: PASSED
- `test_patient_input_schema_validation_accepts_valid`: PASSED
- `test_patient_input_schema_validation_catches_bad_inputs`: PASSED
- `test_risk_tier_assignment_logic`: PASSED
- `test_risk_tier_monotonicity_validator`: PASSED
- `test_preprocessing_pipeline_fit_strictly_on_train`: PASSED

---

## 9. Next Steps for Final Modeling (Stage 3 Execution)

With all data contracts, split manifests, leakage safeguards, schemas, and tier validation utilities locked, the final modeling steps can now proceed with confidence:
1. **Model Training Script (`src/04_train_models.py`):** Train Logistic Regression, Decision Tree, and Random Forest per track and disease.
2. **Evaluation & Comparison (`src/05_evaluate_models.py`):** Compute Recall, F1, ROC-AUC, and PR-AUC; generate comparison tables and ROC/PR curves.
3. **Explainability (`src/06_explainability_shap.py`):** Generate global beeswarm and patient-level waterfall plots.
4. **Interactive Triage Tool (`src/predict_patient.py`):** Provide an end-to-end function/CLI taking patient biometrics and outputting Risk Category + Top 3 Contributing Factors.
