# Chronic Disease Predictor Model
### Predictive Risk Identification for Diabetes and Heart Disease

A dual-track machine learning system and clinical decision-support platform that predicts patient **Risk Categories** (`Low`, `Medium`, or `High`) separately for **Diabetes** and **Heart Disease**, and identifies the **exact biometric variables, clinical laboratory readings, and lifestyle factors pushing that patient toward that category** using SHAP feature attribution.

---

## 1. Quick Start

### Python API
Run an instant patient risk evaluation directly from the command line or any Python application:
```bash
python -c "from models.chronic_disease_predictor.predictor import ChronicDiseasePredictor as P; print(P().predict({'HighBP': 1, 'HighChol': 1, 'BMI': 32.0, 'Smoker': 0, 'PhysActivity': 1, 'HvyAlcoholConsump': 0, 'GenHlth': 3, 'PhysHlth': 2, 'DiffWalk': 0, 'Sex': 1, 'Age': 8}))"
```

### Command-Line Interface (CLI)
Evaluate pre-configured patient profiles:
```bash
python src/predict_patient.py --mode survey --input examples/high_risk_patient.json
```

### Interactive Web Application (Streamlit)
Launch the clinical decision-support dashboard:
```bash
streamlit run app/streamlit_app.py
```

---

## 2. System Overview

### Core Capabilities
1. **Packaged Model System (`models/chronic_disease_predictor/`):**
   - Clean, modular `ChronicDiseasePredictor` class exposing `.predict(patient_dict, mode="survey" | "clinical" | "both")`.
   - Returns calibrated probabilities, risk tiers (`Low`, `Medium`, `High`), Top 3 risk-raising factors with SHAP attributions, Top 2 risk-lowering factors, and statutory medical disclaimers.
   - Comprehensive `model_card.md` documenting verified training, test metrics, and confidence intervals.
   - Full model index cataloging all 12 candidate models across both tracks in `models/MODEL_INDEX.md`.

2. **Interactive Patient Decision Support CLI (`src/predict_patient.py`):**
   - **Survey Mode (`--mode survey`):** Population survey biometrics (CDC BRFSS 2015, 20 features) without requiring invasive blood tests.
   - **Clinical Mode (`--mode clinical`):** Diagnostic continuous laboratory assays (Pima Diabetes: Glucose, Insulin, Skinfold; Cleveland Heart: Angiographic vessels, Resting BP, Serum Cholesterol, Thallium scintigraphy). Formats exact readings with medical units: e.g., `Glucose = 185.0 mg/dL -> raises risk (+2.23)`.
   - **Dual-Track Mode (`--mode both`):** Comprehensive side-by-side evaluation comparing population vs clinical laboratory risk.
   - **Graceful Fallback:** If a user runs clinical mode with survey biometrics omitting laboratory tests, the system automatically detects missing lab readings and executes survey mode with an explicit notice.
   - **Physiological Validation:** Rejects impossible inputs (e.g., negative glucose, impossible BMI) against formal clinical bounds in `config/feature_schema.yaml`.
   - **Explainability:** Generates patient-specific SHAP waterfall plots (`.png`) and structured prediction records (`.json`).
   - **Medical Disclaimer:** Enforces mandatory clinical safety notices.

3. **Interactive Web Application (`app/streamlit_app.py`):**
   - Streamlit interface with quick-load buttons for Low, Medium, and High risk profiles, mode switcher, dynamic input forms, color-coded risk tier badges, calibrated probability gauges, and embedded waterfall plots.

4. **Dual-Track Modeling Architecture:**
   - **Track A (Primary Population Tool):** CDC BRFSS 2015 ($N=253,680$). 20 survey-derived lifestyle, demographic, and biometric predictors.
   - **Track B (Clinical Validation Benchmark):** Diagnostic continuous lab assays from UCI Pima Indians Diabetes ($N=768$) and UCI Cleveland Heart Disease ($N=297$).

---

## 3. Installation

This project runs on Python 3.9+ with a dedicated virtual environment:

```bash
# 1. Navigate to the project root directory
cd "Predictive Risk Identification for Diabetes and Heart Disease "

# 2. Create and activate a clean virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install pinned dependencies
pip install -r requirements.txt
```

---

## 3. One-Command Complete Execution (`run_all.sh`)

To execute the entire end-to-end pipeline from data splitting to final validation and testing in one command:

```bash
./run_all.sh
```

### Pipeline Execution Sequence
1. `src/03_model_preprocessing.py`: Partitions raw datasets into model-ready train and test sets (`data/model_ready/`) with zero data leakage.
2. `src/04_train_models.py`: Trains and tunes Logistic Regression, Decision Tree, and Random Forest models across tracks, applying Platt probability calibration (`CalibratedClassifierCV`).
3. `src/05_evaluate_models.py`: Evaluates candidates on untouched test sets, selects champions, calibrates risk tiers, and exports `champion_models.csv`.
4. `src/06_explainability_shap.py`: Computes Track A SHAP beeswarms, bar plots, and prototypical waterfall plots.
5. `src/07_track_b_shap.py`: Computes Track B clinical SHAP explanations, out-of-fold tier calibration analysis, and repeated cross-validation metrics.
6. `src/08_validation.py`: Runs 1,000-resample bootstrap 95% CIs, repeated 5-fold CV, subgroup fairness auditing (Age & Sex), BRFSS 2022 external generalization sanity check, baseline comparisons, and SHAP stability across seeds.
7. `pytest tests/ -v`: Runs the complete automated test suite (19 passing unit and integration tests).

---

## 4. How to Use the Patient Risk CLI (`src/predict_patient.py`)

The triage CLI supports three distinct modes:

### Mode 1: Population Survey Mode (`--mode survey`)
Evaluates survey-reported lifestyle and biometric indicators (Track A):
```bash
# Evaluate pre-configured test patients
python src/predict_patient.py --mode survey --input examples/low_risk_patient.json
python src/predict_patient.py --mode survey --input examples/medium_risk_patient.json
python src/predict_patient.py --mode survey --input examples/high_risk_patient.json

# Launch interactive survey wizard
python src/predict_patient.py --mode survey --interactive
```

### Mode 2: Clinical Laboratory Mode (`--mode clinical`)
Evaluates biochemical lab assays and diagnostic readings (Track B) with clinical unit formatting:
```bash
# Evaluate pre-configured clinical patients
python src/predict_patient.py --mode clinical --input examples/clinical_patient_low.json
python src/predict_patient.py --mode clinical --input examples/clinical_patient_medium.json
python src/predict_patient.py --mode clinical --input examples/clinical_patient_high.json

# Launch interactive clinical lab wizard
python src/predict_patient.py --mode clinical --interactive
```

### Mode 3: Dual-Track Mode (`--mode both`)
Evaluates population risk and clinical laboratory risk side-by-side:
```bash
python src/predict_patient.py --mode both --input examples/comprehensive_patient_both.json
```

### Graceful Fallback Example
If a patient JSON containing only survey questions is evaluated in clinical mode:
```bash
python src/predict_patient.py --mode clinical --input examples/low_risk_patient.json
```
**Output:**
```
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
[Notice] Clinical laboratory measurements (e.g. Glucose, Cholesterol, resting BP) are absent.
[Graceful Fallback] Executing Survey Mode (Track A - CDC BRFSS Population Risk) instead.
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
```

### Clinical Output Sample
```
================================================================================
PATIENT RISK ASSESSMENT SUMMARY: CLINICAL MODE (TRACK B - LAB TESTS) (ID: CLIN-HIGH-001)
================================================================================

Diabetes risk: HIGH (probability 0.91)
  1. Glucose = 185 mg/dL -> raises risk (+2.23)
  2. BMI = 38.5 kg/m² -> raises risk (+0.52)
  3. Age = 62 years -> raises risk (+0.43)
  Lowering risk: Pregnancies = 0 (-0.41)

Heart Disease risk: HIGH (probability 0.98)
  1. ca = 2 vessels -> raises risk (+0.38)
  2. thalach = 115 bpm -> raises risk (+0.32)
  3. thal = Reversible defect -> raises risk (+0.30)
  Lowering risk: fbs = >120 mg/dL (-0.03)

--------------------------------------------------------------------------------
CLINICAL SAFETY NOTICE:
These are key contributing factors from a statistical model, not proven causes, and this is not a medical diagnosis.
================================================================================
```

---

## 5. Interactive Demo Web Application (`app/streamlit_app.py`)

Launch the web app locally:
```bash
streamlit run app/streamlit_app.py
```
Or run headlessly for test verification:
```bash
streamlit run app/streamlit_app.py --server.headless true
```

**Features:**
- Toggle between **Survey Mode (Track A)** and **Clinical Mode (Track B)**.
- Single-click buttons to load **Low Risk**, **Medium Risk**, and **High Risk** sample profiles.
- Form inputs dynamically bound to ranges and units from `config/feature_schema.yaml`.
- Real-time display of calibrated risk probabilities, color-coded badges, Top 3 risk-increasing factors, Top 2 risk-decreasing factors, and interactive SHAP waterfall plots.
- Prominent clinical disclaimer banners.

---

## 6. Real Computed Results & Validation Summary

### Champion Models (`outputs/tables/champion_models.csv`)
- **Track A Diabetes:** Logistic Regression (`clf__C: 0.01`). Test ROC-AUC = **0.8190**, Test Recall = **74.79%** (operating threshold $\tau^* = 0.1473$), Test PR-AUC = **0.3913**, Test F1 = **0.4431**.
- **Track A Heart Disease:** Logistic Regression (`clf__C: 0.01`). Test ROC-AUC = **0.8458**, Test Recall = **75.52%** (operating threshold $\tau^* = 0.1101$), Test PR-AUC = **0.3594**, Test F1 = **0.3900**.
- **Track B Diabetes:** Logistic Regression (`clf__C: 0.10`). Test ROC-AUC = **0.8081**, Test Recall = **74.07%** (operating threshold $\tau^* = 0.3318$), Test PR-AUC = **0.6639**, Test F1 = **0.6452**.
- **Track B Heart Disease:** Logistic Regression (`clf__C: 1.00`). Test ROC-AUC = **0.9275**, Test Recall = **71.43%** (operating threshold $\tau^* = 0.5672$), Test PR-AUC = **0.9423**, Test F1 = **0.8333**.

### Bootstrap 95% Confidence Intervals (1,000 Resamples on Held-Out Test Sets)
- **Track A Diabetes:** Recall: `[0.7374, 0.7578]`, ROC-AUC: `[0.8142, 0.8238]`, F1: `[0.4350, 0.4506]`.
- **Track A Heart:** Recall: `[0.7435, 0.7669]`, ROC-AUC: `[0.8409, 0.8510]`, F1: `[0.3812, 0.3986]`.
- **Track B Diabetes:** Recall: `[0.6296, 0.8462]`, ROC-AUC: `[0.7342, 0.8689]`.
- **Track B Heart:** Recall: `[0.5357, 0.8750]`, ROC-AUC: `[0.8495, 0.9822]`.

### Calibrated Risk Tiers Monotonicity (`config/risk_tiers.yaml`)
Observed disease rates on held-out test sets rise strictly Low < Medium < High:
- **Track A Diabetes:** Low (**2.69%**) < Medium (**17.26%**) < High (**40.83%**). High tier lift = **2.93x** baseline. True positive capture in Med+High = **90.64%**.
- **Track A Heart:** Low (**1.51%**) < Medium (**11.36%**) < High (**33.60%**). High tier lift = **3.57x** baseline. True positive capture in Med+High = **91.32%**.
- **Track B Diabetes:** Low (**9.09%**) < Medium (**48.33%**) < High (**67.86%**).
- **Track B Heart:** Low (**14.81%**) < Medium (**67.86%**) < High (**100.0%**).

### Longitudinal External Validation (BRFSS 2022 Cohort, $N=60,000$)
- **External ROC-AUC:** **0.8283** (PR-AUC: **0.3186**).
- **Finding:** Demonstrates robust discriminative power across 7 years of survey administration without retraining. Missing questionnaire items (`HighBP`, `HighChol`) were imputed to population baselines, reducing raw sensitivity to $45.14\%$ at the strict 2015 cutoff.

### Feature Attribution Stability Across Random Seeds
- **Diabetes:** **92.83%** Top-3 feature overlap across random seeds (Jaccard: `0.8925`).
- **Heart Disease:** **91.17%** Top-3 feature overlap across random seeds (Jaccard: `0.8685`).

---

## 7. Repository File Map

```
├── app/
│   └── streamlit_app.py              # Interactive clinical web demo app
├── config/
│   ├── feature_schema.yaml           # Patient input feature schema & valid clinical ranges
│   ├── model_config.yaml             # Modeling tracks, CV rules, and leakage constraints
│   └── risk_tiers.yaml               # Calibrated clinical risk tier cutoffs
├── data/
│   ├── clean/                        # Cleaned data files from Stages 1 & 2
│   ├── raw/                          # Raw BRFSS, Pima, Cleveland, and 2022 cohorts
│   └── model_ready/                  # Train/test partitions & split indices
├── docs/
│   └── modeling_decisions.md         # Stage 3 methodology & safeguards manifesto
├── examples/
│   ├── low_risk_patient.json         # Prototypical Low Risk survey profile
│   ├── medium_risk_patient.json      # Prototypical Medium Risk survey profile
│   ├── high_risk_patient.json        # Prototypical High Risk survey profile
│   ├── clinical_patient_low.json     # Prototypical Low Risk lab profile
│   ├── clinical_patient_medium.json  # Prototypical Medium Risk lab profile
│   ├── clinical_patient_high.json    # Prototypical High Risk lab profile
│   └── comprehensive_patient_both.json # Dual-track profile (survey + labs)
├── models/
│   ├── chronic_disease_predictor/    # Packaged Chronic Disease Predictor Model
│   │   ├── predictor.py              # Main ChronicDiseasePredictor class
│   │   ├── model_card.md             # Model Card with verified clinical metrics
│   │   ├── feature_schema.yaml       # Packaged feature validation schema
│   │   ├── risk_tiers.yaml           # Packaged calibrated risk tiers
│   │   └── *.joblib                  # Champion model artifacts (symlinked)
│   ├── all_models/
│   │   ├── track_a_survey/           # All 6 Track A candidate models
│   │   ├── track_b_clinical/         # All 6 Track B candidate models
│   │   └── stage2_unsupervised/      # Stage 2 unsupervised health profiles & docs
│   └── MODEL_INDEX.md                # Master catalog of all 12 candidate models
├── outputs/
│   ├── audit/                        # Phase 0 & Phase 5 audit reports & checklist
│   ├── figures/                      # High-res ROC, PR, calibration, SHAP, and subgroup plots
│   ├── tables/                       # Comparison metrics, champions, CIs, and validation tables
│   └── patient_predictions/          # Output JSON records and patient waterfall PNGs
├── src/
│   ├── 03_model_preprocessing.py     # Preprocessing & split generation
│   ├── 04_train_models.py            # Model training & calibration pipeline
│   ├── 05_evaluate_models.py         # Evaluation, champion selection & tier tuning
│   ├── 06_explainability_shap.py     # Track A SHAP global & local feature attributions
│   ├── 07_track_b_shap.py            # Track B SHAP explanations & repeated CV metrics
│   ├── 08_validation.py              # Robust validation, CIs, subgroups, baselines, stability
│   ├── predict_patient.py            # Production patient triage & CLI tool
│   └── risk_utils.py                 # Core validation & tier assignment utilities
├── tests/
│   ├── test_stage3_prep.py           # Pre-modeling contracts & schema unit tests
│   ├── test_final_system.py          # End-to-end model, prediction, and tier tests
│   ├── test_phase3_validation.py     # Clinical mode, fallback, CI files, and leakage tests
│   └── test_chronic_disease_predictor.py # Verification of packaged predictor & index
├── requirements.txt                  # Pinned production dependencies
├── run_all.sh                        # Automated one-command execution pipeline
└── README.md                         # Complete project documentation
```

---

## 8. Troubleshooting & FAQ

### Issue: `ModuleNotFoundError: No module named 'streamlit'`
**Fix:** Ensure you are using the project's virtual environment:
```bash
source .venv/bin/activate
pip install -r requirements.txt
```

### Issue: `[Validation Error] Patient input rejected`
**Cause:** The patient entry violates physiological bounds defined in `config/feature_schema.yaml` (e.g., `Glucose < 40 mg/dL` or `BMI > 90 kg/m²`).  
**Fix:** Verify patient readings against clinical standard ranges. For missing optional biometrics, leave the fields omitted in JSON and default population values will be imputed.

### Issue: Why does clinical mode fall back to survey mode?
**Cause:** The input JSON contains survey indicators (e.g., `HighBP`, `HighChol`) but omits laboratory measurements (e.g., `Glucose`, `chol`, `trestbps`).  
**Behavior:** The system detects missing lab tests and automatically falls back to Survey Mode (Track A) to provide safe triage rather than crashing. To run in clinical mode, provide lab measurements from `examples/clinical_patient_high.json`.

### Issue: Streamlit Headless Mode Warning in Containerized / SSH Environments
**Fix:** Run Streamlit with the headless flag enabled:
```bash
streamlit run app/streamlit_app.py --server.headless true --server.port 8501
```

---

## 9. Clinical Safety Disclaimer

> **MANDATORY CLINICAL SAFETY NOTICE:**  
> These are key contributing factors from a statistical machine learning model, not proven medical causes, and this tool is not a medical diagnosis. Predictions are intended strictly for clinical decision support, risk stratification, and triage facilitation under the supervision of qualified healthcare professionals.
