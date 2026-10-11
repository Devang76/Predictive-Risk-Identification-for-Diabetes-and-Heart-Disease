# Data Cleaning & Quality Audit Log

**Project:** Predictive Risk Identification for Diabetes and Heart Disease  
**Pipeline Step:** 02_clean  
**Raw Data Sources:** `data/raw/diabetes.csv`, `data/raw/heart_cleveland_upload.csv`  
**Cleaned Data Targets:** `data/clean/pima_clean.csv`, `data/clean/cleveland_clean.csv`  

---

## 1. Executive Summary

This document explains every data quality check, biological validation rule, and cleaning decision applied to both the Pima Indians Diabetes and Cleveland Heart Disease datasets. All raw files were treated as read-only and preserved intact.

| Metric | Pima Indians Diabetes | Cleveland Heart Disease |
| :--- | :---: | :---: |
| **Raw File** | `data/raw/diabetes.csv` | `data/raw/heart_cleveland_upload.csv` |
| **Clean File** | `data/clean/pima_clean.csv` | `data/clean/cleveland_clean.csv` |
| **Total Rows** | 768 | 297 |
| **Total Features** | 9 | 14 |
| **Duplicate Rows** | 0 | 0 |
| **Explicit Missing (Before)** | 0 | 0 |
| **Missing Cells (After Cleaning)** | **652** | **0** |
| **Rows with >= 1 Missing Value** | 376 (48.96%) | 0 (0.0%) |
| **Target Variable** | `Outcome` (0 = 65.1%, 1 = 34.9%) | `condition` (0 = 53.87%, 1 = 46.13%) |

---

## 2. Pima Indians Diabetes (`data/raw/diabetes.csv`)

### A. The "Hidden Missing Values" Problem
At raw ingestion, Pandas reported **0 missing values** (`NaN`). However, exploratory domain inspection revealed that numeric `0` was used as a recording placeholder for uncollected clinical tests.

In a living human:
1. **Glucose cannot be 0 mg/dL** (causes fatal hypoglycemia).
2. **Blood Pressure cannot be 0 mmHg** (absence of blood circulation).
3. **Skinfold Thickness cannot be 0 mm** (biological impossibility).
4. **Serum Insulin cannot be 0 μU/mL** (basal insulin is always present in living humans).
5. **BMI cannot be 0 kg/m²** (a person must have mass and height).

Conversely, **`Pregnancies == 0` is completely valid** (nulliparous women who have never given birth). Therefore, zeros in `Pregnancies` were preserved.

### B. Zero Replacement Audit
The following impossible zeros were replaced with `np.nan`:

| Column | Zeros Replaced | Total Rows | % Missingness | Clinical Meaning |
| :--- | :---: | :---: | :---: | :--- |
| **Glucose** | 5 | 768 | 0.65% | Fasting plasma glucose unmeasured |
| **BloodPressure** | 35 | 768 | 4.56% | Diastolic blood pressure unrecorded |
| **BMI** | 11 | 768 | 1.43% | Height or weight unmeasured |
| **SkinThickness** | 227 | 768 | 29.56% | Triceps skin fold omitted |
| **Insulin** | 374 | 768 | 48.70% | 2-hour serum insulin test omitted |
| **Total Missing Cells** | **652** | - | - | 376 rows have at least one NaN |

### C. Critical Decision: No Imputation During Cleaning
- **Decision:** `pima_clean.csv` preserves `np.nan` values and performs **zero imputation** at this stage.
- **Why:** Imputing missing values on the entire dataset prior to splitting into train/test sets causes **data leakage**. Test data information would leak into the training distribution via global means/medians.
- **Where Imputation Will Happen:** Inside a scikit-learn `Pipeline` (e.g., using `SimpleImputer` or `KNNImputer`) fitted *exclusively* on the training split during cross-validation.

### D. Deep Dive: Handling Heavy Missingness in Insulin (~49%) and SkinThickness (~30%)
- **The Challenge:** Dropping every row with missing data (complete-case analysis) would discard **376 out of 768 rows (48.96%)**, severely cutting our sample size in half and inducing severe selection bias.
- **Modeling Strategy:**
  1. **Tree-Based Models (XGBoost, LightGBM, CatBoost):** These algorithms natively branch missing values into dedicated split directions (missing-incorporated-in-splits). They learn patterns directly without artificial imputation.
  2. **Imputation with Missing Indicators:** For linear models (Logistic Regression) or distance-based models (KNN, SVM), we will use `SimpleImputer(strategy='median', add_indicator=True)`. The added binary indicator feature explicitly informs the model that the value was unobserved, capturing potentially informative missingness.
  3. **Ablation Study:** During feature selection and modeling, we will benchmark models trained **with** vs. **without** the `Insulin` and `SkinThickness` columns to ensure high missingness does not degrade generalization.

### E. Post-Cleaning Range & Anomaly Audit
- `Glucose`: 44.0 to 199.0 mg/dL (physiologically valid range).
- `BloodPressure`: Min = 24.0 mmHg, Max = 122.0 mmHg. A diastolic BP of 24 mmHg is extraordinarily low (profound hypotension/measurement artifact) but retained for modeling robustness.
- `SkinThickness`: Max = 99.0 mm (9.9 cm is an extreme outlier for triceps skinfold; retained, but tree models or robust scaling will prevent undue influence).
- `Insulin`: Max = 846.0 μU/mL (extreme hyperinsulinemia outlier; retained).
- `BMI`: 18.2 to 67.1 kg/m² (valid range from normal to Class III severe obesity).

---

## 3. Cleveland Heart Disease (`data/raw/heart_cleveland_upload.csv`)

### A. Data Integrity & Categorical Encodings
- **Rows & Duplicates:** 297 unique patient rows; 0 duplicates; 0 missing values.
- **Class Balance:** 160 patients with no heart disease (53.87%) vs. 137 with heart disease (46.13%). Well-balanced binary distribution.
- **Categorical Code Designation:**
  - `cp` (chest pain type: 0, 1, 2, 3)
  - `restecg` (resting ECG results: 0, 1, 2)
  - `slope` (slope of peak exercise ST segment: 0, 1, 2)
  - `ca` (number of major vessels colored by fluoroscopy: 0, 1, 2, 3)
  - `thal` (thallium stress test result: 0, 1, 2)
  - `sex` (0 = female, 1 = male)
  - `fbs` (fasting blood sugar > 120 mg/dl: 0 = false, 1 = true)
  - `exang` (exercise-induced angina: 0 = no, 1 = yes)
  *Note:* In modeling, nominal variables (`cp`, `restecg`, `thal`) should be one-hot encoded; binary/ordinal variables can remain as integer codes.

### B. Continuous Measurements & Outlier Audit
- `age`: 29 to 77 years.
- `trestbps`: 94 to 200 mm Hg resting blood pressure.
- `thalach`: 71 to 202 bpm maximum heart rate.
- `oldpeak`: 0.0 to 6.2 mm ST depression.
- `chol`: 126 to 564 mg/dL serum cholesterol.

### C. Cholesterol Outlier (> 500 mg/dL)
- Record identified:
  - **Row Index 79**: Age 67, Female, Resting BP 115 mmHg, Serum Cholesterol **564 mg/dL**, Heart Disease Condition = 0.
- **Clinical & Analytical Decision:**
  - A serum cholesterol of 564 mg/dL is clinically extreme but biologically genuine (consistent with severe familial hypercholesterolemia).
  - The row is **kept intact** in `data/clean/cleveland_clean.csv`.
  - Machine learning models using distance metrics will use `RobustScaler` (median and IQR) rather than `StandardScaler` so this single patient does not distort feature scaling.

---

## 4. CDC BRFSS 2015 Health Indicators (`data/raw/diabetes_binary_health_indicators_BRFSS2015.csv`)

### A. Dataset Overview & Multi-Target Context
The Behavioral Risk Factor Surveillance System (BRFSS) 2015 dataset contains large-scale annual health survey data collected by the CDC. Uniquely, this dataset contains **two disease targets**:
1. **Primary Target (`Diabetes_binary`)**:
   - `0.0` (No Diabetes): **218,334** (86.07%)
   - `1.0` (Diabetes or Prediabetes): **35,346** (13.93%)
   - *Class Imbalance Ratio:* ~6.2 to 1 (reflects true general population prevalence).
2. **Secondary Target (`HeartDiseaseorAttack`)**:
   - `0.0` (No Heart Disease or Myocardial Infarction): **229,787** (90.58%)
   - `1.0` (Reported Heart Disease or Attack): **23,893** (9.42%)
   - *Class Imbalance Ratio:* ~9.6 to 1.

### B. Missing Values
- **Explicit NaN Count:** **0** across all 22 columns.
- **Hidden Zero Check:** Unlike the Pima clinical dataset, zeros in this dataset represent standard binary survey answers (`0 = No`, `1 = Yes`). Zero does not indicate missing values here.

### C. Critical Decision: Retaining Duplicate Rows (Do Not Drop)
- **Duplicate Count:** **24,206 identical rows (9.54%)** out of 253,680 total records.
- **Decision:** **All duplicate rows are strictly retained in `data/clean/brfss_clean.csv`.**
- **Plain-English Rationale:**
  - In a survey of 253,680 people covering 22 discrete features (where 16 are simple binary Yes/No flags and the remaining 6 are small integer ordinal tiers), it is completely expected that multiple distinct respondents will report the exact same combination of demographic and lifestyle responses.
  - For example, two distinct individuals who are both non-smokers, physically active, female, within age category 9 (60-64 years), with college education and high income, will have identical rows.
  - Dropping these rows would discard real people, distort the true population prevalence, and artificially bias risk estimates away from the most common real-world demographic profiles.

### D. Feature Range and Validity Audit
Every feature with predefined survey coding was audited for boundary compliance:
- **`Age` (1 to 13):**
  - **Important Clarification:** In BRFSS, `Age` is **NOT** chronological age in years. It is an **ordinal 5-year age bracket** (`1 = 18-24`, `2 = 25-29`, `3 = 30-34`, ..., `13 = 80 or older`).
  - **Audit Result:** Strictly bounded between 1.0 and 13.0 across all 253,680 rows. Zero out-of-range entries.
- **`GenHlth` (1 to 5):**
  - Self-reported general health scale (`1 = Excellent`, `2 = Very Good`, `3 = Good`, `4 = Fair`, `5 = Poor`).
  - **Audit Result:** Strictly bounded between 1.0 and 5.0. Zero out-of-range entries.
- **`Income` (1 to 8):**
  - Ordinal income brackets (`1 = Less than $10,000` up to `8 = $75,000 or more`).
  - **Audit Result:** Strictly bounded between 1.0 and 8.0. Zero out-of-range entries.
- **`Education` (1 to 6):**
  - Education levels (`1 = Never attended / Kindergarten only` up to `6 = 4+ years of college`).
  - **Audit Result:** Strictly bounded between 1.0 and 6.0. Zero out-of-range entries.
- **`BMI` (12.0 to 98.0 kg/m²):**
  - Minimum is 12.0 kg/m² (severe underweight).
  - Maximum is 98.0 kg/m².
  - **Extreme Outlier Flag (BMI > 60):** **805 respondents (0.32%)** have BMI > 60 kg/m² (ranging from 61.0 to 98.0 kg/m²).
  - **Decision:** These represent genuine cases of Class III super-morbid obesity documented in US health surveillance. They are **kept in the dataset** to ensure real-world epidemiological validity, with tree-based models and robust feature transformations protecting model stability.

---

## 5. Summary Table of Files Generated

- [pima_clean.csv](file:///Users/devangmestry/Predictive Risk Identification for Diabetes and Heart Disease /data/clean/pima_clean.csv)
- [cleveland_clean.csv](file:///Users/devangmestry/Predictive Risk Identification for Diabetes and Heart Disease /data/clean/cleveland_clean.csv)
- [brfss_clean.csv](file:///Users/devangmestry/Predictive Risk Identification for Diabetes and Heart Disease /data/clean/brfss_clean.csv)
- [cleaning_summary.csv](file:///Users/devangmestry/Predictive Risk Identification for Diabetes and Heart Disease /outputs/tables/cleaning_summary.csv)
