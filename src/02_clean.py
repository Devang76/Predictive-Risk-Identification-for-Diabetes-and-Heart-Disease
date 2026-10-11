"""
src/02_clean.py
Data Cleaning & Quality Audit Pipeline
Predictive Risk Identification for Diabetes and Heart Disease

Tasks:
1. PIMA Indians Diabetes (data/raw/diabetes.csv):
   - Audit shape, missing values (NaN), duplicates, describe().
   - Detect biologically impossible 0s in Glucose, BloodPressure, SkinThickness, Insulin, BMI.
   - Count zeros per column (with percentages) and replace with np.nan. Keep Pregnancies 0s intact.
   - No imputation (deferred to modeling pipeline to prevent data leakage).
   - Audit min/max after cleaning and flag implausible extremes.
   - Save to data/clean/pima_clean.csv (NaNs preserved).

2. Cleveland Heart Disease (data/raw/heart_cleveland_upload.csv):
   - Audit shape, missing values, duplicates, class balance.
   - Confirm dtypes and designate categorical feature codes.
   - Audit min/max of continuous features (age, trestbps, chol, thalach, oldpeak).
   - Flag clinical extremes (chol > 500 mg/dL outlier candidate; retained and reported).
   - Save to data/clean/cleveland_clean.csv.

3. Outputs:
   - outputs/tables/cleaning_summary.csv
   - reports/cleaning_log.md
"""

from pathlib import Path
import numpy as np
import pandas as pd

# Define paths relative to repository root
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
CLEAN_DIR = BASE_DIR / "data" / "clean"
TABLES_DIR = BASE_DIR / "outputs" / "tables"
REPORTS_DIR = BASE_DIR / "reports"

# Ensure target directories exist
CLEAN_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

PIMA_RAW_PATH = RAW_DIR / "diabetes.csv"
CLEVELAND_RAW_PATH = RAW_DIR / "heart_cleveland_upload.csv"


def clean_pima():
    print("=" * 80)
    print(" 1. PIMA INDIANS DIABETES CLEANING & AUDIT")
    print(f" Source: {PIMA_RAW_PATH.relative_to(BASE_DIR)}")
    print("=" * 80)

    # Load raw data - never modify raw files
    df_raw = pd.read_csv(PIMA_RAW_PATH)
    rows_initial, cols_initial = df_raw.shape
    missing_initial = int(df_raw.isnull().sum().sum())
    duplicates = int(df_raw.duplicated().sum())

    print(f"\n[A] Initial Shape: {df_raw.shape} ({rows_initial} rows, {cols_initial} columns)")
    print(f"[B] Explicit NaN Values Before Cleaning: {missing_initial}")
    print(f"[C] Duplicate Rows: {duplicates}")

    print("\n[D] Summary Statistics Before Cleaning (.describe()):")
    print(df_raw.describe().round(2))

    # Identify biologically impossible zeros
    zero_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    print("\n[E] Zero Count & Percentage Audit for Biologically Impossible Fields:")
    print("    (Note: Pregnancies zeros are kept intact as valid nulliparous counts)")
    print(f"    {'Feature':<16} | {'Zeros':<8} | {'Total':<8} | {'% Missing':<12}")
    print("    " + "-" * 52)

    zero_stats = {}
    for col in zero_cols:
        cnt = int((df_raw[col] == 0).sum())
        pct = (cnt / rows_initial) * 100
        zero_stats[col] = (cnt, pct)
        print(f"    {col:<16} | {cnt:<8} | {rows_initial:<8} | {pct:>6.2f}%")

    # Replace impossible zeros with NaN
    df_clean = df_raw.copy()
    df_clean[zero_cols] = df_clean[zero_cols].replace(0, np.nan)
    missing_after = int(df_clean.isnull().sum().sum())
    rows_with_na = int(df_clean.isnull().any(axis=1).sum())

    print(f"\n[F] Missing Values After Replacing Zeros with NaN:")
    print(f"    Total NaN cells introduced: {missing_after}")
    print(f"    Rows with >= 1 missing value: {rows_with_na} ({(rows_with_na / rows_initial) * 100:.2f}%)")
    print("    Breakdown per column:")
    print(df_clean.isnull().sum())

    # Min/Max audit and implausible values check
    print("\n[G] Min / Max After Cleaning & Plausibility Audit:")
    print(f"    {'Feature':<25} | {'Min':<10} | {'Max':<10} | {'Clinical Assessment':<35}")
    print("    " + "-" * 88)

    plausibility_notes = {
        "Pregnancies": ("0", "17", "Valid; 17 is high parity but documented"),
        "Glucose": ("44.0", "199.0", "Valid; 44 mg/dL hypoglycemic, 199 high"),
        "BloodPressure": ("24.0", "122.0", "Min 24 mmHg is severe hypotension/flagged"),
        "SkinThickness": ("7.0", "99.0", "Max 99.0 mm (9.9cm) is an extreme caliper outlier"),
        "Insulin": ("14.0", "846.0", "Max 846 μU/mL is severe hyperinsulinemia/outlier"),
        "BMI": ("18.2", "67.1", "Valid; 18.2 slight underweight to 67.1 severe class III obesity"),
        "DiabetesPedigreeFunction": ("0.078", "2.42", "Valid genetic risk score range"),
        "Age": ("21", "81", "Valid adult cohort age range"),
        "Outcome": ("0", "1", "Binary target: 0=Non-diabetic (65.1%), 1=Diabetic (34.9%)")
    }

    for col in df_clean.columns:
        c_min = df_clean[col].min()
        c_max = df_clean[col].max()
        val_min = f"{c_min:.3f}" if isinstance(c_min, float) else str(c_min)
        val_max = f"{c_max:.3f}" if isinstance(c_max, float) else str(c_max)
        assessment = plausibility_notes.get(col, ("", "", "Plausible"))[2]
        print(f"    {col:<25} | {val_min:<10} | {val_max:<10} | {assessment:<35}")

    # Save cleaned file (preserving NaNs)
    pima_out = CLEAN_DIR / "pima_clean.csv"
    df_clean.to_csv(pima_out, index=False)
    print(f"\n[H] Saved clean dataset to: {pima_out.relative_to(BASE_DIR)}")

    summary_info = {
        "dataset": "Pima Indians Diabetes",
        "file_raw": str(PIMA_RAW_PATH.name),
        "file_clean": str(pima_out.name),
        "target_col": "Outcome",
        "rows": rows_initial,
        "cols": cols_initial,
        "duplicates": duplicates,
        "missing_before": missing_initial,
        "missing_after": missing_after,
        "rows_with_missing_after": rows_with_na
    }

    return df_clean, summary_info, zero_stats


def clean_cleveland():
    print("\n" + "=" * 80)
    print(" 2. CLEVELAND HEART DISEASE CLEANING & AUDIT")
    print(f" Source: {CLEVELAND_RAW_PATH.relative_to(BASE_DIR)}")
    print("=" * 80)

    # Load raw data - never modify raw files
    df_raw = pd.read_csv(CLEVELAND_RAW_PATH)
    rows_initial, cols_initial = df_raw.shape
    missing_initial = int(df_raw.isnull().sum().sum())
    duplicates = int(df_raw.duplicated().sum())

    print(f"\n[A] Initial Shape: {df_raw.shape} ({rows_initial} rows, {cols_initial} columns)")
    print(f"[B] Explicit NaN Values: {missing_initial}")
    print(f"[C] Duplicate Rows: {duplicates}")

    # Class balance of target: condition
    cond_counts = df_raw["condition"].value_counts().to_dict()
    cond_pcts = (df_raw["condition"].value_counts(normalize=True) * 100).to_dict()
    print("\n[D] Target Class Balance ('condition'):")
    print(f"    0 (No Disease):    {cond_counts.get(0, 0)} ({cond_pcts.get(0, 0.0):.2f}%)")
    print(f"    1 (Heart Disease): {cond_counts.get(1, 0)} ({cond_pcts.get(1, 0.0):.2f}%)")

    # Confirm Data Types & Designate Categorical Codes
    print("\n[E] Column Data Types & Feature Categorization:")
    categorical_codes = ["cp", "restecg", "slope", "ca", "thal", "sex", "fbs", "exang"]
    continuous_features = ["age", "trestbps", "chol", "thalach", "oldpeak"]
    target_feature = ["condition"]

    print(f"    Continuous features ({len(continuous_features)}): {continuous_features}")
    print(f"    Categorical codes ({len(categorical_codes)}): {categorical_codes}")
    print(f"    Target ({len(target_feature)}): {target_feature}")

    # Min/Max Audit for Continuous Features and Outlier Inspection
    print("\n[F] Min / Max Audit of Continuous Clinical Measurements:")
    print(f"    {'Feature':<12} | {'Min':<8} | {'Max':<8} | {'Clinical Unit / Meaning'}")
    print("    " + "-" * 65)

    ranges = {
        "age": ("years", 29, 77),
        "trestbps": ("mm Hg resting BP", 94, 200),
        "chol": ("mg/dl serum cholesterol", 126, 564),
        "thalach": ("bpm maximum heart rate", 71, 202),
        "oldpeak": ("ST depression in mm", 0.0, 6.2)
    }

    for col in continuous_features:
        c_min = df_raw[col].min()
        c_max = df_raw[col].max()
        unit = ranges[col][0]
        val_min = f"{c_min:.1f}" if isinstance(c_min, float) else str(c_min)
        val_max = f"{c_max:.1f}" if isinstance(c_max, float) else str(c_max)
        print(f"    {col:<12} | {val_min:<8} | {val_max:<8} | {unit}")

    # Flag cholesterol candidate outliers
    chol_outliers = df_raw[df_raw["chol"] > 500]
    print(f"\n[G] Outlier Flagging (Serum Cholesterol > 500 mg/dL):")
    if not chol_outliers.empty:
        print(f"    Found {len(chol_outliers)} record(s) with chol > 500 mg/dL:")
        for idx, row in chol_outliers.iterrows():
            print(f"    - Row Index {idx}: Age={int(row['age'])}, Sex={int(row['sex'])} ({'Female' if row['sex']==0 else 'Male'}), "
                  f"Cholesterol={int(row['chol'])} mg/dL, Resting BP={int(row['trestbps'])} mmHg, Condition={int(row['condition'])}")
        print("    Clinical Decision: Retained in dataset (valid extreme familial hypercholesterolemia case). Reported in log.")
    else:
        print("    No records found with chol > 500 mg/dL.")

    # Save cleaned file
    df_clean = df_raw.copy()
    clev_out = CLEAN_DIR / "cleveland_clean.csv"
    df_clean.to_csv(clev_out, index=False)
    print(f"\n[H] Saved clean dataset to: {clev_out.relative_to(BASE_DIR)}")

    summary_info = {
        "dataset": "Cleveland Heart Disease",
        "file_raw": str(CLEVELAND_RAW_PATH.name),
        "file_clean": str(clev_out.name),
        "target_col": "condition",
        "rows": rows_initial,
        "cols": cols_initial,
        "duplicates": duplicates,
        "missing_before": missing_initial,
        "missing_after": missing_initial,
        "rows_with_missing_after": 0
    }

    return df_clean, summary_info, chol_outliers


def generate_cleaning_summary_table(summaries):
    table_path = TABLES_DIR / "cleaning_summary.csv"
    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(table_path, index=False)
    print("\n" + "=" * 80)
    print(f" 3. SAVED CLEANING SUMMARY TABLE -> {table_path.relative_to(BASE_DIR)}")
    print("=" * 80)
    print(summary_df.to_string(index=False))


def generate_cleaning_log_report(pima_summary, clev_summary, pima_zero_stats, chol_outliers):
    report_path = REPORTS_DIR / "cleaning_log.md"

    chol_row = chol_outliers.iloc[0] if not chol_outliers.empty else None
    chol_details = (
        f"- **Row Index {chol_row.name}**: Age {int(chol_row['age'])}, "
        f"{'Female' if chol_row['sex'] == 0 else 'Male'}, "
        f"Resting BP {int(chol_row['trestbps'])} mmHg, "
        f"Serum Cholesterol **{int(chol_row['chol'])} mg/dL**, "
        f"Heart Disease Condition = {int(chol_row['condition'])}."
        if chol_row is not None else "None"
    )

    report_content = f"""# Data Cleaning & Quality Audit Log

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
| **Total Rows** | {pima_summary['rows']} | {clev_summary['rows']} |
| **Total Features** | {pima_summary['cols']} | {clev_summary['cols']} |
| **Duplicate Rows** | {pima_summary['duplicates']} | {clev_summary['duplicates']} |
| **Explicit Missing (Before)** | {pima_summary['missing_before']} | {clev_summary['missing_before']} |
| **Missing Cells (After Cleaning)** | **{pima_summary['missing_after']}** | **{clev_summary['missing_after']}** |
| **Rows with >= 1 Missing Value** | {pima_summary['rows_with_missing_after']} ({pima_summary['rows_with_missing_after']/pima_summary['rows']*100:.2f}%) | 0 (0.0%) |
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
| **Glucose** | {pima_zero_stats['Glucose'][0]} | {pima_summary['rows']} | {pima_zero_stats['Glucose'][1]:.2f}% | Fasting plasma glucose unmeasured |
| **BloodPressure** | {pima_zero_stats['BloodPressure'][0]} | {pima_summary['rows']} | {pima_zero_stats['BloodPressure'][1]:.2f}% | Diastolic blood pressure unrecorded |
| **BMI** | {pima_zero_stats['BMI'][0]} | {pima_summary['rows']} | {pima_zero_stats['BMI'][1]:.2f}% | Height or weight unmeasured |
| **SkinThickness** | {pima_zero_stats['SkinThickness'][0]} | {pima_summary['rows']} | {pima_zero_stats['SkinThickness'][1]:.2f}% | Triceps skin fold omitted |
| **Insulin** | {pima_zero_stats['Insulin'][0]} | {pima_summary['rows']} | {pima_zero_stats['Insulin'][1]:.2f}% | 2-hour serum insulin test omitted |
| **Total Missing Cells** | **{pima_summary['missing_after']}** | - | - | 376 rows have at least one NaN |

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
  {chol_details}
- **Clinical & Analytical Decision:**
  - A serum cholesterol of 564 mg/dL is clinically extreme but biologically genuine (consistent with severe familial hypercholesterolemia).
  - The row is **kept intact** in `data/clean/cleveland_clean.csv`.
  - Machine learning models using distance metrics will use `RobustScaler` (median and IQR) rather than `StandardScaler` so this single patient does not distort feature scaling.

---

## 4. Summary Table of Files Generated

- [pima_clean.csv](file://{CLEAN_DIR / 'pima_clean.csv'})
- [cleveland_clean.csv](file://{CLEAN_DIR / 'cleveland_clean.csv'})
- [cleaning_summary.csv](file://{TABLES_DIR / 'cleaning_summary.csv'})
"""

    with open(report_path, "w") as f:
        f.write(report_content.strip() + "\n")

    print(f"\n[I] Generated detailed Markdown report -> {report_path.relative_to(BASE_DIR)}")


def main():
    pima_clean, pima_summary, pima_zero_stats = clean_pima()
    clev_clean, clev_summary, chol_outliers = clean_cleveland()

    summaries = [pima_summary, clev_summary]
    generate_cleaning_summary_table(summaries)
    generate_cleaning_log_report(pima_summary, clev_summary, pima_zero_stats, chol_outliers)

    print("\n" + "=" * 80)
    print(" ALL DATA CLEANING AND AUDIT TASKS COMPLETED SUCCESSFULLY.")
    print("=" * 80)


if __name__ == "__main__":
    main()
