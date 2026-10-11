# Project Revision Changelog: Mid-Report V2

**Document:** `mid_report_v2.pdf` (regenerated from `mid_report_v2.docx`)  
**Previous Version:** `mid_report.pdf` (13 pages)  
**Current Version:** `mid_report_v2.pdf` (**18 pages**, strictly satisfying the 5–20 page constraint)  
**Execution Environment:** Python 3 (`.venv`), random_state = 42  
**Date of Revision:** October 2026  

---

## 1. Summary of Changes

All requested modifications from the assignment requirements and user specification have been implemented, verified, and regenerated directly from the underlying data. No statistics or numbers were hand-typed or invented; all tables and metrics were recomputed by `src/project_pipeline.py`, formatted into `mid_report_v2.docx` by `src/build_report.py`, and compiled into `mid_report_v2.pdf` via Microsoft Word. The laboratory notebook was also synchronized and executed via `src/build_notebook.py` into `project_mid_report.ipynb`.

---

## 2. Detailed Number Changes (Old -> New)

| Section / Metric | Old Value in `mid_report.pdf` | New Value in `mid_report_v2.pdf` | Data Source & Rationale |
| :--- | :--- | :--- | :--- |
| **Section 2: Total Workspace Records** | "over 1.3 million records" | **1,590,066 records** across all 8 files; **699,877 records** across 4 selected files | Recomputed exact row counts from Table 0 inventory. |
| **Section 3.1: Pima Hidden Zeros** | 652 zeros | **652 zeros across 376 individual records** (Glucose: 5 [0.65%], BP: 35 [4.56%], Skin: 227 [29.56%], Insulin: 374 [48.70%], BMI: 11 [1.43%]) | Computed directly from `data/raw/diabetes.csv`. Nulliparous Pregnancies (111 rows, 14.45%) confirmed valid. |
| **Section 3.2: Pima Missing Table** | Combined generic missingness table | **Table 1A added**: Dedicated Pima table with clinical mechanisms and explicit handling decisions for each variable | Explicitly justified decisions for Insulin (48.70%) and SkinThickness (29.56%): MICE/KNN with indicator flag in ML; excluded from cross-study rules. |
| **Section 3.2: Structural Missingness Table** | Missing from report | **Table 1B added**: Full missingness per column across all 26,065 integrated records | Documents structural missingness by design: Cholesterol (98.86% missing), Glucose (97.07% missing), BP continuous (96.05% missing), Insulin (98.49% missing), Heart Disease (2.95% missing in Pima). |
| **Section 3.2: BRFSS 2022 Imputation State** | Ambiguous imputation scope | **Explicitly documented**: BRFSS 2022 is a standalone missing-data demonstration; **NO imputation** was applied to the 25,000 BRFSS 2015 transaction dataset used for mining | BRFSS 2015 dataset had 0 missing values across all 25,000 sampled transactions. |
| **Section 3.4: Numeric Variables Coverage** | 5 variables (Age, BMI, BP, Chol, Gluc) | **11 variables (Table 3)**: Added Insulin (394), SkinThickness (541), Pedigree Function (768), Pregnancies (768), MentHlth (25,000), PhysHlth (25,000) | Covers every continuous and discrete numeric variable used in the study with parametric and non-parametric statistics. |
| **Section 3.4: Categorical Variables Coverage** | 7 variables | **14 variables (Table 4)**: Added Smoker, PhysActivity, Fruits, Veggies, DiffWalk, Stroke, GenHlth (1-5), Income (1-8) | Complete accounting of all lifestyle, demographic, and binary outcome indicators appearing in mined rules. |
| **Section 3.5: Figure 4 Heart Disease Takeaway** | "~9.3% heart disease" | **9.89% heart disease prevalence** (2,503 positive cases / 25,297 non-null records; 9.46% in BRFSS sample alone) | Resolved inconsistency between Table 4 (9.89%) and Figure 4 takeaway; explicitly stated the non-null denominator (25,297). |
| **Section 3.5: Figure 5 Correlation Matrix** | `high_glucose` included (r = 0.97 with diabetes outcome) | **`high_glucose` dropped from heatmap** | Removed artificial collinearity artifact. Explained in takeaway that in BRFSS `high_glucose` was mapped directly from `Diabetes_binary` as an operational proxy. |
| **Section 3.5: Figure Titles Clipping** | Figures 3, 4, 5, 6, 7 titles clipped at upper canvas edge | **Zero clipping across all figures** (`top5_mean_rgb = 255.0` border padding verified) | Shortened/repositioned in-image titles, set `y=0.96-0.98`, adjusted top subplot margins (`top=0.84-0.91`), and saved with `bbox_inches="tight"`. |
| **Section 3.6: Normalization Table (Table 5)** | 3 variables (Age, BMI, BP) | **6 variables**: Added Cholesterol (297), Glucose (763), and Insulin (394) | Recomputed Min-Max ranges, Z-score parameters, and skewness reduction. |
| **Section 3.6: Insulin Log-Skewness Value** | Text: -0.08, Figure 6: -0.09 | **-0.09 reported consistently** across Table 5, Figure 6 title, and text | Computed exact skewness: -0.0878, which rounds to -0.09. |
| **Section 3.6: Age Log-Transformation Suitability** | Listed without suitability note | **Explicitly flagged as contraindicated**: Skewness worsens from **-0.36 to -1.01** | Added explanation: Raw age is already mildly left-skewed (-0.36); log1p compresses the right end, worsening asymmetry by 180%. Min-Max or Z-score must be used instead. |
| **Section 4: Binary Transaction Items Table** | Missing from report | **Table 6 added**: Complete inventory of all 15 binary transaction items | Grouped by type: Threshold-Discretized (4 items), One-Hot Indicator (1 item), Original Binary Survey / Outcome Indicators (10 items), with clinical thresholds and support. |
| **Section 4: Mining Scope Justification** | Missing justification | **Added clear statement**: Mining was run on BRFSS 2015 (N=25,000) because all 15 items were measured simultaneously | Explains why continuous lab thresholds (glucose, insulin, cholesterol) do not appear in population rules due to cross-cohort structural missingness. |
| **Section 4.3: Heart Disease Rule 1 Lift Interpretation** | "2.43-fold elevation in heart attack odds" | **"2.43-fold elevation in heart attack probability relative to baseline prevalence"** | Corrected statistical terminology: Lift is a ratio of probabilities (relative risk: rule confidence divided by baseline consequent prevalence), NOT an odds ratio. |
| **Section 4.4: Clinical Cohort Mining Benchmarks** | Absent | **Table 10 added**: Auxiliary frequent pattern mining on Pima and Cleveland cohorts | Pima rule: `{BMI>=30, Age>=35, Glucose>125} -> {Diabetes}` (Conf = 77.1%, Lift = 2.21). Cleveland rule: `{HighChol>=240, Male, ExAngina} -> {HeartDisease}` (Conf = 90.2%, Lift = 1.96). |

---

## 3. Updated Codebase Files

1. **`src/project_pipeline.py`:**
   - Preprocesses Pima (hidden zero conversion, Table 1A generation).
   - Generates structural missingness audit table (`integrated_missing_summary.csv`).
   - Computes comprehensive descriptive statistics for all 11 numeric variables (Table 3) and all 14 categorical variables (Table 4).
   - Generates high-resolution figures with unclipped titles, white-space margins, and drops `high_glucose` proxy from Figure 5.
   - Computes Table 5 normalization summary across 6 continuous features.
   - Inventories all 15 binary transaction items (Table 6).
   - Executes Apriori and FP-Growth benchmarks, support sensitivity analysis, and runs auxiliary mining on Pima and Cleveland cohorts.

2. **`src/build_report.py`:**
   - Builds `mid_report_v2.docx` with all new tables (Tables 0 through 10) and unclipped figures (Figures 1 through 7).
   - Automatically converts `mid_report_v2.docx` to `mid_report_v2.pdf` using native Microsoft Word via `docx2pdf`.
   - Programmatically verifies page count: **18 pages** (strictly between 5 and 20 pages).

3. **`src/build_notebook.py` & `project_mid_report.ipynb`:**
   - Assembles and executes the Jupyter notebook top-to-bottom without errors, reproducing all tables, figures, and markdown explanations in lockstep with the report.

---

## 4. Verification Check

- **File Created:** `mid_report_v2.pdf` (18 pages)
- **Word Source Created:** `mid_report_v2.docx`
- **Tables Present:** Table 0, Table 1A, Table 1B, Table 1C, Table 2, Table 3, Table 4, Table 5, Table 6, Table 7, Table 8, Table 9, Table 10 (Every table has a bold numbered caption).
- **Figures Present:** Figure 1, Figure 2, Figure 3, Figure 4, Figure 5, Figure 6, Figure 7 (Every figure has a bold numbered caption and clinical takeaway).
- **Page Count Bounds:** 5 <= 18 <= 20 : **PASSED [OK]**
- **Recomputation Integrity:** All numbers computed directly from raw data with `random_state = 42`. Zero hand-typed or invented statistics.
