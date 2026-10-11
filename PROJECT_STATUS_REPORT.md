# Project Audit & Technical Status Report
**Project Title:** Predictive Risk Identification for Diabetes and Heart Disease  
**Audit Date:** October 2026  
**Auditor Role:** Project Auditor & Technical Writer  
**Repository Working Directory:** `/Users/devangmestry/Predictive Risk Identification for Diabetes and Heart Disease `  

---

## Executive Summary

This report provides an objective, evidence-based project audit of the entire codebase, datasets, notebooks, generated outputs, figures, tables, and existing reports within the repository.

The ultimate objective of this project is to build machine learning models that:
1. **Predict patient risk levels** (e.g., Low vs. High risk) separately for **Diabetes** and **Heart Disease**.
2. **Explain the exact patient-level drivers** pushing any individual into the high-risk category (using SHAP or equivalent attribution methods), alongside global feature importance.
3. Provide an interactive way to input new patient data and immediately output their risk label and top contributing clinical factors.

### Current Overall Completion Estimate: **50%**
> **One-Line Justification:** The project has completed thorough, high-quality data cleaning, dataset integration, exploratory data analysis, frequent pattern mining, unsupervised clustering, dimensionality reduction, and outlier detection (Stages 1 & 2), but has **not yet implemented supervised predictive modeling, model comparison, evaluation metrics (F1/ROC-AUC/Recall), SHAP explainability, or the new-patient inference interface** (Stage 3).

---

## 1. Work Completed So Far

The project repository reflects two major completed project milestones:
- **Stage 1 (Mid-Report / Preprocessing & Mining):** Documented in [mid_report_v2.pdf](file:///Users/devangmestry/Predictive%20Risk%20Identification%20for%20Diabetes%20and%20Heart%20Disease%20/reports/mid_report_v2.pdf) (18 pages), [project_mid_report.ipynb](file:///Users/devangmestry/Predictive%20Risk%20Identification%20for%20Diabetes%20and%20Heart%20Disease%20/reports/project_mid_report.ipynb), and orchestrated by [src/project_pipeline.py](file:///Users/devangmestry/Predictive%20Risk%20Identification%20for%20Diabetes%20and%20Heart%20Disease%20/src/project_pipeline.py).
- **Stage 2 (Unsupervised Learning & Anomaly Diagnostics):** Documented in [stage2_report.pdf](file:///Users/devangmestry/Predictive%20Risk%20Identification%20for%20Diabetes%20and%20Heart%20Disease%20/stage2_report.pdf) (20 pages), [project_stage2_report.ipynb](file:///Users/devangmestry/Predictive%20Risk%20Identification%20for%20Diabetes%20and%20Heart%20Disease%20/project_stage2_report.ipynb), and orchestrated by [src/stage2_pipeline.py](file:///Users/devangmestry/Predictive%20Risk%20Identification%20for%20Diabetes%20and%20Heart%20Disease%20/src/stage2_pipeline.py).

Below is the itemized breakdown of each completed analytical task.

### 1.1 Data Loading and Description
- **Execution Files:** [src/project_pipeline.py](file:///Users/devangmestry/Predictive%20Risk%20Identification%20for%20Diabetes%20and%20Heart%20Disease%20/src/project_pipeline.py#L52-L104), [eda_analysis.py](file:///Users/devangmestry/Predictive%20Risk%20Identification%20for%20Diabetes%20and%20Heart%20Disease%20/eda_analysis.py).
- **Summary Table:** [outputs/tables/dataset_inventory.csv](file:///Users/devangmestry/Predictive%20Risk%20Identification%20for%20Diabetes%20and%20Heart%20Disease%20/outputs/tables/dataset_inventory.csv).
- **What Was Done:**
  - Complete inventory of all 8 CSV files across [data/raw/](file:///Users/devangmestry/Predictive%20Risk%20Identification%20for%20Diabetes%20and%20Heart%20Disease%20/data/raw) and [data/raw/Indicators of Heart Disease/](file:///Users/devangmestry/Predictive%20Risk%20Identification%20for%20Diabetes%20and%20Heart%20Disease%20/data/raw/Indicators%20of%20Heart%20Disease).
  - Calculated exact record counts: **1,590,066 total raw records** across all 8 files.
  - Selected 4 primary benchmarks (**699,877 records**):
    1. **UCI Pima Indians Diabetes:** [data/raw/diabetes.csv](file:///Users/devangmestry/Predictive%20Risk%20Identification%20for%20Diabetes%20and%20Heart%20Disease%20/data/raw/diabetes.csv) (768 rows, 9 columns; continuous clinical lab tests; 34.90% diabetes prevalence).
    2. **UCI Cleveland Heart Disease:** [data/raw/heart_cleveland_upload.csv](file:///Users/devangmestry/Predictive%20Risk%20Identification%20for%20Diabetes%20and%20Heart%20Disease%20/data/raw/heart_cleveland_upload.csv) (297 rows, 14 columns; cardiac catheterization & exercise testing; 46.13% disease prevalence).
    3. **CDC BRFSS 2015 Health Indicators:** [data/raw/diabetes_binary_health_indicators_BRFSS2015.csv](file:///Users/devangmestry/Predictive%20Risk%20Identification%20for%20Diabetes%20and%20Heart%20Disease%20/data/raw/diabetes_binary_health_indicators_BRFSS2015.csv) (253,680 rows, 22 columns; dual-target nationwide survey; 13.93% diabetes, 9.42% heart disease).
    4. **CDC BRFSS 2022 Heart Disease:** [data/raw/Indicators of Heart Disease/2022/heart_2022_with_nans.csv](file:///Users/devangmestry/Predictive%20Risk%20Identification%20for%20Diabetes%20and%20Heart%20Disease%20/data/raw/Indicators%20of%20Heart%20Disease/2022/heart_2022_with_nans.csv) (445,132 rows, 40 columns; used to benchmark real survey missing-data mechanisms).

### 1.2 Data Preprocessing (Cleaning, Missing Values, Integration, Statistics, Visualizations, Normalization)

#### A. Biological Auditing and Hidden Missing Values
- **Execution Files:** [src/02_clean.py](file:///Users/devangmestry/Predictive%20Risk%20Identification%20for%20Diabetes%20and%20Heart%20Disease%20/src/02_clean.py), [reports/cleaning_log.md](file:///Users/devangmestry/Predictive%20Risk%20Identification%20for%20Diabetes%20and%20Heart%20Disease%20/reports/cleaning_log.md).
- **Summary Table:** [outputs/tables/pima_missing_summary.csv](file:///Users/devangmestry/Predictive%20Risk%20Identification%20for%20Diabetes%20and%20Heart%20Disease%20/outputs/tables/pima_missing_summary.csv).
- **Key Results:**
  - Discovered that raw Pima had 0 explicit `NaN` values because clinical omissions were encoded as numeric `0`.
  - Identified **652 hidden zeros across 376 individual patient rows (48.96% of cohort)**:
    - `Glucose == 0`: 5 rows (0.65%)
    - `BloodPressure == 0`: 35 rows (4.56%)
    - `BMI == 0`: 11 rows (1.43%)
    - `SkinThickness == 0`: 227 rows (29.56%)
    - `Insulin == 0`: 374 rows (48.70%)
  - Replaced biologically impossible zeros with `np.nan` and saved clean version to [data/clean/pima_clean.csv](file:///Users/devangmestry/Predictive%20Risk%20Identification%20for%20Diabetes%20and%20Heart%20Disease%20/data/clean/pima_clean.csv).
  - Validated that `Pregnancies == 0` (111 rows, 14.45%) represents valid nulliparous women and preserved those zeros intact.
  - In Cleveland ([data/clean/cleveland_clean.csv](file:///Users/devangmestry/Predictive%20Risk%20Identification%20for%20Diabetes%20and%20Heart%20Disease%20/data/clean/cleveland_clean.csv)), verified 0 missing values across all 297 records, flagged 1 severe hypercholesterolemia outlier (Cholesterol = 564 mg/dL), and justified keeping it.

#### B. Survey Missing Data Audit & Duplicate Retention
- **Execution Files:** [src/02b_clean_brfss.py](file:///Users/devangmestry/Predictive%20Risk%20Identification%20for%20Diabetes%20and%20Heart%20Disease%20/src/02b_clean_brfss.py), [outputs/tables/missing_values_summary.csv](file:///Users/devangmestry/Predictive%20Risk%20Identification%20for%20Diabetes%20and%20Heart%20Disease%20/outputs/tables/missing_values_summary.csv).
- **Key Results:**
  - CDC Heart 2022 audit demonstrated that listwise deletion drops **199,110 rows (44.73% of data)**, creating severe selection bias. Preserving cases with observed target variables dropped only 0.69% of records.
  - In CDC BRFSS 2015 ([data/clean/brfss_clean.csv](file:///Users/devangmestry/Predictive%20Risk%20Identification%20for%20Diabetes%20and%20Heart%20Disease%20/data/clean/brfss_clean.csv)), **24,206 duplicate rows (9.54%)** were identified. A clinical rationale was established to **retain all duplicates**, as discrete survey responses from 253,680 individuals over 22 bounded questions naturally produce identical demographic/lifestyle profiles.

#### C. Dataset Integration & Structural Missingness Audit
- **Execution Files:** [src/project_pipeline.py](file:///Users/devangmestry/Predictive%20Risk%20Identification%20for%20Diabetes%20and%20Heart%20Disease%20/src/project_pipeline.py#L308-L389).
- **Summary Tables:** [outputs/tables/integration_summary.csv](file:///Users/devangmestry/Predictive%20Risk%20Identification%20for%20Diabetes%20and%20Heart%20Disease%20/outputs/tables/integration_summary.csv), [outputs/tables/integrated_missing_summary.csv](file:///Users/devangmestry/Predictive%20Risk%20Identification%20for%20Diabetes%20and%20Heart%20Disease%20/outputs/tables/integrated_missing_summary.csv).
- **Clean Output File:** [data/clean/integrated_dataset.csv](file:///Users/devangmestry/Predictive%20Risk%20Identification%20for%20Diabetes%20and%20Heart%20Disease%20/data/clean/integrated_dataset.csv) (**26,065 rows, 13 harmonized columns**).
- **Key Results:**
  - Vertically merged Pima ($N=768$), Cleveland ($N=297$), and a stratified sample of CDC BRFSS 2015 ($N=25,000$).
  - Mapped age brackets to estimated chronological midpoints (e.g., bracket 7 $\rightarrow$ 52.0 years).
  - Explicitly documented **structural missingness by design**:
    - `cholesterol` continuous: 98.86% missing (only in Cleveland)
    - `glucose` continuous: 97.07% missing (only in Pima)
    - `blood_pressure` continuous: 96.05% missing (only in Pima & Cleveland)
    - `insulin` continuous: 98.49% missing (only in Pima)
    - `heart_disease_outcome`: 2.95% missing (unassessed in Pima)

#### D. Descriptive Statistics & Visualizations
- **Summary Tables:** [outputs/tables/numeric_descriptive_stats.csv](file:///Users/devangmestry/Predictive%20Risk%20Identification%20for%20Diabetes%20and%20Heart%20Disease%20/outputs/tables/numeric_descriptive_stats.csv) (11 continuous features), [outputs/tables/categorical_descriptive_stats.csv](file:///Users/devangmestry/Predictive%20Risk%20Identification%20for%20Diabetes%20and%20Heart%20Disease%20/outputs/tables/categorical_descriptive_stats.csv) (14 discrete/binary features).
- **Generated Figures:**
  - [Figure 1: Histograms of Key Numeric Variables](file:///Users/devangmestry/Predictive%20Risk%20Identification%20for%20Diabetes%20and%20Heart%20Disease%20/outputs/figures/figure1_numeric_histograms.png)
  - [Figure 2: Fasting Glucose vs. BMI by Diabetes Diagnosis](file:///Users/devangmestry/Predictive%20Risk%20Identification%20for%20Diabetes%20and%20Heart%20Disease%20/outputs/figures/figure2_glucose_vs_bmi.png)
  - [Figure 3: Age vs. Cholesterol by Heart Disease Condition](file:///Users/devangmestry/Predictive%20Risk%20Identification%20for%20Diabetes%20and%20Heart%20Disease%20/outputs/figures/figure3_age_vs_cholesterol.png)
  - [Figure 4: Prevalence of Key Categorical Risk Factors and Outcomes](file:///Users/devangmestry/Predictive%20Risk%20Identification%20for%20Diabetes%20and%20Heart%20Disease%20/outputs/figures/figure4_categorical_distributions.png)
  - [Figure 5: Correlation Heatmap](file:///Users/devangmestry/Predictive%20Risk%20Identification%20for%20Diabetes%20and%20Heart%20Disease%20/outputs/figures/figure5_correlation_heatmap.png) (dropped `high_glucose` operational proxy to eliminate artificial $r=0.97$ circularity)
  - [Figure 6: Log Transformation of Serum Insulin](file:///Users/devangmestry/Predictive%20Risk%20Identification%20for%20Diabetes%20and%20Heart%20Disease%20/outputs/figures/figure6_log_transform_comparison.png)

#### E. Normalization & Transformation Analysis
- **Summary Table:** [outputs/tables/normalization_transformation_summary.csv](file:///Users/devangmestry/Predictive%20Risk%20Identification%20for%20Diabetes%20and%20Heart%20Disease%20/outputs/tables/normalization_transformation_summary.csv).
- **Key Results:**
  - Evaluated Min-Max scaling, Z-score standardization, and Natural Log (`log1p`) transformation across 6 continuous biometrics:
    - `Insulin`: Skewness improved from **+2.17 to -0.09** (highly recommended for linear models).
    - `BMI`: Skewness reduced from **+2.00 to +0.69**.
    - `Glucose`: Skewness reduced from **+0.53 to -0.06**.
    - `BloodPressure`: Skewness reduced from **+0.91 to +0.34**.
    - `Cholesterol`: Skewness reduced from **+1.12 to +0.06**.
    - `Age`: Explicitly identified as **contraindicated for log-transformation** because raw age is slightly left-skewed ($-0.36$), and `log1p` worsens skewness to **$-1.01$** (180% distortion increase).

### 1.3 Frequent Pattern Mining
- **Execution Files:** [src/project_pipeline.py](file:///Users/devangmestry/Predictive%20Risk%20Identification%20for%20Diabetes%20and%20Heart%20Disease%20/src/project_pipeline.py#L694-L897).
- **Summary Tables:**
  - [binary_items_inventory.csv](file:///Users/devangmestry/Predictive%20Risk%20Identification%20for%20Diabetes%20and%20Heart%20Disease%20/outputs/tables/binary_items_inventory.csv) (15 transaction items created with clinical cutoffs).
  - [support_variation_summary.csv](file:///Users/devangmestry/Predictive%20Risk%20Identification%20for%20Diabetes%20and%20Heart%20Disease%20/outputs/tables/support_variation_summary.csv) (Support sensitivity across 0.40, 0.30, 0.20, 0.10, 0.05).
  - [top_diabetes_rules_by_lift.csv](file:///Users/devangmestry/Predictive%20Risk%20Identification%20for%20Diabetes%20and%20Heart%20Disease%20/outputs/tables/top_diabetes_rules_by_lift.csv) & [top_diabetes_rules_by_confidence.csv](file:///Users/devangmestry/Predictive%20Risk%20Identification%20for%20Diabetes%20and%20Heart%20Disease%20/outputs/tables/top_diabetes_rules_by_confidence.csv).
  - [top_heart_disease_rules_by_lift.csv](file:///Users/devangmestry/Predictive%20Risk%20Identification%20for%20Diabetes%20and%20Heart%20Disease%20/outputs/tables/top_heart_disease_rules_by_lift.csv) & [top_heart_disease_rules_by_confidence.csv](file:///Users/devangmestry/Predictive%20Risk%20Identification%20for%20Diabetes%20and%20Heart%20Disease%20/outputs/tables/top_heart_disease_rules_by_confidence.csv).
  - [pima_mined_rules.csv](file:///Users/devangmestry/Predictive%20Risk%20Identification%20for%20Diabetes%20and%20Heart%20Disease%20/outputs/tables/pima_mined_rules.csv) & [cleveland_mined_rules.csv](file:///Users/devangmestry/Predictive%20Risk%20Identification%20for%20Diabetes%20and%20Heart%20Disease%20/outputs/tables/cleveland_mined_rules.csv).
- **Figure:** [Figure 7: Itemsets and Rules vs. Support](file:///Users/devangmestry/Predictive%20Risk%20Identification%20for%20Diabetes%20and%20Heart%20Disease%20/outputs/figures/figure7_itemsets_rules_vs_support.png).
- **Key Results:**
  - Proved exact itemset equivalence between Apriori and FP-Growth on 25,000 transactions at `min_support = 0.20` (both produced exactly 72 frequent itemsets).
  - FP-Growth executed faster ($0.0031\text{ s}$ vs. $0.0041\text{ s}$).
  - Top Population Diabetes Rule: `HighChol, BMI>=30, Age>=50 -> Diabetes` (Support: 5.08%, Confidence: 35.66%, **Lift: 2.58** vs. 13.93% baseline prevalence).
  - Top Population Heart Disease Rule: `HighChol, HighBP, Age>=50 -> HeartDisease` (Support: 5.34%, Confidence: 23.00%, **Lift: 2.43** vs. 9.46% baseline prevalence).
  - Clinical Lab Validation Rules:
    - Pima: `Age>=35, Glucose>125, BMI>=30 -> Diabetes` (Confidence: **77.06%**, **Lift: 2.21**).
    - Cleveland: `Male, HighChol>=240, ExAngina -> HeartDisease` (Confidence: **90.24%**, **Lift: 1.96**).

### 1.4 Unsupervised Clustering, Dimensionality Reduction & Outlier Detection
- **Execution Files:** [src/stage2_pipeline.py](file:///Users/devangmestry/Predictive%20Risk%20Identification%20for%20Diabetes%20and%20Heart%20Disease%20/src/stage2_pipeline.py), [src/build_stage2_report.py](file:///Users/devangmestry/Predictive%20Risk%20Identification%20for%20Diabetes%20and%20Heart%20Disease%20/src/build_stage2_report.py).
- **Output Tables:**
  - [stage2_clustering_metrics_comparison.csv](file:///Users/devangmestry/Predictive%20Risk%20Identification%20for%20Diabetes%20and%20Heart%20Disease%20/outputs/tables/stage2_clustering_metrics_comparison.csv)
  - [stage2_cluster_health_profiles.csv](file:///Users/devangmestry/Predictive%20Risk%20Identification%20for%20Diabetes%20and%20Heart%20Disease%20/outputs/tables/stage2_cluster_health_profiles.csv)
  - [stage2_clinical_cluster_profiles.csv](file:///Users/devangmestry/Predictive%20Risk%20Identification%20for%20Diabetes%20and%20Heart%20Disease%20/outputs/tables/stage2_clinical_cluster_profiles.csv)
  - [stage2_pca_loadings.csv](file:///Users/devangmestry/Predictive%20Risk%20Identification%20for%20Diabetes%20and%20Heart%20Disease%20/outputs/tables/stage2_pca_loadings.csv)
  - [stage2_clustering_reduced_vs_raw.csv](file:///Users/devangmestry/Predictive%20Risk%20Identification%20for%20Diabetes%20and%20Heart%20Disease%20/outputs/tables/stage2_clustering_reduced_vs_raw.csv)
  - [stage2_outlier_summary.csv](file:///Users/devangmestry/Predictive%20Risk%20Identification%20for%20Diabetes%20and%20Heart%20Disease%20/outputs/tables/stage2_outlier_summary.csv)
  - [stage2_outlier_removal_metrics.csv](file:///Users/devangmestry/Predictive%20Risk%20Identification%20for%20Diabetes%20and%20Heart%20Disease%20/outputs/tables/stage2_outlier_removal_metrics.csv)
  - [stage2_parameter_dictionary.csv](file:///Users/devangmestry/Predictive%20Risk%20Identification%20for%20Diabetes%20and%20Heart%20Disease%20/outputs/tables/stage2_parameter_dictionary.csv)
- **Figures:** [stage2_fig1_kmeans_elbow_silhouette.png](file:///Users/devangmestry/Predictive%20Risk%20Identification%20for%20Diabetes%20and%20Heart%20Disease%20/outputs/figures/stage2_fig1_kmeans_elbow_silhouette.png) through [stage2_fig10_outlier_removal_impact.png](file:///Users/devangmestry/Predictive%20Risk%20Identification%20for%20Diabetes%20and%20Heart%20Disease%20/outputs/figures/stage2_fig10_outlier_removal_impact.png).
- **Key Results:**
  - **Clustering Comparison:** Evaluated K-Means ($k=3$), Hierarchical Agglomerative (Ward linkage, $k=3$), and DBSCAN ($\epsilon=2.4$, $\text{minPts}=5$). K-Means ($k=3$) achieved Silhouette = 0.1534, Davies-Bouldin = 2.1061, Calinski-Harabasz = 860.45.
  - **Patient Phenotypes Discovered (BRFSS $N=5,000$):**
    - *Cluster 0 (Low Risk, 46.8%):* Avg Age 49.6, BMI 26.9, Hypertension 8.0%, High Cholesterol 18.1%, **Diabetes 3.2%**, **Heart Disease 1.8%**.
    - *Cluster 1 (Moderate Risk, 39.1%):* Avg Age 65.7, BMI 29.5, Hypertension 78.3%, High Cholesterol 67.6%, **Diabetes 21.6%**, **Heart Disease 14.4%**.
    - *Cluster 2 (High Risk / Multi-Morbidity, 14.1%):* Avg Age 59.1, BMI 31.0, 21.4 poor physical days/month, 13.0 poor mental days/month, **Diabetes 27.0%**, **Heart Disease 23.0%**.
  - **Dimensionality Reduction (PCA & t-SNE):**
    - PC1 (24.78% variance): Driven by self-reported health deterioration (`GenHlth` loading +0.47, `PhysHlth` +0.38, `Income` -0.38).
    - PC2 (13.91% variance): Driven by vascular aging (`Age` +0.57, `HighChol` +0.46, `HighBP` +0.42).
    - Top 6 components retain **73.98% of variance**.
    - Re-clustering on the 6 PCA components improved Silhouette score by **+33.4% (from 0.1534 to 0.2046)** and reduced Davies-Bouldin by **-19.0% (from 2.1061 to 1.7054)**.
  - **Outlier Detection (Isolation Forest & LOF):**
    - Identified 175 extreme outliers (3.5% contamination rate).
    - Outliers exhibited severe multi-morbidity: average BMI 33.4 (max 82.0), 24.9 days/month of poor physical health (vs. 3.4 for inliers), **Diabetes prevalence 29.7%** (vs. 13.2%), and **Heart Disease prevalence 25.1%** (vs. 9.1%).
    - Re-running clustering after outlier removal demonstrated centroid contraction toward realistic population medians.

### 1.5 Supervised Model Training and Evaluation
- **Status:** **NOT STARTED**
- **Evidence:** Verified by exhaustive search across all `.py` and `.ipynb` files. No code imports `LogisticRegression`, `DecisionTreeClassifier`, or `RandomForestClassifier`. No model training, hyperparameter tuning, train-test splitting, or metric computations exist in the repo.

### 1.6 Explainability Work (SHAP / Feature Attribution)
- **Status:** **NOT STARTED**
- **Evidence:** `shap==0.49.1` is listed in [requirements.txt](file:///Users/devangmestry/Predictive%20Risk%20Identification%20for%20Diabetes%20and%20Heart%20Disease%20/requirements.txt), and SHAP is described in the Stage 3 future roadmap in [stage2_report.pdf](file:///Users/devangmestry/Predictive%20Risk%20Identification%20for%20Diabetes%20and%20Heart%20Disease%20/stage2_report.pdf#L711) and [src/build_stage2_notebook.py](file:///Users/devangmestry/Predictive%20Risk%20Identification%20for%20Diabetes%20and%20Heart%20Disease%20/src/build_stage2_notebook.py#L431). However, no actual SHAP code, beeswarm plots, or waterfall plots have been implemented or executed.

---

## 2. Current State vs. The End Product (Gap Analysis)

| Requirement | Status | Evidence (File or Output) | What is Missing |
| :--- | :---: | :--- | :--- |
| **Clean, merged, model-ready dataset** | **Partial** | [data/clean/integrated_dataset.csv](file:///Users/devangmestry/Predictive%20Risk%20Identification%20for%20Diabetes%20and%20Heart%20Disease%20/data/clean/integrated_dataset.csv), [pima_clean.csv](file:///Users/devangmestry/Predictive%20Risk%20Identification%20for%20Diabetes%20and%20Heart%20Disease%20/data/clean/pima_clean.csv), [cleveland_clean.csv](file:///Users/devangmestry/Predictive%20Risk%20Identification%20for%20Diabetes%20and%20Heart%20Disease%20/data/clean/cleveland_clean.csv), [brfss_clean.csv](file:///Users/devangmestry/Predictive%20Risk%20Identification%20for%20Diabetes%20and%20Heart%20Disease%20/data/clean/brfss_clean.csv) | The merged dataset has 96–98% structural missingness across continuous lab values. Dedicated, leakage-free feature matrices ($X, y$) split into stratified train/test sets have not been created for supervised modeling. |
| **Risk prediction model for diabetes** | **Not started** | None | No supervised classification model (Logistic Regression, Decision Tree, Random Forest) has been fitted on diabetes labels. |
| **Risk prediction model for heart disease** | **Not started** | None | No supervised classification model has been fitted on heart disease labels. |
| **Model comparison & best-model selection** | **Not started** | None | No comparison table, cross-validation benchmarking, or selection logic exists between candidate model families. |
| **Evaluation with F1, ROC-AUC, Recall** | **Not started** | None | No classification metrics, PR-AUC, confusion matrices, or ROC curves have been generated for supervised predictions. |
| **Global feature importance** | **Partial** | [stage2_pca_loadings.csv](file:///Users/devangmestry/Predictive%20Risk%20Identification%20for%20Diabetes%20and%20Heart%20Disease%20/outputs/tables/stage2_pca_loadings.csv), [top_diabetes_rules_by_lift.csv](file:///Users/devangmestry/Predictive%20Risk%20Identification%20for%20Diabetes%20and%20Heart%20Disease%20/outputs/tables/top_diabetes_rules_by_lift.csv) | Global importance exists only via PCA component loadings and association rule lift. No supervised model feature importances (e.g., Gini importance, logistic odds ratios, or mean absolute SHAP values) exist. |
| **Patient-level explanation of which variable causes high risk** | **Not started** | None | No SHAP TreeExplainer / LinearExplainer, local waterfall plots, or patient attribution calculations exist. |
| **A way to input a new patient and get a risk label plus top contributing variables** | **Not started** | None | No interactive inference function, CLI tool, REST endpoint, or GUI/web app exists to accept user inputs and output risk predictions + explanations. |
| **Final report and presentation** | **Partial** | [stage2_report.pdf](file:///Users/devangmestry/Predictive%20Risk%20Identification%20for%20Diabetes%20and%20Heart%20Disease%20/stage2_report.pdf) (20 pages), [mid_report_v2.pdf](file:///Users/devangmestry/Predictive%20Risk%20Identification%20for%20Diabetes%20and%20Heart%20Disease%20/reports/mid_report_v2.pdf) (18 pages) | Mid-term and Stage 2 reports are complete, but the final project report (Stage 3) and presentation slides covering supervised models and SHAP explanations have not been authored. |

### Overall Completion Estimate: **50%**
*Justification:* The exploratory, data engineering, statistical, pattern mining, and unsupervised clustering/outlier foundations are 100% complete and rigorously documented, but the core predictive machine learning models, patient-level SHAP explanations, and interactive inference tools have not yet been started.

---

## 3. Key Findings So Far

### 3.1 Data Characteristics & Prevalence
- **Class Imbalance in CDC Survey Cohort ($N=253,680$):**
  - Diabetes prevalence is **13.93%** (35,346 positive cases; ~6.2 to 1 imbalance).
  - Heart disease prevalence is **9.42%** (23,893 positive cases; ~9.6 to 1 imbalance).
- **Clinical Cohorts:**
  - UCI Pima ($N=768$): **34.90%** diabetes prevalence (268 positive cases).
  - UCI Cleveland ($N=297$): **46.13%** heart disease prevalence (137 positive cases).

### 3.2 Most Influential Variables Identified So Far
1. **Age (Vascular & Cardiometabolic Aging):**
   - In PCA, Age has the single highest positive loading on **PC2 (+0.5740)**.
   - Appears in virtually all high-lift population association rules (`Age>=50`).
2. **Hypertension (`HighBP` / `BloodPressure`):**
   - High positive loading on **PC2 (+0.4215)** and **PC1 (+0.3034)**.
   - Present in 78.3% of Moderate-Risk cluster patients and 62.6% of High-Risk cluster patients, compared to only 8.0% of Low-Risk individuals.
3. **Hypercholesterolemia (`HighChol` / `chol`):**
   - High loading on **PC2 (+0.4574)**.
   - Combined with hypertension and age, increases heart disease probability by **2.43x baseline** (Lift: 2.43).
   - In Cleveland clinical testing, `HighChol >= 240 mg/dL` paired with exercise-induced angina predicted heart disease with **90.24% confidence**.
4. **General & Physical Impairment (`GenHlth`, `PhysHlth`, `DiffWalk`):**
   - Single highest loading on **PC1 (+0.4692 for GenHlth, +0.3838 for PhysHlth)**.
   - Delineates the extreme multi-morbidity cluster (Cluster 2) and accounts for the largest gap between normal inliers (3.4 poor physical days/month) and high-acuity outliers (24.9 days/month).
5. **Continuous Biomarkers (in Clinical Lab Subsets):**
   - **Glucose:** Values $>125\text{ mg/dL}$ in Pima predict diabetes with **77.06% confidence** when combined with $\text{BMI} \ge 30$ and $\text{Age} \ge 35$.
   - **Maximum Heart Rate (`thalach`) & ST Depression (`oldpeak`):** In Cleveland, patients in the high-risk cardiac cluster averaged an ST depression of $1.74\text{ mm}$ (vs. $0.53\text{ mm}$ for low-risk) and a lower peak heart rate of $133.8\text{ bpm}$ (vs. $161.9\text{ bpm}$).

---

## 4. Problems and Limitations

1. **Extreme Structural Missingness in Integrated Data (Cross-Study Gap):**
   - The unified file [data/clean/integrated_dataset.csv](file:///Users/devangmestry/Predictive%20Risk%20Identification%20for%20Diabetes%20and%20Heart%20Disease%20/data/clean/integrated_dataset.csv) has over **96% to 98% missingness** for continuous laboratory biometrics (glucose, insulin, continuous blood pressure, serum cholesterol) because the CDC BRFSS survey ($N=25,000$ in sample) only records binary self-reported items (`HighBP`, `HighChol`), while Pima and Cleveland only record lab tests.
   - *Impact on Modeling:* A single model cannot be trained naively on `integrated_dataset.csv` without either dropping all continuous lab features or suffering massive imputation artifacts. **Separate models must be trained per disease cohort or trained strictly on features common across datasets.**
2. **Severe Class Imbalance in Survey Data:**
   - Both target outcomes are heavily skewed in the population survey (~90% negative vs. ~10% positive). Standard accuracy will be misleading (a naive classifier predicting "No Disease" would achieve >90% accuracy). Supervised modeling must prioritize **Recall, F1-score, and ROC-AUC**, paired with class-weight balancing or SMOTE.
3. **Sample Size Disparity:**
   - UCI Cleveland ($N=297$) and UCI Pima ($N=768$) are very small clinical cohorts, whereas CDC BRFSS has 253,680 rows. Models trained on Cleveland risk high variance or overfitting without strong regularization.
4. **Target Proxy Circularity Risk (`high_glucose` in BRFSS):**
   - In earlier iterations of the pipeline, `high_glucose` was mapped directly from `Diabetes_binary` as an operational survey proxy, creating an artificial Pearson correlation of $r=0.97$. This was properly removed from the correlation heatmap ([Figure 5](file:///Users/devangmestry/Predictive%20Risk%20Identification%20for%20Diabetes%20and%20Heart%20Disease%20/outputs/figures/figure5_correlation_heatmap.png)), but it must **never be used as a predictive feature** in supervised diabetes models to prevent data leakage.
5. **Pima Heavy Missingness in Insulin (48.7%) and Skinfold (29.6%):**
   - Nearly half of Pima patients lack 2-hour serum insulin tests. Complete-case deletion reduces sample size to 392 rows. Tree-based models (which handle missingness natively) or median imputation with missing indicator flags must be used.
6. **Cross-Sectional Survey Limitations:**
   - The CDC BRFSS data is self-reported, exposing it to recall bias and lack of longitudinal follow-up.

---

## 5. Remaining Roadmap (Path to Expected End Product)

To complete the project from its current 50% state to 100%, the following 5 sequential phases are required:

```
[Phase 1: Feature Matrices & Leakage-Free Splitting]
                        │
                        ▼
[Phase 2: Supervised Model Training (LogReg, DT, RF)]
                        │
                        ▼
[Phase 3: Model Evaluation & Best Model Selection]
                        │
                        ▼
[Phase 4: Global & Local Explainability (SHAP)]
                        │
                        ▼
[Phase 5: Interactive Patient Inference Tool & Final Report]
```

### Phase 1: Feature Engineering & Dataset Preparation (Estimated Effort: 4–6 Hours)
- **Goal:** Create clean, leakage-free feature matrices ($X, y$) for **Diabetes** and **Heart Disease**.
- **Strategy:**
  - Build two dedicated datasets:
    1. **Population Model Datasets (from CDC BRFSS clean):** Using features measured across survey respondents (Age, BMI, HighBP, HighChol, Smoker, PhysActivity, GenHlth, etc.).
    2. **Clinical Benchmark Datasets:** Dedicated models trained on Pima (clinical diabetes) and Cleveland (clinical heart disease) utilizing continuous laboratory metrics.
  - Implement a stratified 80/20 train-test split with fixed `random_state=42`.
  - Encapsulate preprocessing (imputation + scaling) inside scikit-learn `Pipeline` objects fitted strictly on training folds to prevent data leakage.
- **Files to Create:** `src/03_model_preprocessing.py`, `data/clean/diabetes_model_ready.csv`, `data/clean/heart_model_ready.csv`.

### Phase 2: Supervised Model Training & Cross-Validation (Estimated Effort: 6–8 Hours)
- **Goal:** Train the planned model families with a separate model per disease:
  - **Logistic Regression** (with L2 regularization and balanced class weights)
  - **Decision Tree Classifier** (tuned `max_depth` and `min_samples_split` to avoid overfitting)
  - **Random Forest Classifier** (ensemble of 150+ trees with balanced sub-sampling)
- **Methodology:** 5-fold Stratified Cross-Validation on the training split. Tune key hyperparameters using `GridSearchCV` or `RandomizedSearchCV`.
- **Files to Create:** `src/04_train_models.py`, `models/diabetes_rf.joblib`, `models/diabetes_lr.joblib`, `models/heart_rf.joblib`, `models/heart_lr.joblib`.

### Phase 3: Comprehensive Evaluation & Best Model Selection (Estimated Effort: 5–6 Hours)
- **Goal:** Evaluate all models on held-out test sets using required metrics: **F1-Score**, **ROC-AUC**, and **Recall (Sensitivity)**.
- **Outputs:**
  - Model comparison summary table ranking candidate algorithms per disease.
  - ROC curves and Precision-Recall (PR) curves saved to `outputs/figures/`.
  - Confusion matrices highlighting the reduction of clinical false negatives.
  - Selection of the champion model for each disease.
- **Files to Create:** `src/05_evaluate_models.py`, `outputs/tables/model_comparison_metrics.csv`, `outputs/figures/stage3_roc_pr_curves.png`.

### Phase 4: Model Explainability via SHAP (Estimated Effort: 6–8 Hours)
- **Goal:** Fulfill the critical project requirement of explaining **which exact variable(s) push a patient into the high-risk category**.
- **Implementation:**
  - Compute SHAP values using `shap.TreeExplainer` for Random Forest and `shap.LinearExplainer` for Logistic Regression.
  - **Global Explainability:** Generate SHAP Summary Beeswarm plots showing feature importance and directional risk impact across the cohort.
  - **Patient-Level Explainability:** Generate SHAP Waterfall / Force plots for individual patient profiles (e.g., Patient A at 85% risk pushed by high BMI + hypertension vs. Patient B pushed by smoking + high cholesterol).
- **Files to Create:** `src/06_explainability_shap.py`, `outputs/figures/shap_diabetes_summary.png`, `outputs/figures/shap_heart_summary.png`, `outputs/figures/shap_patient_waterfall_examples.png`.

### Phase 5: Interactive Patient Triage Inference Tool & Final Deliverables (Estimated Effort: 6–8 Hours)
- **Goal:** Provide a practical interface to input a new patient's biometrics and immediately receive:
  1. Risk Classification (Low Risk vs. High Risk, with predicted probability percentage).
  2. The top 3 contributing variables pushing that specific patient toward the high-risk outcome.
- **Implementation:**
  - A standalone CLI script or lightweight web UI (e.g., Streamlit or interactive HTML/Python widget) accepting user input and calling the serialized pipeline + SHAP explainer.
- **Final Deliverables:**
  - Stage 3 Final Academic Report: `stage3_final_report.docx` and `stage3_final_report.pdf`.
  - Synchronized Jupyter Notebook: `project_final_stage3_report.ipynb`.
  - Project presentation slide deck / executive summary.
- **Files to Create:** `src/predict_patient.py`, `src/build_final_report.py`, `src/build_final_notebook.py`.

### Total Estimated Remaining Effort: **27 – 36 Working Hours** (Approx. 4–5 Days).

---

## 6. Verification and Audit Sign-off
- **Auditor:** Project Auditor & Technical Writer
- **Audit Verification Status:** Verified against the physical workspace filesystem, executed data pipelines, and generated tables.
- **Report Location:** [PROJECT_STATUS_REPORT.md](file:///Users/devangmestry/Predictive%20Risk%20Identification%20for%20Diabetes%20and%20Heart%20Disease%20/PROJECT_STATUS_REPORT.md)
