"""
src/build_notebook.py
Builds and executes project_mid_report.ipynb with full markdown explanations,
reproducible code cells, and embedded outputs.
"""

from pathlib import Path
import nbformat as nbf
from nbclient import NotebookClient

BASE_DIR = Path(__file__).resolve().parent.parent
NOTEBOOK_PATH = BASE_DIR / "project_mid_report.ipynb"


def create_notebook():
    nb = nbf.v4.new_notebook()
    nb.metadata = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.9"
        }
    }

    cells = []

    # -------------------------------------------------------------
    # Title & Introduction
    # -------------------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell("""# Predictive Risk Identification for Diabetes and Heart Disease
## Data Mining Course Project Report & Laboratory Notebook (Revised Edition)

**Topic:** Clinical Risk Factor Identification and Association Rule Mining for Cardiometabolic Diseases  
**Team Members:** Data Mining Project Team  
**Primary Environment:** Python 3, Pandas, NumPy, Matplotlib, Seaborn, Scikit-Learn, MLxtend  
**Random State:** 42 (Enforced across all stochastic sampling and cross-validation)  

---

### Project Overview
Heart disease and type 2 diabetes mellitus are the leading causes of illness and mortality worldwide. Fortunately, both conditions often develop slowly over decades and share common lifestyle, metabolic, and hemodynamic risk factors, including obesity, high blood pressure, and atherogenic dyslipidemia. 

In this project, we apply modern data mining techniques to:
1. **Clean, audit, and preprocess** clinical biometric datasets and large-scale public health surveys, uncovering 652 "hidden zeros" in medical tests.
2. **Handle missing values and structural missingness** scientifically by evaluating listwise deletion versus principled target-preserving imputation on real surveys.
3. **Harmonize and vertically integrate** diverse data sources into a unified clinical-epidemiological cohort of 26,065 records.
4. **Discover synergistic risk combinations** using Association Rule Mining (**Apriori** and **FP-Growth** algorithms).
5. **Translate mined rules into actionable clinical insights** that digital health applications (such as *Dietify*) can operationalize for early risk detection and preventative lifestyle interventions.
"""))

    # -------------------------------------------------------------
    # Imports & Global Configuration
    # -------------------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell("""## 0. Setup and Environment Configuration
Here we import all required data science, visualization, and data mining libraries. We also enforce a fixed random seed (`random_state=42`) to make sure every calculation, sample, and result is 100% reproducible every time this notebook is run.
"""))

    cells.append(nbf.v4.new_code_cell("""import os
import time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules

# Set random seed for full reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Configure visual styling
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    "font.sans-serif": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "figure.titlesize": 12
})

# Define key paths
RAW_DIR = Path("data/raw")
CLEAN_DIR = Path("data/clean")
TBL_DIR = Path("outputs/tables")
FIG_DIR = Path("outputs/figures")

print("Environment successfully initialized with random_state=42.")
"""))

    # -------------------------------------------------------------
    # Step 0: Dataset Inspection & Selection
    # -------------------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell("""## 1. Dataset Inspection and Selection
### What are we doing here?
Our project workspace contains eight candidate datasets. Before modifying any data, we inspect every file on disk, examine its dimensions, file size, and contents, and report the exact number of records available.

### Why does this matter?
Inspecting files first ensures we choose the highest quality, most relevant data sources while setting aside redundant or derivative splits.
"""))

    cells.append(nbf.v4.new_code_cell("""# Inspect all available CSV datasets in the repository
dataset_inventory = pd.read_csv(TBL_DIR / "dataset_inventory.csv")
print("Table 0: Comprehensive Inventory of Available Datasets")
print(f"Total Records Across ALL 8 Files:     {dataset_inventory['Rows'].sum():,}")
print(f"Total Records Across 4 Selected Files: {dataset_inventory[dataset_inventory['Selected']]['Rows'].sum():,}")
dataset_inventory
"""))

    cells.append(nbf.v4.new_markdown_cell("""### Justification of File Selection
Based on our inspection, we select four primary files:
1. **`diabetes.csv` (UCI Pima Indians Diabetes, 768 rows):** Selected because it contains direct continuous clinical lab tests (fasting glucose, 2-hour serum insulin, diastolic blood pressure, triceps skinfold thickness, and BMI). It is the premier clinical benchmark for diabetes risk.
2. **`heart_cleveland_upload.csv` (UCI Cleveland Heart Disease, 297 rows):** Selected because it provides rich gold-standard cardiac diagnostic features (resting blood pressure, serum cholesterol, exercise-induced angina, ST-segment depression, and fluoroscopy major vessel counts).
3. **`diabetes_binary_health_indicators_BRFSS2015.csv` (CDC BRFSS 2015, 253,680 rows):** Selected because it contains a massive population sample measuring **BOTH** `Diabetes_binary` AND `HeartDiseaseorAttack` on the exact same respondents. A stratified sample of 25,000 transactions is used for association rule mining.
4. **`heart_2022_with_nans.csv` (CDC BRFSS 2022 with NaNs, 445,132 rows):** Selected specifically as a standalone test of real-world survey missing data science.
"""))

    # -------------------------------------------------------------
    # Part 1.1: Data Cleaning & Pima Zeros
    # -------------------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell("""## 2. Data Cleaning: Uncovering Hidden Zeros & Fixing Anomaly Values
### The "Hidden Zeros" Discovery in Pima Diabetes
In living humans, fasting plasma glucose, diastolic blood pressure, triceps skinfold thickness, serum insulin, and BMI cannot physically equal zero. When the Pima dataset was compiled, clinical staff entered `0` for unmeasured tests instead of leaving them blank. If left uncorrected, models interpret these as true biological measurements. We convert these impossible zeros to `NaN` so Python properly treats them as missing.

*Note:* `Pregnancies == 0` (111 records, 14.45%) represents valid nulliparous females and is kept intact.
"""))

    cells.append(nbf.v4.new_code_cell("""# 1. Pima Diabetes Cleaning & Hidden Zeros Audit
pima_raw = pd.read_csv(RAW_DIR / "diabetes.csv")
zero_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

print("--- Pima Zeros Audit ---")
for col in zero_cols:
    cnt = (pima_raw[col] == 0).sum()
    pct = (cnt / len(pima_raw)) * 100
    print(f"{col:15}: {cnt:3} zeros ({pct:5.2f}% missing)")

# Replace impossible zeros with NaN; keep Pregnancies zeros intact
pima_clean = pima_raw.copy()
pima_clean[zero_cols] = pima_clean[zero_cols].replace(0, np.nan)
pima_clean.to_csv(CLEAN_DIR / "pima_clean.csv", index=False)

pima_miss_table = pd.read_csv(TBL_DIR / "pima_missing_summary.csv")
print("\\nTable 1A: Pima Missing Values Audit and Handling Decisions")
pima_miss_table
"""))

    cells.append(nbf.v4.new_code_cell("""# 2. Cleveland Heart Disease Cleaning & Outlier Audit
clev_raw = pd.read_csv(RAW_DIR / "heart_cleveland_upload.csv")
print("Cleveland Shape:", clev_raw.shape)
print("Cleveland Duplicate Rows:", clev_raw.duplicated().sum())
print("Cleveland Missing Values:", clev_raw.isnull().sum().sum())

# Check for extreme clinical values
high_chol = clev_raw[clev_raw["chol"] > 500]
print(f"\\nOutlier Audit: Patients with Serum Cholesterol > 500 mg/dL: {len(high_chol)}")
print(high_chol[["age", "sex", "chol", "trestbps", "condition"]])

# Save cleaned Cleveland dataset
clev_raw.to_csv(CLEAN_DIR / "cleveland_clean.csv", index=False)
"""))

    cells.append(nbf.v4.new_markdown_cell("""### Cleaning Decisions Summary:
- **Pima Dataset:** Converted 652 impossible zeros across 5 columns into `NaN` (affecting 376 individual records). No imputation is performed during cleaning; raw NaNs are preserved in `data/clean/pima_clean.csv` to prevent data leakage.
- **Cleveland Dataset:** Confirmed 0 duplicate rows and 0 missing values. Retained row 79 (serum cholesterol 564 mg/dL) as an authentic clinical case of familial hypercholesterolemia.
"""))

    # -------------------------------------------------------------
    # Part 1.2: Missing Values Analysis & Real NaN Handling
    # -------------------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell("""## 3. Missing Value Science: Structural Missingness & Real-World Survey NaN Handling
### Structural Missingness in Harmonized Data
When integrating diverse datasets, certain variables exist only in specific studies. For instance, continuous serum cholesterol was collected only in Cleveland, fasting glucose only in Pima, and heart disease was not assessed in Pima. Table 1B documents this structural missingness per column across all 26,065 integrated records.
"""))

    cells.append(nbf.v4.new_code_cell("""# Table 1B: Integrated Harmonized Dataset Structural Missingness
integ_miss_table = pd.read_csv(TBL_DIR / "integrated_missing_summary.csv")
print("Table 1B: Structural Missingness per Column of Integrated Dataset")
integ_miss_table
"""))

    cells.append(nbf.v4.new_code_cell("""# Real NaN Handling Demonstration using CDC Heart 2022
h22 = pd.read_csv(RAW_DIR / "Indicators of Heart Disease" / "2022" / "heart_2022_with_nans.csv")

total_rows = len(h22)
rows_after_dropna = len(h22.dropna())
rows_lost_listwise = total_rows - rows_after_dropna
pct_lost_listwise = (rows_lost_listwise / total_rows) * 100

print(f"Total Raw Records in Heart 2022:                    {total_rows:,}")
print(f"Records remaining under naive listwise deletion:     {rows_after_dropna:,}")
print(f"Records DESTROYED by naive dropna (Listwise Loss):   {rows_lost_listwise:,} ({pct_lost_listwise:.2f}%)")

# Principled approach: drop only if the primary target outcome is missing
h22_clean_target = h22.dropna(subset=["HadHeartAttack"]).copy()
rows_dropped_target = total_rows - len(h22_clean_target)
print(f"\\nPrincipled Approach: Drop ONLY missing target rows:  {rows_dropped_target:,} rows ({rows_dropped_target/total_rows*100:.2f}%)")
print(f"Records PRESERVED for analysis:                      {len(h22_clean_target):,} (99.31%)")
"""))

    cells.append(nbf.v4.new_markdown_cell("""### Key Lesson from Real NaN Handling:
Naive listwise deletion (`df.dropna()`) discards **199,110 people (44.73% of the entire cohort!)**, creating severe selection bias. Our principled approach drops records ONLY when the primary outcome is missing (0.69%), preserving 99.31% of the cohort.
*Important:* BRFSS 2022 is a standalone missing-data demonstration; NO imputation was applied to the 25,000 BRFSS 2015 transaction dataset used for mining, which had zero missing values.
"""))

    # -------------------------------------------------------------
    # Part 1.3: Dataset Integration
    # -------------------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell("""## 4. Dataset Integration (Vertical Harmonization)
### Why is a 1-to-1 Horizontal Patient Match Impossible?
You cannot merge row 1 of Pima with row 1 of Cleveland because they represent distinct patient populations:
- **Pima Indians:** 768 adult females of Pima Indian heritage in Arizona (1965-1980).
- **Cleveland Clinic:** 297 cardiac patients in Ohio undergoing angiography.
- **CDC BRFSS:** Nationwide telephone survey of random civilian US households in 2015.

There are no shared patient IDs. Merging them side-by-side would fabricate fictional hybrid individuals. Instead, we execute **Vertical Harmonization**: stacking cohorts using shared clinical and demographic concepts (Age, Sex, BMI, Blood Pressure, Cholesterol, Glucose, and Disease Outcomes).
"""))

    cells.append(nbf.v4.new_code_cell("""# Display Table 2: Integration Accounting Summary
integration_summary = pd.read_csv(TBL_DIR / "integration_summary.csv")
print("Table 2: Dataset Integration Accounting Log")
integration_summary
"""))

    cells.append(nbf.v4.new_code_cell("""# Load the integrated dataset
integrated_df = pd.read_csv(CLEAN_DIR / "integrated_dataset.csv")
print(f"Integrated Dataset Shape: {integrated_df.shape} (26,065 records)")
integrated_df.head()
"""))

    # -------------------------------------------------------------
    # Part 1.4: Descriptive Statistics
    # -------------------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell("""## 5. Descriptive Statistics of All Study Variables
We compute descriptive statistics across EVERY continuous and categorical variable utilized in this project:
- **Table 3:** Parametric and non-parametric statistics for all 11 continuous/discrete numeric variables.
- **Table 4:** Counts and percentages for all categorical, survey, and outcome indicators.
"""))

    cells.append(nbf.v4.new_code_cell("""# Table 3: Numeric Descriptive Statistics
num_stats = pd.read_csv(TBL_DIR / "numeric_descriptive_stats.csv")
print("Table 3: Comprehensive Descriptive Statistics for Continuous Clinical Variables")
num_stats
"""))

    cells.append(nbf.v4.new_code_cell("""# Table 4: Categorical Descriptive Statistics
cat_stats = pd.read_csv(TBL_DIR / "categorical_descriptive_stats.csv")
print("Table 4: Comprehensive Descriptive Statistics for Categorical and Outcome Indicators")
cat_stats
"""))

    # -------------------------------------------------------------
    # Part 1.5: Visualizations
    # -------------------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell("""## 6. Exploratory Data Visualizations
Here we generate captioned figures illustrating empirical distributions, bivariate relationships, and correlations across the cohort.
"""))

    cells.append(nbf.v4.new_code_cell("""# Figure 1: Histograms of Numeric Variables
fig, axes = plt.subplots(2, 3, figsize=(12, 6.5))
axes = axes.flatten()
num_vars = ["age", "bmi", "blood_pressure", "cholesterol", "glucose"]
colors = ["#2b5c8f", "#d95f02", "#7570b3", "#1b9e77", "#e7298a"]

for i, var in enumerate(num_vars):
    data = integrated_df[var].dropna()
    sns.histplot(data, kde=True, ax=axes[i], color=colors[i], bins=25, edgecolor="black")
    axes[i].set_title(f"Distribution of {var.replace('_', ' ').title()}", fontweight="bold", fontsize=10)
    axes[i].set_xlabel(var.replace('_', ' ').title(), fontsize=9)
    axes[i].set_ylabel("Count", fontsize=9)

axes[5].axis("off")
fig.suptitle("Figure 1: Histograms of Key Clinical and Demographic Numeric Variables", fontsize=11, fontweight="bold", y=0.98)
fig.subplots_adjust(top=0.91, hspace=0.35, wspace=0.25)
plt.savefig(FIG_DIR / "figure1_numeric_histograms.png", dpi=300, bbox_inches="tight")
plt.show()
"""))
    cells.append(nbf.v4.new_markdown_cell("""**Takeaway for Figure 1:** Clinical variables exhibit pronounced positive skewness (especially BMI, glucose, and blood pressure), demonstrating that while the healthy majority clusters near baseline medians, a substantial high-risk cohort experiences severe cardiometabolic stress.
"""))

    cells.append(nbf.v4.new_code_cell("""# Figure 2: Glucose vs BMI colored by Diabetes Diagnosis (Pima)
pima_clean = pd.read_csv(CLEAN_DIR / "pima_clean.csv")
fig, ax = plt.subplots(figsize=(7, 4.8))
scatter_df = pima_clean.dropna(subset=["Glucose", "BMI", "Outcome"])
sns.scatterplot(
    data=scatter_df,
    x="Glucose",
    y="BMI",
    hue="Outcome",
    palette={0: "#2b5c8f", 1: "#d95f02"},
    alpha=0.75,
    s=45,
    edgecolor="k",
    ax=ax
)
ax.axvline(125, color="red", linestyle="--", linewidth=1.2, label="Diabetic Glucose (>125 mg/dL)")
ax.axhline(30, color="darkgreen", linestyle=":", linewidth=1.2, label="Obese BMI (>=30 kg/m²)")
ax.set_title("Figure 2: Fasting Glucose vs. BMI Colored by Diabetes Diagnosis", fontweight="bold", fontsize=11, pad=10)
ax.set_xlabel("Fasting Plasma Glucose (mg/dL)", fontsize=9.5)
ax.set_ylabel("Body Mass Index (kg/m²)", fontsize=9.5)
ax.legend(labels=["Non-Diabetic (0)", "Diabetic (1)", "Diabetic Glucose (>125)", "Obesity (BMI>=30)"], loc="upper left", fontsize=8)
plt.savefig(FIG_DIR / "figure2_glucose_vs_bmi.png", dpi=300, bbox_inches="tight")
plt.show()
"""))
    cells.append(nbf.v4.new_markdown_cell("""**Takeaway for Figure 2:** Patients with both elevated fasting glucose (>125 mg/dL) and obesity (BMI ≥ 30 kg/m²) fall almost exclusively into the diabetic group, visually proving that glycemic and adiposity thresholds jointly drive metabolic disease onset.
"""))

    cells.append(nbf.v4.new_code_cell("""# Figure 3: Age vs Cholesterol colored by Heart Disease Condition (Cleveland)
clev_clean = pd.read_csv(CLEAN_DIR / "cleveland_clean.csv")
fig, ax = plt.subplots(figsize=(7, 4.8))
sns.scatterplot(
    data=clev_clean,
    x="age",
    y="chol",
    hue="condition",
    palette={0: "#2b5c8f", 1: "#d95f02"},
    alpha=0.8,
    s=50,
    edgecolor="k",
    ax=ax
)
ax.axhline(240, color="darkred", linestyle="--", linewidth=1.2, label="High Cholesterol (>=240 mg/dL)")
ax.set_title("Figure 3: Age vs. Cholesterol Colored by Heart Disease Condition", fontweight="bold", fontsize=11, pad=10)
ax.set_xlabel("Patient Age (Years)", fontsize=9.5)
ax.set_ylabel("Serum Cholesterol (mg/dL)", fontsize=9.5)
ax.legend(labels=["No Heart Disease (0)", "Heart Disease (1)", "High Cholesterol (>=240)"], loc="upper left", fontsize=8)
plt.savefig(FIG_DIR / "figure3_age_vs_cholesterol.png", dpi=300, bbox_inches="tight")
plt.show()
"""))
    cells.append(nbf.v4.new_markdown_cell("""**Takeaway for Figure 3:** Heart disease patients concentrate heavily in the upper-right quadrant of advanced age (≥50 years) and elevated cholesterol (≥240 mg/dL), confirming the compounding vascular damage of aging and lipid deposition.
"""))

    cells.append(nbf.v4.new_code_cell("""# Figure 4: Categorical Distributions
fig, axes = plt.subplots(1, 4, figsize=(13.5, 3.8))
cat_plots = [
    ("sex", ["Female (0)", "Male (1)"], "#2b5c8f", "Sex Distribution"),
    ("high_bp", ["Normal BP (0)", "High BP (1)"], "#d95f02", "Hypertension Status"),
    ("diabetes_outcome", ["No Diab (0)", "Diabetes (1)"], "#7570b3", "Diabetes Outcome"),
    ("heart_disease_outcome", ["No Heart Dis (0)", "Heart Dis (1)"], "#e7298a", "Heart Disease (N=25,297)")
]
for i, (col, labels, colr, title) in enumerate(cat_plots):
    s = integrated_df[col].dropna()
    counts = s.value_counts().sort_index()
    ax = axes[i]
    bars = ax.bar([labels[int(k)] for k in counts.index], counts.values, color=colr, edgecolor="k", alpha=0.8)
    ax.set_title(title, fontweight="bold", fontsize=10)
    ax.set_ylabel("Count" if i == 0 else "", fontsize=9)
    for b in bars:
        h = b.get_height()
        ax.annotate(f"{h:,}\\n({h/len(s)*100:.1f}%)", (b.get_x() + b.get_width() / 2, h), ha="center", va="bottom", fontsize=7.5, fontweight="bold")
    ax.set_ylim(0, max(counts.values) * 1.25)

fig.suptitle("Figure 4: Prevalence of Key Categorical Risk Factors and Outcomes", fontsize=11, fontweight="bold", y=0.98)
fig.subplots_adjust(top=0.86, wspace=0.28)
plt.savefig(FIG_DIR / "figure4_categorical_distributions.png", dpi=300, bbox_inches="tight")
plt.show()
"""))
    cells.append(nbf.v4.new_markdown_cell("""**Takeaway for Figure 4:** The cohort exhibits realistic epidemiological class balance: 14.43% diabetes prevalence (3,761 / 26,065) and exactly 9.89% heart disease prevalence among evaluated individuals (2,503 / 25,297 non-null records; 9.46% in the BRFSS sample alone).
"""))

    cells.append(nbf.v4.new_code_cell("""# Figure 5: Correlation Heatmap (dropping high_glucose proxy to prevent artificial 0.97 circularity)
fig, ax = plt.subplots(figsize=(7.5, 5.5))
corr_cols = ["age", "sex", "bmi", "high_bp", "high_chol", "diabetes_outcome", "heart_disease_outcome"]
corr_matrix = integrated_df[corr_cols].corr()
sns.heatmap(
    corr_matrix,
    annot=True,
    fmt=".2f",
    cmap="vlag",
    vmin=-0.3,
    vmax=0.6,
    cbar_kws={"label": "Pearson Correlation (r)"},
    ax=ax,
    linewidths=0.5
)
ax.set_title("Figure 5: Correlation Matrix of Clinical Predictors and Disease Outcomes", fontweight="bold", fontsize=11, pad=10)
plt.savefig(FIG_DIR / "figure5_correlation_heatmap.png", dpi=300, bbox_inches="tight")
plt.show()
"""))
    cells.append(nbf.v4.new_markdown_cell("""**Takeaway for Figure 5:** Age (r=0.22), hypertension (r=0.21), and hypercholesterolemia (r=0.19) exhibit the strongest correlations with heart disease, while BMI (r=0.23) and hypertension (r=0.24) dominate diabetes. High_glucose was removed from this heatmap because in BRFSS it was derived from the diabetes label as an operational proxy (creating an artificial r=0.97 artifact).
"""))

    # -------------------------------------------------------------
    # Part 1.6: Normalization & Transformation
    # -------------------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell("""## 7. Feature Normalization, Transformation & Skewness Analysis
We apply Min-Max scaling [0, 1], Z-score standardization (mean=0, std=1), and natural log transformation (`log1p`).
Table 5 summarizes the results across all six continuous clinical features.

### Why Log Transformation is Contraindicated for Age:
Log transformation is effective for right-skewed variables (Insulin: +2.17 -> -0.09; Cholesterol: +1.12 -> +0.06; Glucose: +0.53 -> -0.06; BMI: +2.00 -> +0.69). For Age, the empirical distribution is already mildly left-skewed (-0.36) due to older adult participation in public health surveys. Applying log1p compresses higher ages, exacerbating negative skewness from -0.36 to -1.01 (a 180% worsening of symmetry). Therefore, log transformation is contraindicated for age.
"""))

    cells.append(nbf.v4.new_code_cell("""# Table 5: Normalization & Transformation Summary
norm_table = pd.read_csv(TBL_DIR / "normalization_transformation_summary.csv")
print("Table 5: Feature Normalization, Standardization, and Skewness Reduction Summary")
norm_table
"""))

    cells.append(nbf.v4.new_code_cell("""# Figure 6: Log Transformation Before & After (Insulin)
pima_clean = pd.read_csv(CLEAN_DIR / "pima_clean.csv")
raw_insulin = pima_clean["Insulin"].dropna()
log_insulin = np.log1p(raw_insulin)

fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2))
sns.histplot(raw_insulin, kde=True, color="#d95f02", ax=axes[0], bins=25, edgecolor="k")
axes[0].set_title(f"Original Serum Insulin\\n(Skewness = +{raw_insulin.skew():.2f} [Heavy Right Tail])", fontweight="bold", fontsize=9.5)
axes[0].set_xlabel("Serum Insulin (μU/mL)", fontsize=9)
axes[0].set_ylabel("Count", fontsize=9)

sns.histplot(log_insulin, kde=True, color="#2b5c8f", ax=axes[1], bins=25, edgecolor="k")
axes[1].set_title(f"Log-Transformed Insulin: log1p(x)\\n(Skewness = {log_insulin.skew():.2f} [Near-Normal])", fontweight="bold", fontsize=9.5)
axes[1].set_xlabel("log(1 + Insulin)", fontsize=9)
axes[1].set_ylabel("Count", fontsize=9)

fig.suptitle("Figure 6: Serum Insulin Before and After Natural Log Transformation: log1p(x)", fontsize=11, fontweight="bold", y=0.98)
fig.subplots_adjust(top=0.84, wspace=0.25)
plt.savefig(FIG_DIR / "figure6_log_transform_comparison.png", dpi=300, bbox_inches="tight")
plt.show()
"""))
    cells.append(nbf.v4.new_markdown_cell("""**Takeaway for Figure 6:** Applying log1p compresses the extreme right tail of serum insulin, reducing skewness from severe positive asymmetry (+2.17) to near-normal symmetry (-0.09), stabilizing variance for statistical and regression modeling.
"""))

    # -------------------------------------------------------------
    # Part 2: Frequent Pattern Mining
    # -------------------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell("""## 8. Frequent Pattern Mining (Apriori & FP-Growth)
### Discretization & Binary Items Created
Continuous variables were discretized using established clinical guidelines: BMI >= 30 (Obesity), Age >= 50 (Vascular Risk Threshold), GenHlth in {Fair, Poor}, and Income <= $25,000. Table 6 documents all 15 binary transaction items created.

*Mining Cohort Scope:* Mining was run on the 25,000 BRFSS 2015 transactions because all 15 lifestyle and clinical indicators were uniformly measured on every respondent. Continuous lab thresholds (e.g., Pima glucose > 125, Cleveland cholesterol >= 240) do not appear in population rules due to cross-cohort structural missingness between clinical laboratory records and public health surveys.
"""))

    cells.append(nbf.v4.new_code_cell("""# Table 6: Complete Inventory of Binary Items Created for Mining
bin_items = pd.read_csv(TBL_DIR / "binary_items_inventory.csv")
print("Table 6: Binary Items Created for Frequent Pattern Mining")
bin_items
"""))

    cells.append(nbf.v4.new_code_cell("""# 1. Benchmark Apriori vs FP-Growth on 25,000 Transactions at min_support = 0.20
brfss_raw = pd.read_csv(RAW_DIR / "diabetes_binary_health_indicators_BRFSS2015.csv")
sample_df = brfss_raw.sample(n=25000, random_state=RANDOM_STATE).copy()

bool_df = pd.DataFrame()
bool_df["HighBP"] = sample_df["HighBP"] == 1.0
bool_df["HighChol"] = sample_df["HighChol"] == 1.0
bool_df["Smoker"] = sample_df["Smoker"] == 1.0
bool_df["PhysActivity"] = sample_df["PhysActivity"] == 1.0
bool_df["Fruits"] = sample_df["Fruits"] == 1.0
bool_df["Veggies"] = sample_df["Veggies"] == 1.0
bool_df["DiffWalk"] = sample_df["DiffWalk"] == 1.0
bool_df["Stroke"] = sample_df["Stroke"] == 1.0
bool_df["Sex_Male"] = sample_df["Sex"] == 1.0
bool_df["BMI>=30"] = sample_df["BMI"] >= 30.0
bool_df["Age>=50"] = sample_df["Age"] >= 7.0
bool_df["GenHlth_PoorFair"] = sample_df["GenHlth"] >= 4.0
bool_df["Income_Low"] = sample_df["Income"] <= 4.0
bool_df["Diabetes"] = sample_df["Diabetes_binary"] == 1.0
bool_df["HeartDisease"] = sample_df["HeartDiseaseorAttack"] == 1.0

t0 = time.time()
fi_ap = apriori(bool_df, min_support=0.2, use_colnames=True)
t_ap = time.time() - t0

t0 = time.time()
fi_fp = fpgrowth(bool_df, min_support=0.2, use_colnames=True)
t_fp = time.time() - t0

ap_set = set(frozenset(x) for x in fi_ap["itemsets"])
fp_set = set(frozenset(x) for x in fi_fp["itemsets"])

print(f"Algorithm comparison at min_support=0.2:")
print(f"  Apriori:   {len(fi_ap)} itemsets in {t_ap:.4f}s")
print(f"  FP-Growth: {len(fi_fp)} itemsets in {t_fp:.4f}s")
print(f"  Exact Itemset Match: {ap_set == fp_set}")
"""))

    cells.append(nbf.v4.new_code_cell("""# Table 7 & Figure 7: Sensitivity Analysis (Varying Minimum Support)
supp_table = pd.read_csv(TBL_DIR / "support_variation_summary.csv")
print("Table 7: Support Variation Summary")
display(supp_table)

# Plot Figure 7
fig, ax1 = plt.subplots(figsize=(7.2, 4.2))
color1 = "#2b5c8f"
color2 = "#d95f02"

ax1.set_xlabel("Minimum Support Threshold", fontweight="bold", fontsize=9.5)
ax1.set_ylabel("Number of Frequent Itemsets", color=color1, fontweight="bold", fontsize=9.5)
line1 = ax1.plot(supp_table["Min_Support"], supp_table["Frequent_Itemsets"], marker="o", color=color1, linewidth=2, label="Frequent Itemsets")
ax1.tick_params(axis="y", labelcolor=color1)
ax1.invert_xaxis()

ax2 = ax1.twinx()
ax2.set_ylabel("Number of Association Rules (Lift >= 1.0)", color=color2, fontweight="bold", fontsize=9.5)
line2 = ax2.plot(supp_table["Min_Support"], supp_table["Total_Rules_Lift_ge_1"], marker="s", color=color2, linewidth=2, linestyle="--", label="Association Rules")
ax2.tick_params(axis="y", labelcolor=color2)

lines = line1 + line2
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc="upper right", fontsize=8.5)
plt.title("Figure 7: Itemsets and Association Rules vs. Minimum Support Threshold", fontweight="bold", fontsize=11, pad=10)
plt.savefig(FIG_DIR / "figure7_itemsets_rules_vs_support.png", dpi=300, bbox_inches="tight")
plt.show()
"""))

    cells.append(nbf.v4.new_markdown_cell("""**Takeaway for Figure 7:** Decreasing minimum support from 0.40 to 0.05 triggers an exponential surge in frequent itemsets (17 to 611) and association rules (24 to 5,836), successfully uncovering highly specific, high-risk disease phenotypes that affect smaller clinical sub-populations.
"""))

    cells.append(nbf.v4.new_code_cell("""# Tables 8 & 9: Top Mined Rules for Diabetes and Heart Disease
diab_lift = pd.read_csv(TBL_DIR / "top_diabetes_rules_by_lift.csv")
heart_lift = pd.read_csv(TBL_DIR / "top_heart_disease_rules_by_lift.csv")

print("Table 8: Top Mined Association Rules Targeting Diabetes (Ranked by Lift)")
display(diab_lift.head(8))

print("\\nTable 9: Top Mined Association Rules Targeting Heart Disease (Ranked by Lift)")
display(heart_lift.head(8))
"""))

    cells.append(nbf.v4.new_code_cell("""# Table 10: Clinical Benchmark Rules (Pima Lab & Cleveland Angiography)
pima_rules = pd.read_csv(TBL_DIR / "pima_mined_rules.csv").head(3)
pima_rules["Cohort"] = "Pima Indians (Lab)"
clev_rules = pd.read_csv(TBL_DIR / "cleveland_mined_rules.csv").head(3)
clev_rules["Cohort"] = "Cleveland (Angio)"
aux_rules = pd.concat([pima_rules, clev_rules], ignore_index=True)[
    ["Cohort", "antecedents_str", "consequents_str", "support", "confidence", "lift"]
]
print("Table 10: Clinical Association Rules Discovered on Pima and Cleveland Cohorts")
aux_rules
"""))

    cells.append(nbf.v4.new_markdown_cell("""### Plain-English Translation of Key Rules
Lift is mathematically defined as the ratio of rule confidence to baseline consequent prevalence: Lift(A -> B) = P(B|A) / P(B). It represents a multiplicative relative risk / ratio of probabilities, NOT an odds ratio.

1. **Diabetes Rule 1 ({Age >= 50, BMI >= 30, HighChol} -> {Diabetes}, Lift = 2.58, Confidence = 35.7%):** An adult aged 50 or older who is clinically obese and has high cholesterol is **2.58 times more likely to have diabetes** than an average person in the population. Over one in three individuals with this exact risk triad suffers from diabetes.
2. **Diabetes Rule 2 ({Age >= 50, HighBP, BMI >= 30} -> {Diabetes}, Lift = 2.52, Confidence = 34.7%):** When aging and obesity are compounded by chronic hypertension, diabetes likelihood surges by more than 150% above population baseline.
3. **Heart Disease Rule 1 ({Age >= 50, HighBP, HighChol} -> {HeartDisease}, Lift = 2.43, Confidence = 23.0%):** Older adults burdened by both hypertension and hypercholesterolemia suffer a **2.43-fold elevation in heart attack probability** relative to population baseline prevalence (a probability ratio of 2.43, not an odds ratio), with nearly one-quarter (23.0%) experiencing major cardiac disease.
4. **Clinical Benchmarks:** In Pima, `{BMI>=30, Age>=35, Glucose>125} -> {Diabetes}` achieves 77.1% confidence and lift 2.21. In Cleveland, `{HighChol>=240, Male, ExAngina} -> {HeartDisease}` achieves 90.2% confidence and lift 1.96.
"""))

    # -------------------------------------------------------------
    # Strategic Insights & Conclusion
    # -------------------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell("""## 9. Strategic Insights for Digital Health Apps (e.g., *Dietify*)
A modern mobile health platform like **Dietify** can embed these exact mined association rules into its proactive recommendation engine:
1. **Automated Risk Triggering:** When a user logs an elevated BMI (≥30) and self-reports high blood pressure, Dietify does not wait for a formal physician diagnosis. The app immediately triggers personalized dietary alerts recommending reduced sodium intake and glycemic control meal plans.
2. **Preventative Lab Screening Triggers:** When Dietify identifies the `{Age >= 50, HighBP, HighChol}` pattern, the app prompts the user with an in-app alert to schedule an annual fasting blood glucose panel, HbA1c test, and ECG stress evaluation.
3. **Behavioral Nudging:** Identifying the strong association between physical limitation (`DiffWalk`) and disease emergence enables Dietify to recommend low-impact restorative physical activities (e.g., chair yoga, aquatic therapy, brisk walking intervals) to interrupt the sedentary cardiometabolic spiral.

---

### Project Limitations
1. **Association is NOT Causation:** Frequent pattern mining reveals co-occurrence; it does not prove that high blood pressure directly causes diabetes. Both stem from shared upstream cellular dysfunction and chronic caloric excess.
2. **Survey and Sampling Bias:** The Pima dataset is restricted to Native American females in Arizona; the Cleveland dataset is derived from hospital inpatients; and the BRFSS relies on self-reported telephone responses subject to recall bias.
3. **Structural Missingness:** Because continuous laboratory metrics could not be imputed across phone surveys without synthetic bias, transaction rules are interpreted within their respective survey and clinical operational scopes.

---

## 10. Conclusion
Through systematic data cleaning, hidden-zero auditing (652 zeros across 376 records in Pima), principled missing data science (showing listwise deletion destroys 44.7% of survey cohorts), and vertical harmonization, we created a unified cardiometabolic cohort of 26,065 records. Frequent pattern mining successfully extracted statistically robust, clinically interpretable risk rules for both diabetes and heart disease. These rules confirm the deadly synergy of the metabolic syndrome triad (Obesity + Hypertension + Hypercholesterolemia) and offer immediate actionable value for digital health monitoring platforms.
"""))

    nb.cells = cells
    with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"Created notebook at: {NOTEBOOK_PATH}")
    return nb


def execute_notebook():
    print(f"Executing notebook {NOTEBOOK_PATH.name} top-to-bottom...")
    with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
        nb = nbf.read(f, as_version=4)

    client = NotebookClient(nb, timeout=600, kernel_name="python3")
    client.execute()

    with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print("Notebook executed successfully without errors!")


if __name__ == "__main__":
    create_notebook()
    execute_notebook()
