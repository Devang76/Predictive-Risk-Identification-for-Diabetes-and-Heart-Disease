"""
Exploratory Data Analysis (EDA) Script
Predictive Risk Identification for Diabetes and Heart Disease

This script loads each dataset CSV file, prints:
- Shape (number of rows and columns)
- Column names
- Data types
- First 5 rows (head)
- Descriptive statistics (describe)
- Outcome column distribution (with bar charts saved to 'eda_plots/')
- Zero counts for Glucose, BloodPressure, and BMI in the Pima dataset
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set visual styling
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({'font.sans-serif': 'DejaVu Sans', 'font.size': 11})

# Directory to save generated bar charts
PLOTS_DIR = "eda_plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

# Define datasets to inspect
DATASETS = [
    {
        "name": "Pima Indians Diabetes",
        "path": "diabetes.csv",
        "outcome_col": "Outcome",
        "labels": {0: "No Diabetes (0)", 1: "Diabetes (1)"}
    },
    {
        "name": "Cleveland Heart Disease",
        "path": "heart_cleveland_upload.csv",
        "outcome_col": "condition",
        "labels": {0: "No Disease (0)", 1: "Heart Disease (1)"}
    },
    {
        "name": "CDC BRFSS 2015 Diabetes (Binary)",
        "path": "Diabetes Health Indicators Dataset/diabetes_binary_health_indicators_BRFSS2015.csv",
        "outcome_col": "Diabetes_binary",
        "labels": {0: "No Diabetes (0)", 1: "Diabetes / Prediabetes (1)"}
    },
    {
        "name": "CDC BRFSS 2015 Diabetes (50-50 Split)",
        "path": "Diabetes Health Indicators Dataset/diabetes_binary_5050split_health_indicators_BRFSS2015.csv",
        "outcome_col": "Diabetes_binary",
        "labels": {0: "No Diabetes (0)", 1: "Diabetes / Prediabetes (1)"}
    },
    {
        "name": "CDC BRFSS 2015 Diabetes (0-1-2 Indicator)",
        "path": "Diabetes Health Indicators Dataset/diabetes_012_health_indicators_BRFSS2015.csv",
        "outcome_col": "Diabetes_012",
        "labels": {0: "No Diabetes (0)", 1: "Prediabetes (1)", 2: "Diabetes (2)"}
    },
    {
        "name": "CDC BRFSS 2020 Heart Disease (Cleaned)",
        "path": "Indicators of Heart Disease/2020/heart_2020_cleaned.csv",
        "outcome_col": "HeartDisease",
        "labels": {"No": "No Heart Disease", "Yes": "Heart Disease"}
    },
    {
        "name": "CDC BRFSS 2022 Heart Disease (No NaNs)",
        "path": "Indicators of Heart Disease/2022/heart_2022_no_nans.csv",
        "outcome_col": "HadHeartAttack",
        "labels": {"No": "No Heart Attack", "Yes": "Had Heart Attack"}
    },
    {
        "name": "CDC BRFSS 2022 Heart Disease (With NaNs)",
        "path": "Indicators of Heart Disease/2022/heart_2022_with_nans.csv",
        "outcome_col": "HadHeartAttack",
        "labels": {"No": "No Heart Attack", "Yes": "Had Heart Attack"}
    }
]

def analyze_dataset(info):
    name = info["name"]
    path = info["path"]
    outcome_col = info["outcome_col"]
    labels = info.get("labels", {})

    print("\n" + "=" * 80)
    print(f" DATASET: {name}")
    print(f" File Path: {path}")
    print("=" * 80)

    if not os.path.exists(path):
        print(f"Warning: File not found at {path}")
        return None

    # Step 1: Load CSV with pandas
    df = pd.read_csv(path)

    # Step 2: Shape
    print(f"\n[1] SHAPE (Rows, Columns): {df.shape}")
    print(f"    - Number of Patients / Records: {df.shape[0]:,}")
    print(f"    - Number of Features / Columns: {df.shape[1]}")

    # Step 3: Column Names
    print(f"\n[2] COLUMN NAMES ({len(df.columns)} total):")
    print(list(df.columns))

    # Step 4: Data Types
    print(f"\n[3] DATA TYPES:")
    print(df.dtypes)

    # Step 5: First 5 Rows
    print(f"\n[4] FIRST 5 ROWS (.head()):")
    print(df.head())

    # Step 6: Descriptive Statistics
    print(f"\n[5] DESCRIPTIVE STATISTICS (.describe()):")
    print(df.describe())

    # Step 7: Bar Chart of Outcome Column
    if outcome_col in df.columns:
        counts = df[outcome_col].value_counts(dropna=False)
        percentages = df[outcome_col].value_counts(normalize=True, dropna=False) * 100

        print(f"\n[6] OUTCOME COLUMN DISTRIBUTION ('{outcome_col}'):")
        for val, cnt in counts.items():
            lbl = labels.get(val, str(val))
            pct = percentages[val]
            print(f"    - {lbl}: {cnt:,} ({pct:.2f}%)")

        # Create Plot
        fig, ax = plt.subplots(figsize=(7, 5))
        plot_df = counts.reset_index()
        plot_df.columns = [outcome_col, "count"]
        plot_df["display_label"] = plot_df[outcome_col].map(lambda x: labels.get(x, str(x)))

        bar_colors = ["#2b5c8f", "#d95f02", "#7570b3", "#e7298a"][:len(plot_df)]
        bars = ax.bar(plot_df["display_label"], plot_df["count"], color=bar_colors, edgecolor="black", alpha=0.85)

        # Annotate bars with counts and percentages
        total = len(df)
        for bar in bars:
            height = bar.get_height()
            pct = (height / total) * 100
            ax.annotate(f'{height:,}\n({pct:.1f}%)',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 5), textcoords="offset points",
                        ha='center', va='bottom', fontsize=10, fontweight='bold')

        ax.set_title(f"Target Distribution: {name}\nColumn: '{outcome_col}'", fontsize=13, fontweight='bold', pad=12)
        ax.set_ylabel("Patient Count", fontsize=11)
        ax.set_xlabel("Class", fontsize=11)
        ax.set_ylim(0, max(plot_df["count"]) * 1.18)

        safe_filename = name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_") + "_outcome.png"
        plot_path = os.path.join(PLOTS_DIR, safe_filename)
        plt.tight_layout()
        plt.savefig(plot_path, dpi=200)
        plt.close()
        print(f"    -> Saved Bar Chart to: {plot_path}")
    else:
        print(f"\n[6] Outcome column '{outcome_col}' not found in DataFrame.")

    return df

def analyze_pima_zeros():
    print("\n" + "=" * 80)
    print(" PIMA INDIANS DIABETES: ZERO COUNT ANALYSIS")
    print("=" * 80)
    path = "diabetes.csv"
    if not os.path.exists(path):
        print("Pima dataset not found.")
        return

    df = pd.read_csv(path)
    vital_cols = ["Glucose", "BloodPressure", "BMI"]

    print("\nBiological context: A living person cannot have a blood glucose level of 0 mg/dL,")
    print("a blood pressure of 0 mmHg, or a BMI of 0. In this dataset, zeros in these columns")
    print("represent missing/unrecorded medical measurements.\n")

    print(f"{'Feature':<18} | {'Zero Count':<12} | {'Total Rows':<12} | {'Percentage Missing':<18}")
    print("-" * 68)
    for col in vital_cols:
        zeros = (df[col] == 0).sum()
        total = len(df)
        pct = (zeros / total) * 100
        print(f"{col:<18} | {zeros:<12} | {total:<12} | {pct:.2f}%")

    # Extra context for skin thickness and insulin
    other_zeros = ["SkinThickness", "Insulin"]
    print("\nOther zero-encoded missing values in Pima:")
    for col in other_zeros:
        zeros = (df[col] == 0).sum()
        pct = (zeros / len(df)) * 100
        print(f"{col:<18} | {zeros:<12} | {len(df):<12} | {pct:.2f}%")

if __name__ == "__main__":
    print("Starting Exploratory Data Analysis for all datasets...")
    for dataset_info in DATASETS:
        analyze_dataset(dataset_info)

    analyze_pima_zeros()
    print("\nEDA Completed successfully! All charts saved in:", PLOTS_DIR)
