#!/usr/bin/env python3
"""
scripts/generate_report_figures.py
Generates missing figures required for the Final Project Report:
1. reports/figures/fig_stage_flow.png: End-to-end architectural flow diagram
2. reports/figures/fig_model_comparison_bar.png: Grouped bar chart comparing F1 and ROC-AUC across all 12 models
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
REPORTS_FIG_DIR = BASE_DIR / "reports" / "figures"
REPORTS_FIG_DIR.mkdir(parents=True, exist_ok=True)

def generate_flow_diagram():
    fig, ax = plt.subplots(figsize=(14, 8), dpi=300)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8.5)
    ax.axis("off")

    # Colors
    c_blue = "#1F497D"
    c_light_blue = "#E9F0F8"
    c_teal = "#0E8A8A"
    c_light_teal = "#E6F5F5"
    c_orange = "#C55A11"
    c_light_orange = "#FDF2E9"
    c_purple = "#5B2C6F"
    c_light_purple = "#F4ECF7"
    c_green = "#1E8449"
    c_light_green = "#EAFAF1"

    # Box 1: Stage 1 Data Audit & Engineering
    rect1 = patches.FancyBboxPatch((0.5, 4.5), 3.8, 3.4, boxstyle="round,pad=0.2",
                                  ec=c_blue, fc=c_light_blue, lw=2)
    ax.add_patch(rect1)
    ax.text(2.4, 7.6, "STAGE 1: AUDIT & FOUNDATION", ha="center", va="center",
            fontsize=11, fontweight="bold", color=c_blue)
    stage1_text = (
        "• Data Audits & Missingness Analysis\n"
        "  - Pima: 652 hidden biological zeros flagged\n"
        "  - Cleveland: 297 verified clinical records\n"
        "  - BRFSS 2015: 253,680 survey records\n"
        "• Feature Normalization Benchmarks\n"
        "  - Min-Max vs Z-Score vs Log Transform\n"
        "• Multi-Cohort Integration (N=26,065)\n"
        "• Frequent Pattern Mining\n"
        "  - Apriori vs FP-Growth algorithms\n"
        "  - Multi-morbidity rules (Lift up to 2.58)"
    )
    ax.text(0.7, 5.8, stage1_text, ha="left", va="center", fontsize=8.5, color="#222222", linespacing=1.3)

    # Box 2: Stage 2 Unsupervised Mining
    rect2 = patches.FancyBboxPatch((4.9, 4.5), 3.8, 3.4, boxstyle="round,pad=0.2",
                                  ec=c_teal, fc=c_light_teal, lw=2)
    ax.add_patch(rect2)
    ax.text(6.8, 7.6, "STAGE 2: UNSUPERVISED MINING", ha="center", va="center",
            fontsize=11, fontweight="bold", color=c_teal)
    stage2_text = (
        "• Phenotypic Clustering Analysis\n"
        "  - K-Means (k=3 champion, Silhouette 0.153)\n"
        "  - Hierarchical (Ward) & DBSCAN (eps=2.4)\n"
        "  - 3 Phenotypes: Young, Metabolic, Multi-morbid\n"
        "• Dimensionality Reduction & Manifolds\n"
        "  - PCA: 6 components retain 74.0% variance\n"
        "  - t-SNE 2D unrolls non-linear risk geometry\n"
        "• Outlier & Anomaly Detection\n"
        "  - Isolation Forest flags 175 multi-crisis cases\n"
        "  - Distinguishes recording artifacts from risk"
    )
    ax.text(5.1, 5.8, stage2_text, ha="left", va="center", fontsize=8.5, color="#222222", linespacing=1.3)

    # Box 3: Stage 3 Supervised ML
    rect3 = patches.FancyBboxPatch((9.3, 4.5), 4.2, 3.4, boxstyle="round,pad=0.2",
                                  ec=c_orange, fc=c_light_orange, lw=2)
    ax.add_patch(rect3)
    ax.text(11.4, 7.6, "STAGE 3: SUPERVISED ML PIPELINES", ha="center", va="center",
            fontsize=11, fontweight="bold", color=c_orange)
    stage3_text = (
        "• Dual Modeling Tracks (Separate Models)\n"
        "  - Track A: Population Survey (BRFSS N=253k)\n"
        "  - Track B: Clinical Diagnostics (Pima, Cleveland)\n"
        "• 12 Candidate Models (LogReg, DT, RF)\n"
        "  - 5-Fold Stratified CV on training set only\n"
        "  - Sigmoid probability calibration\n"
        "  - 75% Sensitivity triage operating thresholds\n"
        "• Calibrated 3-Tier Risk Assignment\n"
        "  - Low, Medium, High risk tiers\n"
        "  - Lift up to 3.57x baseline prevalence"
    )
    ax.text(9.5, 5.8, stage3_text, ha="left", va="center", fontsize=8.5, color="#222222", linespacing=1.3)

    # Box 4: Explainability & Robust Validation
    rect4 = patches.FancyBboxPatch((1.5, 0.4), 5.0, 3.2, boxstyle="round,pad=0.2",
                                  ec=c_purple, fc=c_light_purple, lw=2)
    ax.add_patch(rect4)
    ax.text(4.0, 3.3, "EXPLAINABILITY & VALIDATION", ha="center", va="center",
            fontsize=11, fontweight="bold", color=c_purple)
    val_text = (
        "• Global & Patient-Level SHAP Attributions\n"
        "  - Exact biometric risk contributions (+ / -)\n"
        "  - Waterfall plots & beeswarm summary plots\n"
        "  - Explanation stability across seeds (87.3% overlap)\n"
        "• Comprehensive Statistical Auditing\n"
        "  - 1,000-resample bootstrap 95% confidence intervals\n"
        "  - Demographic equity: Subgroups by age and biological sex\n"
        "  - External longitudinal check: BRFSS 2022 (N=60,000)"
    )
    ax.text(1.7, 1.8, val_text, ha="left", va="center", fontsize=8.5, color="#222222", linespacing=1.3)

    # Box 5: Production Deployment & Clinical Interface
    rect5 = patches.FancyBboxPatch((7.5, 0.4), 5.5, 3.2, boxstyle="round,pad=0.2",
                                  ec=c_green, fc=c_light_green, lw=2)
    ax.add_patch(rect5)
    ax.text(10.25, 3.3, "CLINICAL DECISION SUPPORT SYSTEM", ha="center", va="center",
            fontsize=11, fontweight="bold", color=c_green)
    deploy_text = (
        "• Multi-Modal Patient Triage CLI (src/predict_patient.py)\n"
        "  - Modes: --mode survey | --mode clinical | --mode both\n"
        "  - Real biometric units (mg/dL, mmHg, kg/m²)\n"
        "  - Automated graceful fallback for missing lab fields\n"
        "• Interactive Web Application (app/streamlit_app.py)\n"
        "  - Dynamic feature inputs, live gauges, tier badges\n"
        "  - Embedded SHAP waterfalls & cluster phenotype mapping\n"
        "• Clinical Disclaimers & Patient Safety Enforced"
    )
    ax.text(7.7, 1.8, deploy_text, ha="left", va="center", fontsize=8.5, color="#222222", linespacing=1.3)

    # Connecting Arrows
    arrow_props = dict(facecolor="#555555", edgecolor="#333333", width=2.0, headwidth=7, shrink=0.08)

    # Stage 1 -> Stage 2
    ax.annotate("", xy=(4.9, 6.2), xytext=(4.3, 6.2), arrowprops=arrow_props)
    # Stage 2 -> Stage 3
    ax.annotate("", xy=(9.3, 6.2), xytext=(8.7, 6.2), arrowprops=arrow_props)
    # Stage 3 -> Validation
    ax.annotate("", xy=(5.0, 3.6), xytext=(9.8, 4.5), arrowprops=arrow_props)
    # Stage 3 -> Deployment
    ax.annotate("", xy=(10.0, 3.6), xytext=(11.0, 4.5), arrowprops=arrow_props)
    # Validation -> Deployment
    ax.annotate("", xy=(7.5, 2.0), xytext=(6.5, 2.0), arrowprops=arrow_props)

    plt.tight_layout()
    out_path = REPORTS_FIG_DIR / "fig_stage_flow.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] Generated {out_path}")

def generate_model_comparison_bar():
    csv_path = BASE_DIR / "outputs" / "tables" / "model_comparison_metrics.csv"
    df = pd.read_csv(csv_path)

    # Prepare labels and data
    model_labels = []
    f1_scores = []
    roc_aucs = []

    label_map = {
        ("track_a", "diabetes", "logistic_regression"): "Trk A Diab: LogReg *",
        ("track_a", "diabetes", "decision_tree"): "Trk A Diab: DecTree",
        ("track_a", "diabetes", "random_forest"): "Trk A Diab: RandFor",
        ("track_a", "heart", "logistic_regression"): "Trk A Heart: LogReg *",
        ("track_a", "heart", "decision_tree"): "Trk A Heart: DecTree",
        ("track_a", "heart", "random_forest"): "Trk A Heart: RandFor",
        ("track_b", "diabetes", "logistic_regression"): "Trk B Diab: LogReg *",
        ("track_b", "diabetes", "decision_tree"): "Trk B Diab: DecTree",
        ("track_b", "diabetes", "random_forest"): "Trk B Diab: RandFor",
        ("track_b", "heart", "logistic_regression"): "Trk B Heart: LogReg *",
        ("track_b", "heart", "decision_tree"): "Trk B Heart: DecTree",
        ("track_b", "heart", "random_forest"): "Trk B Heart: RandFor",
    }

    for _, row in df.iterrows():
        key = (row["Track"], row["Disease"], row["Model"])
        label = label_map.get(key, f"{row['Track']} {row['Disease']} {row['Model']}")
        model_labels.append(label)
        f1_scores.append(row["Test_F1"])
        roc_aucs.append(row["Test_ROC_AUC"])

    x = np.arange(len(model_labels))
    width = 0.38

    fig, ax = plt.subplots(figsize=(13, 6.5), dpi=300)
    rects1 = ax.bar(x - width/2, f1_scores, width, label="Test F1-Score (Operating Threshold)", color="#1F497D", alpha=0.9)
    rects2 = ax.bar(x + width/2, roc_aucs, width, label="Test ROC-AUC", color="#C55A11", alpha=0.9)

    ax.set_ylabel("Metric Score", fontsize=11, fontweight="bold")
    ax.set_title("Supervised Model Benchmark: Test F1-Score and ROC-AUC Across All 12 Candidate Models\n(* Asterisk denotes designated Champion Model for each disease track)",
                 fontsize=12, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(model_labels, rotation=45, ha="right", fontsize=9.5)
    ax.set_ylim(0, 1.1)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(frameon=True, facecolor="#F9F9F9", edgecolor="#CCCCCC", loc="upper left", fontsize=10)

    # Attach score values above bars
    for rect in rects1:
        height = rect.get_height()
        ax.annotate(f"{height:.2f}",
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=7.5, color="#1F497D", fontweight="bold")

    for rect in rects2:
        height = rect.get_height()
        ax.annotate(f"{height:.2f}",
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=7.5, color="#C55A11", fontweight="bold")

    # Group divider lines
    for div_x in [2.5, 5.5, 8.5]:
        ax.axvline(div_x, color="#888888", linestyle=":", linewidth=1.2)

    plt.tight_layout()
    out_path = REPORTS_FIG_DIR / "fig_model_comparison_bar.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] Generated {out_path}")

if __name__ == "__main__":
    generate_flow_diagram()
    generate_model_comparison_bar()
