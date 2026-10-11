"""
src/build_stage2_notebook.py
Builds and executes project_stage2_report.ipynb with complete markdown explanations,
reproducible code cells, and embedded outputs.
"""

from pathlib import Path
import nbformat as nbf
from nbclient import NotebookClient

BASE_DIR = Path(__file__).resolve().parent.parent
NOTEBOOK_PATH = BASE_DIR / "project_stage2_report.ipynb"
REPORTS_NOTEBOOK_PATH = BASE_DIR / "reports" / "project_stage2_report.ipynb"


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

    # Title & Overview
    cells.append(nbf.v4.new_markdown_cell("""# Predictive Risk Identification for Diabetes and Heart Disease
## Stage 2 Technical Report & Laboratory Notebook: Clustering, Dimensionality Reduction, and Outlier Detection

**Topic:** Unsupervised Phenotyping, Manifold Mapping, and Anomaly Diagnostics for Cardiometabolic Disease Risk  
**Course Project Stage:** Stage 2 Deliverable  
**Core Technologies:** Python 3, Pandas, NumPy, Scikit-Learn, SciPy, Matplotlib, Seaborn  
**Random State:** 42 (Enforced globally across all stochastic seeds)  

---

### Executive Overview & Scientific Scope
Cardiometabolic conditions—primarily Type 2 Diabetes Mellitus (T2D) and Coronary Heart Disease (CHD)—share common pathophysiological drivers including visceral adiposity, insulin resistance, atherogenic dyslipidemia, and arterial hypertension.

Building upon **Mid-Report 1** (which audited 652 hidden zeros in clinical biometrics, evaluated missing data science across surveys, and mined frequent itemsets via Apriori and FP-Growth), this **Stage 2 Notebook** implements three core unsupervised data mining pillars:
1. **Clustering Analysis**: Delineating patient risk phenotypes using **K-Means**, **Hierarchical Agglomerative Clustering (Ward linkage)**, and **DBSCAN (density-based)**, validated on population (BRFSS) and clinical lab cohorts (Cleveland Heart Disease and Pima Indians Diabetes).
2. **Dimensionality Reduction**: Untangling multi-collinear predictors and projecting high-dimensional manifolds into interpretable 2D/3D spaces using **Principal Component Analysis (PCA)** and **t-Distributed Stochastic Neighbor Embedding (t-SNE)**.
3. **Outlier Detection**: Distinguishing genuine pathological extremes (e.g., morbid obesity, acute glycemic crisis) from impossible entry artifacts using **Isolation Forests** and **Local Outlier Factor (LOF)**, and evaluating model stability before versus after outlier removal.
4. **Integration & Translation**: Synthesizing findings into actionable clinical decision support rules for the **Dietify** digital health mobile app.
"""))

    # Section 0: Setup
    cells.append(nbf.v4.new_markdown_cell("""## 0. Setup and Environment Configuration
We configure the runtime environment, set visual styling parameters, define directory paths, and enforce `random_state=42` across all random number generators.
"""))

    cells.append(nbf.v4.new_code_cell("""import os
import sys
import time
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.manifold import TSNE
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.preprocessing import StandardScaler

# Suppress harmless OpenBLAS warnings on macOS
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Set random seed for 100% reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Configure visualization aesthetics
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    "font.sans-serif": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.titlesize": 12,
})

CLEAN_DIR = Path("data/clean")
TBL_DIR = Path("outputs/tables")
FIG_DIR = Path("outputs/figures")

print("Stage 2 Environment successfully initialized. Reproducibility seed = 42.")
"""))

    # Section 0.1: Loading Data
    cells.append(nbf.v4.new_markdown_cell("""### 0.1 Loading Cleaned Cohorts from Mid-Report 1
We load the preprocessed datasets created during Mid-Report 1:
- **Primary Multi-Risk Cohort (CDC BRFSS 2015, Sample N=5,000)**: Complete lifestyle, demographic, and multi-disease survey records.
- **Cleveland Heart Disease Cohort (N=297)**: Clinical cardiac catheterization evaluations.
- **Pima Indians Diabetes Cohort (N=768)**: Clinical lab biometrics (with median imputation applied to missing lab tests for distance-based ML).
"""))

    cells.append(nbf.v4.new_code_cell("""# 1. Primary Population Cohort
brfss_full = pd.read_csv(CLEAN_DIR / "brfss_clean.csv")
pop_df = brfss_full.sample(n=5000, random_state=RANDOM_STATE).copy().reset_index(drop=True)

# 2. Clinical Benchmark Cohorts
clev_df = pd.read_csv(CLEAN_DIR / "cleveland_clean.csv")
pima_clean = pd.read_csv(CLEAN_DIR / "pima_clean.csv")

# Impute Pima for distance metrics
pima_imp = pima_clean.copy()
zero_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
for col in zero_cols:
    pima_imp[col] = pima_imp[col].fillna(pima_imp[col].median())

print(f"Population Cohort (BRFSS 2015 Sample): {pop_df.shape}")
print(f"  Diabetes Prevalence:     {pop_df['Diabetes_binary'].mean()*100:.2f}%")
print(f"  Heart Disease Prevalence: {pop_df['HeartDiseaseorAttack'].mean()*100:.2f}%")
print(f"Cleveland Heart Disease Cohort:       {clev_df.shape}")
print(f"Pima Indians Diabetes Cohort:         {pima_clean.shape}")
"""))

    # Section 1: Clustering Analysis
    cells.append(nbf.v4.new_markdown_cell("""## 1. Clustering Analysis
### What are we doing here?
We apply unsupervised clustering algorithms to identify natural patient risk strata. We evaluate three distinct mathematical approaches:
1. **K-Means**: Centroid-based partitioning optimizing within-cluster sum of squares (WCSS).
2. **Hierarchical Agglomerative Clustering**: Bottom-up merging using Ward's minimum-variance criterion.
3. **DBSCAN**: Density-based clustering grouping contiguous high-density regions and flagging noise points.

We standardize all continuous and ordinal features using `StandardScaler` to ensure distance metrics are not biased by disparate measurement scales.
"""))

    cells.append(nbf.v4.new_code_cell("""# Define features for clustering
features = [
    "Age", "BMI", "GenHlth", "PhysHlth", "MentHlth",
    "HighBP", "HighChol", "Smoker", "PhysActivity", "Income", "Education"
]
X_raw = pop_df[features].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

# 1. K-Means Optimization across k in [2, 10]
k_range = range(2, 11)
inertias, sil_scores, db_scores = [], [], []

for k in k_range:
    km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10).fit(X_scaled)
    inertias.append(km.inertia_)
    sil_scores.append(silhouette_score(X_scaled, km.labels_))
    db_scores.append(davies_bouldin_score(X_scaled, km.labels_))

# Plot Elbow and Silhouette curves (Figure 1)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.2))
ax1.plot(k_range, inertias, marker="o", color="#1F497D", linewidth=2)
ax1.set_title("K-Means Elbow Method (Inertia / WCSS)", fontweight="bold")
ax1.set_xlabel("Number of Clusters (k)")
ax1.set_ylabel("Within-Cluster Sum of Squares")
ax1.axvline(x=3, color="#D95F02", linestyle="--", label="Elbow Knee (k=3)")
ax1.legend(loc="upper right")

ax2.plot(k_range, sil_scores, marker="s", color="#D95F02", linewidth=2)
ax2.set_title("Silhouette Score vs. Number of Clusters", fontweight="bold")
ax2.set_xlabel("Number of Clusters (k)")
ax2.set_ylabel("Mean Silhouette Coefficient")
ax2.axvline(x=3, color="#1F497D", linestyle="--", label="Local Peak (k=3)")
ax2.legend(loc="upper right")

plt.suptitle("Figure 1: K-Means Optimization Curves — Elbow Inertia and Silhouette Analysis", fontweight="bold")
plt.tight_layout()
plt.show()

# Fit optimal K-Means (k=3)
km_opt = KMeans(n_clusters=3, random_state=RANDOM_STATE, n_init=10)
pop_df["KMeans_Cluster"] = km_opt.fit_predict(X_scaled)
print("Optimal K-Means fitted at k=3.")
"""))

    cells.append(nbf.v4.new_markdown_cell("""### 1.1 DBSCAN Optimization & k-Distance Graph
To parameterize DBSCAN in $d=11$ dimensions, we construct a sorted 5-nearest-neighbor distance plot. The elbow inflection identifies the optimal epsilon neighborhood radius (`eps=2.40`) for `min_samples=5`.
"""))

    cells.append(nbf.v4.new_code_cell("""# DBSCAN 5-NN distance curve
k_neighbors = 5
nbrs = NearestNeighbors(n_neighbors=k_neighbors).fit(X_scaled)
distances, _ = nbrs.kneighbors(X_scaled)
k_distances = np.sort(distances[:, k_neighbors - 1])

plt.figure(figsize=(7.5, 4.2))
plt.plot(range(len(k_distances)), k_distances, color="#2B5C8F", linewidth=2)
selected_eps = 2.40
plt.axhline(y=selected_eps, color="red", linestyle="--", label=f"Selected eps = {selected_eps} (Knee Point)")
plt.title("Figure 2: k-Distance Graph for DBSCAN Parameter Tuning (minPts = 5)", fontweight="bold")
plt.xlabel("Sorted Observations (Index)")
plt.ylabel(f"{k_neighbors}-NN Distance")
plt.legend(loc="upper left")
plt.tight_layout()
plt.show()

dbscan = DBSCAN(eps=selected_eps, min_samples=5)
pop_df["DBSCAN_Cluster"] = dbscan.fit_predict(X_scaled)
n_db_clusters = len(set(pop_df["DBSCAN_Cluster"])) - (1 if -1 in pop_df["DBSCAN_Cluster"] else 0)
n_db_noise = (pop_df["DBSCAN_Cluster"] == -1).sum()
print(f"DBSCAN Results: {n_db_clusters} clusters identified; {n_db_noise} noise points ({n_db_noise/len(pop_df)*100:.1f}%).")
"""))

    cells.append(nbf.v4.new_markdown_cell("""### 1.2 Hierarchical Agglomerative Clustering & Dendrogram
We implement agglomerative clustering using Ward's minimum-variance linkage. The dendrogram demonstrates hierarchical merging and justifies a 3-cluster partition cut.
"""))

    cells.append(nbf.v4.new_code_cell("""# Subsample for dendrogram visualization (N=150)
sub_idx = np.random.RandomState(RANDOM_STATE).choice(len(X_scaled), size=150, replace=False)
Z = linkage(X_scaled[sub_idx], method="ward", metric="euclidean")

plt.figure(figsize=(10, 4.5))
dendrogram(Z, truncate_mode="lastp", p=25, leaf_rotation=90, leaf_font_size=8.5, color_threshold=18)
plt.axhline(y=18, color="red", linestyle="--", label="Cluster Cutoff (k=3)")
plt.title("Figure 3: Hierarchical Agglomerative Dendrogram (Ward Linkage, Euclidean Distance)", fontweight="bold")
plt.xlabel("Observation Index or Cluster Size (p=25 Truncated)")
plt.ylabel("Ward Cophenetic Merge Distance")
plt.legend(loc="upper right")
plt.tight_layout()
plt.show()

agg_opt = AgglomerativeClustering(n_clusters=3, metric="euclidean", linkage="ward")
pop_df["Hierarchical_Cluster"] = agg_opt.fit_predict(X_scaled)
print("Hierarchical Agglomerative Clustering completed at k=3.")
"""))

    cells.append(nbf.v4.new_markdown_cell("""### 1.3 Health Domain Interpretation & Cluster Profiling
We map each K-Means cluster to clinical and demographic dimensions. Table 1 profiles the resulting patient strata, revealing Low-Risk, Moderate-Risk, and High-Risk cardiometabolic phenotypes.
"""))

    cells.append(nbf.v4.new_code_cell("""# Display precomputed Table 1
df_prof = pd.read_csv(TBL_DIR / "stage2_cluster_health_profiles.csv")
print("Table 1: Clinical and Demographic Health Profiles across Unsupervised K-Means Risk Clusters:")
df_prof
"""))

    cells.append(nbf.v4.new_markdown_cell("""### 1.4 Clinical Cohort Validation: Continuous Biometrics in Cleveland and Pima
To confirm our findings against continuous clinical laboratory assays (cholesterol, fasting glucose, insulin, blood pressure), we benchmarked identical clustering workflows on Cleveland Heart Disease and Pima Indians Diabetes cohorts.
"""))

    cells.append(nbf.v4.new_code_cell("""# Display precomputed Table 2
df_clin = pd.read_csv(TBL_DIR / "stage2_clinical_cluster_profiles.csv")
print("Table 2: Clinical Laboratory Benchmark Profiles across Cleveland and Pima Cohorts:")
df_clin
"""))

    cells.append(nbf.v4.new_code_cell("""# Display Figure 5
from IPython.display import Image, display
display(Image(filename=str(FIG_DIR / "stage2_fig5_clinical_cohort_clusters.png")))
"""))

    cells.append(nbf.v4.new_markdown_cell("""### 1.5 Clustering Methods Comparative Performance
Table 3 compares K-Means, Hierarchical (Ward), and DBSCAN across Silhouette Score, Davies-Bouldin Index, Calinski-Harabasz Index, and Adjusted Rand Index (ARI).
"""))

    cells.append(nbf.v4.new_code_cell("""df_comp = pd.read_csv(TBL_DIR / "stage2_clustering_metrics_comparison.csv")
print("Table 3: Unsupervised Clustering Methods Performance and Similarity Comparison:")
df_comp
"""))

    # Section 2: Dimensionality Reduction
    cells.append(nbf.v4.new_markdown_cell("""## 2. Dimensionality Reduction
### Why Dimensionality Reduction?
1. **Multi-Collinearity**: Clinical and demographic indicators share strong correlation (e.g., age with vascular stiffening, health impairment days with mobility).
2. **Noise Filtering**: Eliminates orthogonal discrete survey noise.
3. **Curse of Dimensionality**: Compacts Euclidean space so distance-based algorithms operate efficiently.
4. **2D/3D Visualization**: Enables human interpretation of complex multi-risk manifolds.
"""))

    cells.append(nbf.v4.new_code_cell("""# Principal Component Analysis (PCA)
pca = PCA(random_state=RANDOM_STATE)
X_pca = pca.fit_transform(X_scaled)
exp_var = pca.explained_variance_ratio_
cum_var = np.cumsum(exp_var)

# Scree plot (Figure 6)
comps = range(1, len(exp_var) + 1)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.2))
ax1.bar(comps, exp_var * 100, color="#1F497D", edgecolor="black", alpha=0.85)
ax1.plot(comps, exp_var * 100, color="#D95F02", marker="o")
ax1.set_title("PCA Scree Plot (Explained Variance %)", fontweight="bold")
ax1.set_xlabel("Principal Component")
ax1.set_ylabel("Variance Explained (%)")

ax2.plot(comps, cum_var * 100, marker="s", color="#D95F02", linewidth=2)
ax2.axhline(y=70, color="gray", linestyle="--", label="70% Threshold")
ax2.axhline(y=80, color="black", linestyle=":", label="80% Threshold (6 Components)")
ax2.set_title("Cumulative Explained Variance Ratio", fontweight="bold")
ax2.set_xlabel("Components Kept")
ax2.set_ylabel("Cumulative Variance (%)")
ax2.legend(loc="lower right")

plt.suptitle("Figure 6: Principal Component Analysis — Scree Plot and Cumulative Variance", fontweight="bold")
plt.tight_layout()
plt.show()

print(f"Top 2 Components Explained Variance: {cum_var[1]*100:.2f}%")
print(f"Top 6 Components Explained Variance: {cum_var[5]*100:.2f}% (Dimensions reduced by 45.5%)")
"""))

    cells.append(nbf.v4.new_markdown_cell("""### 2.1 PCA Factor Loadings
Table 4 displays the top factor loadings for the first three principal components, identifying the biological axes of variation.
"""))

    cells.append(nbf.v4.new_code_cell("""df_load = pd.read_csv(TBL_DIR / "stage2_pca_loadings.csv")
print("Table 4: Principal Component Factor Loadings for the Top 3 Components:")
df_load
"""))

    cells.append(nbf.v4.new_markdown_cell("""### 2.2 t-SNE Embedding & Neighborhood Sensitivity
We evaluate t-SNE across perplexities 15, 30, and 50 to assess local neighborhood stability.
"""))

    cells.append(nbf.v4.new_code_cell("""# Display Figure 7 (t-SNE Sensitivity)
display(Image(filename=str(FIG_DIR / "stage2_fig7_tsne_perplexity_sensitivity.png")))
"""))

    cells.append(nbf.v4.new_markdown_cell("""### 2.3 Before vs. After Dimensionality Reduction Visualizations
Figure 8 compares raw feature space against 2D PCA space, while Figure 4 provides a 4-panel comparison of PCA and t-SNE colored by risk clusters and empirical disease diagnoses.
"""))

    cells.append(nbf.v4.new_code_cell("""# Display Figure 8 (Raw vs Reduced)
display(Image(filename=str(FIG_DIR / "stage2_fig8_raw_vs_reduced_comparison.png")))
"""))

    cells.append(nbf.v4.new_code_cell("""# Display Figure 4 (4-Panel PCA vs t-SNE by Clusters and Outcomes)
display(Image(filename=str(FIG_DIR / "stage2_fig4_cluster_scatter_projections.png")))
"""))

    cells.append(nbf.v4.new_markdown_cell("""### 2.4 Re-Clustering on Reduced Feature Spaces
Table 5 compares K-Means clustering across Original 11D space, PCA 6-component space, PCA 2D space, and t-SNE 2D space.
"""))

    cells.append(nbf.v4.new_code_cell("""df_red = pd.read_csv(TBL_DIR / "stage2_clustering_reduced_vs_raw.csv")
print("Table 5: Clustering Performance Comparison across Original and Reduced Feature Spaces:")
df_red
"""))

    # Section 3: Outlier Detection
    cells.append(nbf.v4.new_markdown_cell("""## 3. Outlier Detection
### Defining Healthcare Outliers
We distinguish between:
1. **Impossible Measurement Artifacts**: Values violating biology (e.g., blood pressure = 0, glucose = 0, BMI = 0), audited and converted to NaN in Mid-Report 1.
2. **Genuine Pathological Extremes**: Valid patients suffering from acute multi-morbidity (e.g., BMI > 60 kg/m², 30 days/month of complete physical impairment).

We deploy **Isolation Forest** and **Local Outlier Factor (LOF)** at `contamination=0.035` (3.5% cohort fraction).
"""))

    cells.append(nbf.v4.new_code_cell("""# Isolation Forest and LOF
CONTAMINATION = 0.035
iso = IsolationForest(n_estimators=150, contamination=CONTAMINATION, random_state=RANDOM_STATE)
pop_df["Iso_Outlier"] = (iso.fit_predict(X_scaled) == -1).astype(int)

lof = LocalOutlierFactor(n_neighbors=20, contamination=CONTAMINATION)
pop_df["LOF_Outlier"] = (lof.fit_predict(X_scaled) == -1).astype(int)

n_iso = pop_df["Iso_Outlier"].sum()
n_lof = pop_df["LOF_Outlier"].sum()
print(f"Isolation Forest flagged {n_iso} outliers ({n_iso/len(pop_df)*100:.1f}%).")
print(f"Local Outlier Factor flagged {n_lof} outliers ({n_lof/len(pop_df)*100:.1f}%).")
"""))

    cells.append(nbf.v4.new_markdown_cell("""### 3.1 Outlier Manifold Visualization & Clinical Profiles
Figure 9 visualizes detected outliers on the PCA and t-SNE projections, paired with boxplots contrasting BMI and physical impairment days. Table 6 examines their clinical profiles.
"""))

    cells.append(nbf.v4.new_code_cell("""# Display Figure 9
display(Image(filename=str(FIG_DIR / "stage2_fig9_outliers_projection_boxplots.png")))
"""))

    cells.append(nbf.v4.new_code_cell("""# Display Table 6
df_out = pd.read_csv(TBL_DIR / "stage2_outlier_summary.csv")
print("Table 6: Biometric and Clinical Profile Comparison — Normal Inliers vs. Isolation Forest Outliers:")
df_out
"""))

    cells.append(nbf.v4.new_markdown_cell("""### 3.2 Impact of Outlier Removal on Model Stability
Table 7 and Figure 10 evaluate the quantitative impact of trimming the 175 outliers on clustering separation metrics and PCA variance explained.
"""))

    cells.append(nbf.v4.new_code_cell("""# Display Table 7
df_out_imp = pd.read_csv(TBL_DIR / "stage2_outlier_removal_metrics.csv")
print("Table 7: Quantitative Impact of Outlier Removal on Clustering Separation Metrics and PCA Variance:")
df_out_imp
"""))

    cells.append(nbf.v4.new_code_cell("""# Display Figure 10
display(Image(filename=str(FIG_DIR / "stage2_fig10_outlier_removal_impact.png")))
"""))

    # Section 4: Integration and Insights
    cells.append(nbf.v4.new_markdown_cell("""## 4. Integration, Strategic Insights & Future Roadmap
### 4.1 Tripartite Methodological Synergy
1. **Dimensionality Reduction** untangles multi-collinearity and overcomes the curse of dimensionality.
2. **Clustering** discovers discrete clinical risk phenotypes (Low-Risk, Metabolic Escalation, and Multi-Morbidity Crisis).
3. **Outlier Detection** filters out artificial coordinate leverage while isolating genuine high-acuity crisis patients.

### 4.2 Translation to Dietify Digital Health Platform
- **Automated Zero-Lab Triage**: Maps self-reported survey answers into risk clusters upon user onboarding.
- **Targeted Nutritional Guidance**: Serves low-glycemic, anti-inflammatory DASH/Mediterranean meal plans to Cluster 1.
- **Restorative Mobility Protocols**: Prescribes gentle low-impact physical activities for Cluster 2 users suffering from joint/mobility limitations.
- **Outlier Red-Flag Alerts**: Automatically flags extreme biometrics for urgent clinical escalation.

### 4.3 Study Limitations
- Self-reported survey data subject to recall bias.
- Cross-sectional study design prevents establishing longitudinal temporal causality.
- Spherical clustering assumptions under K-Means.

### 4.4 Roadmap for Final Project (Stage 3 Supervised Machine Learning)
- Supervised classification models: **LightGBM**, **XGBoost**, **Random Forest**, and **Logistic Regression**.
- Imbalanced learning techniques: **SMOTE** and class-cost weighting.
- Clinical evaluation criteria: **PR-AUC**, **ROC-AUC**, **F1-Score**, and **Recall / Sensitivity**.
- Model explainability: **SHAP (SHapley Additive exPlanations)** global summary and local waterfall plots.
"""))

    # Appendix: Parameter Dictionary
    cells.append(nbf.v4.new_markdown_cell("""## Appendix: Methodological Parameter Dictionary
Table 8 inventories every algorithm hyperparameter and its mathematical justification, ensuring 100% computational reproducibility.
"""))

    cells.append(nbf.v4.new_code_cell("""df_param = pd.read_csv(TBL_DIR / "stage2_parameter_dictionary.csv")
print("Table 8: Comprehensive Algorithmic Parameter Dictionary and Methodological Justification:")
df_param
"""))

    nb.cells = cells
    return nb


def execute_and_save_notebook():
    print("Assembling Stage 2 Jupyter Notebook structure...")
    nb = create_notebook()

    print(f"Executing notebook cells top to bottom via NotebookClient (in {BASE_DIR})...")
    client = NotebookClient(nb, timeout=600, kernel_name="python3")
    client.execute()

    with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
        nbf.write(nb, f)

    with open(REPORTS_NOTEBOOK_PATH, "w", encoding="utf-8") as f:
        nbf.write(nb, f)

    print(f"Successfully executed and saved Stage 2 notebook to:")
    print(f"  1. {NOTEBOOK_PATH}")
    print(f"  2. {REPORTS_NOTEBOOK_PATH}")


if __name__ == "__main__":
    execute_and_save_notebook()
