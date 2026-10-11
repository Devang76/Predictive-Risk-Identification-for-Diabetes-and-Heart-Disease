# Stage 2 Unsupervised Models & Artifacts

This directory is designated for unsupervised model artifacts (clustering estimators, PCA transformers, and outlier detectors).

In the current project architecture:
- Stage 2 clustering analysis was executed on a standardized population cohort ($N=5,000$).
- Discovered phenotypic patient health profiles are serialized as structured tabular profiles:
  - `outputs/tables/stage2_cluster_health_profiles.csv` (General population K-Means $k=3$ cluster centroids)
  - `outputs/tables/stage2_clinical_cluster_profiles.csv` (Pima and Cleveland secondary cluster profiles)
  - `outputs/tables/stage2_pca_loadings.csv` (Principal Component Analysis loading matrix)
  - `outputs/tables/stage2_outlier_summary.csv` (Isolation Forest multi-crisis anomaly profile)
- Low-dimensional visual manifolds are stored in:
  - `outputs/figures/stage2_fig1_kmeans_elbow_silhouette.png`
  - `outputs/figures/stage2_fig4_cluster_scatter_projections.png`
  - `outputs/figures/stage2_fig6_pca_scree_loadings.png`
  - `outputs/figures/stage2_fig9_outliers_projection_boxplots.png`

The final application and CLI query `stage2_cluster_health_profiles.csv` directly for nearest-centroid Euclidean phenotype matching.
