# Final Project Report v2 Build Log & Data Source Cross-Check

**Project:** Predictive Risk Identification for Diabetes and Heart Disease (Chronic Disease Predictor Model)  
**Document Generated:** `reports/final_project_report_v2.docx` & `reports/final_project_report_v2.pdf`  
**Edition:** Version 2.0 (Post-Packaging & Reorganization Release)  
**Execution Timestamp:** 2026-10-10  
**Verification Standard:** 100% Grounded in Real Project Output Files (Zero Fabricated or Rounded Data)

---

## 1. Table-to-Source Mapping and Cross-Checks

| Report Table Number & Title | Source Project File | Key Verified Numbers / Cross-Check | Verification Status |
| :--- | :--- | :--- | :---: |
| **Table 1: Comprehensive Dataset Inventory** | `outputs/tables/dataset_inventory.csv` | BRFSS 2015: 253,680 rows, 22 cols; Pima: 768 rows; Cleveland: 297 rows; BRFSS 2022: 445,132 rows | **VERIFIED [OK]** |
| **Table 2: Pima Hidden Zeros Audit** | `outputs/tables/pima_missing_summary.csv` | Glucose: 5 zeros (0.65%); BP: 35 zeros (4.56%); SkinThickness: 227 zeros (29.56%); Insulin: 374 zeros (48.70%); BMI: 11 zeros (1.43%); Total 652 | **VERIFIED [OK]** |
| **Table 3: Integrated Structural Missingness** | `outputs/tables/integrated_missing_summary.csv` | N=26,065; Age/Sex: 0% missing; Cholesterol: 98.86% missing; Insulin: 98.49% missing; BloodPressure (continuous): 96.05% missing | **VERIFIED [OK]** |
| **Table 4: Feature Skewness & Normalization** | `outputs/tables/normalization_transformation_summary.csv` | Insulin skew: +2.17 -> -0.09; BMI: +2.00 -> +0.69; Cholesterol: +1.12 -> +0.06; Age: -0.36 -> -1.01 | **VERIFIED [OK]** |
| **Table 5: Frequent Pattern Support Sensitivity** | `outputs/tables/support_variation_summary.csv` | Support 0.40: 17 itemsets, 0 target rules; Support 0.05: 611 itemsets, 5,836 total rules, 45 diabetes rules, 17 heart rules | **VERIFIED [OK]** |
| **Table 6: Top Multi-Morbidity Association Rules** | `outputs/tables/top_diabetes_rules_by_confidence.csv` | HighChol, BMI>=30, Age>=50 -> Diabetes: Support = 5.08%, Confidence = 35.66%, Lift = 2.584 | **VERIFIED [OK]** |
| **Table 7: Clustering Algorithm Benchmark** | `outputs/tables/stage2_clustering_metrics_comparison.csv` | K-Means (k=3): Silhouette = 0.1534, DB = 2.1061, CH = 860.45; Hierarchical: Silhouette = 0.1193; DBSCAN: 166 noise points | **VERIFIED [OK]** |
| **Table 8: Discovered Patient Phenotypes** | `outputs/tables/stage2_cluster_health_profiles.csv` | Cluster 0: N=2,339 (46.8%), Diab=3.2%, Heart=1.8%; Cluster 1: N=1,956 (39.1%), Diab=21.6%, Heart=14.4%; Cluster 2: N=705 (14.1%), Diab=27.0%, Heart=23.0% | **VERIFIED [OK]** |
| **Table 9: Dimensionality Reduction Comparison** | `outputs/tables/stage2_clustering_reduced_vs_raw.csv` | Raw 11D: Silhouette = 0.1534; PCA 6D: Silhouette = 0.2046, DB = 1.7054 (73.98% variance); PCA 2D: Silhouette = 0.4050 | **VERIFIED [OK]** |
| **Table 10: Supervised Model Benchmark (12 Models)** | `outputs/tables/model_comparison_metrics.csv` | Trk A Diab LogReg: ROC-AUC=0.8190, PR-AUC=0.3913, Recall=0.7479, F1=0.4431; Trk A Heart LogReg: ROC-AUC=0.8458, PR-AUC=0.3594, Recall=0.7552, F1=0.3900 | **VERIFIED [OK]** |
| **Table 11: Clinical Value-Add vs. Baselines** | `outputs/tables/baseline_comparison.csv` | Trk A Diab: Champion ROC-AUC 0.8190 vs Simple 3-Feature 0.7709 (+0.0481) vs Naive 0.5000; Trk B Heart: Champion 0.9275 vs Simple 0.5915 (+0.3360) | **VERIFIED [OK]** |
| **Table 12: Champion Models & Thresholds** | `outputs/tables/champion_models.csv` | Trk A Diab thresh: 0.1473; Trk A Heart thresh: 0.1101; Trk B Diab thresh: 0.3318; Trk B Heart thresh: 0.5672 | **VERIFIED [OK]** |
| **Table 13: Risk Tier Calibration Summary** | `outputs/tables/risk_tier_calibration_summary.csv` | Trk A Diab: Low Prev=2.69%, Med=17.26%, High=40.83% (Lift=2.93x, Med+High Capture=90.64%); Trk A Heart: Low=1.51%, Med=11.36%, High=33.60% (Lift=3.57x) | **VERIFIED [OK]** |
| **Table 14: Top 5 Global SHAP Features** | `outputs/tables/shap_feature_importance.csv` | Trk A Diab: GenHlth (0.5359), Age (0.3916), HighBP (0.3629), BMI (0.3488), HighChol (0.2815); Trk A Heart: Age (0.6726), GenHlth (0.4540), Sex (0.4057) | **VERIFIED [OK]** |
| **Table 15: Individual Patient Assessments** | Example JSONs & Test Patient Data | PT-LOW-001 (Diab=0.01, Heart=0.00); PT-MED-002 (Diab=0.11, Heart=0.08); PT-HIGH-003 (Diab=0.76, Heart=0.65); False Neg #15 (Prob=0.2320, True=1) | **VERIFIED [OK]** |
| **Table 16: Robustness Validation Verdicts** | `outputs/tables/validation_summary.md` | Statistical CIs (PASS), Sensitivity >=75% (PASS), Cross-Fold (PASS), Demographics (PASS), External 2022 (PASS), Stability (PASS), Small-Sample (WEAK) | **VERIFIED [OK]** |
| **Table 17: Champion Hyperparameters** | Model Joblib Objects (`models/*.joblib`) | All champions: L2 penalty, lbfgs solver, balanced class_weight, sigmoid Platt calibration | **VERIFIED [OK]** |
| **Table 18: Software Dependencies** | Environment Configuration | scikit-learn 1.6.1, shap 0.49.1, pandas 2.2.3, numpy 2.0.2, streamlit 1.50.0, pytest 8.3.5 | **VERIFIED [OK]** |

---

## 2. Figure-to-Source Mapping and Visual Cross-Checks

| Report Figure Number & Caption | Source Image File | Description and Visual Takeaway | Verification Status |
| :--- | :--- | :--- | :---: |
| **Figure 1: End-to-End System Architecture** | `reports/figures/fig_stage_flow.png` | Comprehensive multi-stage flow diagram connecting Stage 1 through Stage 3 deployment | **VERIFIED [OK]** |
| **Figure 2: K-Means WCSS Elbow & Silhouette** | `reports/figures/fig_stage2_kmeans.png` | Diagnostic elbow knee at k=3 (Inertia=45,710.2, Silhouette=0.1534) | **VERIFIED [OK]** |
| **Figure 3: Unsupervised Patient Phenotypes** | `reports/figures/fig_stage2_clusters.png` | 2D PCA and t-SNE projections of Low, Moderate, and High risk patient clusters | **VERIFIED [OK]** |
| **Figure 4: PCA Scree & Factor Loadings** | `reports/figures/fig_stage2_pca.png` | Scree variance curve and loading matrix for top components | **VERIFIED [OK]** |
| **Figure 5: Isolation Forest Outliers** | `reports/figures/fig_stage2_outliers.png` | 175 multi-crisis outliers with extreme physical health impairment | **VERIFIED [OK]** |
| **Figure 6: Model Benchmark Bar Chart** | `reports/figures/fig_model_comparison_bar.png` | Grouped bar chart comparing F1 and ROC-AUC across all 12 pipelines | **VERIFIED [OK]** |
| **Figure 7: Overlaid ROC Curves** | `reports/figures/fig_roc_curves.png` | ROC curves across all tracks (ROC-AUC from 0.819 to 0.928) | **VERIFIED [OK]** |
| **Figure 8: Overlaid Precision-Recall Curves** | `reports/figures/fig_pr_curves.png` | PR curves illustrating precision-recall trade-offs | **VERIFIED [OK]** |
| **Figure 9: Confusion Matrices** | `reports/figures/fig_confusion_matrices.png` | True/False Positives and Negatives at calibrated operating thresholds | **VERIFIED [OK]** |
| **Figure 10: Probability Calibration Curves** | `reports/figures/fig_calibration_curves.png` | Platt sigmoid reliability diagrams before and after calibration | **VERIFIED [OK]** |
| **Figure 11-12: Track A Diabetes SHAP** | `reports/figures/fig_shap_bar_track_a_diabetes.png` & `fig_shap_beeswarm_track_a_diabetes.png` | Mean \|SHAP\| bar chart and beeswarm distribution for Track A Diabetes | **VERIFIED [OK]** |
| **Figure 13-14: Track A Heart SHAP** | `reports/figures/fig_shap_bar_track_a_heart.png` & `fig_shap_beeswarm_track_a_heart.png` | Mean \|SHAP\| bar chart and beeswarm distribution for Track A Heart Disease | **VERIFIED [OK]** |
| **Figure 15-16: Track B Clinical SHAP** | `reports/figures/fig_shap_bar_track_b_diabetes.png` & `fig_shap_bar_track_b_heart.png` | Fasting glucose, BMI, thallium scan, and heart rate attributions | **VERIFIED [OK]** |
| **Figure 17-20: Patient Waterfall Plots** | `reports/figures/fig_waterfall_*.png` | Local SHAP waterfall attributions for Low, Med, High patients | **VERIFIED [OK]** |
| **Figure 21-22: Subgroup Equity Plots** | `reports/figures/fig_subgroup_age_diabetes.png` & `fig_subgroup_sex_diabetes.png` | Sensitivity comparisons across 13 age brackets and biological sex | **VERIFIED [OK]** |

---

## 3. Executive Cross-Check Summary

- **Total Tables in Report:** 18
- **Total Figures in Report:** 22
- **Data Completeness:** 100% of reported values matched exact figures in source CSV and JSON artifacts.
- **Leakage Audit:** High glucose feature strictly banned from Track A; zero test-set overlap confirmed.
- **Clinical Safety Disclaimer:** Embedded prominently in Executive Summary and Patient Case Studies sections.
