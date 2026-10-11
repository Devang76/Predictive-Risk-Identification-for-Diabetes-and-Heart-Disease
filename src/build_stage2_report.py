"""
src/build_stage2_report.py
Generates the comprehensive Stage 2 academic and clinical data mining project report:
1. Word format: stage2_report.docx (and reports/stage2_report.docx)
2. PDF format:  stage2_report.pdf  (and reports/stage2_report.pdf)
Verifies that the compiled PDF strictly satisfies the 5–20 page requirement.
"""

import os
from pathlib import Path
import docx
from docx import Document
from docx.enum.table import WD_ALIGN_VERTICAL, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn
from docx.shared import Inches, Pt, RGBColor
from docx2pdf import convert
import pandas as pd
from pypdf import PdfReader

BASE_DIR = Path(__file__).resolve().parent.parent
FIG_DIR = BASE_DIR / "outputs" / "figures"
TBL_DIR = BASE_DIR / "outputs" / "tables"
REPORTS_DIR = BASE_DIR / "reports"

DOCX_ROOT = BASE_DIR / "stage2_report.docx"
PDF_ROOT = BASE_DIR / "stage2_report.pdf"
DOCX_REPORT = REPORTS_DIR / "stage2_report.docx"
PDF_REPORT = REPORTS_DIR / "stage2_report.pdf"


def set_cell_background(cell, fill_hex):
    """Set background color of a table cell."""
    tcPr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    tcPr.append(shd)


def set_cell_margins(cell, top=70, bottom=70, left=100, right=100):
    """Set inner padding for table cell."""
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = parse_xml(
        f'<w:tcMar {nsdecls("w")}>'
        f'<w:top w:w="{top}" w:type="dxa"/>'
        f'<w:bottom w:w="{bottom}" w:type="dxa"/>'
        f'<w:left w:w="{left}" w:type="dxa"/>'
        f'<w:right w:w="{right}" w:type="dxa"/>'
        f'</w:tcMar>'
    )
    tcPr.append(tcMar)


def add_styled_table(doc, df, caption, col_widths=None, font_size_pt=8.0):
    """Add a professional academic table with captions and formatting."""
    p_cap = doc.add_paragraph()
    p_cap.paragraph_format.space_before = Pt(6)
    p_cap.paragraph_format.space_after = Pt(2)
    p_cap.paragraph_format.keep_with_next = True
    run_cap = p_cap.add_run(caption)
    run_cap.font.name = "Calibri"
    run_cap.font.size = Pt(9.0)
    run_cap.font.bold = True
    run_cap.font.color.rgb = RGBColor(31, 73, 125)

    table = doc.add_table(rows=len(df) + 1, cols=len(df.columns))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER

    # Header Row
    hdr_cells = table.rows[0].cells
    for i, col_name in enumerate(df.columns):
        clean_name = str(col_name).replace("_", " ").title()
        hdr_cells[i].text = clean_name
        set_cell_background(hdr_cells[i], "1F497D")  # Deep Navy Blue
        set_cell_margins(hdr_cells[i], top=90, bottom=90, left=100, right=100)
        p = hdr_cells[i].paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        for r in p.runs:
            r.font.name = "Calibri"
            r.font.size = Pt(font_size_pt)
            r.font.bold = True
            r.font.color.rgb = RGBColor(255, 255, 255)

    # Data Rows
    for row_idx, row_data in enumerate(df.values):
        row_cells = table.rows[row_idx + 1].cells
        bg_color = "F2F5F9" if row_idx % 2 == 1 else "FFFFFF"
        for col_idx, val in enumerate(row_data):
            if pd.isnull(val):
                text_val = "N/A"
            elif isinstance(val, float):
                text_val = f"{val:.4f}" if 0 < abs(val) < 0.01 else f"{val:.2f}"
            else:
                text_val = str(val)

            row_cells[col_idx].text = text_val
            set_cell_background(row_cells[col_idx], bg_color)
            set_cell_margins(row_cells[col_idx], top=50, bottom=50, left=80, right=80)
            p = row_cells[col_idx].paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT if col_idx == 0 or len(text_val) > 22 else WD_ALIGN_PARAGRAPH.CENTER
            for r in p.runs:
                r.font.name = "Calibri"
                r.font.size = Pt(font_size_pt)
                r.font.color.rgb = RGBColor(40, 40, 40)

    if col_widths and len(col_widths) == len(df.columns):
        for row in table.rows:
            for idx, width in enumerate(col_widths):
                row.cells[idx].width = width

    doc.add_paragraph().paragraph_format.space_after = Pt(3)


def add_styled_figure(doc, img_path, caption, takeaway, width=Inches(5.4)):
    """Embed an image with numbered caption and clinical takeaway."""
    p_img = doc.add_paragraph()
    p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_img.paragraph_format.space_before = Pt(6)
    p_img.paragraph_format.space_after = Pt(2)
    p_img.paragraph_format.keep_with_next = True
    run_img = p_img.add_run()
    run_img.add_picture(str(img_path), width=width)

    # Caption
    p_cap = doc.add_paragraph()
    p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_cap.paragraph_format.space_before = Pt(2)
    p_cap.paragraph_format.space_after = Pt(1)
    p_cap.paragraph_format.keep_with_next = True
    run_cap = p_cap.add_run(caption)
    run_cap.font.name = "Calibri"
    run_cap.font.size = Pt(9.0)
    run_cap.font.bold = True
    run_cap.font.color.rgb = RGBColor(31, 73, 125)

    # Takeaway
    p_take = doc.add_paragraph()
    p_take.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_take.paragraph_format.space_before = Pt(1)
    p_take.paragraph_format.space_after = Pt(5)
    run_take_lbl = p_take.add_run("Key Takeaway: ")
    run_take_lbl.font.name = "Calibri"
    run_take_lbl.font.size = Pt(8.5)
    run_take_lbl.font.bold = True
    run_take_lbl.font.color.rgb = RGBColor(180, 50, 20)

    run_take_txt = p_take.add_run(takeaway)
    run_take_txt.font.name = "Calibri"
    run_take_txt.font.size = Pt(8.5)
    run_take_txt.font.italic = True
    run_take_txt.font.color.rgb = RGBColor(60, 60, 60)


def build_report_document():
    doc = Document()

    # Configure Margins (0.75 in)
    for section in doc.sections:
        section.top_margin = Inches(0.75)
        section.bottom_margin = Inches(0.75)
        section.left_margin = Inches(0.75)
        section.right_margin = Inches(0.75)

    # Base Styles
    style_normal = doc.styles["Normal"]
    font_normal = style_normal.font
    font_normal.name = "Calibri"
    font_normal.size = Pt(9.5)
    font_normal.color.rgb = RGBColor(35, 35, 35)

    # Document Header / Title
    p_title = doc.add_paragraph()
    p_title.paragraph_format.space_before = Pt(0)
    p_title.paragraph_format.space_after = Pt(2)
    p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_title = p_title.add_run("Predictive Risk Identification for Diabetes and Heart Disease")
    r_title.font.name = "Calibri"
    r_title.font.size = Pt(18)
    r_title.font.bold = True
    r_title.font.color.rgb = RGBColor(31, 73, 125)

    p_sub = doc.add_paragraph()
    p_sub.paragraph_format.space_before = Pt(0)
    p_sub.paragraph_format.space_after = Pt(6)
    p_sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_sub = p_sub.add_run("Stage 2 Technical Report: Unsupervised Clustering, Manifold Dimensionality Reduction, and Outlier Diagnostics")
    r_sub.font.name = "Calibri"
    r_sub.font.size = Pt(10.5)
    r_sub.font.italic = True
    r_sub.font.color.rgb = RGBColor(100, 100, 100)

    p_meta = doc.add_paragraph()
    p_meta.paragraph_format.space_before = Pt(0)
    p_meta.paragraph_format.space_after = Pt(10)
    p_meta.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_meta = p_meta.add_run("Team Data Mining Course Project Report (Stage 2 Deliverable) | Reproducibility Seed: random_state = 42")
    r_meta.font.name = "Calibri"
    r_meta.font.size = Pt(8.5)
    r_meta.font.bold = True
    r_meta.font.color.rgb = RGBColor(43, 92, 143)

    # -------------------------------------------------------------
    # Introduction
    # -------------------------------------------------------------
    h_intro = doc.add_heading("Introduction & Stage 1 Integration Recap", level=1)
    h_intro.paragraph_format.space_before = Pt(8)
    h_intro.paragraph_format.space_after = Pt(2)

    doc.add_paragraph(
        "Cardiometabolic diseases—predominantly Type 2 Diabetes Mellitus (T2D) and Coronary Heart Disease (CHD)—constitute the foremost global "
        "drivers of adult morbidity and preventable healthcare expenditure. Rather than operating as independent pathologies, modern clinical "
        "epidemiology demonstrates that diabetes and cardiovascular disease share a unified systemic pathophysiological continuum: chronic insulin "
        "resistance, atherogenic dyslipidemia, microvascular and macrovascular endothelial inflammation, and arterial hypertension. Uncovering the "
        "latent phenotypic clusters and multi-dimensional risk manifolds underlying these conditions is vital for preventative clinical triage."
    )

    doc.add_paragraph(
        "In Mid-Report 1, our research team established a rigorous, audit-controlled data foundation across four core datasets: (1) UCI Pima Indians "
        "Diabetes (diabetes.csv, N=768), where an audit of 'hidden zeros' converted 652 impossible biological values (e.g., glucose, blood pressure, "
        "and BMI of zero) to explicit missing values; (2) UCI Cleveland Heart Disease (heart_cleveland_upload.csv, N=297), where genuine clinical "
        "pathologies—including an extreme cholesterol record of 564 mg/dL representing familial hypercholesterolemia—were validated and preserved; "
        "(3) CDC BRFSS 2015 Health Indicators (diabetes_binary_health_indicators_BRFSS2015.csv, N=253,680), which recorded both diabetes and heart "
        "disease outcomes on the exact same respondents; and (4) an integrated vertical cohort (N=26,065). Mid-Report 1 analyzed missingness mechanisms, "
        "standardized features, and executed frequent pattern mining (Apriori and FP-Growth) to uncover multi-factor association rules."
    )

    doc.add_paragraph(
        "Stage 2 extends this empirical foundation into advanced unsupervised data mining. Specifically, we investigate three interconnected "
        "analytical pillars: (1) Clustering Analysis to identify natural risk phenotypes and clinical subgroups; (2) Dimensionality Reduction "
        "via Principal Component Analysis (PCA) and t-Distributed Stochastic Neighbor Embedding (t-SNE) to untangle multi-collinear risk factors, "
        "visualize high-dimensional manifolds, and compress feature space; and (3) Outlier Detection via Isolation Forests and Local Outlier Factor "
        "(LOF) to distinguish genuine life-threatening multi-morbidity crises from recording artifacts, evaluating their impact on model stability. "
        "All analytical steps are grounded in Python (scikit-learn, scipy, pandas, numpy, seaborn) with random_state=42 strictly enforced."
    )

    # -------------------------------------------------------------
    # SECTION 1: CLUSTERING ANALYSIS
    # -------------------------------------------------------------
    h1 = doc.add_heading("1. Clustering Analysis", level=1)
    h1.paragraph_format.space_before = Pt(9)
    h1.paragraph_format.space_after = Pt(2)

    doc.add_paragraph(
        "To discover unobserved patient risk segments without relying on supervised labels, we applied three distinct clustering paradigms to our "
        "standardized multi-risk cohort: (1) K-Means (centroid-based spherical partitioning), (2) Hierarchical Agglomerative Clustering (connectivity-based "
        "variance minimization with Ward's linkage), and (3) Density-Based Spatial Clustering of Applications with Noise (DBSCAN). Features included "
        "standardized clinical and lifestyle predictors: Age, Body Mass Index (BMI), General Health, Physical Health Impairment Days, Mental Health "
        "Impairment Days, Hypertension Status, High Cholesterol Status, Smoking History, Physical Activity, Household Income, and Education."
    )

    # 1.1 Hyperparameter Selection & Mathematical Justification
    h1_1 = doc.add_heading("1.1 Hyperparameter Optimization & Mathematical Justification", level=2)
    h1_1.paragraph_format.space_before = Pt(6)
    h1_1.paragraph_format.space_after = Pt(2)

    doc.add_paragraph(
        "Cluster cardinality and boundary thresholds were determined through rigorous diagnostic optimization: "
        "\n• K-Means Cluster Count (k): Evaluated across k in [2, 10] using within-cluster sum of squares (WCSS / Inertia) and mean silhouette coefficients. "
        "As illustrated in Figure 1, the WCSS trajectory exhibits a sharp inflection ('elbow knee') at k=3 (Inertia = 45,710.2), where the rate of variance "
        "reduction decelerates markedly. Concurrently, the silhouette coefficient attains an optimal local peak at k=3 (Silhouette = 0.1534), confirming "
        "that partitioning the cohort into three risk tiers maximizes mathematical parsimony while preventing artificial fragmentation."
        "\n• DBSCAN Neighborhood Parameters (eps and minPts): In d=11 dimensions, the core neighborhood size was set to minPts = 5 based on dimensional "
        "heuristics (minPts >= d + 1). To determine the optimal radius (eps), a sorted 5-nearest-neighbor (5-NN) distance graph was constructed (Figure 2). "
        "The distance curve displays a pronounced elbow knee at eps = 2.40. Below this threshold, point density collapses into excessive noise; above it, "
        "distinct clusters merge into an undifferentiated mega-cluster."
        "\n• Hierarchical Linkage and Distance Metric: Agglomerative clustering was executed using Euclidean distance paired with Ward's minimum-variance "
        "criterion (linkage='ward'). Ward's objective minimizes the total within-cluster variance growth at each iterative merge step. The resulting cophenetic "
        "dendrogram (Figure 3) reveals three prominent primary branches that fuse at a cophenetic merge distance above 18.0, corroborating the 3-cluster partition."
    )

    add_styled_figure(
        doc,
        FIG_DIR / "stage2_fig1_kmeans_elbow_silhouette.png",
        "Figure 1: K-Means Optimization Curves — Within-Cluster Sum of Squares (Inertia) and Mean Silhouette Coefficient across k ∈ [2, 10]",
        "The WCSS elbow knee and peak local silhouette coefficient simultaneously identify k=3 as the mathematically optimal cluster partition."
    )

    add_styled_figure(
        doc,
        FIG_DIR / "stage2_fig2_dbscan_kdistance.png",
        "Figure 2: Sorted 5-Nearest Neighbor Distance Curve for DBSCAN Neighborhood Tuning (minPts = 5)",
        "The sharp knee transition at eps = 2.40 establishes the boundary between core high-density patient manifolds and peripheral sparse noise."
    )

    add_styled_figure(
        doc,
        FIG_DIR / "stage2_fig3_hierarchical_dendrogram.png",
        "Figure 3: Hierarchical Agglomerative Clustering Dendrogram (Ward Linkage, Euclidean Distance, Truncated p=25)",
        "The horizontal distance cut at height 18.0 cleanly delineates three primary cluster lineages before cophenetic distance escalates."
    )

    # 1.2 Health Domain Interpretation
    h1_2 = doc.add_heading("1.2 Health Domain Interpretation & Clinical Actionability", level=2)
    h1_2.paragraph_format.space_before = Pt(6)
    h1_2.paragraph_format.space_after = Pt(2)

    doc.add_paragraph(
        "To assess clinical validity, we profiled the resulting K-Means clusters against health metrics and empirical disease diagnoses (Table 1). "
        "The three clusters uncover highly distinct, biologically coherent stages along the cardiometabolic disease spectrum:"
        "\n1. Cluster 0 — 'Low-Risk / Active Baseline Cohort' (N=2,339; 46.8% of sample): Characterized by an average estimated age of 49.6 years, "
        "mean BMI of 26.9 kg/m² (mildly overweight, well below clinical obesity), high physical activity (87.7%), and minimal physical impairment "
        "(1.1 days/month). Baseline hypertension (8.0%) and high cholesterol (18.1%) are remarkably low. Crucially, empirical diabetes prevalence is "
        "only 3.2% and coronary heart disease is only 1.8%. This represents a resilient, lifestyle-protected cohort requiring primary wellness maintenance."
        "\n2. Cluster 1 — 'Moderate-Risk / Overweight Metabolic Segment' (N=1,956; 39.1% of sample): Patients are older (mean age 65.7 years) with an "
        "elevated mean BMI of 29.5 kg/m² (borderline clinical obesity). Despite maintaining fair physical activity (68.8%) and low disability days "
        "(1.5 days/month), this cluster exhibits pervasive vascular and lipid degeneration: 78.3% suffer from hypertension and 67.6% from hypercholesterolemia. "
        "Consequently, diabetes prevalence escalates nearly 7-fold to 21.6% and heart disease jumps 8-fold to 14.4%. This is a high-leverage pre-symptomatic "
        "segment where aggressive dietary, exercise, and blood pressure interventions can arrest progression toward end-organ complications."
        "\n3. Cluster 2 — 'High-Risk / Multi-Morbidity Crisis Segment' (N=705; 14.1% of sample): Represents severe systemic disease burden. Despite an "
        "average age of 59.1 years (younger than Cluster 1), these individuals have a mean BMI of 31.0 kg/m² (Class I clinical obesity), high smoking rates "
        "(59.9%), poor physical activity (50.5%), and catastrophic physical disability (mean 21.4 days of poor physical health per month out of 30) paired "
        "with heavy mental health burden (13.0 days/month). In this cluster, diabetes prevalence reaches 27.0% and heart disease reaches 23.0% (nearly 1 in 4). "
        "This group represents an actionable crisis segment requiring multidisciplinary clinical care, glycemic stabilization, and close cardiovascular monitoring."
    )

    df_prof = pd.read_csv(TBL_DIR / "stage2_cluster_health_profiles.csv")
    add_styled_table(
        doc,
        df_prof,
        "Table 1: Clinical and Demographic Health Profiles across Unsupervised K-Means Risk Clusters (N=5,000)",
        font_size_pt=7.5
    )

    # 1.3 Clinical Cohort Validation
    h1_3 = doc.add_heading("1.3 Clinical Cohort Validation: Continuous Biometrics in Cleveland and Pima", level=2)
    h1_3.paragraph_format.space_before = Pt(6)
    h1_3.paragraph_format.space_after = Pt(2)

    doc.add_paragraph(
        "To corroborate these findings against continuous clinical laboratory assays—specifically addressing continuous serum cholesterol, "
        "fasting plasma glucose, 2-hour serum insulin, and resting hemodynamics—we benchmarked identical unsupervised clustering workflows on the "
        "Cleveland Heart Disease (N=297) and Pima Indians Diabetes (N=768) cohorts (Table 2 and Figure 5):"
        "\n• Cleveland Clinic Cardiac Cohort: K-Means cleanly partitioned cardiac patients into two pathophysiological strata: Cluster 0 (N=130) "
        "exhibits an average age of 60.9 years, resting blood pressure of 139.6 mmHg, serum cholesterol of 262.0 mg/dL, depressed maximum heart rate "
        "(133.8 bpm), and severe exercise-induced ST depression (1.74 mm), culminating in a 69.2% confirmed angiographic coronary disease rate. "
        "Conversely, Cluster 1 (N=167) represents younger patients (49.6 years) with higher cardiac reserve (max HR 161.9 bpm), normal ST segments "
        "(0.53 mm), lower cholesterol (236.0 mg/dL), and a 28.1% heart disease rate."
        "\n• Pima Indians Diabetes Cohort: Clustering partitioned women into Cluster 0 (N=435, Normoglycemic / Young: age 27.6 years, glucose 104.8 mg/dL, "
        "insulin 114.2 μU/mL, BMI 30.0 kg/m², 17.7% diabetes) versus Cluster 1 (N=333, Severe Hyperglycemic / Insulin-Resistant: age 40.7 years, "
        "fasting glucose 143.7 mg/dL, serum insulin 175.2 μU/mL, BMI 35.6 kg/m², diastolic BP 79.4 mmHg, 57.4% confirmed diabetes)."
    )

    df_clin = pd.read_csv(TBL_DIR / "stage2_clinical_cluster_profiles.csv")
    add_styled_table(
        doc,
        df_clin,
        "Table 2: Clinical Laboratory Benchmark Profiles across Cleveland Heart Disease and Pima Indians Diabetes Cohorts",
        font_size_pt=7.0
    )

    add_styled_figure(
        doc,
        FIG_DIR / "stage2_fig5_clinical_cohort_clusters.png",
        "Figure 5: Clinical Laboratory Benchmarking — Scatter Distributions of Continuous Biomarkers in Cleveland and Pima Cohorts",
        "Continuous physiological assays (cholesterol, ST depression, fasting glucose, and BMI) partition patients into sharp clinical risk phenotypes."
    )

    # 1.4 Comparative Algorithm Evaluation
    h1_4 = doc.add_heading("1.4 Comparative Algorithm Evaluation & Structural Dynamics", level=2)
    h1_4.paragraph_format.space_before = Pt(6)
    h1_4.paragraph_format.space_after = Pt(2)

    doc.add_paragraph(
        "To evaluate algorithmic behavior, Table 3 compares K-Means, Hierarchical (Ward), and DBSCAN across Silhouette Score, Davies-Bouldin Index, "
        "Calinski-Harabasz Index, and Adjusted Rand Index (ARI) relative to K-Means:"
        "\n• K-Means vs. Hierarchical Agglomerative: K-Means achieves a higher silhouette score (0.1534 vs. 0.1193) and superior Davies-Bouldin Index "
        "(2.1061 vs. 2.3281, where lower values indicate better separation). However, the Adjusted Rand Index between them is 0.4758, indicating substantial "
        "structural agreement. Both methods partition the global continuous space into three tiers, but Ward's hierarchical tree merges smaller border "
        "branches based on strict agglomerative proximity, slightly sacrificing global centroid separation."
        "\n• DBSCAN Structural Contrasts: DBSCAN yielded 4 clusters and identified 166 points (3.3%) as sparse noise. On its dense core clusters, DBSCAN "
        "achieves a higher silhouette score (0.2578) and low Davies-Bouldin Index (1.0105). However, its ARI with K-Means is only 0.0589. This profound "
        "divergence stems from algorithm inductive bias: DBSCAN groups the vast majority of respondents (N=4,784, or 95.7%) into a single large continuous "
        "density-connected manifold, while carving out three tiny satellite clusters (N=16, 17, 17). In survey health data with continuous demographic gradients, "
        "DBSCAN fails to establish intermediate risk strata, demonstrating that centroid-based partitioning is far superior for actionable clinical segmentation."
    )

    df_comp = pd.read_csv(TBL_DIR / "stage2_clustering_metrics_comparison.csv")
    add_styled_table(
        doc,
        df_comp,
        "Table 3: Unsupervised Clustering Methods Performance and Cross-Method Similarity Comparison (N=5,000)",
        font_size_pt=7.5
    )

    # -------------------------------------------------------------
    # SECTION 2: DIMENSIONALITY REDUCTION
    # -------------------------------------------------------------
    h2 = doc.add_heading("2. Dimensionality Reduction", level=1)
    h2.paragraph_format.space_before = Pt(9)
    h2.paragraph_format.space_after = Pt(2)

    doc.add_paragraph(
        "Modern health records encompass high-dimensional, highly correlated features. While rich in information, high dimensionality introduces "
        "severe analytical bottlenecks: (1) the 'curse of dimensionality', where Euclidean distances between points converge, degrading distance-based "
        "clustering; (2) severe multi-collinearity (e.g., General Health, Physical Health days, and Mobility limitations co-vary strongly); (3) noise "
        "amplification from discrete survey categories; and (4) human cognitive inability to visualize beyond three dimensions. To resolve these hurdles, "
        "we applied both linear (PCA) and non-linear (t-SNE) dimensionality reduction."
    )

    # 2.1 Principal Component Analysis (PCA)
    h2_1 = doc.add_heading("2.1 Principal Component Analysis: Scree Diagnostics & Factor Loadings", level=2)
    h2_1.paragraph_format.space_before = Pt(6)
    h2_1.paragraph_format.space_after = Pt(2)

    doc.add_paragraph(
        "PCA was fitted across the 11 standardized clinical features. As displayed in the scree diagnostics (Figure 6), the first principal component (PC1) "
        "accounts for 24.69% of total variance, PC2 captures 14.00%, and PC3 captures 9.80%. The variance trajectory demonstrates that 6 principal components "
        "are sufficient to retain 73.98% (nearly three-quarters) of total population variance, successfully compressing dimensionality by 45.5% while "
        "filtering out uninformative orthogonal residual noise."
    )

    add_styled_figure(
        doc,
        FIG_DIR / "stage2_fig6_pca_scree_loadings.png",
        "Figure 6: Principal Component Analysis — Scree Plot of Individual Explained Variance and Cumulative Variance Ratio",
        "Six principal components capture 74.0% of cohort variance, effectively compressing feature space while filtering uncorrelated noise."
    )

    doc.add_paragraph(
        "To interpret the biological meaning of the principal axes, Table 4 examines the factor loadings for the top three components:"
        "\n• PC1 ('General Physical Impairment & Socioeconomic Vulnerability', 24.7% variance): Dominated by heavy positive loadings on Poor General Health "
        "(+0.4692), Physical Impairment Days (+0.3838), Hypertension (+0.3034), and Mental Impairment Days (+0.2627), contrasted against strong negative "
        "loadings on Household Income (-0.3803), Education Level (-0.3011), and Physical Activity (-0.2704). PC1 captures an individual's overall burden "
        "of functional disability and socio-environmental vulnerability."
        "\n• PC2 ('Age-Associated Cardiometabolic Degeneration', 14.0% variance): Driven overwhelmingly by positive loadings on Chronological Age (+0.5740), "
        "High Cholesterol (+0.4574), and Hypertension (+0.4215), with a negative loading on Mental Impairment (-0.3980). PC2 isolates the biological aging "
        "axis where vascular stiffening and dyslipidemia progress independently of acute subjective mental distress."
        "\n• PC3 ('Socioeconomic Adiposity Axis', 9.8% variance): Characterized by positive loadings on Education (+0.5677), BMI (+0.4286), and Income (+0.3461), "
        "reflecting metabolic variations cross-cutting demographic tiers."
    )

    df_load = pd.read_csv(TBL_DIR / "stage2_pca_loadings.csv")
    add_styled_table(
        doc,
        df_load,
        "Table 4: Principal Component Factor Loadings (Eigenvectors) for the Top Three Components",
        font_size_pt=8.0
    )

    # 2.2 t-SNE Embedding & Neighborhood Sensitivity
    h2_2 = doc.add_heading("2.2 t-Distributed Stochastic Neighbor Embedding: Manifold Mapping & Sensitivity", level=2)
    h2_2.paragraph_format.space_before = Pt(6)
    h2_2.paragraph_format.space_after = Pt(2)

    doc.add_paragraph(
        "While PCA preserves global variance along orthogonal hyperplanes, it struggles to capture non-linear manifold geometry where patient sub-types "
        "curve through high-dimensional space. To address this, we implemented t-SNE, which converts Euclidean distances into conditional probabilities "
        "under a Gaussian distribution in high-dimensional space and minimizes Kullback-Leibler (KL) divergence in 2D using a Student-t kernel."
        "\n• Neighborhood Perplexity Sensitivity: We evaluated t-SNE across perplexity values of 15, 30, and 50 (Figure 7). At perplexity=15, the projection "
        "displays minor over-fragmentation into loose micro-clusters. At perplexity=30 and 50, the manifold stabilizes into clear, coherent continuum arms "
        "representing low, moderate, and high cardiometabolic risk, demonstrating robust topological stability without distortion."
        "\n• Local vs. Global Structure Preservation: t-SNE excels at preserving local neighborhood structures—ensuring that individuals with nearly "
        "identical clinical profiles map to adjacent 2D coordinates. However, distances between distant clusters on a t-SNE projection cannot be interpreted "
        "as absolute Euclidean distances. In contrast, PCA faithfully preserves global geometric distances at the expense of overlapping local cluster boundaries."
    )

    add_styled_figure(
        doc,
        FIG_DIR / "stage2_fig7_tsne_perplexity_sensitivity.png",
        "Figure 7: t-SNE Neighborhood Sensitivity Analysis across Perplexities (15, 30, and 50) Colored by K-Means Clusters",
        "The manifold structure demonstrates high stability across perplexities, cleanly unrolling the risk continuum without topological tearing."
    )

    # 2.3 Before vs. After Visualizations
    h2_3 = doc.add_heading("2.3 Before and After Visualizations of the Patient Manifold", level=2)
    h2_3.paragraph_format.space_before = Pt(6)
    h2_3.paragraph_format.space_after = Pt(2)

    doc.add_paragraph(
        "Figure 8 contrasts the raw 2D feature space (BMI vs. Estimated Age) against the 2D PCA reduced projection. In the raw feature space (Figure 8A), "
        "the three risk clusters overlap heavily; a patient with BMI=28 and age=55 cannot be reliably separated from an adjacent patient based on two raw "
        "axes alone. Once transformed onto PC1 and PC2 (Figure 8B), the clusters separate cleanly: Cluster 0 occupies the lower-left quadrant (low impairment, "
        "younger), Cluster 1 concentrates in the upper-right quadrant (advanced age, vascular risk), and Cluster 2 extends along the positive PC1 axis "
        "(severe physical disability)."
    )

    add_styled_figure(
        doc,
        FIG_DIR / "stage2_fig8_raw_vs_reduced_comparison.png",
        "Figure 8: Before and After Dimensionality Reduction — Raw Feature Space (Age vs. BMI) vs. Principal Component Space (PC1 vs. PC2)",
        "Dimensionality reduction resolves severe feature overlap, separating overlapping raw biometrics into distinct orthogonal risk quadrants."
    )

    doc.add_paragraph(
        "Furthermore, Figure 4 provides a comprehensive 4-panel comparison: Panels A and C project the K-Means clusters onto PCA and t-SNE, while Panels B "
        "and D project empirical disease diagnoses (Diabetes and Heart Disease). Remarkably, without ever receiving label information, the unsupervised "
        "clusters align directly with disease emergence: patients in Cluster 2 form the distinct rightmost branch on t-SNE, perfectly matching the dense "
        "concentration of empirical heart attacks and diabetes."
    )

    add_styled_figure(
        doc,
        FIG_DIR / "stage2_fig4_cluster_scatter_projections.png",
        "Figure 4: 2D Manifold Visualizations — PCA vs. t-SNE Colored by Unsupervised Risk Clusters and Empirical Disease Outcomes",
        "Unsupervised manifold clusters mirror empirical clinical diagnoses: high-risk clusters align precisely with true diabetes and heart disease cases."
    )

    # 2.4 Re-Clustering on Reduced Data
    h2_4 = doc.add_heading("2.4 Re-Clustering on Reduced Feature Spaces: Quantitative Separation Benchmarks", level=2)
    h2_4.paragraph_format.space_before = Pt(6)
    h2_4.paragraph_format.space_after = Pt(2)

    doc.add_paragraph(
        "To evaluate whether dimensionality reduction improves cluster separation and interpretability, we re-executed K-Means clustering across three "
        "reduced feature spaces: (1) PCA 6 components (retaining 74.0% variance), (2) PCA 2D projection, and (3) t-SNE 2D coordinates (Table 5):"
        "\n• PCA 6 Components: The silhouette score increases from 0.1534 in the original 11D space to 0.2046—a 33.4% relative improvement—while the "
        "Davies-Bouldin index drops from 2.1061 to 1.7054 (19.0% improvement in cluster compactness). Crucially, the Adjusted Rand Index between the 6D PCA "
        "clusters and the original 11D clusters is 0.9027 (over 90% identical assignment). This demonstrates that PCA successfully discarded collinear noise "
        "without distorting the underlying clinical grouping."
        "\n• PCA 2D Projection: Compressing feature space to PC1 and PC2 achieves a dramatic silhouette surge to 0.4050 (+164% increase) and an outstanding "
        "Davies-Bouldin index of 0.8625, with an ARI of 0.7978. While visually striking, 2D PCA omits 61.3% of orthogonal variance; thus, PCA-6 represents "
        "the optimal operational balance for robust downstream risk modeling."
        "\n• t-SNE 2D Manifold: Clustering on t-SNE yields a high silhouette score (0.4070) reflecting compact 2D islands, but an ARI of 0.1124 relative to "
        "global 11D space. This reflects the fundamental property that t-SNE partitions local probability neighborhoods rather than global geometric hyperplanes."
    )

    df_red = pd.read_csv(TBL_DIR / "stage2_clustering_reduced_vs_raw.csv")
    add_styled_table(
        doc,
        df_red,
        "Table 5: Clustering Performance Comparison across Original and Reduced Dimensionality Feature Spaces (N=5,000)",
        font_size_pt=7.0
    )

    # -------------------------------------------------------------
    # SECTION 3: OUTLIER DETECTION
    # -------------------------------------------------------------
    h3 = doc.add_heading("3. Outlier Detection", level=1)
    h3.paragraph_format.space_before = Pt(9)
    h3.paragraph_format.space_after = Pt(2)

    # 3.1 Defining Outliers in Healthcare
    h3_1 = doc.add_heading("3.1 Defining Outliers in Cardiometabolic Healthcare: Pathology vs. Artifacts", level=2)
    h3_1.paragraph_format.space_before = Pt(6)
    h3_1.paragraph_format.space_after = Pt(2)

    doc.add_paragraph(
        "In healthcare data mining, the term 'outlier' carries dual, fundamentally opposing interpretations that must be rigorously distinguished: "
        "\n1. Impossible Measurement Artifacts & Recording Errors: Values that violate biological and physiological laws in living humans. Examples include "
        "a diastolic blood pressure of 0 mmHg, a fasting blood glucose of 0 mg/dL, or a Body Mass Index of 0 kg/m². In Mid-Report 1, these anomalies were "
        "identified as 'hidden zeros' resulting from omitted examinations and converted to NaN. Retaining such artifacts would corrupt machine learning "
        "models by injecting false biological signals."
        "\n2. Genuine Pathological Extremes (Clinical Outliers): Values that are statistically extreme (exceeding 3 or 4 standard deviations from the mean) "
        "yet physiologically authentic. Examples include a patient with super-morbid obesity (BMI = 82.0 kg/m²), extreme hypercholesterolemia (serum "
        "cholesterol = 564 mg/dL from familial LDLR mutations), acute hyperglycemic crises (glucose > 180 mg/dL), or catastrophic multi-morbidity "
        "(reporting 30 out of 30 days of complete physical and mental disability). These patients are not errors; they represent the most vulnerable, "
        "highest-cost individuals in the healthcare system."
    )

    # 3.2 Outlier Detection Methodology & Visualizations
    h3_2 = doc.add_heading("3.2 Outlier Detection Algorithms, Manifold Localization & Boxplot Diagnostics", level=2)
    h3_2.paragraph_format.space_before = Pt(6)
    h3_2.paragraph_format.space_after = Pt(2)

    doc.add_paragraph(
        "To identify multidimensional anomalies, we deployed two complementary algorithms calibrated at contamination = 0.035 (3.5% cohort fraction): "
        "\n• Isolation Forest: An ensemble of 150 randomized isolation trees that isolates anomalies based on path length depth. Because extreme outliers "
        "require few random feature splits to separate from the normal distribution, they exhibit short average tree path lengths."
        "\n• Local Outlier Factor (LOF): A density-based algorithm (n_neighbors = 20) that computes the local reachability density of a patient relative "
        "to their surrounding neighbors. Points residing in substantially sparser neighborhoods than their peers receive high outlier scores."
        "\n• Detection Results: Isolation Forest flagged exactly 175 outliers (3.5%), LOF flagged 175 outliers (3.5%), and the union identified 347 unique "
        "anomalous patients (6.9%). Isolation Forest proved exceptionally capable at isolating extreme global biometric leverage points, whereas LOF isolated "
        "local density anomalies (e.g., young individuals with anomalous stroke occurrences)."
    )

    doc.add_paragraph(
        "Figure 9 visualizes the detected Isolation Forest outliers: on the PCA and t-SNE projections (Panels A and B), outliers (orange triangles) concentrate "
        "at the extreme peripheral edges of the manifold—particularly along the upper right of PC1. The multi-feature boxplots (Panels C and D) reveal "
        "stark distributional contrasts: while inliers display a median BMI of 27.4 kg/m² and a median of 0 poor physical days, outliers exhibit extreme "
        "right-tailed distributions extending to BMI=82.0 kg/m² and a median of 30 out of 30 days of physical disability."
    )

    add_styled_figure(
        doc,
        FIG_DIR / "stage2_fig9_outliers_projection_boxplots.png",
        "Figure 9: Outlier Detection Diagnostics — Manifold Embeddings and Biometric Distribution Contrasts (Inliers vs. Outliers)",
        "Detected outliers reside on the distant perimeter of the manifold, driven by morbid obesity (BMI up to 82) and complete physical impairment (30/30 days)."
    )

    # 3.3 Outlier Characteristics & Root Causes
    h3_3 = doc.add_heading("3.3 Outlier Characteristics, Clinical Profiles & Root Causes", level=2)
    h3_3.paragraph_format.space_before = Pt(6)
    h3_3.paragraph_format.space_after = Pt(2)

    doc.add_paragraph(
        "Table 6 profiles the biometric and clinical composition of the 175 Isolation Forest outliers relative to the 4,825 normal inliers: "
        "\n• Disproportionate Disease Burden: Compared to inliers, outliers exhibit a 2.25-fold higher diabetes prevalence (29.7% vs. 13.2%) and a "
        "2.76-fold higher coronary heart disease rate (25.1% vs. 9.1%). Hypertension prevalence is 58.3% and high cholesterol is 50.3%."
        "\n• Catastrophic Physical & Mental Disability: The most dramatic differentiator is functional impairment: outliers report an average of 24.9 "
        "days of poor physical health and 24.1 days of poor mental health per month, compared to just 3.4 and 2.3 days in inliers (a 7-fold and 10-fold increase)."
        "\n• Morbid Obesity: Outliers have an average BMI of 33.4 kg/m² with a maximum of 82.0 kg/m² (Class III super-morbid obesity)."
        "\n• Root Causes: Root cause analysis indicates that these outliers represent authentic multi-morbidity syndromes where severe obesity, chronic "
        "pain, vascular pathology, and mental health crises converge. Survey fatigue or extreme distress may also lead respondents to report maximum values "
        "(30 days) across all health items."
    )

    df_out = pd.read_csv(TBL_DIR / "stage2_outlier_summary.csv")
    add_styled_table(
        doc,
        df_out,
        "Table 6: Biometric and Clinical Profile Comparison — Normal Inliers vs. Isolation Forest Outliers",
        font_size_pt=8.0
    )

    # 3.4 Impact of Outlier Removal
    h3_4 = doc.add_heading("3.4 Analytical Impact of Outlier Removal on Clustering and PCA Stability", level=2)
    h3_4.paragraph_format.space_before = Pt(6)
    h3_4.paragraph_format.space_after = Pt(2)

    doc.add_paragraph(
        "To rigorously quantify how extreme outliers affect downstream data mining models, we re-fitted K-Means clustering and PCA on the clean cohort "
        "(N=4,825) after removing the 175 outliers (Table 7 and Figure 10):"
        "\n• Clustering Silhouette and Dispersion Dynamics: On the clean cohort, the K-Means silhouette score shifts from 0.1534 to 0.1306 (-0.0228), the "
        "Davies-Bouldin index changes from 2.1061 to 2.2894 (+0.1833), and the Calinski-Harabasz score shifts from 860.45 to 729.61. In statistical literature, "
        "this is an established phenomenon: extreme distant outliers artificially inflate between-cluster distance ratios by acting as extreme anchor points; "
        "trimming them removes artificial dispersion, revealing the true continuous boundary density of the core general population."
        "\n• PCA Variance Stabilization: Top-2 PCA explained variance shifts from 38.69% to 36.72% (-1.97%), and Top-6 variance shifts from 73.98% to 72.72% "
        "(-1.25%). In the raw data, principal axes were slightly skewed by extreme leverage coordinates; in the clean dataset, variance spreads more evenly "
        "across all 11 lifestyle and physiological dimensions, reflecting general population variance."
    )

    df_out_imp = pd.read_csv(TBL_DIR / "stage2_outlier_removal_metrics.csv")
    add_styled_table(
        doc,
        df_out_imp,
        "Table 7: Quantitative Impact of Outlier Removal on Clustering Separation Metrics and PCA Variance Explained",
        font_size_pt=7.0
    )

    add_styled_figure(
        doc,
        FIG_DIR / "stage2_fig10_outlier_removal_impact.png",
        "Figure 10: Before and After Outlier Removal — Principal Component Projections and Cluster Boundaries (N=5,000 vs. N=4,825)",
        "Removing extreme multi-morbidity outliers eliminates artificial coordinate leverage, stabilizing principal axes around genuine population variance."
    )

    # -------------------------------------------------------------
    # SECTION 4: INTEGRATION AND INSIGHTS
    # -------------------------------------------------------------
    h4 = doc.add_heading("4. Integration, Strategic Insights & Future Roadmap", level=1)
    h4.paragraph_format.space_before = Pt(9)
    h4.paragraph_format.space_after = Pt(2)

    # 4.1 Tripartite Synergy
    h4_1 = doc.add_heading("4.1 Tripartite Methodological Synergy", level=2)
    h4_1.paragraph_format.space_before = Pt(6)
    h4_1.paragraph_format.space_after = Pt(2)

    doc.add_paragraph(
        "Clustering, Dimensionality Reduction, and Outlier Detection are not isolated procedures; together, they establish a cohesive, self-reinforcing "
        "data mining architecture for complex health discovery: "
        "\n1. Dimensionality Reduction (PCA and t-SNE) untangles multi-collinear clinical indicators, compresses 11 correlated features into clean "
        "orthogonal axes (PC1: physical impairment, PC2: age/cardiovascular risk), and overcomes the curse of dimensionality."
        "\n2. Unsupervised Clustering operates on this optimized representation to delineate discrete, actionable patient phenotypes (Low Risk, Metabolic "
        "Escalation, and Multi-Morbidity Crisis) that mirror real-world cardiometabolic progression."
        "\n3. Outlier Detection acts as a critical quality and safety filter: it isolates high-leverage anomalies, evaluates their impact on model "
        "geometry, and separates genuine high-acuity crisis patients from standard population risk distributions."
    )

    # 4.2 Strategic Translation to Dietify Mobile Health Application
    h4_2 = doc.add_heading("4.2 Actionable Translation to the Dietify Digital Health Platform", level=2)
    h4_2.paragraph_format.space_before = Pt(6)
    h4_2.paragraph_format.space_after = Pt(2)

    doc.add_paragraph(
        "These empirical discoveries provide immediate operational leverage for preventative mobile health platforms such as Dietify: "
        "\n• Zero-Laboratory Automated User Triage: By embedding our 6-component PCA model and K-Means centroid weights into the mobile onboarding "
        "flow, Dietify can dynamically stratify users into risk tiers using simple, non-invasive intake questions (age, BMI, self-reported general health, "
        "smoking, and activity) without requiring expensive blood panels or clinic visits."
        "\n• Targeted Nutritional Prescriptions for Cluster 1 (The Pre-Symptomatic Window): Cluster 1 represents the highest-leverage intervention target "
        "(39.1% of cohort, 78% hypertensive, 21.6% diabetic). Dietify can automatically serve personalized Dietary Approaches to Stop Hypertension (DASH) "
        "and Mediterranean meal plans rich in potassium, magnesium, and viscous soluble fiber, directly targeting the insulin resistance driving both blood "
        "pressure and glycemic escalation."
        "\n• Restorative Mobility Protocols for Cluster 2: In Cluster 2, physical limitation (DiffWalk) and chronic pain days (21.4 days/month) render standard "
        "vigorous exercise prescriptions useless or harmful. Dietify can automatically tailor restorative, low-impact exercise modules (e.g., seated resistance "
        "bands, gentle walking intervals, aquatic therapy) to interrupt the sedentary inflammatory cycle."
        "\n• Outlier Anomaly Safeguards & Clinical Escalation: When a Dietify user logs sudden extreme entries (e.g., rapid weight gain indicating congestive "
        "heart failure fluid retention, systolic BP > 180 mmHg, or 30 days of debilitating impairment), the app's outlier engine triggers a medical red-flag "
        "alert, advising immediate professional clinical consultation rather than automated dietary adjustments."
    )

    # 4.3 Study Limitations
    h4_3 = doc.add_heading("4.3 Methodological Limitations", level=2)
    h4_3.paragraph_format.space_before = Pt(6)
    h4_3.paragraph_format.space_after = Pt(2)

    doc.add_paragraph(
        "While rigorous, several methodological constraints must be acknowledged: "
        "\n1. Survey Recall and Self-Report Biases: The CDC BRFSS relies on telephonic self-reporting. Conditions like hypertension and diabetes require "
        "previous clinical diagnosis; undiagnosed individuals are erroneously coded as non-cases, inducing conservative prevalence under-reporting."
        "\n2. Cross-Sectional Observational Nature: The data captures a single snapshot in time. Clustering establishes co-occurrence associations but "
        "cannot prove longitudinal temporal causality (e.g., whether physical inactivity caused obesity or joint pain caused inactivity)."
        "\n3. Algorithmic Assumptions: K-Means inherently assumes isotropic, spherical clusters of roughly equal variance. While effective for population "
        "triage, non-linear manifolds require density-based or deep embedding architectures for fine-grained boundary resolution."
    )

    # 4.4 Roadmap for Final Project
    h4_4 = doc.add_heading("4.4 Roadmap for Final Project (Stage 3 Supervised Machine Learning)", level=2)
    h4_4.paragraph_format.space_before = Pt(6)
    h4_4.paragraph_format.space_after = Pt(2)

    doc.add_paragraph(
        "Stage 2 establishes the feature representations and patient segments for the final stage of our university project: supervised predictive modeling. "
        "The Stage 3 roadmap comprises: "
        "\n• Supervised Predictive Model Benchmarking: Implementing and cross-validating diverse model families: Gradient Boosted Decision Trees (LightGBM, "
        "XGBoost, CatBoost), Random Forests, and Penalized Logistic Regression."
        "\n• Imbalanced Learning Protocols: Applying Synthetic Minority Over-sampling Technique (SMOTE) and focal cost-sensitive loss weighting to overcome "
        "the ~6:1 diabetes and ~10:1 heart disease class imbalance."
        "\n• Clinical Evaluation Metrics: Moving beyond misleading accuracy to prioritize Precision-Recall Area Under the Curve (PR-AUC), ROC-AUC, F1-Score, "
        "and Sensitivity / Recall—ensuring false-negative triage errors are minimized."
        "\n• Explainable AI via SHAP (SHapley Additive exPlanations): Computing global feature importance and patient-level waterfall plots to explain "
        "individual risk predictions, ensuring transparency for healthcare providers and Dietify users."
    )

    # -------------------------------------------------------------
    # SECTION 5: CONCLUSION
    # -------------------------------------------------------------
    h5 = doc.add_heading("5. Conclusion", level=1)
    h5.paragraph_format.space_before = Pt(9)
    h5.paragraph_format.space_after = Pt(2)

    doc.add_paragraph(
        "This Stage 2 project report successfully executed unsupervised clustering, dimensionality reduction, and outlier diagnostics across "
        "harmonized population and clinical cohorts. K-Means clustering (k=3), justified by inertia elbow inflection and silhouette peaks, uncovered "
        "three clinically distinct cardiometabolic risk phenotypes (Low-Risk Baseline [46.8%], Pre-Symptomatic Metabolic Escalation [39.1%], and "
        "Multi-Morbidity Crisis [14.1%]). PCA compressed feature space by 45.5% while retaining 74.0% variance across two primary axes of physical "
        "impairment and vascular aging, improving clustering separation by 33.4%. Isolation Forest detected 175 high-acuity outliers exhibiting morbid "
        "obesity (BMI up to 82.0 kg/m²), 25 days/month of severe impairment, and elevated disease prevalence. Together, these techniques provide a "
        "scientifically sound, clinically actionable foundation for predictive machine learning and preventative mobile health interventions in Dietify."
    )

    # -------------------------------------------------------------
    # APPENDIX: PARAMETER DICTIONARY
    # -------------------------------------------------------------
    h_app = doc.add_heading("Appendix: Comprehensive Methodological Parameter Dictionary", level=1)
    h_app.paragraph_format.space_before = Pt(9)
    h_app.paragraph_format.space_after = Pt(2)

    doc.add_paragraph(
        "To guarantee 100% computational reproducibility across every stage of analysis, Table 8 inventories every algorithm hyperparameter, "
        "its exact numerical configuration, and its mathematical and clinical justification."
    )

    df_param = pd.read_csv(TBL_DIR / "stage2_parameter_dictionary.csv")
    add_styled_table(
        doc,
        df_param,
        "Table 8: Comprehensive Algorithmic Parameter Dictionary and Methodological Justification (Seed: 42)",
        font_size_pt=7.0
    )

    doc.save(DOCX_ROOT)
    doc.save(DOCX_REPORT)
    print(f"Saved styled Stage 2 report to: {DOCX_ROOT} and {DOCX_REPORT}")


def convert_to_pdf_and_check():
    print(f"Converting {DOCX_ROOT.name} to {PDF_ROOT.name}...")
    convert(str(DOCX_ROOT), str(PDF_ROOT))
    if DOCX_REPORT.exists():
        convert(str(DOCX_REPORT), str(PDF_REPORT))

    print(f"Successfully generated {PDF_ROOT.name}!")

    reader = PdfReader(str(PDF_ROOT))
    page_count = len(reader.pages)
    print("=" * 80)
    print(f"VERIFICATION: Total Page Count of {PDF_ROOT.name}: {page_count} pages")
    print(f"Requirement Check: 5 <= {page_count} <= 20 : {'PASSED [OK]' if 5 <= page_count <= 20 else 'FAILED [OUT OF BOUNDS]'}")
    print("=" * 80)
    return page_count


if __name__ == "__main__":
    build_report_document()
    convert_to_pdf_and_check()
