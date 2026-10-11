"""
src/build_status_pdf.py
Generates the comprehensive Project Status Report in Word (.docx) and compiles
into high-fidelity PDF (PROJECT_STATUS_REPORT.pdf) in the project root.
"""

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
DOCX_OUT = BASE_DIR / "PROJECT_STATUS_REPORT.docx"
PDF_OUT = BASE_DIR / "PROJECT_STATUS_REPORT.pdf"
FIG_DIR = BASE_DIR / "outputs" / "figures"
TBL_DIR = BASE_DIR / "outputs" / "tables"


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
    p_cap.paragraph_format.space_before = Pt(8)
    p_cap.paragraph_format.space_after = Pt(3)
    p_cap.paragraph_format.keep_with_next = True
    run_cap = p_cap.add_run(caption)
    run_cap.font.name = "Calibri"
    run_cap.font.size = Pt(9.5)
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
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT if col_idx == 0 or len(text_val) > 25 else WD_ALIGN_PARAGRAPH.CENTER
            for r in p.runs:
                r.font.name = "Calibri"
                r.font.size = Pt(font_size_pt)
                r.font.color.rgb = RGBColor(40, 40, 40)

    if col_widths and len(col_widths) == len(df.columns):
        for row in table.rows:
            for idx, width in enumerate(col_widths):
                row.cells[idx].width = width

    doc.add_paragraph().paragraph_format.space_after = Pt(4)


def add_callout(doc, title, text, bg_hex="EBF2FA", border_hex="1F497D"):
    """Add a highlighted callout box for executive notes."""
    tbl = doc.add_table(rows=1, cols=1)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    cell = tbl.rows[0].cells[0]
    set_cell_background(cell, bg_hex)
    set_cell_margins(cell, top=100, bottom=100, left=140, right=140)

    # Add left border accent
    tcPr = cell._tc.get_or_add_tcPr()
    borders = parse_xml(
        f'<w:tcBorders {nsdecls("w")}>'
        f'<w:top w:val="none"/>'
        f'<w:left w:val="single" w:sz="36" w:space="0" w:color="{border_hex}"/>'
        f'<w:bottom w:val="none"/>'
        f'<w:right w:val="none"/>'
        f'</w:tcBorders>'
    )
    tcPr.append(borders)

    p = cell.paragraphs[0]
    p.paragraph_format.space_before = Pt(3)
    p.paragraph_format.space_after = Pt(2)
    run_t = p.add_run(title + "\n")
    run_t.font.name = "Calibri"
    run_t.font.size = Pt(10.5)
    run_t.font.bold = True
    run_t.font.color.rgb = RGBColor(31, 73, 125)

    run_body = p.add_run(text)
    run_body.font.name = "Calibri"
    run_body.font.size = Pt(9.5)
    run_body.font.italic = False
    run_body.font.color.rgb = RGBColor(40, 40, 40)

    doc.add_paragraph().paragraph_format.space_after = Pt(4)


def add_figure(doc, img_path, caption, takeaway, width=Inches(5.2)):
    """Embed an image with numbered caption and clinical takeaway."""
    if not img_path.exists():
        return
    p_img = doc.add_paragraph()
    p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_img.paragraph_format.space_before = Pt(6)
    p_img.paragraph_format.space_after = Pt(2)
    p_img.paragraph_format.keep_with_next = True
    p_img.add_run().add_picture(str(img_path), width=width)

    p_cap = doc.add_paragraph()
    p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_cap.paragraph_format.space_before = Pt(2)
    p_cap.paragraph_format.space_after = Pt(1)
    p_cap.paragraph_format.keep_with_next = True
    run_cap = p_cap.add_run(caption)
    run_cap.font.name = "Calibri"
    run_cap.font.size = Pt(8.5)
    run_cap.font.bold = True
    run_cap.font.color.rgb = RGBColor(31, 73, 125)

    p_take = doc.add_paragraph()
    p_take.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_take.paragraph_format.space_before = Pt(0)
    p_take.paragraph_format.space_after = Pt(6)
    run_take = p_take.add_run("Key Finding: " + takeaway)
    run_take.font.name = "Calibri"
    run_take.font.size = Pt(8.0)
    run_take.font.italic = True
    run_take.font.color.rgb = RGBColor(90, 90, 90)


def build_status_document():
    doc = Document()

    # Page Margins: 0.8 inches all around
    sections = doc.sections
    for section in sections:
        section.top_margin = Inches(0.8)
        section.bottom_margin = Inches(0.8)
        section.left_margin = Inches(0.8)
        section.right_margin = Inches(0.8)

    # Styles
    styles = doc.styles
    normal_style = styles["Normal"]
    normal_style.font.name = "Calibri"
    normal_style.font.size = Pt(10)
    normal_style.font.color.rgb = RGBColor(40, 40, 40)
    normal_style.paragraph_format.line_spacing = 1.15
    normal_style.paragraph_format.space_after = Pt(4)

    # -------------------------------------------------------------
    # DOCUMENT HEADER / BANNER
    # -------------------------------------------------------------
    p_title = doc.add_paragraph()
    p_title.paragraph_format.space_before = Pt(0)
    p_title.paragraph_format.space_after = Pt(2)
    run_title = p_title.add_run("Predictive Risk Identification for Diabetes and Heart Disease")
    run_title.font.name = "Calibri Light"
    run_title.font.size = Pt(20)
    run_title.font.bold = True
    run_title.font.color.rgb = RGBColor(31, 73, 125)

    p_sub = doc.add_paragraph()
    p_sub.paragraph_format.space_before = Pt(0)
    p_sub.paragraph_format.space_after = Pt(6)
    run_sub = p_sub.add_run("Project Status Report, Gap Analysis & Comprehensive Roadmap")
    run_sub.font.name = "Calibri"
    run_sub.font.size = Pt(12)
    run_sub.font.bold = True
    run_sub.font.color.rgb = RGBColor(217, 95, 2)

    # Metadata Table
    meta_df = pd.DataFrame([
        {"Field": "Role", "Detail": "Project Auditor & Technical Writer"},
        {"Field": "Evaluation Scope", "Detail": "All Code, Data, Notebooks, Models, Outputs, and Reports"},
        {"Field": "Current Status", "Detail": "50% Complete (Stages 1 & 2 Complete; Stage 3 Pending)"},
        {"Field": "Target Models", "Detail": "Logistic Regression, Decision Tree, Random Forest + SHAP"},
        {"Field": "Evaluation Metrics", "Detail": "F1-Score, ROC-AUC, Recall / Sensitivity (Separate Per Disease)"},
        {"Field": "Audit Date", "Detail": "October 2026"}
    ])
    add_styled_table(doc, meta_df, "Project Metadata & Audit Scope", col_widths=[Inches(1.8), Inches(4.8)], font_size_pt=8.5)

    # Executive Summary Callout
    add_callout(
        doc,
        "Executive Summary: Current Completion Level = 50%",
        "One-Line Justification: The project has completed thorough, high-quality data cleaning, dataset integration, "
        "exploratory data analysis, frequent pattern mining, unsupervised clustering, dimensionality reduction, and outlier "
        "diagnostics (Stages 1 & 2), but has NOT yet implemented supervised predictive modeling, model comparison, evaluation "
        "metrics (F1/ROC-AUC/Recall), SHAP explainability, or the interactive new-patient inference interface (Stage 3)."
    )

    # -------------------------------------------------------------
    # SECTION 1: WORK COMPLETED SO FAR
    # -------------------------------------------------------------
    h1 = doc.add_heading("1. Work Completed So Far", level=1)
    h1.paragraph_format.space_before = Pt(10)
    h1.paragraph_format.space_after = Pt(2)
    for r in h1.runs:
        r.font.name = "Calibri Light"
        r.font.size = Pt(14)
        r.font.bold = True
        r.font.color.rgb = RGBColor(31, 73, 125)

    doc.add_paragraph(
        "The project repository reflects two successfully delivered milestones: Stage 1 (Mid-Term Report: Preprocessing, "
        "Missing Data Science, Integration, and Frequent Pattern Mining) and Stage 2 (Clustering, Dimensionality Reduction, "
        "and Anomaly Diagnostics). Below is an evidence-based audit of each completed step."
    )

    # 1.1 Data Loading & Description
    h1_1 = doc.add_heading("1.1 Data Loading & Workspace Inventory", level=2)
    h1_1.paragraph_format.space_before = Pt(6)
    h1_1.paragraph_format.space_after = Pt(2)
    for r in h1_1.runs:
        r.font.color.rgb = RGBColor(31, 73, 125)

    doc.add_paragraph(
        "A complete inventory was conducted across all 8 raw CSV files in the workspace (totaling 1,590,066 records). "
        "Four benchmark datasets totaling 699,877 records were selected for primary clinical and population analysis:"
        "\n• UCI Pima Indians Diabetes (data/raw/diabetes.csv): 768 patient records across 9 clinical lab features (34.90% diabetes)."
        "\n• UCI Cleveland Heart Disease (data/raw/heart_cleveland_upload.csv): 297 patient records across 14 cardiac features (46.13% disease)."
        "\n• CDC BRFSS 2015 Health Indicators: 253,680 survey records across 22 questions with dual disease outcomes (13.93% diabetes, 9.42% heart disease)."
        "\n• CDC Heart 2022 (with NaNs): 445,132 records used as a benchmark for real-world survey missingness mechanisms."
    )

    if (TBL_DIR / "dataset_inventory.csv").exists():
        df_inv = pd.read_csv(TBL_DIR / "dataset_inventory.csv")
        add_styled_table(doc, df_inv, "Table 1: Complete Dataset Inventory in Workspace", font_size_pt=7.5)

    # 1.2 Data Preprocessing
    h1_2 = doc.add_heading("1.2 Data Preprocessing & Missing Data Science", level=2)
    h1_2.paragraph_format.space_before = Pt(6)
    h1_2.paragraph_format.space_after = Pt(2)
    for r in h1_2.runs:
        r.font.color.rgb = RGBColor(31, 73, 125)

    doc.add_paragraph(
        "Data cleaning and quality audits were executed via src/02_clean.py, src/02b_clean_brfss.py, and src/project_pipeline.py:"
        "\n• Pima Hidden Zeros Audit: Uncovered 652 hidden zeros across 376 individual patient rows (48.96% of cohort). These zeros "
        "represented test omissions in continuous biometrics where a true zero is biologically impossible: Insulin (374 rows, 48.70%), "
        "Skinfold Thickness (227 rows, 29.56%), Diastolic BP (35 rows, 4.56%), BMI (11 rows, 1.43%), and Glucose (5 rows, 0.65%). "
        "All impossible zeros were replaced with NaN in data/clean/pima_clean.csv. Pregnancies zeros (111 rows, 14.45%) were preserved as valid nulliparous counts."
        "\n• Survey Missingness & Duplicates: In CDC Heart 2022, listwise deletion was shown to drop 44.73% of records (199,110 rows), "
        "whereas preserving observed targets dropped only 0.69%. In CDC BRFSS 2015, 24,206 duplicate rows (9.54%) were identified and intentionally retained, "
        "as discrete survey profiles from 253,680 people naturally produce identical responses."
        "\n• Vertical Dataset Integration: Vertically merged Pima (768), Cleveland (297), and a stratified sample of BRFSS (25,000) "
        "into data/clean/integrated_dataset.csv (26,065 rows, 13 harmonized columns). Documented structural missingness by design: "
        "lab values are absent in survey data (Cholesterol 98.86%, Glucose 97.07%, BP 96.05%, Insulin 98.49% missing)."
        "\n• Normalization & Transformations: Tested Min-Max, Z-score, and log1p transformations. Insulin skewness improved from +2.17 to -0.09. "
        "Age log-transformation was proven contraindicated, worsening skewness from -0.36 to -1.01."
    )

    if (TBL_DIR / "pima_missing_summary.csv").exists():
        df_pima_miss = pd.read_csv(TBL_DIR / "pima_missing_summary.csv")[["Variable", "Zero_Placeholder_Count", "Percent_Missing", "Clinical_Mechanism"]]
        add_styled_table(doc, df_pima_miss, "Table 2: Pima Indians Diabetes Hidden Zero Audit", font_size_pt=8.0)

    # 1.3 Frequent Pattern Mining
    h1_3 = doc.add_heading("1.3 Frequent Pattern Mining & Association Rules", level=2)
    h1_3.paragraph_format.space_before = Pt(6)
    h1_3.paragraph_format.space_after = Pt(2)
    for r in h1_3.runs:
        r.font.color.rgb = RGBColor(31, 73, 125)

    doc.add_paragraph(
        "Conducted frequent pattern mining on 25,000 transactions (15 binary items) using Apriori and FP-Growth:"
        "\n• Algorithm Benchmark: Verified exact itemset equivalence at min_support = 0.20 (72 identical itemsets). FP-Growth executed faster (0.0031s vs 0.0041s)."
        "\n• Support Sensitivity: Evaluated support thresholds across [0.40, 0.30, 0.20, 0.10, 0.05]. At 0.05, 611 itemsets and 5,836 association rules emerged."
        "\n• Top Population Rules: {HighChol, BMI>=30, Age>=50} -> Diabetes (Confidence: 35.66%, Lift: 2.58 vs. 13.93% baseline). "
        "{HighChol, HighBP, Age>=50} -> HeartDisease (Confidence: 23.00%, Lift: 2.43 vs. 9.46% baseline)."
        "\n• Clinical Benchmark Rules: In Pima, {Age>=35, Glucose>125, BMI>=30} -> Diabetes yielded 77.06% confidence (Lift: 2.21). "
        "In Cleveland, {Male, HighChol>=240, ExAngina} -> HeartDisease yielded 90.24% confidence (Lift: 1.96)."
    )

    # 1.4 Clustering, Dimensionality Reduction & Outlier Detection
    h1_4 = doc.add_heading("1.4 Unsupervised Clustering, Dimensionality Reduction & Outlier Diagnostics", level=2)
    h1_4.paragraph_format.space_before = Pt(6)
    h1_4.paragraph_format.space_after = Pt(2)
    for r in h1_4.runs:
        r.font.color.rgb = RGBColor(31, 73, 125)

    doc.add_paragraph(
        "Executed Stage 2 unsupervised machine learning via src/stage2_pipeline.py:"
        "\n• Clustering Models: Benchmarked K-Means (k=3), Hierarchical Agglomerative (Ward linkage, k=3), and DBSCAN (eps=2.4, minPts=5). "
        "K-Means (k=3) achieved Silhouette = 0.1534, Davies-Bouldin = 2.1061, and Calinski-Harabasz = 860.45. Discovered 3 clinical phenotypes: "
        "Cluster 0 (Low Risk, 46.8%, 3.2% diabetes, 1.8% heart disease), Cluster 1 (Moderate Overweight, 39.1%, 21.6% diabetes, 14.4% heart disease), "
        "and Cluster 2 (Multi-Morbidity Crisis, 14.1%, 27.0% diabetes, 23.0% heart disease)."
        "\n• Dimensionality Reduction: PCA decomposed 11 features into PC1 (24.78% variance; physical deterioration) and PC2 (13.91% variance; vascular aging). "
        "Top 6 PCs retain 73.98% variance. Re-clustering on 6 PCs improved Silhouette by +33.4% (to 0.2046) and reduced Davies-Bouldin by -19.0% (to 1.7054)."
        "\n• Outlier Detection: Isolation Forest identified 175 outliers (3.5% contamination). Outliers represent severe multi-morbidity (avg BMI 33.4, max 82.0, "
        "24.9 days/month impaired, 29.7% diabetes, 25.1% heart disease). Pruning outliers restored cluster centroids toward realistic population medians."
    )

    if (TBL_DIR / "stage2_cluster_health_profiles.csv").exists():
        df_clust_prof = pd.read_csv(TBL_DIR / "stage2_cluster_health_profiles.csv")
        add_styled_table(doc, df_clust_prof, "Table 3: Health and Biometric Profiles across K-Means Clusters", font_size_pt=7.5)

    # Embed Cluster & PCA figure
    add_figure(
        doc,
        FIG_DIR / "stage2_fig4_cluster_scatter_projections.png",
        "Figure 1: 2D Manifold Visualizations — PCA vs. t-SNE Projections Colored by Risk Clusters and Diagnoses",
        "PCA and t-SNE cleanly segregate Low-Risk healthy respondents from High-Risk multi-morbidity patients."
    )

    # 1.5 Supervised Model Training & Explainability Status
    h1_5 = doc.add_heading("1.5 Supervised Machine Learning & Explainability (Current Status)", level=2)
    h1_5.paragraph_format.space_before = Pt(6)
    h1_5.paragraph_format.space_after = Pt(2)
    for r in h1_5.runs:
        r.font.color.rgb = RGBColor(31, 73, 125)

    doc.add_paragraph(
        "Plain-Truth Audit: Supervised model training, evaluation, and explainability work have NOT YET BEEN IMPLEMENTED. "
        "Specifically: no classification models (Logistic Regression, Decision Tree, Random Forest) have been trained or evaluated; "
        "no F1, ROC-AUC, or Recall metrics have been generated; and no SHAP global summary or patient-level waterfall plots exist in the repository."
    )

    # -------------------------------------------------------------
    # SECTION 2: CURRENT STATE VS. END PRODUCT (GAP ANALYSIS)
    # -------------------------------------------------------------
    h2 = doc.add_heading("2. Current State vs. The End Product (Gap Analysis)", level=1)
    h2.paragraph_format.space_before = Pt(10)
    h2.paragraph_format.space_after = Pt(2)
    for r in h2.runs:
        r.font.name = "Calibri Light"
        r.font.size = Pt(14)
        r.font.bold = True
        r.font.color.rgb = RGBColor(31, 73, 125)

    doc.add_paragraph(
        "The table below details every explicit project requirement against actual files and outputs in the workspace:"
    )

    gap_data = [
        {
            "Requirement": "Clean, merged, model-ready dataset",
            "Status": "Partial",
            "Evidence": "data/clean/integrated_dataset.csv, pima_clean.csv, cleveland_clean.csv, brfss_clean.csv",
            "What_is_Missing": "Merged dataset has 96-98% structural missingness for lab values. Dedicated feature matrices (X, y) split into train/test folds have not been built."
        },
        {
            "Requirement": "Risk prediction model for diabetes",
            "Status": "Not started",
            "Evidence": "None",
            "What_is_Missing": "No supervised model (Logistic Regression, Decision Tree, Random Forest) trained for diabetes prediction."
        },
        {
            "Requirement": "Risk prediction model for heart disease",
            "Status": "Not started",
            "Evidence": "None",
            "What_is_Missing": "No supervised model trained for heart disease prediction."
        },
        {
            "Requirement": "Model comparison & best-model selection",
            "Status": "Not started",
            "Evidence": "None",
            "What_is_Missing": "No cross-validation benchmark or selection criteria comparing algorithms."
        },
        {
            "Requirement": "Evaluation with F1, ROC-AUC, Recall",
            "Status": "Not started",
            "Evidence": "None",
            "What_is_Missing": "No evaluation metrics, ROC curves, PR curves, or confusion matrices exist for supervised predictions."
        },
        {
            "Requirement": "Global feature importance",
            "Status": "Partial",
            "Evidence": "stage2_pca_loadings.csv, top_diabetes_rules_by_lift.csv",
            "What_is_Missing": "Only unsupervised PCA loadings and rule lift exist. Supervised Gini importance, logistic weights, and global SHAP values are missing."
        },
        {
            "Requirement": "Patient-level explanation of risk variables",
            "Status": "Not started",
            "Evidence": "None",
            "What_is_Missing": "No SHAP TreeExplainer/LinearExplainer or local waterfall/force plots explaining individual patients."
        },
        {
            "Requirement": "Input a new patient & get risk + top variables",
            "Status": "Not started",
            "Evidence": "None",
            "What_is_Missing": "No interactive script, function, CLI, or UI accepting user inputs and outputting risk predictions + explanations."
        },
        {
            "Requirement": "Final report and presentation",
            "Status": "Partial",
            "Evidence": "stage2_report.pdf (20 pages), mid_report_v2.pdf (18 pages)",
            "What_is_Missing": "Mid-term and Stage 2 reports are complete, but final Stage 3 report and presentation slides are not yet authored."
        }
    ]
    df_gap = pd.DataFrame(gap_data)
    add_styled_table(doc, df_gap, "Table 4: Comprehensive Gap Analysis against Required End Product", font_size_pt=7.5)

    # -------------------------------------------------------------
    # SECTION 3: KEY FINDINGS SO FAR
    # -------------------------------------------------------------
    h3 = doc.add_heading("3. Key Findings So Far", level=1)
    h3.paragraph_format.space_before = Pt(10)
    h3.paragraph_format.space_after = Pt(2)
    for r in h3.runs:
        r.font.name = "Calibri Light"
        r.font.size = Pt(14)
        r.font.bold = True
        r.font.color.rgb = RGBColor(31, 73, 125)

    doc.add_paragraph(
        "Quantitative findings from data preprocessing, pattern mining, and unsupervised clustering:"
        "\n• Baseline Population Disease Burden: In CDC BRFSS (N=253,680), diabetes prevalence is 13.93% (35,346 positive cases) and "
        "heart disease prevalence is 9.42% (23,893 positive cases), reflecting class imbalance ratios of ~6.2:1 and ~9.6:1 respectively."
        "\n• Age & Vascular Aging: Age has the single highest positive loading on PCA Component 2 (+0.5740). Population association rules "
        "indicate that Age>=50 serves as a requisite gatekeeper for cardiometabolic escalation."
        "\n• Hypertension (HighBP): HighBP loads strongly on PC2 (+0.4215) and PC1 (+0.3034). Prevalent in 78.3% of Moderate-Risk "
        "cluster patients and 62.6% of High-Risk cluster patients, compared to only 8.0% of Low-Risk individuals."
        "\n• High Cholesterol (HighChol): Loads heavily on PC2 (+0.4574). Paired with HighBP and Age>=50, it elevates population heart "
        "attack probability by 2.43x baseline (Confidence: 23.00%). In Cleveland angiography, HighChol>=240 paired with exercise angina "
        "predicted heart disease with 90.24% confidence."
        "\n• General & Physical Health Days: GenHlth (+0.4692) and PhysHlth (+0.3838) dominate PC1. Normal inliers report 3.4 days/month "
        "of poor physical health vs. 24.9 days/month for high-acuity outliers."
        "\n• Fasting Glucose in Clinical Cohorts: Fasting glucose >125 mg/dL in Pima predicts diabetes with 77.06% confidence when combined with BMI>=30 and Age>=35."
    )

    # Embed Outlier Boxplot figure
    add_figure(
        doc,
        FIG_DIR / "stage2_fig9_outliers_projection_boxplots.png",
        "Figure 2: Outlier Diagnostics — Manifold Embeddings and Biometric Distribution Contrasts",
        "Isolation Forest outliers exhibit dramatic elevations in BMI (up to 82.0 kg/m²) and physical health impairment (24.9 days/month)."
    )

    # -------------------------------------------------------------
    # SECTION 4: PROBLEMS AND LIMITATIONS
    # -------------------------------------------------------------
    h4 = doc.add_heading("4. Problems, Risks, and Limitations", level=1)
    h4.paragraph_format.space_before = Pt(10)
    h4.paragraph_format.space_after = Pt(2)
    for r in h4.runs:
        r.font.name = "Calibri Light"
        r.font.size = Pt(14)
        r.font.bold = True
        r.font.color.rgb = RGBColor(31, 73, 125)

    doc.add_paragraph(
        "Critical analytical and technical constraints identified during the audit:"
        "\n1. Structural Missingness in Merged Cohort: In integrated_dataset.csv, continuous lab tests (glucose, insulin, continuous BP, cholesterol) "
        "are absent in over 96% to 98% of rows because the CDC survey records only binary questions. Training a single model directly on the merged "
        "dataset would introduce severe bias. Separate models must be trained per disease cohort or trained on survey-common features."
        "\n2. Severe Class Imbalance: Both disease targets in the survey cohort are heavily skewed (~90% negative vs ~10% positive). "
        "Standard accuracy is misleading (a naive dummy classifier achieves >90% accuracy). Evaluation must strictly prioritize Recall, F1, and ROC-AUC."
        "\n3. Small Clinical Cohort Sizes: UCI Cleveland (N=297) and UCI Pima (N=768) are small clinical datasets compared to CDC BRFSS (N=253,680), "
        "making clinical models vulnerable to overfitting if regularized hyperparameters are not enforced."
        "\n4. Target Proxy Circularity Risk: high_glucose in BRFSS was mapped directly from Diabetes_binary (r=0.97). While removed from the EDA "
        "heatmap, it must strictly be excluded as an input feature in supervised modeling to avoid trivial target leakage."
        "\n5. Survey Recall Bias: CDC BRFSS relies on self-reported phone survey responses, subject to recall bias and lack of longitudinal verification."
    )

    # -------------------------------------------------------------
    # SECTION 5: REMAINING ROADMAP
    # -------------------------------------------------------------
    h5 = doc.add_heading("5. Prioritized Roadmap to Complete the End Product", level=1)
    h5.paragraph_format.space_before = Pt(10)
    h5.paragraph_format.space_after = Pt(2)
    for r in h5.runs:
        r.font.name = "Calibri Light"
        r.font.size = Pt(14)
        r.font.bold = True
        r.font.color.rgb = RGBColor(31, 73, 125)

    doc.add_paragraph(
        "To advance the project from 50% to 100% completion, the following 5 sequential phases must be executed:"
    )

    roadmap_data = [
        {
            "Phase": "Phase 1: Feature Matrices & Leakage-Free Splitting",
            "Key_Tasks": "Build separate model-ready datasets (Population Survey vs. Clinical Lab); implement stratified 80/20 train/test splits; build scikit-learn Pipelines with median/robust scaling.",
            "Files_to_Create_or_Modify": "src/03_model_preprocessing.py, data/clean/diabetes_model_ready.csv, data/clean/heart_model_ready.csv",
            "Effort_Estimate": "4 - 6 Hours"
        },
        {
            "Phase": "Phase 2: Supervised Model Training",
            "Key_Tasks": "Train separate models for Diabetes and Heart Disease across Logistic Regression, Decision Tree, and Random Forest; tune hyperparameters via GridSearchCV; handle class imbalance via class_weight='balanced' and SMOTE.",
            "Files_to_Create_or_Modify": "src/04_train_models.py, models/diabetes_*.joblib, models/heart_*.joblib",
            "Effort_Estimate": "6 - 8 Hours"
        },
        {
            "Phase": "Phase 3: Model Evaluation & Best Selection",
            "Key_Tasks": "Evaluate held-out test sets on F1-Score, ROC-AUC, Recall (Sensitivity); generate comparative benchmark tables, ROC curves, PR curves, and confusion matrices; select champion model.",
            "Files_to_Create_or_Modify": "src/05_evaluate_models.py, outputs/tables/model_comparison_metrics.csv, outputs/figures/stage3_roc_pr_curves.png",
            "Effort_Estimate": "5 - 6 Hours"
        },
        {
            "Phase": "Phase 4: Global & Local SHAP Explainability",
            "Key_Tasks": "Compute SHAP values using TreeExplainer and LinearExplainer; generate global summary beeswarm plots; implement patient-level waterfall plots explaining exact variables pushing risk.",
            "Files_to_Create_or_Modify": "src/06_explainability_shap.py, outputs/figures/shap_summary_plots.png, outputs/figures/shap_patient_waterfall.png",
            "Effort_Estimate": "6 - 8 Hours"
        },
        {
            "Phase": "Phase 5: Interactive Patient Inference Tool & Final Report",
            "Key_Tasks": "Develop CLI/interactive script where user enters patient biometrics and gets predicted risk + top 3 pushing factors; compile final Stage 3 report and presentation slides.",
            "Files_to_Create_or_Modify": "src/predict_patient.py, src/build_final_report.py, stage3_final_report.docx/.pdf, final_presentation.pptx",
            "Effort_Estimate": "6 - 8 Hours"
        }
    ]
    df_road = pd.DataFrame(roadmap_data)
    add_styled_table(doc, df_road, "Table 5: Prioritized Step-by-Step Implementation Roadmap", font_size_pt=7.5)

    add_callout(
        doc,
        "Total Estimated Remaining Effort",
        "Total Estimated Implementation Time: 27 – 36 Working Hours (approximately 4 to 5 working days).\n"
        "Execution Path: Preprocessing -> Model Training -> Multi-Metric Evaluation -> SHAP Explainability -> Interactive Triage Interface -> Final Report."
    )

    # -------------------------------------------------------------
    # SIGN-OFF
    # -------------------------------------------------------------
    p_sign = doc.add_paragraph()
    p_sign.paragraph_format.space_before = Pt(12)
    run_sign = p_sign.add_run("Audit Sign-off:\nVerified against repository filesystem, data pipelines, output tables, and compiled reports.\nReport Output: PROJECT_STATUS_REPORT.pdf | Date: October 2026")
    run_sign.font.size = Pt(8.5)
    run_sign.font.italic = True
    run_sign.font.color.rgb = RGBColor(120, 120, 120)

    doc.save(DOCX_OUT)
    print(f"Saved Word source to: {DOCX_OUT}")


def convert_docx_to_pdf():
    print(f"Converting {DOCX_OUT.name} to {PDF_OUT.name} via Microsoft Word (docx2pdf)...")
    convert(str(DOCX_OUT), str(PDF_OUT))
    print(f"Successfully generated {PDF_OUT.name}!")

    if PDF_OUT.exists():
        reader = PdfReader(str(PDF_OUT))
        print("=" * 80)
        print(f"VERIFICATION: Total Page Count of {PDF_OUT.name}: {len(reader.pages)} pages")
        print(f"File Size: {PDF_OUT.stat().st_size / 1024:.1f} KB")
        print(f"File Location: {PDF_OUT}")
        print("=" * 80)
        return len(reader.pages)
    return 0


if __name__ == "__main__":
    build_status_document()
    convert_docx_to_pdf()
