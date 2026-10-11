"""
src/build_report.py
Generates the revised comprehensive academic and clinical project report:
1. Word format: mid_report_v2.docx
2. PDF format:  mid_report_v2.pdf (converted via native Microsoft Word)
Verifies that page count is strictly within 5-20 pages.
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
DOCX_PATH = BASE_DIR / "mid_report_v2.docx"
PDF_PATH = BASE_DIR / "mid_report_v2.pdf"


def set_cell_background(cell, fill_hex):
    """Set background color of a table cell."""
    tcPr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    tcPr.append(shd)


def set_cell_margins(cell, top=80, bottom=80, left=120, right=120):
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
    # Caption
    p_cap = doc.add_paragraph()
    p_cap.paragraph_format.space_before = Pt(7)
    p_cap.paragraph_format.space_after = Pt(2)
    p_cap.paragraph_format.keep_with_next = True
    run_cap = p_cap.add_run(caption)
    run_cap.font.name = "Calibri"
    run_cap.font.size = Pt(9.0)
    run_cap.font.bold = True
    run_cap.font.color.rgb = RGBColor(43, 92, 143)

    # Table
    table = doc.add_table(rows=len(df) + 1, cols=len(df.columns))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER

    # Header Row
    hdr_cells = table.rows[0].cells
    for i, col_name in enumerate(df.columns):
        clean_name = str(col_name).replace("_", " ").title()
        hdr_cells[i].text = clean_name
        set_cell_background(hdr_cells[i], "1F497D")  # Deep Navy Blue
        set_cell_margins(hdr_cells[i], top=100, bottom=100, left=110, right=110)
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
            if isinstance(val, float):
                text_val = f"{val:.4f}" if 0 < abs(val) < 0.01 else f"{val:.2f}"
            else:
                text_val = str(val)

            row_cells[col_idx].text = text_val
            set_cell_background(row_cells[col_idx], bg_color)
            set_cell_margins(row_cells[col_idx], top=60, bottom=60, left=90, right=90)
            p = row_cells[col_idx].paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT if col_idx == 0 or len(text_val) > 20 else WD_ALIGN_PARAGRAPH.CENTER
            for r in p.runs:
                r.font.name = "Calibri"
                r.font.size = Pt(font_size_pt)
                r.font.color.rgb = RGBColor(40, 40, 40)

    # Apply widths if provided
    if col_widths and len(col_widths) == len(df.columns):
        for row in table.rows:
            for idx, width in enumerate(col_widths):
                row.cells[idx].width = width

    doc.add_paragraph().paragraph_format.space_after = Pt(4)


def add_styled_figure(doc, img_path, caption, takeaway, width=Inches(5.4)):
    """Embed an image with numbered caption and 1-2 sentence clinical takeaway."""
    p_img = doc.add_paragraph()
    p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_img.paragraph_format.space_before = Pt(7)
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
    run_cap.font.size = Pt(9)
    run_cap.font.bold = True
    run_cap.font.color.rgb = RGBColor(43, 92, 143)

    # Takeaway
    p_take = doc.add_paragraph()
    p_take.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_take.paragraph_format.space_before = Pt(1)
    p_take.paragraph_format.space_after = Pt(6)
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

    # -------------------------------------------------------------
    # Document Header / Title
    # -------------------------------------------------------------
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
    p_sub.paragraph_format.space_after = Pt(8)
    p_sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_sub = p_sub.add_run("A Data Mining Study on Preprocessing, Missing Data Science, and Association Rule Mining")
    r_sub.font.name = "Calibri"
    r_sub.font.size = Pt(10.5)
    r_sub.font.italic = True
    r_sub.font.color.rgb = RGBColor(100, 100, 100)

    # Metadata Box
    p_meta = doc.add_paragraph()
    p_meta.paragraph_format.space_before = Pt(0)
    p_meta.paragraph_format.space_after = Pt(12)
    p_meta.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_meta = p_meta.add_run("Team Data Mining Course Project Report (Revised Edition) | Random State = 42 | Deliverable Package")
    r_meta.font.name = "Calibri"
    r_meta.font.size = Pt(8.5)
    r_meta.font.bold = True
    r_meta.font.color.rgb = RGBColor(43, 92, 143)

    # -------------------------------------------------------------
    # 1. Introduction
    # -------------------------------------------------------------
    h1 = doc.add_heading("1. Introduction", level=1)
    h1.paragraph_format.space_before = Pt(10)
    h1.paragraph_format.space_after = Pt(3)

    doc.add_paragraph(
        "Type 2 diabetes mellitus and coronary heart disease represent the leading drivers of morbidity, healthcare expenditure, "
        "and premature mortality worldwide. While historically diagnosed and managed as distinct medical entities, modern clinical "
        "epidemiology recognizes that diabetes and cardiovascular disease are deeply interconnected manifestations of shared "
        "cardiometabolic dysfunction. They share fundamental biological drivers, including chronic insulin resistance, visceral adiposity, "
        "atherogenic dyslipidemia, systemic vascular endothelial inflammation, and chronic arterial hypertension. Timely identification of "
        "individuals harboring synergistic risk factors offers the highest clinical leverage to arrest disease progression before irreversible "
        "end-organ damage, myocardial infarction, or stroke occurs."
    )

    doc.add_paragraph(
        "This project establishes a rigorous, reproducible data mining pipeline to uncover multi-factorial cardiometabolic risk patterns. "
        "Specifically, we investigate how specialized clinical biometric examinations (e.g., fasting plasma glucose, 2-hour serum insulin, "
        "and diastolic blood pressure) and large-scale public health surveillance indicators (e.g., CDC BRFSS lifestyle and demographic surveys) "
        "can be systematically audited, preprocessed, harmonized, and mined. Using unsupervised frequent pattern mining algorithms "
        "(Apriori and FP-Growth), we discover multi-factor risk rules that delineate high-risk patient subgroups and evaluate how digital health "
        "platforms—such as Dietify—can operationalize these discoveries for automated, preventative lifestyle and dietary interventions."
    )

    # -------------------------------------------------------------
    # 2. Data Description & File Selection
    # -------------------------------------------------------------
    h2 = doc.add_heading("2. Data Description & Study Inventory", level=1)
    h2.paragraph_format.space_before = Pt(10)
    h2.paragraph_format.space_after = Pt(3)

    inv_df = pd.read_csv(TBL_DIR / "dataset_inventory.csv")
    total_all_records = int(inv_df["Rows"].sum())
    total_selected_records = int(inv_df[inv_df["Selected"]]["Rows"].sum())

    doc.add_paragraph(
        f"To establish both high clinical resolution and broad epidemiological generalizability, our project workspace inventories "
        f"eight candidate data files totaling exactly {total_all_records:,} records across all available files. From this inventory, four datasets "
        f"comprising exactly {total_selected_records:,} records were selected for primary clinical and survey benchmarking. "
        f"Table 0 inventories all candidate files, detailing their dimensions, disk footprints, selection statuses, and analytical roles."
    )

    add_styled_table(doc, inv_df, "Table 0: Comprehensive Inventory of Available Project Datasets and Study Selection Decisions", font_size_pt=7.5)

    doc.add_paragraph(
        "Primary Selection Rationale:\n"
        "1. UCI Pima Indians Diabetes (diabetes.csv, 768 records): Selected because it provides continuous, gold-standard physiological lab tests "
        "(fasting plasma glucose, 2-hour serum insulin, diastolic blood pressure, triceps skinfold thickness, and BMI) across 768 adult females of Pima heritage.\n"
        "2. UCI Cleveland Heart Disease (heart_cleveland_upload.csv, 297 records): Selected for its clinical angiography measurements, including resting blood pressure, "
        "serum cholesterol, exercise-induced angina, and fluoroscopy vessel counts across 297 cardiac patients.\n"
        "3. CDC BRFSS 2015 Health Indicators (diabetes_binary_health_indicators_BRFSS2015.csv, 253,680 records): Selected for its massive population sample "
        "recording BOTH diabetes diagnosis (Diabetes_binary) AND heart disease status (HeartDiseaseorAttack) on the exact same individuals. A representative stratified "
        "sample of 25,000 transactions was utilized for computationally tractible association rule mining.\n"
        "4. CDC BRFSS 2022 Raw with NaNs (heart_2022_with_nans.csv, 445,132 records): Retained strictly as a standalone demonstration of authentic survey "
        "missing data science, evaluating listwise deletion versus principled imputation."
    )

    # -------------------------------------------------------------
    # 3. Data Preprocessing
    # -------------------------------------------------------------
    h3 = doc.add_heading("3. Data Preprocessing", level=1)
    h3.paragraph_format.space_before = Pt(10)
    h3.paragraph_format.space_after = Pt(3)

    # 3.1 Data Cleaning & Impossible Zeros
    h3_1 = doc.add_heading("3.1 Data Cleaning & The 'Hidden Zeros' Audit", level=2)
    h3_1.paragraph_format.space_before = Pt(7)
    h3_1.paragraph_format.space_after = Pt(2)

    doc.add_paragraph(
        "At initial ingestion, standard Pandas integrity checks reported zero explicit missing values (NaN) in the Pima Indians dataset. However, "
        "domain-specific physiological auditing revealed that numerical zeros were recorded as placeholders for unperformed clinical examinations. "
        "In living humans, fasting plasma glucose, diastolic blood pressure, triceps skinfold thickness, serum insulin, and BMI cannot physically "
        "equal zero; a recorded value of zero reflects severe biological impossibility. Consequently, all zero entries in these five attributes were audited "
        "and converted to np.nan. In total, exactly 652 hidden zeros were uncovered across 376 individual patient records (Glucose: 5 [0.65%], "
        "BloodPressure: 35 [4.56%], SkinThickness: 227 [29.56%], Insulin: 374 [48.70%], and BMI: 11 [1.43%]). Conversely, zero values in Pregnancies "
        "(111 records [14.45%]) represent valid nulliparous females and were preserved intact. Raw NaNs were preserved in data/clean/pima_clean.csv "
        "without premature imputation to prevent data leakage."
    )

    doc.add_paragraph(
        "In the Cleveland dataset (heart_cleveland_upload.csv), zero duplicate rows and zero missing values were confirmed across all 297 patients. "
        "Outlier screening identified a single patient (row index 79: a 67-year-old female) with a serum cholesterol of 564 mg/dL. While far exceeding "
        "standard reference ranges, extreme cholesterol levels above 500 mg/dL represent authentic familial hypercholesterolemia. Rather than discarding "
        "this vital clinical pathology, the record was retained in data/clean/cleveland_clean.csv, with downstream algorithms employing robust scaling."
    )

    # 3.2 Missing Values & Structural Missingness
    h3_2 = doc.add_heading("3.2 Missing Values Analysis, Clinical Mechanisms & Structural Missingness", level=2)
    h3_2.paragraph_format.space_before = Pt(7)
    h3_2.paragraph_format.space_after = Pt(2)

    doc.add_paragraph(
        "Missing data in health sciences stems from diverse mechanisms: test invasiveness, clinical protocol skips, survey respondent refusal, "
        "or structural discrepancies between study designs. Table 1A outlines the exact missing counts, percentages, clinical mechanisms, and explicit handling "
        "decisions for the Pima Indians dataset after converting hidden zeros to NaN."
    )

    pima_miss_df = pd.read_csv(TBL_DIR / "pima_missing_summary.csv")
    add_styled_table(doc, pima_miss_df, "Table 1A: Pima Indians Diabetes Hidden Zeros Audit, Missingness Frequencies, Clinical Mechanisms, and Principled Handling Decisions", font_size_pt=7.5)

    doc.add_paragraph(
        "Justified Decisions for High-Missingness Pima Variables:\n"
        "1. Serum Insulin (48.70% Missing): The 2-hour post-challenge serum insulin test requires an uncomfortable, costly oral glucose tolerance protocol, "
        "omitted for nearly half the cohort. Performing listwise deletion (dropping rows with missing insulin) would permanently destroy 48.7% of the sample, "
        "introducing catastrophic selection bias toward subjects with specialized medical attention. Simple univariate median imputation would artificially spike "
        "distributional variance at an arbitrary central point. Therefore, in predictive modeling, Multiple Imputation by Chained Equations (MICE) or tree-based "
        "algorithms with native missingness branching (e.g., LightGBM/XGBoost) should be used, paired with an explicit 'Insulin_unmeasured' indicator flag. "
        "For population-level association rule mining across the integrated cohort, Insulin is excluded because it was never collected in the BRFSS surveillance survey.\n"
        "2. Triceps Skinfold Thickness (29.56% Missing): Caliper measurement was frequently skipped due to patient discomfort or equipment calibration delays. "
        "Skinfold thickness is strongly collinear with BMI (Pearson r = 0.65). In predictive modeling, regression imputation conditional on BMI and age preserves "
        "physiological variance far better than univariate central imputation. Like insulin, Skinfold Thickness is excluded from cross-cohort association rules due to absence in BRFSS."
    )

    doc.add_paragraph(
        "Harmonized Cohort Structural Missingness (Missingness by Design):\n"
        "When integrating disparate data sources, vertical concatenation introduces structural missingness: variables present in one study protocol were simply "
        "never collected in another. Table 1B details the missingness per column across the 26,065 records of our harmonized dataset."
    )

    integ_miss_df = pd.read_csv(TBL_DIR / "integrated_missing_summary.csv")
    add_styled_table(doc, integ_miss_df, "Table 1B: Harmonized Cohort Structural Missingness Analysis per Column and Analytical Treatment in Mining", font_size_pt=7.5)

    doc.add_paragraph(
        "Treatment of Structural Missingness in Frequent Pattern Mining:\n"
        "Association rule mining operates on boolean transaction matrices where item presence is evaluated across all transactions. If mining were executed "
        "across all 26,065 integrated rows, continuous lab features like cholesterol (98.86% missing) or glucose (97.07% missing) would appear in fewer than "
        "3% of transactions, artificially depressing support and generating degenerate missingness rules. Furthermore, Pima rows lack heart disease evaluations "
        "(768 missing rows, 2.95%). Therefore, our primary frequent pattern mining is executed specifically on the 25,000 BRFSS transactions where all 15 candidate "
        "lifestyle, demographic, and binary outcome indicators were uniformly measured on every single individual (0% missing). Auxiliary mining is separately "
        "conducted on Pima and Cleveland to extract laboratory-specific rules without cross-study imputation artifacts."
    )

    doc.add_paragraph(
        "CDC BRFSS 2022 Standalone Real Survey Missingness Demonstration:\n"
        "To evaluate real-world missing data protocols, a standalone empirical test was conducted on heart_2022_with_nans.csv (445,132 raw survey responses). "
        "If an analyst applies naive listwise deletion (df.dropna()), exactly 199,110 records—representing 44.73% of the entire cohort—are permanently destroyed. "
        "Because sicker, older, or economically disadvantaged individuals are systematically more likely to skip survey items, listwise deletion introduces severe "
        "demographic selection bias. In contrast, our principled approach discards records ONLY when the primary clinical outcome (HadHeartAttack) is missing "
        "(3,065 rows, or 0.69%), preserving 442,067 records (99.31%) while applying median and mode imputation to missing clinical predictors. Table 1C documents "
        "the most frequently skipped survey items. Crucially, BRFSS 2022 serves exclusively as a standalone missing-data demonstration; NO imputation was applied "
        "to the BRFSS 2015 data used for mining, which was complete at ingestion."
    )

    miss_df = pd.read_csv(TBL_DIR / "missing_values_summary.csv").head(10)
    add_styled_table(doc, miss_df, "Table 1C: CDC BRFSS 2022 Real Survey Missing Value Accounting, Clinical Mechanisms, and Planned Imputation Strategy", font_size_pt=7.5)

    # 3.3 Dataset Integration
    h3_3 = doc.add_heading("3.3 Dataset Integration & Vertical Harmonization", level=2)
    h3_3.paragraph_format.space_before = Pt(7)
    h3_3.paragraph_format.space_after = Pt(2)

    doc.add_paragraph(
        "Why a Horizontal (Patient-to-Patient) Join is Impossible:\n"
        "A horizontal row-by-row join across Pima, Cleveland, and BRFSS is scientifically invalid because these datasets represent completely separate patient populations, "
        "study designs, geographical catchment areas, and historical decades. The Pima cohort comprises 768 Native American females examined between 1965 and 1980 in Arizona; "
        "the Cleveland cohort represents 297 cardiac inpatients undergoing angiography in Ohio; and the BRFSS represents civilian households surveyed via random-digit telephone "
        "dialing across the United States. There are zero shared patient identifiers. Horizontally merging row 1 of Pima with row 1 of Cleveland would fabricate a fictitious, "
        "contradictory hybrid individual."
    )

    doc.add_paragraph(
        "Consequently, we executed vertical harmonization: mapping shared clinical and demographic concepts into a standardized data schema. Harmonized variables "
        "include age, biological sex, BMI, hypertension (high_bp), hypercholesterolemia (high_chol), hyperglycemia (high_glucose), diabetes diagnosis, and heart disease outcome. "
        "Table 2 details the exact accounting of raw, kept, and sampled rows, confirming a unified harmonized cohort of 26,065 records."
    )

    int_df = pd.read_csv(TBL_DIR / "integration_summary.csv")
    add_styled_table(doc, int_df, "Table 2: Unified Dataset Integration Accounting and Harmonization Log", font_size_pt=8.0)

    # 3.4 Descriptive Statistics
    h3_4 = doc.add_heading("3.4 Descriptive Statistics of All Study Variables", level=2)
    h3_4.paragraph_format.space_before = Pt(7)
    h3_4.paragraph_format.space_after = Pt(2)

    doc.add_paragraph(
        "To satisfy rigorous exploratory requirements, descriptive statistics were computed across EVERY continuous and categorical variable utilized in this project. "
        "Table 3 presents parametric and non-parametric central tendency and dispersion metrics for all 11 continuous/discrete numeric attributes. Table 4 presents complete "
        "frequency distributions and relative percentages for all categorical, survey, and binary outcome indicators."
    )

    num_df = pd.read_csv(TBL_DIR / "numeric_descriptive_stats.csv")
    add_styled_table(doc, num_df, "Table 3: Comprehensive Descriptive Statistics for All Continuous and Discrete Numeric Variables", font_size_pt=7.5)

    cat_df = pd.read_csv(TBL_DIR / "categorical_descriptive_stats.csv")
    add_styled_table(doc, cat_df, "Table 4: Comprehensive Descriptive Statistics for All Categorical, Survey, and Outcome Indicators", font_size_pt=7.5)

    # 3.5 Visualizations
    h3_5 = doc.add_heading("3.5 Exploratory Data Visualizations", level=2)
    h3_5.paragraph_format.space_before = Pt(7)
    h3_5.paragraph_format.space_after = Pt(2)

    doc.add_paragraph(
        "Figures 1 through 5 illustrate empirical distributions, bivariate relationships, and pairwise correlations across the harmonized cohort. "
        "Every figure is paired with an evaluative clinical takeaway."
    )

    add_styled_figure(
        doc,
        FIG_DIR / "figure1_numeric_histograms.png",
        "Figure 1: Histograms and Kernel Density Estimates of Key Continuous Clinical Variables",
        "Clinical variables exhibit pronounced positive skewness (especially BMI, glucose, and blood pressure), demonstrating that while the healthy majority clusters near baseline medians, a substantial high-risk cohort experiences severe cardiometabolic stress.",
        width=Inches(5.4)
    )

    add_styled_figure(
        doc,
        FIG_DIR / "figure2_glucose_vs_bmi.png",
        "Figure 2: Bivariate Scatter Plot of Fasting Glucose vs. BMI Stratified by Diabetes Outcome (Pima)",
        "Patients exhibiting both fasting hyperglycemia (>125 mg/dL) and clinical obesity (BMI >= 30 kg/m²) concentrate almost exclusively in the diabetic group, visually proving that glycemic and adiposity thresholds jointly drive metabolic disease emergence.",
        width=Inches(4.6)
    )

    add_styled_figure(
        doc,
        FIG_DIR / "figure3_age_vs_cholesterol.png",
        "Figure 3: Bivariate Scatter Plot of Age vs. Serum Cholesterol Stratified by Cardiac Condition (Cleveland)",
        "Diagnosed heart disease patients cluster heavily in the upper-right quadrant of advanced age (>=50 years) and elevated cholesterol (>=240 mg/dL), confirming the compounding vascular pathology of chronological aging and atherogenic lipid deposition.",
        width=Inches(4.6)
    )

    add_styled_figure(
        doc,
        FIG_DIR / "figure4_categorical_distributions.png",
        "Figure 4: Prevalence and Class Balance of Categorical Risk Factors and Health Outcomes",
        "The cohort exhibits realistic epidemiological class balance: 14.43% diabetes prevalence (3,761 / 26,065) and exactly 9.89% heart disease prevalence among evaluated individuals (2,503 / 25,297 non-null records; 9.46% in the BRFSS sample alone).",
        width=Inches(5.4)
    )

    add_styled_figure(
        doc,
        FIG_DIR / "figure5_correlation_heatmap.png",
        "Figure 5: Pearson Correlation Matrix of Harmonized Clinical Variables and Health Outcomes",
        "Age (r=0.22), hypertension (r=0.21), and hypercholesterolemia (r=0.19) exhibit the strongest correlations with heart disease, while BMI (r=0.23) and hypertension (r=0.24) dominate diabetes. High_glucose was removed from this heatmap because in BRFSS it was derived from the diabetes label as an operational proxy (creating an artificial r=0.97 artifact).",
        width=Inches(4.6)
    )

    # 3.6 Normalization & Transformation
    h3_6 = doc.add_heading("3.6 Feature Normalization, Transformation & Skewness Analysis", level=2)
    h3_6.paragraph_format.space_before = Pt(7)
    h3_6.paragraph_format.space_after = Pt(2)

    doc.add_paragraph(
        "Clinical variables operate on vastly differing numerical scales: age spans decades (21-82), blood pressure spans tens of mmHg (24-200), "
        "and cholesterol spans hundreds of mg/dL (126-564). Unscaled features distort distance-based and gradient-descent algorithms. We applied "
        "Min-Max scaling [0, 1] to bound features for neural architectures and Z-score Standardization (mean=0, std=1) for linear algorithms. "
        "Furthermore, right-skewed variables received a natural logarithmic transformation: log1p(x) = ln(1 + x). Table 5 details the transformation effects "
        "across all six clinical features."
    )

    norm_df = pd.read_csv(TBL_DIR / "normalization_transformation_summary.csv")
    add_styled_table(doc, norm_df, "Table 5: Feature Normalization, Standardization, and Skewness Reduction Summary", font_size_pt=7.5)

    doc.add_paragraph(
        "Why Log Transformation is Contraindicated for Age:\n"
        "Log transformation compresses extreme positive values, making it highly effective for variables with long right tails (Insulin: skewness reduced from +2.17 to -0.09; "
        "Cholesterol: +1.12 to +0.06; Glucose: +0.53 to -0.06; BMI: +2.00 to +0.69). However, for Age, the raw distribution is already mildly left-skewed (skewness = -0.36) "
        "due to higher survey completion among older adults. Applying log1p compresses the upper age values, severely exacerbating negative skewness from -0.36 to -1.01 "
        "(a 180% worsening of distributional symmetry). Therefore, log transformation is contraindicated for age; standard Min-Max scaling or Z-score standardization "
        "must be used instead."
    )

    add_styled_figure(
        doc,
        FIG_DIR / "figure6_log_transform_comparison.png",
        "Figure 6: Distribution of Serum Insulin Before and After Natural Log Transformation: log1p(x)",
        "Applying log1p compresses the extreme right tail of serum insulin, reducing skewness from severe positive asymmetry (+2.17) to near-normal symmetry (-0.09), stabilizing variance for statistical and regression modeling.",
        width=Inches(4.8)
    )

    # -------------------------------------------------------------
    # 4. Frequent Pattern Mining
    # -------------------------------------------------------------
    h4 = doc.add_heading("4. Frequent Pattern Mining & Association Rules", level=1)
    h4.paragraph_format.space_before = Pt(10)
    h4.paragraph_format.space_after = Pt(3)

    doc.add_paragraph(
        "To uncover non-linear, multi-factorial risk combinations, we performed Association Rule Mining. Continuous variables were discretized "
        "using established medical criteria: BMI >= 30.0 kg/m² (WHO clinical obesity), Age >= 50 years (epidemiological cardiovascular risk threshold), "
        "GenHlth in {Fair, Poor}, and Income <= $25,000. Categorical survey indicators were one-hot encoded, and disease outcomes were retained as binary targets. "
        "Table 6 inventories all 15 binary transaction items created for frequent pattern mining."
    )

    bin_df = pd.read_csv(TBL_DIR / "binary_items_inventory.csv")
    add_styled_table(doc, bin_df, "Table 6: Complete Inventory of Binary Items Created for Frequent Pattern Mining Grouped by Construction Type", font_size_pt=7.5)

    doc.add_paragraph(
        "Mining Cohort Scope & Justification:\n"
        "Frequent pattern mining was conducted on the 25,000 BRFSS 2015 transactions because all 15 lifestyle, demographic, and binary outcome indicators "
        "were uniformly measured on every single individual. Continuous clinical lab thresholds (such as Pima glucose > 125 mg/dL, Pima insulin > 166 μU/mL, "
        "and Cleveland cholesterol >= 240 mg/dL) do not appear in these population rules due to cross-cohort structural missingness between clinical laboratory records "
        "and large-scale telephone health surveys."
    )

    # 4.1 Apriori vs FP-Growth
    h4_1 = doc.add_heading("4.1 Algorithmic Benchmark: Apriori vs. FP-Growth", level=2)
    h4_1.paragraph_format.space_before = Pt(7)
    h4_1.paragraph_format.space_after = Pt(2)

    doc.add_paragraph(
        "We benchmarked the Apriori and FP-Growth algorithms on the 25,000 transaction cohort at a minimum support threshold of 0.20. Both algorithms produced "
        "exactly 72 identical frequent itemsets, providing empirical verification of mathematical correctness. In terms of execution speed, Apriori completed "
        "in 0.0045 seconds, whereas FP-Growth required 1.9270 seconds. While FP-growth avoids candidate generation in theoretical computer science, MLxtend's "
        "Apriori implementation utilizes highly optimized C-level NumPy bitwise AND operations across dense boolean arrays, outperforming the pure-Python "
        "pointer-overhead of recursive FP-tree construction on wide binary matrices."
    )

    # 4.2 Support Variation
    h4_2 = doc.add_heading("4.2 Sensitivity Analysis: Varying Minimum Support", level=2)
    h4_2.paragraph_format.space_before = Pt(7)
    h4_2.paragraph_format.space_after = Pt(2)

    doc.add_paragraph(
        "To evaluate pattern discovery depth, minimum support was varied across [0.40, 0.30, 0.20, 0.10, 0.05]. Table 7 and Figure 7 document "
        "the resulting combinatorial explosion of frequent itemsets and association rules."
    )

    supp_df = pd.read_csv(TBL_DIR / "support_variation_summary.csv")
    add_styled_table(doc, supp_df, "Table 7: Frequent Itemsets and Association Rules Discovered Across Decreasing Minimum Support Thresholds", font_size_pt=8.0)

    add_styled_figure(
        doc,
        FIG_DIR / "figure7_itemsets_rules_vs_support.png",
        "Figure 7: Growth Trajectory of Frequent Itemsets and Association Rules as Support Decreases",
        "Decreasing minimum support from 0.40 to 0.05 triggers an exponential surge in frequent itemsets (17 to 611) and association rules (24 to 5,836), successfully uncovering highly specific, high-risk disease phenotypes that affect smaller clinical sub-populations.",
        width=Inches(4.8)
    )

    # 4.3 Mined Rules
    h4_3 = doc.add_heading("4.3 Discovered Clinical Association Rules", level=2)
    h4_3.paragraph_format.space_before = Pt(7)
    h4_3.paragraph_format.space_after = Pt(2)

    doc.add_paragraph(
        "Tables 8 and 9 present the top mined association rules specifically targeting Diabetes and Heart Disease as consequents, ranked by Lift "
        "and Confidence at minimum support = 0.05."
    )

    diab_lift = pd.read_csv(TBL_DIR / "top_diabetes_rules_by_lift.csv").head(8)
    add_styled_table(doc, diab_lift, "Table 8: Top Mined Association Rules Targeting Diabetes (Ranked by Lift)", font_size_pt=8.0)

    heart_lift = pd.read_csv(TBL_DIR / "top_heart_disease_rules_by_lift.csv").head(8)
    add_styled_table(doc, heart_lift, "Table 9: Top Mined Association Rules Targeting Heart Disease (Ranked by Lift)", font_size_pt=8.0)

    doc.add_paragraph(
        "Plain-English Translation of Key Rules:\n"
        "Mathematically, Lift is defined as the ratio of rule confidence to baseline consequent prevalence: Lift(A -> B) = P(B|A) / P(B). "
        "It represents a multiplicative relative risk / ratio of probabilities, NOT an odds ratio.\n"
        "1. Diabetes Rule 1 ({Age >= 50, BMI >= 30, HighChol} -> {Diabetes}, Lift = 2.58, Confidence = 35.7%): An adult aged 50 or older who is clinically "
        "obese and has high cholesterol is 2.58 times more likely to have diabetes than a randomly selected individual from the general population. More than "
        "one in three individuals with this exact risk triad suffers from diabetes.\n"
        "2. Diabetes Rule 2 ({Age >= 50, HighBP, BMI >= 30} -> {Diabetes}, Lift = 2.52, Confidence = 34.7%): Chronic hypertension compounded by obesity and "
        "aging elevates diabetes likelihood by more than 150% above population baseline.\n"
        "3. Heart Disease Rule 1 ({Age >= 50, HighBP, HighChol} -> {HeartDisease}, Lift = 2.43, Confidence = 23.0%): This rule encapsulates the classic "
        "atherosclerotic vascular triad. Older adults burdened by both hypertension and hypercholesterolemia suffer a 2.43-fold elevation in heart attack "
        "probability relative to population baseline prevalence (a probability ratio of 2.43, not an odds ratio), with nearly one-quarter (23.0%) experiencing cardiac disease."
    )

    # 4.4 Auxiliary Mining on Clinical Benchmarks
    h4_4 = doc.add_heading("4.4 Domain-Specific Clinical Rules (Pima & Cleveland Benchmarks)", level=2)
    h4_4.paragraph_format.space_before = Pt(7)
    h4_4.paragraph_format.space_after = Pt(2)

    doc.add_paragraph(
        "To complement population survey mining and overcome structural missingness, we separately mined frequent patterns on the Pima (clinical lab) "
        "and Cleveland (cardiac angiography) cohorts. Table 10 presents the top discovered clinical rules."
    )

    pima_rules = pd.read_csv(TBL_DIR / "pima_mined_rules.csv").head(3)
    pima_rules["Cohort"] = "Pima Indians (Lab)"
    clev_rules = pd.read_csv(TBL_DIR / "cleveland_mined_rules.csv").head(3)
    clev_rules["Cohort"] = "Cleveland (Angio)"
    aux_rules = pd.concat([pima_rules, clev_rules], ignore_index=True)[
        ["Cohort", "antecedents_str", "consequents_str", "support", "confidence", "lift"]
    ]
    add_styled_table(doc, aux_rules, "Table 10: Clinical Association Rules Discovered on Pima (Lab) and Cleveland (Angiography) Cohorts", font_size_pt=8.0)

    doc.add_paragraph(
        "Clinical Benchmark Takeaways:\n"
        "- In the Pima clinical lab cohort, the rule {BMI >= 30, Age >= 35, Glucose > 125 mg/dL} -> {Diabetes} achieves 77.1% confidence and a lift of 2.21, "
        "confirming that laboratory hyperglycemia combined with obesity provides exceptionally high predictive confidence.\n"
        "- In the Cleveland angiography cohort, the rule {Male, HighChol >= 240 mg/dL, Exercise Angina} -> {HeartDisease} achieves 90.2% confidence and a lift of 1.96, "
        "demonstrating that exertional chest pain combined with severe hypercholesterolemia in men identifies severe anatomical coronary artery disease with over 90% certainty."
    )

    # -------------------------------------------------------------
    # 5. Discussion & Digital Health Insights
    # -------------------------------------------------------------
    h5 = doc.add_heading("5. Discussion, Strategic Insights & Dietify Integration", level=1)
    h5.paragraph_format.space_before = Pt(10)
    h5.paragraph_format.space_after = Pt(3)

    doc.add_paragraph(
        "The empirical association rules mined across both survey transactions and clinical cohorts demonstrate the compounding synergy of the Metabolic Syndrome triad: "
        "Obesity (BMI >= 30), Hypertension (HighBP), and Dyslipidemia (HighChol). While individual isolated risk factors elevate disease probability modestly, "
        "their co-occurrence multiplies disease probability non-linearly."
    )

    doc.add_paragraph(
        "Actionable Deployment in Digital Health Applications (e.g., Dietify):\n"
        "A preventative mobile health platform like Dietify can embed these exact mined association rules into its clinical decision support engine:\n"
        "- Automated Risk Triggering: When a user logs a BMI >= 30 and records an elevated blood pressure reading, Dietify's rules engine does not wait for an official "
        "physician diagnosis. It immediately flags the user profile and serves personalized low-glycemic, sodium-restricted meal plans to target insulin resistance.\n"
        "- Clinical Screening Recommendations: When an individual matches the {Age >= 50, HighBP, HighChol} rule antecedent, Dietify issues an in-app alert prompting "
        "the user to schedule an annual fasting blood glucose panel, HbA1c test, and ECG stress evaluation with their healthcare provider.\n"
        "- Behavioral Nudging: Identifying the strong association between physical limitation (DiffWalk) and disease emergence enables Dietify to recommend low-impact "
        "restorative physical activities (e.g., chair yoga, aquatic therapy, brisk walking intervals) to interrupt the sedentary cardiometabolic spiral."
    )

    doc.add_paragraph(
        "Critical Project Limitations:\n"
        "1. Association vs. Causation: Association rule mining identifies co-occurrence patterns; it does not prove causality. Chronic hypertension does not necessarily "
        "cause diabetes directly; rather, both conditions stem from common upstream cellular pathology, including hyperinsulinemia, chronic systemic inflammation, and caloric surplus.\n"
        "2. Demographic and Sampling Bias: The Pima dataset is restricted to Native American females in Arizona; the Cleveland dataset is derived from hospital inpatients "
        "undergoing catheterization; and the BRFSS relies on self-reported telephone responses subject to recall and social desirability biases.\n"
        "3. Cross-Cohort Structural Missingness: Due to protocol heterogeneity, continuous laboratory metrics could not be imputed across phone surveys without introducing "
        "massive synthetic bias. Mined transaction rules must be interpreted within their respective survey and clinical operational scopes."
    )

    # -------------------------------------------------------------
    # 6. Conclusion
    # -------------------------------------------------------------
    h6 = doc.add_heading("6. Conclusion", level=1)
    h6.paragraph_format.space_before = Pt(10)
    h6.paragraph_format.space_after = Pt(3)

    doc.add_paragraph(
        "This project established an end-to-end, reproducible data mining pipeline for cardiometabolic disease risk identification. By auditing hidden zeros "
        "(652 zeros across 376 records in Pima), rigorously evaluating survey missing data science (showing listwise deletion destroys 44.7% of survey cohorts), "
        "and vertically harmonizing multi-center datasets, we built a robust analytical foundation. Unsupervised frequent pattern mining successfully discovered "
        "high-confidence, high-lift association rules governing diabetes and heart disease emergence. These discoveries confirm that digital health monitoring "
        "platforms can utilize self-reportable cardiometabolic risk triads to trigger early, personalized preventive interventions before severe clinical disease manifests."
    )

    doc.save(DOCX_PATH)
    print(f"Saved styled report to: {DOCX_PATH}")


def convert_to_pdf_and_check():
    print(f"Converting {DOCX_PATH.name} to {PDF_PATH.name}...")
    convert(str(DOCX_PATH), str(PDF_PATH))
    print(f"Successfully generated {PDF_PATH.name}!")

    reader = PdfReader(str(PDF_PATH))
    page_count = len(reader.pages)
    print("=" * 80)
    print(f"VERIFICATION: Total Page Count of {PDF_PATH.name}: {page_count} pages")
    print(f"Requirement Check: 5 <= {page_count} <= 20 : {'PASSED [OK]' if 5 <= page_count <= 20 else 'FAILED [OUT OF BOUNDS]'}")
    print("=" * 80)
    return page_count


if __name__ == "__main__":
    build_report_document()
    convert_to_pdf_and_check()
