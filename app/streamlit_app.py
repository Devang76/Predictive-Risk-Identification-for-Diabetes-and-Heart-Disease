"""
app/streamlit_app.py
Interactive Clinical Decision Support Web Application
Project: Chronic Disease Predictor Model (Predictive Risk Identification for Diabetes and Heart Disease)

Features:
1. Mode Switcher:
   - Survey Mode (Track A - CDC BRFSS Population Survey)
   - Clinical Mode (Track B - Continuous Diagnostic Lab Tests)
2. One-Click Example Patient Loading:
   - Low-Risk, Medium-Risk, and High-Risk prototypical profiles.
3. Real-Time Risk Categorization & Triage:
   - Color-coded risk tier badges (Low=Green, Medium=Amber, High=Red).
   - Calibrated disease probability gauges.
   - Top 3 risk-increasing biometrics and Top 2 risk-decreasing biometrics.
   - High-resolution interactive SHAP waterfall attribution plots.
4. Stage 2 Cluster Resemblance:
   - Maps patient to nearest Stage 2 metabolic/comorbidity cluster profile.
5. Clinical Safety & Medical Disclaimers.
"""

from pathlib import Path
import json
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import yaml

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.predict_patient import format_reading, predict_and_explain_disease
from src.risk_utils import load_yaml, validate_patient_input

CONFIG_DIR = BASE_DIR / "config"
EXAMPLES_DIR = BASE_DIR / "examples"
TABLES_DIR = BASE_DIR / "outputs" / "tables"
PREDICTIONS_DIR = BASE_DIR / "outputs" / "patient_predictions"
PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

DISCLAIMER_TEXT = (
    "These are key contributing factors from a statistical model, not proven causes, and this is not a medical diagnosis. "
    "Predictions are intended strictly for clinical decision support, risk stratification, and triage facilitation under "
    "the supervision of qualified healthcare professionals."
)

st.set_page_config(
    page_title="Chronic Disease Predictor Model",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data
def load_schemas():
    return load_yaml(CONFIG_DIR / "feature_schema.yaml")


@st.cache_data
def load_risk_tiers():
    return load_yaml(CONFIG_DIR / "risk_tiers.yaml")


@st.cache_data
def load_stage2_cluster_profiles():
    profile_path = TABLES_DIR / "stage2_cluster_health_profiles.csv"
    if profile_path.exists():
        return pd.read_csv(profile_path)
    return None


def get_cluster_resemblance(patient_dict: dict, mode: str) -> str:
    """Find closest Stage 2 cluster profile based on available biometrics."""
    profiles = load_stage2_cluster_profiles()
    if profiles is None:
        return "Stage 2 cluster profiles not loaded."

    # Compare key common features: Age, BMI, Hypertension
    bmi = float(patient_dict.get("BMI", 27.0))
    if mode == "survey":
        age_cat = int(patient_dict.get("Age", 7))
        # Approximate age from category
        age_approx = 20.0 + (age_cat - 1) * 5.0
        hi_bp = float(patient_dict.get("HighBP", 0)) * 100.0
    else:
        age_approx = float(patient_dict.get("age", patient_dict.get("Age", 50)))
        sbp = float(patient_dict.get("trestbps", patient_dict.get("BloodPressure", 120)))
        hi_bp = 100.0 if sbp >= 130.0 else 0.0

    scores = []
    for _, row in profiles.iterrows():
        d_age = (age_approx - float(row["Avg_Age_Years"])) / 15.0
        d_bmi = (bmi - float(row["Avg_BMI"])) / 6.0
        d_bp = (hi_bp - float(row["Hypertension_Rate_Pct"])) / 40.0
        dist = np.sqrt(d_age**2 + d_bmi**2 + d_bp**2)
        scores.append((dist, row["Clinical_Designation"]))

    scores.sort(key=lambda x: x[0])
    return scores[0][1]


def load_example_patient_data(example_type: str, mode: str) -> dict:
    """Load pre-configured example patient profile into session dictionary."""
    if mode == "survey":
        filename = f"{example_type.lower()}_risk_patient.json"
    else:
        filename = f"clinical_patient_{example_type.lower()}.json"

    filepath = EXAMPLES_DIR / filename
    if filepath.exists():
        with open(filepath, "r") as f:
            return json.load(f)
    return {}


# ==============================================================================
# UI HEADER & HERO
# ==============================================================================
st.title("🩺 Chronic Disease Predictor Model")
st.markdown(
    "**Clinical Decision Support for Diabetes and Coronary Heart Disease**  \n"
    "Calculates calibrated disease probability $P(y=1)$, assigns validated tri-level risk tiers "
    "(`Low`, `Medium`, `High`), and computes local patient-specific SHAP attribution drivers."
)

st.warning(f"⚠️ **Clinical Safety Notice:** {DISCLAIMER_TEXT}")

# ==============================================================================
# SIDEBAR CONTROLS & PATIENT LOADING
# ==============================================================================
st.sidebar.header("🛠️ Patient Triage Controls")

mode_selection = st.sidebar.radio(
    "Select Evaluation Mode:",
    ["Survey Mode (Track A - CDC BRFSS)", "Clinical Mode (Track B - Diagnostic Labs)"],
    index=0,
)
current_mode = "survey" if "Survey" in mode_selection else "clinical"

st.sidebar.markdown("---")
st.sidebar.subheader("📋 Load Prototypical Patient")
col_b1, col_b2, col_b3 = st.sidebar.columns(3)

if "patient_input" not in st.session_state:
    st.session_state.patient_input = load_example_patient_data("low", current_mode)

if col_b1.button("🟢 Low Risk", use_container_width=True):
    st.session_state.patient_input = load_example_patient_data("low", current_mode)
    st.rerun()

if col_b2.button("🟡 Med Risk", use_container_width=True):
    st.session_state.patient_input = load_example_patient_data("medium", current_mode)
    st.rerun()

if col_b3.button("🔴 High Risk", use_container_width=True):
    st.session_state.patient_input = load_example_patient_data("high", current_mode)
    st.rerun()

patient_data = st.session_state.patient_input

# ==============================================================================
# INPUT FORM CREATION
# ==============================================================================
st.subheader(f"Patient Biometric Entry ({mode_selection})")
form_inputs = {}

with st.form("patient_assessment_form"):
    if current_mode == "survey":
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("##### 🧬 Vital Signs & Diagnosis")
            form_inputs["HighBP"] = st.selectbox(
                "High Blood Pressure Diagnosis",
                [0, 1],
                index=[0, 1].index(int(patient_data.get("HighBP", 0))),
                format_func=lambda x: "Yes" if x == 1 else "No",
            )
            form_inputs["HighChol"] = st.selectbox(
                "High Cholesterol Diagnosis",
                [0, 1],
                index=[0, 1].index(int(patient_data.get("HighChol", 0))),
                format_func=lambda x: "Yes" if x == 1 else "No",
            )
            form_inputs["CholCheck"] = st.selectbox(
                "Cholesterol Checked in 5 Years",
                [0, 1],
                index=[0, 1].index(int(patient_data.get("CholCheck", 1))),
                format_func=lambda x: "Yes" if x == 1 else "No",
            )
            form_inputs["BMI"] = st.number_input(
                "Body Mass Index (BMI, kg/m²)",
                min_value=12.0,
                max_value=90.0,
                value=float(patient_data.get("BMI", 27.0)),
                step=0.5,
            )
            form_inputs["Stroke"] = st.selectbox(
                "History of Stroke",
                [0, 1],
                index=[0, 1].index(int(patient_data.get("Stroke", 0))),
                format_func=lambda x: "Yes" if x == 1 else "No",
            )

        with c2:
            st.markdown("##### 🏃 Lifestyle & Daily Habits")
            form_inputs["Smoker"] = st.selectbox(
                "Smoking History (>=100 Cigarettes)",
                [0, 1],
                index=[0, 1].index(int(patient_data.get("Smoker", 0))),
                format_func=lambda x: "Yes" if x == 1 else "No",
            )
            form_inputs["PhysActivity"] = st.selectbox(
                "Physical Activity (Past 30 Days)",
                [0, 1],
                index=[0, 1].index(int(patient_data.get("PhysActivity", 1))),
                format_func=lambda x: "Yes" if x == 1 else "No",
            )
            form_inputs["Fruits"] = st.selectbox(
                "Daily Fruit Consumption (>=1/day)",
                [0, 1],
                index=[0, 1].index(int(patient_data.get("Fruits", 1))),
                format_func=lambda x: "Yes" if x == 1 else "No",
            )
            form_inputs["Veggies"] = st.selectbox(
                "Daily Vegetable Consumption (>=1/day)",
                [0, 1],
                index=[0, 1].index(int(patient_data.get("Veggies", 1))),
                format_func=lambda x: "Yes" if x == 1 else "No",
            )
            form_inputs["HvyAlcoholConsump"] = st.selectbox(
                "Heavy Alcohol Consumption",
                [0, 1],
                index=[0, 1].index(int(patient_data.get("HvyAlcoholConsump", 0))),
                format_func=lambda x: "Yes" if x == 1 else "No",
            )

        with c3:
            st.markdown("##### 👤 General Health & Demographics")
            gen_labels = {1: "1 - Excellent", 2: "2 - Very Good", 3: "3 - Good", 4: "4 - Fair", 5: "5 - Poor"}
            form_inputs["GenHlth"] = st.selectbox(
                "General Health Self-Rating",
                [1, 2, 3, 4, 5],
                index=[1, 2, 3, 4, 5].index(int(patient_data.get("GenHlth", 2))),
                format_func=lambda x: gen_labels[x],
            )
            form_inputs["PhysHlth"] = st.slider(
                "Days of Poor Physical Health (Past 30)",
                0,
                30,
                int(patient_data.get("PhysHlth", 0)),
            )
            form_inputs["MentHlth"] = st.slider(
                "Days of Poor Mental Health (Past 30)",
                0,
                30,
                int(patient_data.get("MentHlth", 0)),
            )
            form_inputs["DiffWalk"] = st.selectbox(
                "Difficulty Walking / Stairs",
                [0, 1],
                index=[0, 1].index(int(patient_data.get("DiffWalk", 0))),
                format_func=lambda x: "Yes" if x == 1 else "No",
            )
            form_inputs["Sex"] = st.selectbox(
                "Biological Sex",
                [0, 1],
                index=[0, 1].index(int(patient_data.get("Sex", 0))),
                format_func=lambda x: "Male" if x == 1 else "Female",
            )
            age_brackets = {
                1: "18-24", 2: "25-29", 3: "30-34", 4: "35-39", 5: "40-44", 6: "45-49",
                7: "50-54", 8: "55-59", 9: "60-64", 10: "65-69", 11: "70-74", 12: "75-79", 13: "80+"
            }
            form_inputs["Age"] = st.selectbox(
                "Age Bracket",
                list(age_brackets.keys()),
                index=list(age_brackets.keys()).index(int(patient_data.get("Age", 7))),
                format_func=lambda x: f"{x} ({age_brackets[x]} yrs)",
            )
            form_inputs["AnyHealthcare"] = int(patient_data.get("AnyHealthcare", 1))
            form_inputs["NoDocbcCost"] = int(patient_data.get("NoDocbcCost", 0))
            form_inputs["Education"] = int(patient_data.get("Education", 5))
            form_inputs["Income"] = int(patient_data.get("Income", 6))

    else:
        # Clinical Mode Inputs
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("##### 🩸 Diabetes Assays (Pima Protocol)")
            form_inputs["Glucose"] = st.number_input(
                "Fasting Plasma Glucose (mg/dL)",
                min_value=40.0,
                max_value=300.0,
                value=float(patient_data.get("Glucose", 117.0)),
                step=1.0,
            )
            form_inputs["BloodPressure"] = st.number_input(
                "Diastolic Blood Pressure (mmHg)",
                min_value=30.0,
                max_value=160.0,
                value=float(patient_data.get("BloodPressure", 72.0)),
                step=1.0,
            )
            form_inputs["BMI"] = st.number_input(
                "Body Mass Index (BMI, kg/m²)",
                min_value=15.0,
                max_value=75.0,
                value=float(patient_data.get("BMI", 32.0)),
                step=0.5,
            )
            form_inputs["Pregnancies"] = st.number_input(
                "Number of Pregnancies",
                min_value=0,
                max_value=20,
                value=int(patient_data.get("Pregnancies", 1)),
                step=1,
            )
            form_inputs["Insulin"] = st.number_input(
                "2-Hour Serum Insulin (μU/mL)",
                min_value=10.0,
                max_value=900.0,
                value=float(patient_data.get("Insulin", 125.0)),
                step=5.0,
            )
            form_inputs["SkinThickness"] = st.number_input(
                "Triceps Skinfold (mm)",
                min_value=5.0,
                max_value=100.0,
                value=float(patient_data.get("SkinThickness", 29.0)),
                step=1.0,
            )
            form_inputs["DiabetesPedigreeFunction"] = st.number_input(
                "Diabetes Pedigree Function",
                min_value=0.05,
                max_value=3.0,
                value=float(patient_data.get("DiabetesPedigreeFunction", 0.372)),
                step=0.01,
            )
            form_inputs["Age"] = st.number_input(
                "Age (Years)",
                min_value=21,
                max_value=100,
                value=int(patient_data.get("Age", 35)),
                step=1,
            )
            form_inputs["age"] = form_inputs["Age"]

        with c2:
            st.markdown("##### 🫀 Cardiac Hemodynamics (Cleveland)")
            form_inputs["trestbps"] = st.number_input(
                "Resting Blood Pressure (mmHg)",
                min_value=70.0,
                max_value=240.0,
                value=float(patient_data.get("trestbps", 130.0)),
                step=1.0,
            )
            form_inputs["chol"] = st.number_input(
                "Serum Cholesterol (mg/dL)",
                min_value=100.0,
                max_value=600.0,
                value=float(patient_data.get("chol", 240.0)),
                step=5.0,
            )
            form_inputs["thalach"] = st.number_input(
                "Max Heart Rate Achieved (bpm)",
                min_value=60.0,
                max_value=230.0,
                value=float(patient_data.get("thalach", 150.0)),
                step=1.0,
            )
            form_inputs["oldpeak"] = st.number_input(
                "ST Depression (Exercise vs Rest, mm)",
                min_value=0.0,
                max_value=10.0,
                value=float(patient_data.get("oldpeak", 0.8)),
                step=0.1,
            )
            form_inputs["sex"] = st.selectbox(
                "Biological Sex",
                [0, 1],
                index=[0, 1].index(int(patient_data.get("sex", 1))),
                format_func=lambda x: "Male" if x == 1 else "Female",
            )
            form_inputs["fbs"] = st.selectbox(
                "Fasting Blood Sugar > 120 mg/dL",
                [0, 1],
                index=[0, 1].index(int(patient_data.get("fbs", 0))),
                format_func=lambda x: "Yes" if x == 1 else "No",
            )

        with c3:
            st.markdown("##### 🔬 Angiography & Stress Testing")
            cp_labels = {0: "Typical angina", 1: "Atypical angina", 2: "Non-anginal pain", 3: "Asymptomatic"}
            form_inputs["cp"] = st.selectbox(
                "Chest Pain Type",
                [0, 1, 2, 3],
                index=[0, 1, 2, 3].index(int(patient_data.get("cp", 0))),
                format_func=lambda x: cp_labels[x],
            )
            ecg_labels = {0: "Normal", 1: "ST-T wave abnormality", 2: "LV hypertrophy"}
            form_inputs["restecg"] = st.selectbox(
                "Resting ECG Results",
                [0, 1, 2],
                index=[0, 1, 2].index(int(patient_data.get("restecg", 0))),
                format_func=lambda x: ecg_labels[x],
            )
            form_inputs["exang"] = st.selectbox(
                "Exercise-Induced Angina",
                [0, 1],
                index=[0, 1].index(int(patient_data.get("exang", 0))),
                format_func=lambda x: "Yes" if x == 1 else "No",
            )
            slope_labels = {0: "Upsloping", 1: "Flat", 2: "Downsloping"}
            form_inputs["slope"] = st.selectbox(
                "Peak Exercise ST Slope",
                [0, 1, 2],
                index=[0, 1, 2].index(int(patient_data.get("slope", 1))),
                format_func=lambda x: slope_labels[x],
            )
            form_inputs["ca"] = st.selectbox(
                "Major Vessels via Fluoroscopy (0-3)",
                [0, 1, 2, 3],
                index=[0, 1, 2, 3].index(int(patient_data.get("ca", 0))),
            )
            thal_labels = {0: "Normal", 1: "Fixed defect", 2: "Reversible defect"}
            form_inputs["thal"] = st.selectbox(
                "Thallium Scintigraphy",
                [0, 1, 2],
                index=[0, 1, 2].index(int(patient_data.get("thal", 0))),
                format_func=lambda x: thal_labels[x],
            )

    submit_button = st.form_submit_button("🔍 Compute Risk Triage & Explain Factors", use_container_width=True)

# ==============================================================================
# RISK TRIAGE & EXPLANATION OUTPUT
# ==============================================================================
if submit_button:
    track_key = "track_a" if current_mode == "survey" else "track_b"
    patient_id = "web_patient"

    # Schema Validation
    try:
        clean_inputs = validate_patient_input(form_inputs, track=track_key)
    except Exception as exc:
        st.error(f"❌ **Physiological Validation Error:** {exc}")
        st.stop()

    st.success("✅ Patient measurements validated against physiological schema.")

    # Compute Predictions & Attributions
    with st.spinner("Evaluating calibrated models and calculating SHAP attributions..."):
        try:
            diab_res = predict_and_explain_disease(
                patient_clean=clean_inputs,
                disease="diabetes",
                track=track_key,
                output_dir=PREDICTIONS_DIR,
                patient_id=patient_id,
            )
            heart_res = predict_and_explain_disease(
                patient_clean=clean_inputs,
                disease="heart",
                track=track_key,
                output_dir=PREDICTIONS_DIR,
                patient_id=patient_id,
            )
        except Exception as e:
            st.error(f"❌ Prediction failed: {e}")
            st.stop()

    # Stage 2 Cluster Resemblance
    cluster_name = get_cluster_resemblance(clean_inputs, current_mode)

    st.markdown("---")
    st.markdown(f"### 📊 Stage 2 Cluster Resemblance: **{cluster_name}**")

    # Display Columns for Diabetes and Heart Disease
    col_d, col_h = st.columns(2)

    tier_colors = {
        "Low": ("#2e7d32", "🟢 LOW RISK"),
        "Medium": ("#f57f17", "🟡 MEDIUM RISK"),
        "High": ("#c62828", "🔴 HIGH RISK"),
    }

    # 1. Diabetes Card
    with col_d:
        st.subheader("🩸 Diabetes Risk Triage")
        d_color, d_badge = tier_colors.get(diab_res["risk_tier"], ("#333", diab_res["risk_tier"]))
        st.markdown(
            f"<div style='background-color:{d_color};color:white;padding:10px;border-radius:8px;"
            f"font-size:20px;font-weight:bold;text-align:center;'>{d_badge}</div>",
            unsafe_allow_html=True,
        )
        st.metric(
            label="Calibrated Disease Probability",
            value=f"{diab_res['calibrated_probability']:.1%}",
            help=f"Cutoffs: Low < {diab_res['low_cutoff']:.1%}, High >= {diab_res['high_cutoff']:.1%}",
        )
        st.markdown("**Top Factors Raising Risk (+SHAP):**")
        for inc in diab_res["top_risk_increasers"]:
            st.markdown(f"- **{inc['feature']}** = `{inc['formatted_reading']}` *(+{inc['shap_value']:.2f})*")

        st.markdown("**Top Factors Lowering Risk (-SHAP):**")
        if diab_res["top_risk_decreasers"]:
            for dec in diab_res["top_risk_decreasers"]:
                st.markdown(f"- **{dec['feature']}** = `{dec['formatted_reading']}` *({dec['shap_value']:.2f})*")
        else:
            st.markdown("- *None identified*")

        waterfall_path = PREDICTIONS_DIR / f"{patient_id}_{track_key}_diabetes_waterfall.png"
        if waterfall_path.exists():
            st.image(str(waterfall_path), caption="Patient-Specific Diabetes SHAP Waterfall Attribution", use_container_width=True)

    # 2. Heart Disease Card
    with col_h:
        st.subheader("🫀 Heart Disease Risk Triage")
        h_color, h_badge = tier_colors.get(heart_res["risk_tier"], ("#333", heart_res["risk_tier"]))
        st.markdown(
            f"<div style='background-color:{h_color};color:white;padding:10px;border-radius:8px;"
            f"font-size:20px;font-weight:bold;text-align:center;'>{h_badge}</div>",
            unsafe_allow_html=True,
        )
        st.metric(
            label="Calibrated Disease Probability",
            value=f"{heart_res['calibrated_probability']:.1%}",
            help=f"Cutoffs: Low < {heart_res['low_cutoff']:.1%}, High >= {heart_res['high_cutoff']:.1%}",
        )
        st.markdown("**Top Factors Raising Risk (+SHAP):**")
        for inc in heart_res["top_risk_increasers"]:
            st.markdown(f"- **{inc['feature']}** = `{inc['formatted_reading']}` *(+{inc['shap_value']:.2f})*")

        st.markdown("**Top Factors Lowering Risk (-SHAP):**")
        if heart_res["top_risk_decreasers"]:
            for dec in heart_res["top_risk_decreasers"]:
                st.markdown(f"- **{dec['feature']}** = `{dec['formatted_reading']}` *({dec['shap_value']:.2f})*")
        else:
            st.markdown("- *None identified*")

        waterfall_path_h = PREDICTIONS_DIR / f"{patient_id}_{track_key}_heart_waterfall.png"
        if waterfall_path_h.exists():
            st.image(str(waterfall_path_h), caption="Patient-Specific Heart Disease SHAP Waterfall Attribution", use_container_width=True)
