"""
src/predict_patient.py
Patient-Level Risk Triage & Explainability Command-Line Tool
Project: Chronic Disease Predictor Model (Predictive Risk Identification for Diabetes and Heart Disease)

Features & Modes:
1. Multi-Mode Triage Architecture:
   - --mode survey   (Track A: Population survey biometrics from CDC BRFSS)
   - --mode clinical (Track B: Laboratory measurements from Pima Diabetes & Cleveland Heart)
   - --mode both     (Comprehensive: Dual-track evaluation comparing population vs clinical risk)
2. Graceful Fallback:
   - If a user enters survey data in clinical mode without laboratory readings,
     the system detects missing lab fields and gracefully falls back to survey mode with an explicit notice.
3. Clinical Measurement Formatting:
   - In clinical mode, prints exact readings with medical units:
     e.g., "Glucose = 160.0 mg/dL -> raises risk (+0.18)", "trestbps = 145.0 mmHg -> raises risk (+0.14)"
4. Schema Validation:
   - Validates biometrics against config/feature_schema.yaml.
   - Rejects impossible/out-of-bound physiological readings with clinical rationale.
5. Calibrated Tri-Level Risk Triage & SHAP Waterfall Attribution:
   - Computes calibrated probabilities P(y=1) and assigns Low / Medium / High tiers.
   - Generates local patient waterfall plots (saved to outputs/patient_predictions/).
   - Identifies Top 3 risk increasers and Top 2 risk decreasers.
6. Clinical Safety Disclaimer:
   "These are key contributing factors from a statistical model, not proven causes, and this is not a medical diagnosis."
"""

import argparse
import json
from pathlib import Path
import sys
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import yaml

warnings.filterwarnings("ignore")

# Ensure root directory is on sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.risk_utils import assign_risk_tier, load_yaml, validate_patient_input

MODELS_DIR = BASE_DIR / "models"
MODEL_READY_DIR = BASE_DIR / "data" / "model_ready"
CONFIG_DIR = BASE_DIR / "config"
OUTPUTS_DIR = BASE_DIR / "outputs" / "patient_predictions"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

DISCLAIMER_TEXT = (
    "These are key contributing factors from a statistical model, not proven causes, and this is not a medical diagnosis."
)

def get_bundle_path(track: str, disease: str, model_name: str) -> Path:
    """Resolve model bundle path across chronic_disease_predictor, all_models, and legacy models directories."""
    # 1. Chronic disease predictor package
    p_cdp = MODELS_DIR / "chronic_disease_predictor" / f"{track}_{disease}_{model_name}.joblib"
    if p_cdp.exists():
        return p_cdp
    # 2. Track subfolder in all_models
    subfolder = "track_a_survey" if track == "track_a" else "track_b_clinical"
    p_sub = MODELS_DIR / "all_models" / subfolder / f"{track}_{disease}_{model_name}.joblib"
    if p_sub.exists():
        return p_sub
    # 3. Root models/ fallback
    return MODELS_DIR / f"{track}_{disease}_{model_name}.joblib"

# Human-readable labels and decodings for Survey (Track A) variables
SURVEY_VALUE_LABELS = {
    "HighBP": {0: "No", 1: "Yes"},
    "HighChol": {0: "No", 1: "Yes"},
    "CholCheck": {0: "No", 1: "Yes"},
    "Smoker": {0: "No", 1: "Yes"},
    "Stroke": {0: "No", 1: "Yes"},
    "PhysActivity": {0: "No", 1: "Yes"},
    "Fruits": {0: "No", 1: "Yes"},
    "Veggies": {0: "No", 1: "Yes"},
    "HvyAlcoholConsump": {0: "No", 1: "Yes"},
    "AnyHealthcare": {0: "No", 1: "Yes"},
    "NoDocbcCost": {0: "No", 1: "Yes"},
    "DiffWalk": {0: "No", 1: "Yes"},
    "Sex": {0: "Female", 1: "Male"},
    "GenHlth": {1: "Excellent", 2: "Very Good", 3: "Good", 4: "Fair", 5: "Poor"},
    "Age": {
        1: "18-24", 2: "25-29", 3: "30-34", 4: "35-39", 5: "40-44", 6: "45-49",
        7: "50-54", 8: "55-59", 9: "60-64", 10: "65-69", 11: "70-74", 12: "75-79", 13: "80+"
    },
    "Education": {
        1: "No schooling", 2: "Elementary", 3: "Some high school",
        4: "High school grad", 5: "Some college", 6: "College grad"
    },
    "Income": {
        1: "<$10k", 2: "$10k-$15k", 3: "$15k-$20k", 4: "$20k-$25k",
        5: "$25k-$35k", 6: "$35k-$50k", 7: "$50k-$75k", 8: ">=$75k"
    },
}

# Human-readable labels and units for Clinical (Track B) variables
CLINICAL_VALUE_LABELS = {
    "sex": {0: "Female", 1: "Male"},
    "cp": {0: "Typical angina", 1: "Atypical angina", 2: "Non-anginal pain", 3: "Asymptomatic"},
    "fbs": {0: "<=120 mg/dL", 1: ">120 mg/dL"},
    "restecg": {0: "Normal", 1: "ST-T abnormality", 2: "LV hypertrophy"},
    "exang": {0: "No", 1: "Yes"},
    "slope": {0: "Upsloping", 1: "Flat", 2: "Downsloping"},
    "thal": {0: "Normal", 1: "Fixed defect", 2: "Reversible defect"},
}

CLINICAL_UNITS = {
    "Glucose": "mg/dL",
    "BloodPressure": "mmHg",
    "SkinThickness": "mm",
    "Insulin": "μU/mL",
    "BMI": "kg/m²",
    "trestbps": "mmHg",
    "chol": "mg/dL",
    "thalach": "bpm",
    "oldpeak": "mm",
    "ca": "vessels",
    "Age": "years",
    "age": "years",
}


def format_reading(feature: str, val: float, track: str = "track_a") -> str:
    """Format raw patient reading into clear, professional clinical representation."""
    int_val = int(round(val))

    # 1. Clinical Track formatting
    if track in ["track_b", "clinical"]:
        if feature in CLINICAL_VALUE_LABELS and int_val in CLINICAL_VALUE_LABELS[feature]:
            return CLINICAL_VALUE_LABELS[feature][int_val]
        if feature in CLINICAL_UNITS:
            unit = CLINICAL_UNITS[feature]
            if unit in ["mg/dL", "mmHg", "bpm", "years", "vessels"]:
                return f"{int_val} {unit}"
            return f"{val:.1f} {unit}"
        if feature == "DiabetesPedigreeFunction":
            return f"{val:.3f}"
        if feature == "Pregnancies":
            return f"{int_val}"
        return f"{val:.1f}"

    # 2. Survey Track formatting
    if feature in SURVEY_VALUE_LABELS and int_val in SURVEY_VALUE_LABELS[feature]:
        return SURVEY_VALUE_LABELS[feature][int_val]
    if feature == "BMI":
        return f"{val:.1f} kg/m²"
    if feature in ["PhysHlth", "MentHlth"]:
        return f"{int_val} days"
    return str(val)


def get_cached_background(track: str, disease: str, target_col: str, feature_names: list) -> pd.DataFrame:
    """Load reference background dataset for SHAP explainers."""
    train_file = MODEL_READY_DIR / f"{track}_{disease}_train.csv"
    df_train = pd.read_csv(train_file)
    X_train = df_train.drop(columns=[target_col])[feature_names]
    sample_size = min(1500, len(X_train))
    return X_train.sample(n=sample_size, random_state=42) if len(X_train) > sample_size else X_train.copy()


def predict_and_explain_disease(
    patient_clean: dict,
    disease: str,
    track: str = "track_a",
    output_dir: Path = OUTPUTS_DIR,
    patient_id: str = "patient",
) -> dict:
    """Compute calibrated probability, risk tier, SHAP factor rankings, and waterfall plot for one disease."""
    # Find champion model from champion_models.csv
    champ_csv = BASE_DIR / "outputs" / "tables" / "champion_models.csv"
    champions_df = pd.read_csv(champ_csv)
    champ_row = champions_df[(champions_df["Track"] == track) & (champions_df["Disease"] == disease)]
    if champ_row.empty:
        raise ValueError(f"No champion model found for track={track}, disease={disease}")
    model_name = champ_row.iloc[0]["Champion_Model"]
    bundle_path = get_bundle_path(track, disease, model_name)
    bundle = joblib.load(bundle_path)

    calibrated_model = bundle["calibrated_model"]
    pipeline = bundle["pipeline"]
    feature_names = bundle["feature_names"]
    target_col = bundle["target_col"]

    # Map input patient readings into model feature vector
    input_vector = {}
    for col in feature_names:
        if col in patient_clean:
            input_vector[col] = patient_clean[col]
        elif col.lower() in patient_clean:
            input_vector[col] = patient_clean[col.lower()]
        elif col.capitalize() in patient_clean:
            input_vector[col] = patient_clean[col.capitalize()]
        else:
            raise KeyError(f"Feature '{col}' missing for {track} {disease} prediction.")

    patient_df = pd.DataFrame([input_vector])

    # Predict calibrated probability
    cal_prob = float(calibrated_model.predict_proba(patient_df)[0, 1])

    # Assign Risk Tier
    disease_key = "diabetes" if disease == "diabetes" else "heart_disease"
    tier_info = assign_risk_tier(cal_prob, disease=disease_key, track=track)
    risk_tier = tier_info["risk_tier"]

    # Compute SHAP explanation
    X_bg = get_cached_background(track, disease, target_col, feature_names)

    # Feature transformation through pipeline preprocessor
    named_steps = pipeline.named_steps
    X_bg_trans = X_bg.copy()
    patient_trans = patient_df.copy()

    if "imputer" in named_steps:
        X_bg_trans = named_steps["imputer"].transform(X_bg_trans)
        patient_trans = named_steps["imputer"].transform(patient_trans)
    if "scaler" in named_steps:
        X_bg_trans = named_steps["scaler"].transform(X_bg_trans)
        patient_trans = named_steps["scaler"].transform(patient_trans)

    X_bg_trans = pd.DataFrame(X_bg_trans, columns=feature_names)
    patient_trans = pd.DataFrame(patient_trans, columns=feature_names)

    clf = pipeline.named_steps["clf"]
    if model_name == "logistic_regression":
        explainer = shap.LinearExplainer(clf, X_bg_trans)
        raw_exp = explainer(patient_trans)
        shap_vals = raw_exp.values[0]
        base_val = float(raw_exp.base_values[0])
    else:
        explainer = shap.TreeExplainer(clf)
        raw_exp = explainer(patient_trans)
        if len(raw_exp.values.shape) == 3:
            shap_vals = raw_exp.values[0, :, 1]
            base_val = float(raw_exp.base_values[0, 1] if len(raw_exp.base_values.shape) == 2 else raw_exp.base_values[0])
        else:
            shap_vals = raw_exp.values[0]
            base_val = float(raw_exp.base_values[0])

    # Build Explanation for waterfall plot populated with original unscaled biometrics
    single_exp = shap.Explanation(
        values=shap_vals,
        base_values=base_val,
        data=patient_df.values[0],
        feature_names=feature_names,
    )

    # Save local waterfall plot
    plot_filename = f"{patient_id}_{track}_{disease}_waterfall.png"
    plot_path = output_dir / plot_filename
    plt.figure(figsize=(9, 6))
    shap.plots.waterfall(single_exp, max_display=10, show=False)
    track_title = "Clinical Mode" if track == "track_b" else "Survey Mode"
    plt.title(
        f"Patient {patient_id} ({track_title}): {disease.capitalize()} Risk Attribution\n"
        f"Assigned Tier: {risk_tier} (Calibrated Probability: {cal_prob:.1%})",
        fontsize=11, fontweight="bold", pad=10
    )
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    # Extract Top 3 risk increasers (SHAP > 0) and Top 2 risk decreasers (SHAP < 0)
    contributions = []
    for feat, s_val in zip(feature_names, shap_vals):
        raw_val = input_vector[feat]
        reading_str = format_reading(feat, raw_val, track=track)
        contributions.append({
            "feature": feat,
            "raw_value": raw_val,
            "formatted_reading": reading_str,
            "shap_value": float(s_val),
        })

    increasers = sorted([c for c in contributions if c["shap_value"] > 0], key=lambda x: x["shap_value"], reverse=True)[:3]
    decreasers = sorted([c for c in contributions if c["shap_value"] < 0], key=lambda x: x["shap_value"])[:2]

    return {
        "disease": disease,
        "track": track,
        "champion_model": model_name,
        "calibrated_probability": round(cal_prob, 4),
        "risk_tier": risk_tier,
        "low_cutoff": tier_info["low_threshold"],
        "high_cutoff": tier_info["high_threshold"],
        "top_risk_increasers": increasers,
        "top_risk_decreasers": decreasers,
        "waterfall_plot": str(plot_path.relative_to(BASE_DIR)) if str(plot_path).startswith(str(BASE_DIR)) else str(plot_path),
    }


def print_patient_assessment(patient_id: str, diabetes_res: dict, heart_res: dict, mode_label: str = "Survey Mode (Track A)"):
    """Print standardized clinical triage assessment to console."""
    print("\n" + "=" * 80)
    print(f"CHRONIC DISEASE PREDICTOR MODEL - {mode_label.upper()} (ID: {patient_id})")
    print("=" * 80)

    # 1. Diabetes Section
    d_prob = diabetes_res["calibrated_probability"]
    d_tier = diabetes_res["risk_tier"]
    print(f"\nDiabetes risk: {d_tier.upper()} (probability {d_prob:.2f})")
    for idx, inc in enumerate(diabetes_res["top_risk_increasers"], start=1):
        sign = f"+{inc['shap_value']:.2f}"
        print(f"  {idx}. {inc['feature']} = {inc['formatted_reading']} -> raises risk ({sign})")

    dec_strs = [f"{d['feature']} = {d['formatted_reading']} ({d['shap_value']:.2f})" for d in diabetes_res["top_risk_decreasers"]]
    if dec_strs:
        print(f"  Lowering risk: {', '.join(dec_strs)}")
    else:
        print("  Lowering risk: None identified")

    # 2. Heart Disease Section
    h_prob = heart_res["calibrated_probability"]
    h_tier = heart_res["risk_tier"]
    print(f"\nHeart Disease risk: {h_tier.upper()} (probability {h_prob:.2f})")
    for idx, inc in enumerate(heart_res["top_risk_increasers"], start=1):
        sign = f"+{inc['shap_value']:.2f}"
        print(f"  {idx}. {inc['feature']} = {inc['formatted_reading']} -> raises risk ({sign})")

    dec_strs_h = [f"{d['feature']} = {d['formatted_reading']} ({d['shap_value']:.2f})" for d in heart_res["top_risk_decreasers"]]
    if dec_strs_h:
        print(f"  Lowering risk: {', '.join(dec_strs_h)}")
    else:
        print("  Lowering risk: None identified")

    # 3. Mandatory Medical Disclaimer
    print("\n" + "-" * 80)
    print("CLINICAL SAFETY NOTICE:")
    print(DISCLAIMER_TEXT)
    print("=" * 80 + "\n")


def has_clinical_lab_fields(patient_data: dict) -> bool:
    """Check if patient dictionary provides laboratory measurements required for clinical mode."""
    lab_keys = {"Glucose", "glucose", "trestbps", "chol", "thalach", "oldpeak", "ca", "thal", "Insulin", "insulin"}
    return any(k in patient_data and patient_data[k] is not None for k in lab_keys)


def has_survey_fields(patient_data: dict) -> bool:
    """Check if patient dictionary provides survey / population metrics for survey mode."""
    survey_keys = {"HighBP", "HighChol", "GenHlth", "DiffWalk", "PhysActivity", "Smoker"}
    return any(k in patient_data and patient_data[k] is not None for k in survey_keys)


def interactive_patient_entry(mode: str = "survey") -> dict:
    """Interactive wizard for CLI entry."""
    patient = {}
    if mode == "clinical":
        print("\n" + "=" * 80)
        print("INTERACTIVE PATIENT CLINICAL LAB ENTRY WIZARD (Track B - Clinical)")
        print("=" * 80)
        fields = [
            ("Glucose", "Fasting plasma glucose (mg/dL, range 40-300)", 117.0, float),
            ("BloodPressure", "Diastolic blood pressure (mmHg, range 30-160)", 72.0, float),
            ("BMI", "Body Mass Index (kg/m², range 15-75)", 32.0, float),
            ("Pregnancies", "Number of pregnancies (0-20)", 1, int),
            ("Insulin", "Serum insulin (μU/mL, range 10-900)", 125.0, float),
            ("SkinThickness", "Triceps skinfold thickness (mm, range 5-100)", 29.0, float),
            ("DiabetesPedigreeFunction", "Diabetes pedigree function (range 0.05-3.0)", 0.372, float),
            ("Age", "Chronological age (years, range 21-100)", 35, int),
            ("sex", "Biological sex (0=Female, 1=Male)", 1, int),
            ("cp", "Chest pain type (0=Typical angina, 1=Atypical, 2=Non-anginal, 3=Asymptomatic)", 0, int),
            ("trestbps", "Resting blood pressure (mmHg, range 70-240)", 130.0, float),
            ("chol", "Serum cholesterol (mg/dL, range 100-600)", 240.0, float),
            ("fbs", "Fasting blood sugar > 120 mg/dL (0=No, 1=Yes)", 0, int),
            ("restecg", "Resting ECG (0=Normal, 1=ST-T abnormality, 2=LV hypertrophy)", 0, int),
            ("thalach", "Maximum heart rate achieved (bpm, range 60-230)", 150.0, float),
            ("exang", "Exercise-induced angina (0=No, 1=Yes)", 0, int),
            ("oldpeak", "ST depression induced by exercise (mm, range 0.0-10.0)", 0.8, float),
            ("slope", "ST slope (0=Upsloping, 1=Flat, 2=Downsloping)", 1, int),
            ("ca", "Major vessels colored by fluoroscopy (0-3)", 0, int),
            ("thal", "Thallium stress test (0=Normal, 1=Fixed defect, 2=Reversible defect)", 0, int),
        ]
    else:
        print("\n" + "=" * 80)
        print("INTERACTIVE PATIENT BIOMETRIC ENTRY WIZARD (Track A - CDC BRFSS)")
        print("=" * 80)
        fields = [
            ("HighBP", "High Blood Pressure diagnosis (0=No, 1=Yes)", 0, int),
            ("HighChol", "High Cholesterol diagnosis (0=No, 1=Yes)", 0, int),
            ("CholCheck", "Cholesterol checked within 5 years (0=No, 1=Yes)", 1, int),
            ("BMI", "Body Mass Index (kg/m², range: 12.0 - 90.0)", 27.0, float),
            ("Smoker", "Smoked at least 100 cigarettes in lifetime (0=No, 1=Yes)", 0, int),
            ("Stroke", "History of stroke (0=No, 1=Yes)", 0, int),
            ("PhysActivity", "Physical activity in past 30 days (0=No, 1=Yes)", 1, int),
            ("Fruits", "Consumes fruit daily (0=No, 1=Yes)", 1, int),
            ("Veggies", "Consumes vegetables daily (0=No, 1=Yes)", 1, int),
            ("HvyAlcoholConsump", "Heavy alcohol consumption (0=No, 1=Yes)", 0, int),
            ("AnyHealthcare", "Has health insurance / healthcare coverage (0=No, 1=Yes)", 1, int),
            ("NoDocbcCost", "Could not see doctor due to cost in past year (0=No, 1=Yes)", 0, int),
            ("GenHlth", "General health self-rating (1=Excellent, 2=Very Good, 3=Good, 4=Fair, 5=Poor)", 2, int),
            ("MentHlth", "Days of poor mental health in past 30 days (0 - 30)", 0, int),
            ("PhysHlth", "Days of poor physical health in past 30 days (0 - 30)", 0, int),
            ("DiffWalk", "Difficulty walking or climbing stairs (0=No, 1=Yes)", 0, int),
            ("Sex", "Biological Sex (0=Female, 1=Male)", 0, int),
            ("Age", "Age category bracket (1=18-24 ... 7=50-54 ... 13=80+)", 7, int),
            ("Education", "Education level (1=None ... 4=High School Grad ... 6=College Grad)", 5, int),
            ("Income", "Income level (1=<$10k ... 5=$25k-$35k ... 8=>=$75k)", 6, int),
        ]

    for key, prompt, default, val_type in fields:
        while True:
            raw = input(f"{prompt} [{default}]: ").strip()
            if not raw:
                patient[key] = default
                break
            try:
                patient[key] = val_type(raw)
                break
            except ValueError:
                print(f"  [Error] Invalid format. Please enter a valid {val_type.__name__}.")
    return patient


def main():
    parser = argparse.ArgumentParser(
        description="Chronic Disease Predictor Model: Predict patient risk tiers and explain contributing factors for Diabetes and Heart Disease.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input", "-i", type=str, help="Path to input patient JSON file.")
    parser.add_argument("--interactive", action="store_true", help="Launch interactive step-by-step CLI entry wizard.")
    parser.add_argument("--output-dir", "-o", type=str, default=str(OUTPUTS_DIR), help="Directory to save prediction JSON and waterfall PNGs.")
    parser.add_argument("--mode", "-m", type=str, default="survey", choices=["survey", "clinical", "both"], help="Evaluation mode: survey (Track A), clinical (Track B), or both.")
    parser.add_argument("--track", type=str, default=None, choices=["track_a", "track_b"], help="Legacy track flag (aliases to --mode).")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Resolve mode and track alias
    mode = args.mode
    if args.track == "track_a":
        mode = "survey"
    elif args.track == "track_b":
        mode = "clinical"

    if args.interactive:
        patient_raw = interactive_patient_entry(mode=mode)
        patient_id = "interactive_patient"
    elif args.input:
        in_path = Path(args.input)
        if not in_path.exists():
            print(f"[Error] Patient JSON file not found: {in_path}", file=sys.stderr)
            sys.exit(1)
        with open(in_path, "r") as f:
            patient_raw = json.load(f)
        patient_id = patient_raw.get("patient_id", in_path.stem)
    else:
        print("[Error] Must specify either --input <file.json> or --interactive.", file=sys.stderr)
        parser.print_help()
        sys.exit(1)

    # Check for graceful fallback if clinical mode is requested but no lab fields exist
    actual_modes = []
    if mode == "survey":
        actual_modes = ["survey"]
    elif mode == "clinical":
        if not has_clinical_lab_fields(patient_raw) and has_survey_fields(patient_raw):
            print("\n" + "!" * 80)
            print("[Notice] Clinical laboratory measurements (e.g. Glucose, Cholesterol, resting BP) are absent.")
            print("[Graceful Fallback] Executing Survey Mode (Track A - CDC BRFSS Population Risk) instead.")
            print("!" * 80 + "\n")
            actual_modes = ["survey"]
        else:
            actual_modes = ["clinical"]
    elif mode == "both":
        actual_modes = ["survey"]
        if has_clinical_lab_fields(patient_raw):
            actual_modes.append("clinical")
        else:
            print("\n[Notice] No laboratory measurements found in patient input; omitting clinical track in dual-mode evaluation.\n")

    overall_output = {
        "patient_id": patient_id,
        "input_features": patient_raw,
        "modes_evaluated": actual_modes,
        "predictions": {},
        "disclaimer": DISCLAIMER_TEXT,
    }

    for eval_mode in actual_modes:
        track_name = "track_a" if eval_mode == "survey" else "track_b"
        mode_label = "Survey Mode (Track A - CDC BRFSS)" if eval_mode == "survey" else "Clinical Mode (Track B - Lab Tests)"

        # 1. Schema Validation
        try:
            patient_clean = validate_patient_input(patient_raw, track=track_name)
        except Exception as e:
            print(f"\n[Validation Error] Patient input rejected for {eval_mode} mode:\n  {e}", file=sys.stderr)
            sys.exit(1)

        print(f"\nEvaluating Patient: {patient_id} across {mode_label}...")

        # 2. Predict & Explain Diabetes
        diabetes_res = predict_and_explain_disease(
            patient_clean=patient_clean,
            disease="diabetes",
            track=track_name,
            output_dir=out_dir,
            patient_id=f"{patient_id}_{eval_mode}",
        )

        # 3. Predict & Explain Heart Disease
        heart_res = predict_and_explain_disease(
            patient_clean=patient_clean,
            disease="heart",
            track=track_name,
            output_dir=out_dir,
            patient_id=f"{patient_id}_{eval_mode}",
        )

        # 4. Standard Console Output
        print_patient_assessment(patient_id, diabetes_res, heart_res, mode_label=mode_label)

        overall_output["predictions"][eval_mode] = {
            "diabetes": diabetes_res,
            "heart_disease": heart_res,
        }

    # 5. Export JSON Output
    json_path = out_dir / f"{patient_id}_prediction.json"
    with open(json_path, "w") as f:
        json.dump(overall_output, f, indent=2)

    print(f"[OK] Full prediction record saved: {json_path.relative_to(BASE_DIR)}")


if __name__ == "__main__":
    main()
