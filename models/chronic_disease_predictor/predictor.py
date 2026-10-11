"""
models/chronic_disease_predictor/predictor.py
Chronic Disease Predictor Model: Core In-Memory Predictor & Explainability Engine

Exposes:
    ChronicDiseasePredictor: Production inference class for Type 2 Diabetes and
    Coronary Heart Disease risk stratification across population survey and clinical laboratory modes.
"""

from pathlib import Path
import json
import warnings
import yaml
import joblib
import pandas as pd
import numpy as np
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Root path resolution
PACKAGE_DIR = Path(__file__).resolve().parent
MODELS_ROOT = PACKAGE_DIR.parent
BASE_DIR = MODELS_ROOT.parent

DISCLAIMER_TEXT = "These are key contributing factors from a statistical model, not proven causes, and this is not a medical diagnosis."

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
    "GenHlth": {1: "Excellent", 2: "Very Good", 3: "Good", 4: "Fair", 5: "Poor"},
    "DiffWalk": {0: "No", 1: "Yes"},
    "Sex": {0: "Female", 1: "Male"},
    "Age": {
        1: "18-24", 2: "25-29", 3: "30-34", 4: "35-39", 5: "40-44",
        6: "45-49", 7: "50-54", 8: "55-59", 9: "60-64", 10: "65-69",
        11: "70-74", 12: "75-79", 13: "80+"
    },
    "Education": {1: "Never attended school", 2: "Elementary", 3: "Some high school", 4: "High school graduate", 5: "Some college", 6: "College graduate"},
    "Income": {1: "<$10,000", 2: "$10,000-$14,999", 3: "$15,000-$19,999", 4: "$20,000-$24,999", 5: "$25,000-$34,999", 6: "$35,000-$49,999", 7: "$50,000-$74,999", 8: ">=$75,000"}
}

CLINICAL_VALUE_LABELS = {
    "sex": {0: "Female", 1: "Male"},
    "cp": {0: "Typical angina", 1: "Atypical angina", 2: "Non-anginal pain", 3: "Asymptomatic"},
    "fbs": {0: "<=120 mg/dL", 1: ">120 mg/dL"},
    "restecg": {0: "Normal", 1: "ST-T wave abnormality", 2: "Left ventricular hypertrophy"},
    "exang": {0: "No", 1: "Yes"},
    "slope": {0: "Upsloping", 1: "Flat", 2: "Downsloping"},
    "thal": {0: "Normal", 1: "Fixed defect", 2: "Reversible defect"},
}

CLINICAL_UNITS = {
    "Glucose": "mg/dL",
    "BloodPressure": "mmHg",
    "SkinThickness": "mm",
    "Insulin": "uU/mL",
    "BMI": "kg/m²",
    "Age": "years",
    "age": "years",
    "trestbps": "mmHg",
    "chol": "mg/dL",
    "thalach": "bpm",
    "oldpeak": "mm ST depression",
    "ca": "vessels",
}


def format_reading(feature: str, val, track: str = "track_a") -> str:
    """Format numeric patient feature value with units and clinical text labels."""
    try:
        int_val = int(round(float(val)))
    except (ValueError, TypeError):
        return str(val)

    if track in ["track_b", "clinical"]:
        if feature in CLINICAL_VALUE_LABELS and int_val in CLINICAL_VALUE_LABELS[feature]:
            return CLINICAL_VALUE_LABELS[feature][int_val]
        if feature in CLINICAL_UNITS:
            unit = CLINICAL_UNITS[feature]
            if unit in ["mg/dL", "mmHg", "bpm", "years", "vessels"]:
                return f"{int_val} {unit}"
            return f"{float(val):.1f} {unit}"
        if feature == "DiabetesPedigreeFunction":
            return f"{float(val):.3f}"
        if feature == "Pregnancies":
            return f"{int_val}"
        return f"{float(val):.1f}"

    if feature in SURVEY_VALUE_LABELS and int_val in SURVEY_VALUE_LABELS[feature]:
        return SURVEY_VALUE_LABELS[feature][int_val]
    if feature == "BMI":
        return f"{float(val):.1f} kg/m²"
    if feature in ["PhysHlth", "MentHlth"]:
        return f"{int_val} days"
    return str(val)


class ChronicDiseasePredictor:
    """
    Main working model packaging: Chronic Disease Predictor Model.

    Supports:
    - Mode 'survey' (Track A: CDC BRFSS population screening)
    - Mode 'clinical' (Track B: Continuous clinical laboratory measurements)
    - Mode 'both' (Comprehensive dual evaluation)
    """

    def __init__(self, model_dir: Path = None):
        self.model_dir = Path(model_dir) if model_dir else PACKAGE_DIR
        self.all_models_dir = MODELS_ROOT / "all_models"
        self.config_dir = BASE_DIR / "config"
        self.model_ready_dir = BASE_DIR / "data" / "model_ready"

        # Load risk tiers
        risk_tiers_file = self.model_dir / "risk_tiers.yaml"
        if not risk_tiers_file.exists():
            risk_tiers_file = self.config_dir / "risk_tiers.yaml"
        with open(risk_tiers_file, "r") as f:
            self.risk_tiers_config = yaml.safe_load(f)
        self.risk_tiers = self.risk_tiers_config

        # Load feature schema
        schema_file = self.model_dir / "feature_schema.yaml"
        if not schema_file.exists():
            schema_file = self.config_dir / "feature_schema.yaml"
        with open(schema_file, "r") as f:
            self.feature_schema = yaml.safe_load(f)

        champions_csv = BASE_DIR / "outputs" / "tables" / "champion_models.csv"
        self._champions_df = pd.read_csv(champions_csv) if champions_csv.exists() else None

        self._bundles = {}
        self._backgrounds = {}

    def _get_bundle(self, track: str, disease: str) -> dict:
        key = (track, disease)
        if key not in self._bundles:
            filename = f"{track}_{disease}_logistic_regression.joblib"
            # Primary lookup inside chronic_disease_predictor/
            bundle_path = self.model_dir / filename
            if not bundle_path.exists():
                subfolder = "track_a_survey" if track == "track_a" else "track_b_clinical"
                bundle_path = self.all_models_dir / subfolder / filename
            if not bundle_path.exists():
                raise FileNotFoundError(f"Model bundle not found: {bundle_path}")
            self._bundles[key] = joblib.load(bundle_path)
        return self._bundles[key]

    def _get_background(self, track: str, disease: str, target_col: str, feature_names: list) -> pd.DataFrame:
        key = (track, disease)
        if key not in self._backgrounds:
            train_file = self.model_ready_dir / f"{track}_{disease}_train.csv"
            if train_file.exists():
                df_train = pd.read_csv(train_file)
                X_train = df_train.drop(columns=[target_col])[feature_names]
                sample_size = min(1500, len(X_train))
                self._backgrounds[key] = X_train.sample(n=sample_size, random_state=42) if len(X_train) > sample_size else X_train.copy()
            else:
                self._backgrounds[key] = pd.DataFrame(columns=feature_names)
        return self._backgrounds[key]

    def assign_tier(self, prob: float, disease: str, track: str = "track_a") -> dict:
        """Assign calibrated probability to Low / Medium / High tier."""
        tiers_block = self.risk_tiers_config.get("risk_tiers", {})
        track_key = "track_a" if track in ["track_a", "survey"] else "track_b"
        dis_key = "diabetes" if disease == "diabetes" else "heart_disease"
        cfg = tiers_block.get(track_key, {}).get(dis_key, {})

        low_th = float(cfg.get("low_threshold", 0.05))
        high_th = float(cfg.get("high_threshold", 0.20))

        if prob <= low_th:
            tier = "Low"
        elif prob <= high_th:
            tier = "Medium"
        else:
            tier = "High"

        return {
            "risk_tier": tier,
            "low_threshold": low_th,
            "high_threshold": high_th,
            "description": cfg.get("tier_descriptions", {}).get(tier.lower(), "")
        }

    def _predict_single_disease(
        self,
        patient_dict: dict,
        disease: str,
        track: str = "track_a",
        output_dir: Path = None,
        patient_id: str = "patient"
    ) -> dict:
        bundle = self._get_bundle(track, disease)
        calibrated_model = bundle["calibrated_model"]
        pipeline = bundle["pipeline"]
        feature_names = bundle["feature_names"]
        target_col = bundle["target_col"]
        model_name = bundle.get("model_name", "logistic_regression")

        # Map input features
        input_vector = {}
        for col in feature_names:
            if col in patient_dict:
                input_vector[col] = patient_dict[col]
            elif col.lower() in patient_dict:
                input_vector[col] = patient_dict[col.lower()]
            elif col.capitalize() in patient_dict:
                input_vector[col] = patient_dict[col.capitalize()]
            else:
                # Age cross-mapping if available
                if col == "Age" and "age" in patient_dict:
                    val = patient_dict["age"]
                    if val > 14: # continuous years -> ordinal bracket
                        from src.risk_utils import continuous_age_to_bracket
                        input_vector[col] = continuous_age_to_bracket(val)
                    else:
                        input_vector[col] = val
                elif col == "age" and "Age" in patient_dict:
                    val = patient_dict["Age"]
                    if val <= 13: # ordinal bracket -> continuous approximate midpoint
                        from src.risk_utils import bracket_to_continuous_age
                        input_vector[col] = bracket_to_continuous_age(val)
                else:
                    schema_track = "track_a" if track == "track_a" else ("track_b_diabetes" if disease == "diabetes" else "track_b_heart")
                    feats_cfg = self.feature_schema.get(schema_track, {}).get("features", {})
                    if col in feats_cfg and "default" in feats_cfg[col]:
                        input_vector[col] = feats_cfg[col]["default"]
                    else:
                        raise KeyError(f"Feature '{col}' missing for {track} {disease} prediction.")

        patient_df = pd.DataFrame([input_vector])

        # Calibrated probability
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cal_prob = float(calibrated_model.predict_proba(patient_df)[0, 1])

        tier_info = self.assign_tier(cal_prob, disease=disease, track=track)
        risk_tier = tier_info["risk_tier"]

        # Local SHAP attributions
        X_bg = self._get_background(track, disease, target_col, feature_names)
        named_steps = pipeline.named_steps
        X_bg_trans = X_bg.copy()
        patient_trans = patient_df.copy()

        if "imputer" in named_steps and len(X_bg_trans) > 0:
            X_bg_trans = named_steps["imputer"].transform(X_bg_trans)
            patient_trans = named_steps["imputer"].transform(patient_trans)
        if "scaler" in named_steps and len(X_bg_trans) > 0:
            X_bg_trans = named_steps["scaler"].transform(X_bg_trans)
            patient_trans = named_steps["scaler"].transform(patient_trans)

        X_bg_trans = pd.DataFrame(X_bg_trans, columns=feature_names)
        patient_trans = pd.DataFrame(patient_trans, columns=feature_names)

        clf = pipeline.named_steps["clf"]
        if len(X_bg_trans) > 0:
            explainer = shap.LinearExplainer(clf, X_bg_trans)
            raw_exp = explainer(patient_trans)
            shap_vals = raw_exp.values[0]
            base_val = float(raw_exp.base_values[0])
        else:
            shap_vals = np.zeros(len(feature_names))
            base_val = 0.0

        # Save waterfall if requested
        plot_path_str = None
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            plot_path = output_dir / f"{patient_id}_{track}_{disease}_waterfall.png"
            single_exp = shap.Explanation(
                values=shap_vals,
                base_values=base_val,
                data=patient_df.values[0],
                feature_names=feature_names,
            )
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
            plot_path_str = str(plot_path)

        # Top factors
        contributions = []
        for feat, s_val in zip(feature_names, shap_vals):
            raw_val = input_vector[feat]
            reading_str = format_reading(feat, raw_val, track=track)
            contributions.append({
                "feature": feat,
                "raw_value": raw_val,
                "reading": reading_str,
                "formatted_reading": reading_str,
                "shap_value": float(s_val),
            })

        increasers = sorted([c for c in contributions if c["shap_value"] > 0], key=lambda x: x["shap_value"], reverse=True)[:3]
        decreasers = sorted([c for c in contributions if c["shap_value"] < 0], key=lambda x: x["shap_value"])[:2]

        return {
            "disease": disease,
            "track": track,
            "champion_model": model_name,
            "probability": round(cal_prob, 4),
            "calibrated_probability": round(cal_prob, 4),
            "risk_tier": risk_tier,
            "low_cutoff": tier_info["low_threshold"],
            "high_cutoff": tier_info["high_threshold"],
            "top_risk_increasers": increasers,
            "top_risk_decreasers": decreasers,
            "waterfall_plot": plot_path_str,
        }

    def predict(
        self,
        patient_dict: dict,
        mode: str = "survey",
        output_dir: Path = None,
        patient_id: str = None
    ) -> dict:
        """
        Main working model prediction endpoint.

        Parameters
        ----------
        patient_dict : dict
            Patient biometric, demographic, and clinical readings.
        mode : str
            Triage operational mode: 'survey', 'clinical', or 'both'.
        output_dir : Path, optional
            Directory to save local SHAP waterfall diagrams.
        patient_id : str, optional
            Identifier for logging and plot naming.

        Returns
        -------
        dict
            Dictionary containing calibrated disease probabilities, risk tiers,
            top 3 risk increasers, top 2 risk decreasers, and medical disclaimer.
        """
        pid = patient_id or patient_dict.get("patient_id", "patient")
        clean_patient = dict(patient_dict)
        mode = mode.lower()

        out = {
            "model_name": "Chronic Disease Predictor Model",
            "patient_id": pid,
            "modes_evaluated": [mode],
            "predictions": {},
            "disclaimer": DISCLAIMER_TEXT
        }

        if mode in ["survey", "both"]:
            try:
                from src.risk_utils import validate_patient_input
                survey_data = validate_patient_input(clean_patient, track="survey")
            except Exception:
                survey_data = clean_patient
            out["predictions"]["survey"] = {
                "diabetes": self._predict_single_disease(survey_data, "diabetes", track="track_a", output_dir=output_dir, patient_id=f"{pid}_survey"),
                "heart_disease": self._predict_single_disease(survey_data, "heart", track="track_a", output_dir=output_dir, patient_id=f"{pid}_survey"),
            }

        if mode in ["clinical", "both"]:
            try:
                from src.risk_utils import validate_patient_input
                clin_data = validate_patient_input(clean_patient, track="clinical")
            except Exception:
                clin_data = clean_patient
            out["predictions"]["clinical"] = {
                "diabetes": self._predict_single_disease(clin_data, "diabetes", track="track_b", output_dir=output_dir, patient_id=f"{pid}_clinical"),
                "heart_disease": self._predict_single_disease(clin_data, "heart", track="track_b", output_dir=output_dir, patient_id=f"{pid}_clinical"),
            }

        return out
