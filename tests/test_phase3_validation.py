"""
tests/test_phase3_validation.py
Automated Verification Suite for Clinical Mode, Validation Artifacts, and Robustness
Project: Predictive Risk Identification for Diabetes and Heart Disease

Test Cases:
1. Clinical mode execution on prototypical Low, Medium, and High clinical patient profiles.
2. Graceful fallback when clinical mode receives input without laboratory fields.
3. Dual-track (--mode both) execution on comprehensive patient profiles.
4. Schema validation: rejects impossible/out-of-bounds biometrics with informative ValueError.
5. Monotonicity validation across Low < Medium < High tiers for all 4 champions.
6. Existence and schema verification of all Phase 1 & 2 validation tables and reports.
7. Strict data leakage check across model-ready sets and model bundles.
"""

import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pytest

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
MODEL_READY_DIR = BASE_DIR / "data" / "model_ready"
TABLES_DIR = BASE_DIR / "outputs" / "tables"
EXAMPLES_DIR = BASE_DIR / "examples"

from src.predict_patient import (
    get_bundle_path,
    has_clinical_lab_fields,
    predict_and_explain_disease,
)
from src.risk_utils import (
    assign_risk_tier,
    validate_patient_input,
    validate_risk_tier_monotonicity,
)


def test_clinical_mode_runs_successfully(tmp_path):
    """Verify that clinical mode runs cleanly on all clinical example patients."""
    test_files = [
        ("clinical_patient_low.json", "Low"),
        ("clinical_patient_medium.json", "Medium"),
        ("clinical_patient_high.json", "High"),
    ]

    for fname, expected_tier in test_files:
        json_path = EXAMPLES_DIR / fname
        assert json_path.exists(), f"Missing example file: {fname}"
        with open(json_path, "r") as f:
            raw_patient = json.load(f)

        assert has_clinical_lab_fields(raw_patient) is True

        clean_patient = validate_patient_input(raw_patient, track="clinical")

        # Evaluate Clinical Diabetes
        d_res = predict_and_explain_disease(
            patient_clean=clean_patient,
            disease="diabetes",
            track="track_b",
            output_dir=tmp_path,
            patient_id=raw_patient["patient_id"],
        )
        assert 0.0 <= d_res["calibrated_probability"] <= 1.0
        assert d_res["risk_tier"] in ["Low", "Medium", "High"]
        assert len(d_res["top_risk_increasers"]) <= 3
        # Ensure readings contain clinical units
        for inc in d_res["top_risk_increasers"]:
            assert "formatted_reading" in inc
            assert len(inc["formatted_reading"]) > 0

        # Evaluate Clinical Heart Disease
        h_res = predict_and_explain_disease(
            patient_clean=clean_patient,
            disease="heart",
            track="track_b",
            output_dir=tmp_path,
            patient_id=raw_patient["patient_id"],
        )
        assert 0.0 <= h_res["calibrated_probability"] <= 1.0
        assert h_res["risk_tier"] in ["Low", "Medium", "High"]


def test_graceful_fallback_detection_when_labs_missing():
    """Verify that patient dictionaries without labs are correctly detected for graceful fallback."""
    survey_file = EXAMPLES_DIR / "low_risk_patient.json"
    with open(survey_file, "r") as f:
        survey_patient = json.load(f)

    # Missing lab fields
    assert has_clinical_lab_fields(survey_patient) is False

    # Clinical profile has lab fields
    clinical_file = EXAMPLES_DIR / "clinical_patient_high.json"
    with open(clinical_file, "r") as f:
        clinical_patient = json.load(f)
    assert has_clinical_lab_fields(clinical_patient) is True


def test_dual_mode_comprehensive_patient_runs(tmp_path):
    """Verify that comprehensive patient profile evaluates across both survey and clinical tracks."""
    comp_file = EXAMPLES_DIR / "comprehensive_patient_both.json"
    assert comp_file.exists()
    with open(comp_file, "r") as f:
        comp_patient = json.load(f)

    # Validate survey track
    clean_survey = validate_patient_input(comp_patient, track="survey")
    d_survey = predict_and_explain_disease(clean_survey, "diabetes", track="track_a", output_dir=tmp_path, patient_id="comp_pt")
    assert d_survey["risk_tier"] in ["Low", "Medium", "High"]

    # Validate clinical track
    clean_clin = validate_patient_input(comp_patient, track="clinical")
    d_clin = predict_and_explain_disease(clean_clin, "diabetes", track="track_b", output_dir=tmp_path, patient_id="comp_pt")
    assert d_clin["risk_tier"] in ["Low", "Medium", "High"]


def test_invalid_physiological_inputs_rejected_with_informative_error():
    """Verify that impossible biometrics are caught and rejected by schema validation."""
    # 1. Negative Glucose
    with pytest.raises(ValueError, match="physiologically valid minimum"):
        validate_patient_input({"Glucose": 10.0}, track="track_b_diabetes")

    # 2. Extreme BMI exceeding maximum
    with pytest.raises(ValueError, match="physiologically valid maximum"):
        validate_patient_input({"BMI": 120.0}, track="track_a")

    # 3. Invalid binary value
    with pytest.raises(ValueError, match="allowed binary set"):
        validate_patient_input({"HighBP": 5}, track="track_a")

    # 4. Out of bounds ordinal categorical value
    with pytest.raises(ValueError, match="outside permitted category range"):
        validate_patient_input({"GenHlth": 9}, track="track_a")


def test_risk_tier_monotonicity_across_all_four_champions():
    """Verify that observed test set disease rates strictly follow Low < Medium < High for all champions."""
    tasks = [
        ("track_a", "diabetes", "track_a_diabetes_test.csv", "Diabetes_binary"),
        ("track_a", "heart", "track_a_heart_test.csv", "HeartDiseaseorAttack"),
        ("track_b", "diabetes", "track_b_diabetes_test.csv", "Outcome"),
        ("track_b", "heart", "track_b_heart_test.csv", "condition"),
    ]

    for track, disease, test_csv, target_col in tasks:
        champ_csv = TABLES_DIR / "champion_models.csv"
        champs_df = pd.read_csv(champ_csv)
        row = champs_df[(champs_df["Track"] == track) & (champs_df["Disease"] == disease)]
        model_name = row.iloc[0]["Champion_Model"]

        bundle = joblib.load(get_bundle_path(track, disease, model_name))
        calibrated_model = bundle["calibrated_model"]
        feature_names = bundle["feature_names"]

        df_test = pd.read_csv(MODEL_READY_DIR / test_csv)
        X_test = df_test[feature_names]
        y_test = df_test[target_col].astype(int).values

        probs = calibrated_model.predict_proba(X_test)[:, 1]

        # Fetch configured thresholds
        disease_key = "diabetes" if disease == "diabetes" else "heart_disease"
        tier_info = assign_risk_tier(0.5, disease=disease_key, track=track)
        low_t = tier_info["low_threshold"]
        high_t = tier_info["high_threshold"]

        val = validate_risk_tier_monotonicity(y_test, probs, low_t, high_t)
        rates = val["observed_rates"]
        assert rates["Low"] < rates["Medium"] < rates["High"], (
            f"Monotonicity failed for {track} {disease}: Low={rates['Low']}, Med={rates['Medium']}, High={rates['High']}"
        )


def test_all_validation_tables_and_reports_exist():
    """Verify that all Phase 1 and Phase 2 statistical tables and summary reports exist and are non-empty."""
    required_files = [
        TABLES_DIR / "validation_confidence_intervals.csv",
        TABLES_DIR / "validation_repeated_cv.csv",
        TABLES_DIR / "subgroup_performance_diabetes.csv",
        TABLES_DIR / "subgroup_performance_heart.csv",
        TABLES_DIR / "external_validation_2022_heart.csv",
        TABLES_DIR / "baseline_comparison.csv",
        TABLES_DIR / "explanation_stability.csv",
        TABLES_DIR / "validation_summary.md",
        TABLES_DIR / "shap_feature_importance_track_b.csv",
        TABLES_DIR / "track_b_cv_metrics_with_ci.csv",
    ]

    for fpath in required_files:
        assert fpath.exists(), f"Missing required validation artifact: {fpath.name}"
        assert fpath.stat().st_size > 0, f"File is empty: {fpath.name}"


def test_strict_zero_leakage_across_all_model_ready_data():
    """Verify that high_glucose is never present and no train/test data overlaps exist."""
    # 1. No high_glucose in any model-ready dataset
    for csv_file in MODEL_READY_DIR.glob("*.csv"):
        df = pd.read_csv(csv_file, nrows=5)
        assert "high_glucose" not in df.columns, f"Leaky column high_glucose found in {csv_file.name}!"

    # 2. Train and test sets have 0 index overlap
    split_file = MODEL_READY_DIR / "split_indices.json"
    assert split_file.exists()
    with open(split_file, "r") as f:
        split_dict = json.load(f)

    for split_key, indices in split_dict.items():
        train_idx = set(indices["train_indices"])
        test_idx = set(indices["test_indices"])
        overlap = train_idx.intersection(test_idx)
        assert len(overlap) == 0, f"Train/test index overlap detected in {split_key}: {len(overlap)} rows!"
