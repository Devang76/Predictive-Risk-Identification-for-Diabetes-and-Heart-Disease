"""
tests/test_final_system.py
Stage 3 End-to-End System & Safeguard Verification Suite
Project: Predictive Risk Identification for Diabetes and Heart Disease

Tests:
1. All 12 model bundles load cleanly and contain calibrated models and pipelines.
2. Leakage safeguard: Neither high_glucose nor target variables exist in model feature names.
3. predict_patient.py end-to-end functionality on all 3 sample patient JSON profiles.
4. Risk tier calibration monotonicity: Low < Medium < High observed test disease prevalence.
5. Clinical safety: >= 90% true positive capture in Medium + High tiers.
"""

import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pytest
import yaml

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
MODEL_READY_DIR = BASE_DIR / "data" / "model_ready"
CONFIG_DIR = BASE_DIR / "config"
EXAMPLES_DIR = BASE_DIR / "examples"

from src.predict_patient import get_bundle_path, predict_and_explain_disease
from src.risk_utils import assign_risk_tier, validate_patient_input, validate_risk_tier_monotonicity


def test_all_12_model_files_load():
    """Verify that all 12 trained and tuned model bundles exist and load with expected keys."""
    tracks = ["track_a", "track_b"]
    diseases = ["diabetes", "heart"]
    model_names = ["logistic_regression", "decision_tree", "random_forest"]

    for track in tracks:
        subfolder = "track_a_survey" if track == "track_a" else "track_b_clinical"
        for disease in diseases:
            for m in model_names:
                bundle_path = MODELS_DIR / "all_models" / subfolder / f"{track}_{disease}_{m}.joblib"
                if not bundle_path.exists():
                    bundle_path = MODELS_DIR / f"{track}_{disease}_{m}.joblib"
                assert bundle_path.exists(), f"Missing model bundle: {bundle_path.name}"
                bundle = joblib.load(bundle_path)
                assert "pipeline" in bundle, f"'pipeline' missing in {bundle_path.name}"
                assert "calibrated_model" in bundle, f"'calibrated_model' missing in {bundle_path.name}"
                assert "cv_roc_auc" in bundle, f"'cv_roc_auc' missing in {bundle_path.name}"
                assert "feature_names" in bundle, f"'feature_names' missing in {bundle_path.name}"
                assert len(bundle["feature_names"]) > 0


def test_no_leakage_columns_in_model_features():
    """Verify that high_glucose and targets are strictly absent from all model feature sets."""
    banned = ["high_glucose", "Diabetes_binary", "HeartDiseaseorAttack", "Outcome", "condition"]
    model_files = list((MODELS_DIR / "all_models").rglob("*.joblib"))
    if not model_files:
        model_files = list(MODELS_DIR.glob("*.joblib"))
    assert len(model_files) == 12, f"Expected 12 models in all_models/, found {len(model_files)}"
    for bundle_path in model_files:
        bundle = joblib.load(bundle_path)
        feature_names = bundle["feature_names"]
        for b in banned:
            assert b not in feature_names, f"Leaky column '{b}' found in {bundle_path.name} feature names!"


def test_predict_patient_runs_on_all_three_example_patients(tmp_path):
    """Verify that predict_and_explain_disease runs cleanly on Low, Medium, and High example patients."""
    patient_files = [
        ("low_risk_patient.json", "Low"),
        ("medium_risk_patient.json", "Medium"),
        ("high_risk_patient.json", "High"),
    ]

    for fname, expected_tier in patient_files:
        json_path = EXAMPLES_DIR / fname
        assert json_path.exists(), f"Missing example file: {fname}"
        with open(json_path, "r") as f:
            raw_patient = json.load(f)

        clean_patient = validate_patient_input(raw_patient, track="track_a")

        # Test Diabetes
        d_res = predict_and_explain_disease(
            patient_clean=clean_patient,
            disease="diabetes",
            track="track_a",
            output_dir=tmp_path,
            patient_id=raw_patient["patient_id"],
        )
        assert 0.0 <= d_res["calibrated_probability"] <= 1.0
        assert d_res["risk_tier"] == expected_tier, f"Expected {expected_tier} tier for diabetes in {fname}, got {d_res['risk_tier']}"
        assert len(d_res["top_risk_increasers"]) <= 3
        assert len(d_res["top_risk_decreasers"]) <= 2

        # Test Heart Disease
        h_res = predict_and_explain_disease(
            patient_clean=clean_patient,
            disease="heart",
            track="track_a",
            output_dir=tmp_path,
            patient_id=raw_patient["patient_id"],
        )
        assert 0.0 <= h_res["calibrated_probability"] <= 1.0
        assert h_res["risk_tier"] == expected_tier, f"Expected {expected_tier} tier for heart in {fname}, got {h_res['risk_tier']}"
        assert len(h_res["top_risk_increasers"]) <= 3
        assert len(h_res["top_risk_decreasers"]) <= 2


def test_risk_tier_monotonicity_on_untouched_test_sets():
    """Verify that risk tiers on untouched test sets strictly satisfy Low < Medium < High observed prevalence."""
    champ_csv = BASE_DIR / "outputs" / "tables" / "champion_models.csv"
    assert champ_csv.exists(), "champion_models.csv missing"
    champs_df = pd.read_csv(champ_csv)

    with open(CONFIG_DIR / "risk_tiers.yaml", "r") as f:
        tiers_cfg = yaml.safe_load(f)

    task_map = {
        ("track_a", "diabetes"): ("track_a_diabetes_test.csv", "Diabetes_binary", "diabetes"),
        ("track_a", "heart"): ("track_a_heart_test.csv", "HeartDiseaseorAttack", "heart_disease"),
    }

    for (track, disease), (test_csv, target_col, disease_key) in task_map.items():
        champ_model = champs_df[(champs_df["Track"] == track) & (champs_df["Disease"] == disease)].iloc[0]["Champion_Model"]
        bundle = joblib.load(get_bundle_path(track, disease, champ_model))
        calibrated_model = bundle["calibrated_model"]

        df_test = pd.read_csv(MODEL_READY_DIR / test_csv)
        X_test = df_test.drop(columns=[target_col])
        y_test = df_test[target_col].values
        test_probs = calibrated_model.predict_proba(X_test)[:, 1]

        tau_low = tiers_cfg["tracks"][track][disease_key]["low_threshold"]
        tau_high = tiers_cfg["tracks"][track][disease_key]["high_threshold"]

        eval_res = validate_risk_tier_monotonicity(y_test, test_probs, tau_low, tau_high)
        assert eval_res["is_monotonic"], f"Non-monotonic risk tiers observed for {track} {disease} on test set!"
        assert eval_res["satisfies_triage_safety"], f"Triage safety failed (positive capture < 90%) for {track} {disease}"
        assert eval_res["satisfies_high_concentration"], f"High concentration failed (< 2.0x baseline) for {track} {disease}"
        assert eval_res["all_passed"], f"Not all criteria passed for {track} {disease}"
