"""
tests/test_stage3_prep.py
Pytest test suite validating Stage 3 data preparation, leakage safeguards,
patient schema enforcement, and risk tier logic.

Project: Predictive Risk Identification for Diabetes and Heart Disease
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.risk_utils import (
    assign_risk_tier,
    get_preprocessing_pipeline,
    validate_patient_input,
    validate_risk_tier_monotonicity,
)

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_READY_DIR = BASE_DIR / "data" / "model_ready"


def test_no_target_or_cross_target_in_features():
    """Verify that feature matrices do not contain their target or cross-disease target."""
    # 1. Track A Diabetes
    df_diab_tr = pd.read_csv(MODEL_READY_DIR / "track_a_diabetes_train.csv")
    X_diab_cols = [c for c in df_diab_tr.columns if c != "Diabetes_binary"]
    assert "Diabetes_binary" not in X_diab_cols, "Target found in Track A Diabetes features!"
    assert "HeartDiseaseorAttack" not in X_diab_cols, "Cross-disease target found in Track A Diabetes features!"
    assert len(X_diab_cols) == 20, f"Expected 20 features for Track A, got {len(X_diab_cols)}"

    # 2. Track A Heart Disease
    df_heart_tr = pd.read_csv(MODEL_READY_DIR / "track_a_heart_train.csv")
    X_heart_cols = [c for c in df_heart_tr.columns if c != "HeartDiseaseorAttack"]
    assert "HeartDiseaseorAttack" not in X_heart_cols, "Target found in Track A Heart features!"
    assert "Diabetes_binary" not in X_heart_cols, "Cross-disease target found in Track A Heart features!"
    assert len(X_heart_cols) == 20, f"Expected 20 features for Track A, got {len(X_heart_cols)}"

    # 3. Track B Diabetes (Pima)
    df_pima_tr = pd.read_csv(MODEL_READY_DIR / "track_b_diabetes_train.csv")
    X_pima_cols = [c for c in df_pima_tr.columns if c != "Outcome"]
    assert "Outcome" not in X_pima_cols, "Target found in Track B Diabetes features!"
    assert len(X_pima_cols) == 8, f"Expected 8 features for Pima, got {len(X_pima_cols)}"

    # 4. Track B Heart Disease (Cleveland)
    df_clev_tr = pd.read_csv(MODEL_READY_DIR / "track_b_heart_train.csv")
    X_clev_cols = [c for c in df_clev_tr.columns if c != "condition"]
    assert "condition" not in X_clev_cols, "Target found in Track B Heart features!"
    assert len(X_clev_cols) == 13, f"Expected 13 features for Cleveland, got {len(X_clev_cols)}"


def test_no_high_glucose_in_any_dataset():
    """Verify that operational proxy 'high_glucose' is absent from all model-ready datasets."""
    for csv_file in MODEL_READY_DIR.glob("*.csv"):
        if csv_file.name == "model_ready_summary.csv":
            continue
        df = pd.read_csv(csv_file)
        assert "high_glucose" not in df.columns, f"Found banned proxy 'high_glucose' in {csv_file.name}!"


def test_train_test_split_no_overlap():
    """Verify that train and test splits have 0 overlapping indices or rows."""
    indices_path = MODEL_READY_DIR / "split_indices.json"
    assert indices_path.exists(), "split_indices.json manifest missing!"

    with open(indices_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    for track_key, split in manifest.items():
        tr_set = set(split["train_indices"])
        te_set = set(split["test_indices"])
        overlap = tr_set.intersection(te_set)
        assert len(overlap) == 0, f"DATA LEAKAGE: {track_key} has {len(overlap)} overlapping indices!"
        assert len(tr_set) > 0 and len(te_set) > 0, f"Empty split detected in {track_key}!"


def test_patient_input_schema_validation_accepts_valid():
    """Verify that validate_patient_input accepts biologically valid profiles."""
    valid_patient_a = {
        "HighBP": 1,
        "HighChol": 1,
        "CholCheck": 1,
        "BMI": 28.5,
        "Smoker": 0,
        "Stroke": 0,
        "PhysActivity": 1,
        "Fruits": 1,
        "Veggies": 1,
        "HvyAlcoholConsump": 0,
        "AnyHealthcare": 1,
        "NoDocbcCost": 0,
        "GenHlth": 3,
        "MentHlth": 2,
        "PhysHlth": 4,
        "DiffWalk": 0,
        "Sex": 1,
        "Age": 8,
        "Education": 5,
        "Income": 6,
    }
    sanitized = validate_patient_input(valid_patient_a, track="track_a")
    assert sanitized["BMI"] == 28.5
    assert sanitized["HighBP"] == 1
    assert sanitized["GenHlth"] == 3


def test_patient_input_schema_validation_catches_bad_inputs():
    """Verify that validate_patient_input rejects impossible/out-of-bounds readings."""
    # 1. Impossible low BMI
    with pytest.raises(ValueError, match="Impossible clinical reading for 'Body Mass Index'"):
        validate_patient_input({"BMI": 5.0}, track="track_a")

    # 2. Impossible high BMI
    with pytest.raises(ValueError, match="Impossible clinical reading for 'Body Mass Index'"):
        validate_patient_input({"BMI": 120.0}, track="track_a")

    # 3. Out-of-bounds General Health rating (must be 1-5)
    with pytest.raises(ValueError, match="outside permitted category range"):
        validate_patient_input({"GenHlth": 8}, track="track_a")

    # 4. Out-of-bounds Mental Health days (must be 0-30)
    with pytest.raises(ValueError, match="exceeds physiologically valid maximum"):
        validate_patient_input({"MentHlth": 45}, track="track_a")

    # 5. Invalid binary flag
    with pytest.raises(ValueError, match="not in allowed binary set"):
        validate_patient_input({"HighBP": 3}, track="track_a")


def test_risk_tier_assignment_logic():
    """Verify risk tier cutoff assignment for Track A."""
    tier_low = assign_risk_tier(0.05, disease="diabetes", track="track_a")
    assert tier_low["risk_tier"] == "Low"

    tier_med = assign_risk_tier(0.18, disease="diabetes", track="track_a")
    assert tier_med["risk_tier"] == "Medium"

    tier_high = assign_risk_tier(0.45, disease="diabetes", track="track_a")
    assert tier_high["risk_tier"] == "High"


def test_risk_tier_monotonicity_validator():
    """Verify that validate_risk_tier_monotonicity correctly evaluates clinical monotonicity and lift."""
    # Valid calibrated distribution (20 patients):
    # Low (10 patients, 0 pos -> 0.0%)
    # Medium (6 patients, 1 pos -> 16.7%)
    # High (4 patients, 3 pos -> 75.0%)
    # Overall baseline: 4 / 20 = 20.0%
    y_true = np.array([0]*10 + [0]*5 + [1] + [0] + [1]*3)
    y_prob = np.array(
        [0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.07, 0.08, 0.08, 0.09] +  # Low (10)
        [0.12, 0.14, 0.16, 0.18, 0.22, 0.26] +                           # Medium (6)
        [0.35, 0.45, 0.55, 0.65]                                         # High (4)
    )

    res = validate_risk_tier_monotonicity(y_true, y_prob, low_threshold=0.10, high_threshold=0.30)
    assert res["is_monotonic"] is True
    assert res["satisfies_triage_safety"] is True  # 100% of positives are in Medium+High
    assert res["satisfies_high_concentration"] is True  # 75.0% >= 2x 20.0% baseline
    assert res["all_passed"] is True
    assert res["observed_rates"]["Low"] < res["observed_rates"]["Medium"] < res["observed_rates"]["High"]

    # Non-monotonic test case (where Medium has higher rate than High)
    y_true_bad = np.array([0]*10 + [1]*5 + [0] + [0]*4)
    res_bad = validate_risk_tier_monotonicity(y_true_bad, y_prob, low_threshold=0.10, high_threshold=0.30)
    assert res_bad["is_monotonic"] is False
    assert res_bad["all_passed"] is False


def test_preprocessing_pipeline_fit_strictly_on_train():
    """Verify that preprocessing pipelines can be fit strictly on training data."""
    X_train = np.array([[100.0, 25.0], [np.nan, 30.0], [140.0, np.nan], [110.0, 22.0]])
    X_test = np.array([[120.0, np.nan], [np.nan, 28.0]])

    pipe = get_preprocessing_pipeline(model_family="linear", scaler_type="standard")
    X_train_trans = pipe.fit_transform(X_train)
    X_test_trans = pipe.transform(X_test)

    assert not np.isnan(X_train_trans).any(), "NaNs remained after training fit!"
    assert not np.isnan(X_test_trans).any(), "NaNs remained after test transform!"
    assert X_train_trans.shape == X_train.shape
    assert X_test_trans.shape == X_test.shape
