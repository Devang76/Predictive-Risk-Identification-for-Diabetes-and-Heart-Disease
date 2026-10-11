"""
tests/test_chronic_disease_predictor.py
Verification suite for the packaged Chronic Disease Predictor Model.

Tests:
1. ChronicDiseasePredictor class instantiation and model loading.
2. Inference on prototypical example patients (Low, Medium, High).
3. Correct prediction structure (probabilities, risk tiers, top 3/top 2 factors, disclaimer).
4. Dual-track and clinical mode validation.
5. Model Index completeness: every model file listed in models/MODEL_INDEX.md physically exists on disk.
"""

import json
from pathlib import Path
import re
import pytest

BASE_DIR = Path(__file__).resolve().parent.parent
EXAMPLES_DIR = BASE_DIR / "examples"
MODELS_DIR = BASE_DIR / "models"

from models.chronic_disease_predictor.predictor import ChronicDiseasePredictor


@pytest.fixture(scope="module")
def predictor():
    """Instantiate a shared ChronicDiseasePredictor instance."""
    return ChronicDiseasePredictor()


def test_chronic_disease_predictor_initialization(predictor):
    """Verify that ChronicDiseasePredictor loads and initializes configurations."""
    assert predictor is not None
    assert "tracks" in predictor.risk_tiers
    assert "track_a" in predictor.feature_schema
    assert len(predictor._champions_df) == 4


def test_predict_on_three_example_survey_patients(predictor):
    """Verify prediction and attribution across the 3 prototypical survey patients."""
    test_cases = [
        ("low_risk_patient.json", "Low"),
        ("medium_risk_patient.json", "Medium"),
        ("high_risk_patient.json", "High"),
    ]

    for filename, expected_tier in test_cases:
        filepath = EXAMPLES_DIR / filename
        assert filepath.exists(), f"Missing example file: {filename}"
        with open(filepath, "r") as f:
            patient_data = json.load(f)

        res = predictor.predict(patient_data, mode="survey")

        assert "patient_id" in res
        assert "predictions" in res
        assert "survey" in res["predictions"]
        assert "disclaimer" in res
        assert "These are key contributing factors from a statistical model" in res["disclaimer"]

        survey_res = res["predictions"]["survey"]
        assert "diabetes" in survey_res
        assert "heart_disease" in survey_res

        for disease in ["diabetes", "heart_disease"]:
            d_out = survey_res[disease]
            assert 0.0 <= d_out["probability"] <= 1.0
            assert d_out["risk_tier"] == expected_tier
            assert len(d_out["top_risk_increasers"]) <= 3
            assert len(d_out["top_risk_decreasers"]) <= 2

            # Check factor fields
            for inc in d_out["top_risk_increasers"]:
                assert "feature" in inc
                assert "reading" in inc
                assert "shap_value" in inc
                assert inc["shap_value"] >= 0

            for dec in d_out["top_risk_decreasers"]:
                assert "feature" in dec
                assert "reading" in dec
                assert "shap_value" in dec
                assert dec["shap_value"] <= 0


def test_predict_clinical_mode(predictor):
    """Verify that clinical mode operates on clinical patient records."""
    filepath = EXAMPLES_DIR / "clinical_patient_high.json"
    assert filepath.exists()
    with open(filepath, "r") as f:
        patient_data = json.load(f)

    res = predictor.predict(patient_data, mode="clinical")
    assert "clinical" in res["predictions"]
    clin_res = res["predictions"]["clinical"]

    assert clin_res["diabetes"]["risk_tier"] in ["Low", "Medium", "High"]
    assert clin_res["heart_disease"]["risk_tier"] in ["Low", "Medium", "High"]
    assert 0.0 <= clin_res["diabetes"]["probability"] <= 1.0
    assert 0.0 <= clin_res["heart_disease"]["probability"] <= 1.0


def test_predict_both_mode(predictor):
    """Verify dual-track mode on comprehensive patient profile."""
    filepath = EXAMPLES_DIR / "comprehensive_patient_both.json"
    assert filepath.exists()
    with open(filepath, "r") as f:
        patient_data = json.load(f)

    res = predictor.predict(patient_data, mode="both")
    assert "survey" in res["predictions"]
    assert "clinical" in res["predictions"]


def test_every_model_in_model_index_exists():
    """Verify that every model file documented in models/MODEL_INDEX.md physically exists on disk."""
    index_path = MODELS_DIR / "MODEL_INDEX.md"
    assert index_path.exists(), "models/MODEL_INDEX.md missing!"

    content = index_path.read_text()

    # Extract file name and relative location folder from table
    pattern = r"\|\s*`([^`]+\.joblib)`\s*\|\s*`([^`]+)`\s*\|"
    matches = re.findall(pattern, content)

    assert len(matches) == 12, f"Expected 12 model rows in MODEL_INDEX.md, parsed {len(matches)}"

    for filename, location_folder in matches:
        folder_clean = location_folder.strip("/ ")
        model_path = MODELS_DIR / folder_clean / filename
        assert model_path.exists(), f"Model file listed in MODEL_INDEX.md does not exist: {model_path}"
        assert model_path.stat().st_size > 0, f"Model file is empty: {model_path}"
