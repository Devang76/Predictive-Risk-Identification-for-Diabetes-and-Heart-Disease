"""
src/risk_utils.py
Core utility functions for risk tier assignment, clinical monotonicity validation,
patient input schema verification, and leakage-free scikit-learn preprocessing pipelines.

Project: Predictive Risk Identification for Diabetes and Heart Disease
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import yaml
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler

BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_DIR = BASE_DIR / "config"
TIERS_CONFIG_PATH = CONFIG_DIR / "risk_tiers.yaml"
SCHEMA_CONFIG_PATH = CONFIG_DIR / "feature_schema.yaml"
MODEL_CONFIG_PATH = CONFIG_DIR / "model_config.yaml"


def load_yaml(filepath: Union[str, Path]) -> Dict[str, Any]:
    """Load a YAML configuration file safely."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def assign_risk_tier(
    probability: float,
    disease: str = "diabetes",
    track: str = "track_a",
    custom_thresholds: Optional[Tuple[float, float]] = None,
    config_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """
    Convert a predicted disease probability into a clinical risk category (Low, Medium, High).

    Parameters
    ----------
    probability : float
        Predicted probability of disease, bounded in [0.0, 1.0].
    disease : str
        Target disease: 'diabetes' or 'heart_disease'.
    track : str
        Modeling track: 'track_a' (BRFSS population) or 'track_b' (clinical).
    custom_thresholds : tuple of (float, float), optional
        Explicit (low_threshold, high_threshold). Overrides config file if provided.
    config_path : str or Path, optional
        Custom path to risk_tiers.yaml.

    Returns
    -------
    dict
        Structured record containing:
        - risk_tier: 'Low', 'Medium', or 'High'
        - probability: float
        - disease: str
        - track: str
        - low_threshold: float
        - high_threshold: float
    """
    if not (0.0 <= probability <= 1.0):
        raise ValueError(f"Probability must be bounded between 0.0 and 1.0, got: {probability}")

    if custom_thresholds is not None:
        low_t, high_t = custom_thresholds
    else:
        cfg_file = config_path or TIERS_CONFIG_PATH
        tiers_cfg = load_yaml(cfg_file)
        track_cfg = tiers_cfg["tracks"][track][disease]
        low_t = float(track_cfg["low_threshold"])
        high_t = float(track_cfg["high_threshold"])

    if low_t >= high_t:
        raise ValueError(f"Low threshold ({low_t}) must be strictly less than high threshold ({high_t})")

    if probability < low_t:
        tier = "Low"
    elif probability < high_t:
        tier = "Medium"
    else:
        tier = "High"

    return {
        "risk_tier": tier,
        "probability": round(float(probability), 4),
        "disease": disease,
        "track": track,
        "low_threshold": low_t,
        "high_threshold": high_t,
    }


def validate_risk_tier_monotonicity(
    y_true: Union[np.ndarray, pd.Series, List[int]],
    y_pred_prob: Union[np.ndarray, pd.Series, List[float]],
    low_threshold: float,
    high_threshold: float,
    baseline_prevalence: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Validate that risk tiers satisfy required clinical and statistical properties:
    1. Monotonicity: Observed rate in Low < Observed rate in Medium < Observed rate in High.
    2. Triage Safety: At least 90% of all true positive cases fall into Medium + High tiers
       (i.e. Low risk tier contains <= 10% of true positives).
    3. Actionable Concentration: High tier observed rate >= 2.0x baseline prevalence.

    Parameters
    ----------
    y_true : array-like
        Binary ground-truth labels (0 or 1).
    y_pred_prob : array-like
        Predicted positive-class probabilities P(y=1).
    low_threshold : float
        Cutoff separating Low from Medium risk.
    high_threshold : float
        Cutoff separating Medium from High risk.
    baseline_prevalence : float, optional
        Overall population disease prevalence. If None, computed from y_true.

    Returns
    -------
    dict
        Diagnostic summary with counts, observed rates, and validation pass/fail flags.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred_prob = np.asarray(y_pred_prob, dtype=float)

    if len(y_true) != len(y_pred_prob):
        raise ValueError(f"Lengths must match: y_true={len(y_true)}, y_pred_prob={len(y_pred_prob)}")

    total_patients = len(y_true)
    total_positives = int(np.sum(y_true))
    baseline_rate = float(total_positives / total_patients) if baseline_prevalence is None else baseline_prevalence

    # Tier masks
    low_mask = y_pred_prob < low_threshold
    med_mask = (y_pred_prob >= low_threshold) & (y_pred_prob < high_threshold)
    high_mask = y_pred_prob >= high_threshold

    # Patient counts per tier
    n_low = int(np.sum(low_mask))
    n_med = int(np.sum(med_mask))
    n_high = int(np.sum(high_mask))

    # Observed positive cases per tier
    pos_low = int(np.sum(y_true[low_mask]))
    pos_med = int(np.sum(y_true[med_mask]))
    pos_high = int(np.sum(y_true[high_mask]))

    # Observed disease rates per tier
    rate_low = float(pos_low / n_low) if n_low > 0 else 0.0
    rate_med = float(pos_med / n_med) if n_med > 0 else 0.0
    rate_high = float(pos_high / n_high) if n_high > 0 else 0.0

    # Rule 1: Monotonicity
    is_monotonic = bool(rate_low < rate_med < rate_high)

    # Rule 2: Triage Safety (Medium + High captures >= 90% of true positives)
    med_high_pos = pos_med + pos_high
    pos_capture_rate = float(med_high_pos / total_positives) if total_positives > 0 else 1.0
    satisfies_triage_safety = bool(pos_capture_rate >= 0.90)

    # Rule 3: High tier concentration (>= 2x baseline)
    high_tier_lift = float(rate_high / baseline_rate) if baseline_rate > 0 else 0.0
    satisfies_high_concentration = bool(high_tier_lift >= 2.0)

    all_passed = bool(is_monotonic and satisfies_triage_safety and satisfies_high_concentration)

    return {
        "is_monotonic": is_monotonic,
        "satisfies_triage_safety": satisfies_triage_safety,
        "satisfies_high_concentration": satisfies_high_concentration,
        "all_passed": all_passed,
        "tier_counts": {"Low": n_low, "Medium": n_med, "High": n_high},
        "tier_positives": {"Low": pos_low, "Medium": pos_med, "High": pos_high},
        "observed_rates": {
            "Low": round(rate_low, 4),
            "Medium": round(rate_med, 4),
            "High": round(rate_high, 4),
        },
        "true_positives_captured_in_med_plus_high_pct": round(pos_capture_rate * 100, 2),
        "high_tier_lift_vs_baseline": round(high_tier_lift, 2),
        "baseline_prevalence": round(baseline_rate, 4),
    }


def validate_patient_input(
    patient_data: Dict[str, Any],
    track: str = "track_a",
    schema_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """
    Validate a new patient input dictionary against the formal clinical schema.
    Applies default values if an optional biometric is omitted and rejects impossible
    readings with informative clinical error messages.

    Parameters
    ----------
    patient_data : dict
        Key-value mapping of feature name to entered value.
    track : str
        'track_a' (BRFSS population features), 'track_b_diabetes' (Pima), or 'track_b_heart' (Cleveland).
    schema_path : str or Path, optional
        Custom path to feature_schema.yaml.

    Returns
    -------
    dict
        Clean, validated, and type-cast patient feature dictionary.

    Raises
    ------
    ValueError
        If an entered value is out of bounds, has invalid type, or violates categorical codes.
    """
    cfg_file = schema_path or SCHEMA_CONFIG_PATH
    schema_yaml = load_yaml(cfg_file)

    # Resolve track aliases
    if track in ["survey", "track_a"]:
        track_schema = schema_yaml["track_a"]["features"]
    elif track in ["clinical", "track_b"]:
        # Merge both clinical feature sets
        track_schema = {}
        track_schema.update(schema_yaml["track_b_diabetes"]["features"])
        track_schema.update(schema_yaml["track_b_heart"]["features"])
    elif track in schema_yaml:
        track_schema = schema_yaml[track]["features"]
    else:
        raise ValueError(
            f"Unknown track '{track}'. Must be one of: 'survey', 'clinical', 'track_a', 'track_b', {list(schema_yaml.keys())}"
        )

    sanitized: Dict[str, Any] = {}
    missing_imputed: List[str] = []

    # Case-insensitive / alias handling for age across survey and clinical modes
    patient_data_copy = dict(patient_data)
    if track in ["clinical", "track_b", "track_b_diabetes", "track_b_heart"]:
        # In clinical mode, age must be continuous years (min 20/21)
        if "age" in patient_data_copy and patient_data_copy["age"] > 13:
            patient_data_copy["Age"] = patient_data_copy["age"]
        elif "Age" in patient_data_copy and "age" not in patient_data_copy:
            patient_data_copy["age"] = patient_data_copy["Age"]
    elif track in ["survey", "track_a"]:
        # In survey mode, Age is an ordinal bracket 1-13
        if "Age" not in patient_data_copy and "age" in patient_data_copy:
            raw_age = patient_data_copy["age"]
            if raw_age < 25:
                patient_data_copy["Age"] = 1
            elif raw_age < 30:
                patient_data_copy["Age"] = 2
            elif raw_age < 35:
                patient_data_copy["Age"] = 3
            elif raw_age < 40:
                patient_data_copy["Age"] = 4
            elif raw_age < 45:
                patient_data_copy["Age"] = 5
            elif raw_age < 50:
                patient_data_copy["Age"] = 6
            elif raw_age < 55:
                patient_data_copy["Age"] = 7
            elif raw_age < 60:
                patient_data_copy["Age"] = 8
            elif raw_age < 65:
                patient_data_copy["Age"] = 9
            elif raw_age < 70:
                patient_data_copy["Age"] = 10
            elif raw_age < 75:
                patient_data_copy["Age"] = 11
            elif raw_age < 80:
                patient_data_copy["Age"] = 12
            else:
                patient_data_copy["Age"] = 13

    for feat_key, feat_spec in track_schema.items():
        feat_name = feat_spec.get("name", feat_key)
        feat_type = feat_spec.get("type", "numeric")
        default_val = feat_spec.get("default", None)
        units = feat_spec.get("units", "")

        # Check if feature is provided
        if feat_key not in patient_data_copy or patient_data_copy[feat_key] is None:
            if default_val is not None:
                sanitized[feat_key] = default_val
                missing_imputed.append(feat_key)
                continue
            else:
                raise ValueError(f"Missing mandatory feature '{feat_name}' ({feat_key}) with no default available.")

        raw_val = patient_data_copy[feat_key]

        # 1. Binary validation
        if feat_type == "binary":
            try:
                val_int = int(raw_val)
            except (ValueError, TypeError):
                raise ValueError(
                    f"Invalid input for '{feat_name}' ({feat_key}): Value '{raw_val}' must be binary 0 or 1."
                )
            allowed = feat_spec.get("allowed_values", [0, 1])
            if val_int not in allowed:
                raise ValueError(
                    f"Invalid input for '{feat_name}' ({feat_key}): Value {val_int} is not in allowed binary set {allowed}."
                )
            sanitized[feat_key] = val_int

        # 2. Ordinal validation
        elif feat_type == "ordinal":
            try:
                val_int = int(raw_val)
            except (ValueError, TypeError):
                raise ValueError(
                    f"Invalid input for '{feat_name}' ({feat_key}): Value '{raw_val}' must be an integer code."
                )
            allowed = feat_spec.get("allowed_values", [])
            if allowed and val_int not in allowed:
                raise ValueError(
                    f"Invalid input for '{feat_name}' ({feat_key}): Value {val_int} is outside permitted category range {allowed}."
                )
            sanitized[feat_key] = val_int

        # 3. Continuous / Numeric validation
        elif feat_type == "numeric":
            try:
                val_float = float(raw_val)
            except (ValueError, TypeError):
                raise ValueError(
                    f"Invalid input for '{feat_name}' ({feat_key}): Value '{raw_val}' cannot be parsed as a numeric reading."
                )
            min_val = feat_spec.get("min_value", None)
            max_val = feat_spec.get("max_value", None)

            if min_val is not None and val_float < min_val:
                raise ValueError(
                    f"Impossible clinical reading for '{feat_name}' ({feat_key}): Value {val_float} {units} "
                    f"is below physiologically valid minimum ({min_val} {units})."
                )
            if max_val is not None and val_float > max_val:
                raise ValueError(
                    f"Impossible clinical reading for '{feat_name}' ({feat_key}): Value {val_float} {units} "
                    f"exceeds physiologically valid maximum ({max_val} {units})."
                )
            sanitized[feat_key] = val_float

        else:
            sanitized[feat_key] = raw_val

    return sanitized


def get_preprocessing_pipeline(
    model_family: str = "tree",
    imputation_strategy: str = "median",
    scaler_type: str = "standard",
) -> Pipeline:
    """
    Construct a scikit-learn preprocessing Pipeline to ensure zero data leakage.
    Imputation and scaling parameters are strictly learned from training folds only.

    Parameters
    ----------
    model_family : str
        'tree' (e.g. Decision Tree, Random Forest) or 'linear' (Logistic Regression).
        Tree models require only imputation; linear models require imputation + scaling.
    imputation_strategy : str
        Strategy for SimpleImputer: 'median' or 'mean'.
    scaler_type : str
        'standard' (StandardScaler) or 'robust' (RobustScaler).

    Returns
    -------
    Pipeline
        Configured scikit-learn Pipeline.
    """
    steps = [("imputer", SimpleImputer(strategy=imputation_strategy))]

    if model_family.lower() in ["linear", "distance", "logistic_regression", "knn", "svm"]:
        if scaler_type.lower() == "robust":
            steps.append(("scaler", RobustScaler()))
        else:
            steps.append(("scaler", StandardScaler()))

    return Pipeline(steps)
