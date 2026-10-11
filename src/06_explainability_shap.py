"""
src/06_explainability_shap.py
Stage 3 Feature Attribution & Explainability Pipeline using SHAP
Project: Predictive Risk Identification for Diabetes and Heart Disease

Tasks:
1. Explains champion models (and candidate models) using exact SHAP explainers:
   - LinearExplainer for Logistic Regression.
   - TreeExplainer for Decision Tree & Random Forest.
   - Robustly handles different SHAP versions and multi-class array shapes.
2. Uses random background samples (3,000 rows) for computational efficiency.
3. Produces high-resolution global explainability visualizations:
   - outputs/figures/stage3_shap_beeswarm_diabetes.png
   - outputs/figures/stage3_shap_bar_diabetes.png
   - outputs/figures/stage3_shap_beeswarm_heart.png
   - outputs/figures/stage3_shap_bar_heart.png
4. Identifies three representative test patients (Low, Medium, High risk) per disease:
   - Generates local waterfall plots for each patient:
     outputs/figures/stage3_shap_waterfall_{disease}_{tier}.png
5. Exports global mean absolute SHAP importance table:
   - outputs/tables/shap_feature_importance.csv
"""

from pathlib import Path
import warnings
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import yaml
import sys

warnings.filterwarnings("ignore")

# Ensure root directory is on sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.risk_utils import assign_risk_tier

MODEL_READY_DIR = BASE_DIR / "data" / "model_ready"
MODELS_DIR = BASE_DIR / "models"
FIGURES_DIR = BASE_DIR / "outputs" / "figures"
TABLES_DIR = BASE_DIR / "outputs" / "tables"
CONFIG_DIR = BASE_DIR / "config"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42


def get_feature_transformed_data(pipeline, X_df: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    """Transform raw feature dataframe through pipeline preprocessors prior to classifier."""
    named_steps = pipeline.named_steps
    X_curr = X_df.copy()

    if "imputer" in named_steps:
        X_curr = named_steps["imputer"].transform(X_curr)
    if "scaler" in named_steps:
        X_curr = named_steps["scaler"].transform(X_curr)

    return pd.DataFrame(X_curr, columns=feature_names, index=X_df.index)


def get_shap_explainer_and_values(bundle: dict, X_bg: pd.DataFrame, X_eval: pd.DataFrame):
    """
    Construct appropriate SHAP explainer (LinearExplainer or TreeExplainer)
    and extract positive-class SHAP values cleanly across SHAP version formats.
    """
    pipeline = bundle["pipeline"]
    feature_names = bundle["feature_names"]
    clf = pipeline.named_steps["clf"]
    model_name = bundle["model_name"]

    X_bg_trans = get_feature_transformed_data(pipeline, X_bg, feature_names)
    X_eval_trans = get_feature_transformed_data(pipeline, X_eval, feature_names)

    if model_name == "logistic_regression":
        explainer = shap.LinearExplainer(clf, X_bg_trans)
        raw_exp = explainer(X_eval_trans)
        shap_values = raw_exp.values
        base_values = raw_exp.base_values
    else:
        explainer = shap.TreeExplainer(clf)
        raw_exp = explainer(X_eval_trans)
        # Handle 3D array (n_samples, n_features, 2) for binary classification
        if len(raw_exp.values.shape) == 3:
            shap_values = raw_exp.values[:, :, 1]
            base_values = raw_exp.base_values[:, 1] if len(raw_exp.base_values.shape) == 2 else raw_exp.base_values
        else:
            shap_values = raw_exp.values
            base_values = raw_exp.base_values

    # Build clean Explanation object populated with unscaled patient readings for readability
    explanation_pos = shap.Explanation(
        values=shap_values,
        base_values=base_values,
        data=X_eval.values,
        feature_names=feature_names,
    )
    return explainer, explanation_pos, X_eval_trans


def explain_track_disease(track: str, disease: str, target_col: str, train_csv_name: str, test_csv_name: str):
    """Run full SHAP explainability analysis for the champion model of a given track and disease."""
    # Find champion model from champion_models.csv
    champ_csv = TABLES_DIR / "champion_models.csv"
    if not champ_csv.exists():
        raise FileNotFoundError(f"champion_models.csv not found at {champ_csv}")

    champions_df = pd.read_csv(champ_csv)
    champ_row = champions_df[(champions_df["Track"] == track) & (champions_df["Disease"] == disease)]
    if champ_row.empty:
        raise ValueError(f"No champion found for {track} {disease}")

    champ_model_name = champ_row.iloc[0]["Champion_Model"]
    print(f"\n{'=' * 80}")
    print(f"SHAP EXPLAINABILITY: {track.upper()} - {disease.upper()} (Champion: {champ_model_name})")
    print(f"{'=' * 80}")

    ALL_MODELS_DIR = BASE_DIR / "models" / "all_models"
    subfolder = "track_a_survey" if track == "track_a" else "track_b_clinical"
    bundle_path = ALL_MODELS_DIR / subfolder / f"{track}_{disease}_{champ_model_name}.joblib"
    if not bundle_path.exists():
        bundle_path = MODELS_DIR / f"{track}_{disease}_{champ_model_name}.joblib"
    bundle = joblib.load(bundle_path)
    feature_names = bundle["feature_names"]

    # Load data
    df_train = pd.read_csv(MODEL_READY_DIR / train_csv_name)
    df_test = pd.read_csv(MODEL_READY_DIR / test_csv_name)

    X_train = df_train.drop(columns=[target_col])
    X_test = df_test.drop(columns=[target_col])
    y_test = df_test[target_col].astype(int).values

    # Subsample background and evaluation sets for speed if dataset is large
    n_bg = min(3000, len(X_train))
    n_eval = min(3000, len(X_test))
    X_bg = X_train.sample(n=n_bg, random_state=RANDOM_STATE) if len(X_train) > n_bg else X_train.copy()
    X_eval = X_test.sample(n=n_eval, random_state=RANDOM_STATE) if len(X_test) > n_eval else X_test.copy()

    print(f"Generating SHAP values with {len(X_bg)} background samples and {len(X_eval)} evaluation samples...")
    explainer, explanation_pos, X_eval_trans = get_shap_explainer_and_values(bundle, X_bg, X_eval)

    # 1. Global Mean Absolute SHAP Importance Table
    mean_abs_shap = np.abs(explanation_pos.values).mean(axis=0)
    feat_imp = pd.DataFrame({
        "Track": track,
        "Disease": disease,
        "Model": champ_model_name,
        "Feature": feature_names,
        "Mean_Absolute_SHAP": np.round(mean_abs_shap, 5),
    }).sort_values(by="Mean_Absolute_SHAP", ascending=False).reset_index(drop=True)
    feat_imp["Rank"] = range(1, len(feat_imp) + 1)

    print("\nTOP 10 MOST INFLUENTIAL FEATURES:")
    print(feat_imp.head(10)[["Rank", "Feature", "Mean_Absolute_SHAP"]].to_string(index=False))

    # 2. Global Beeswarm Plot
    plt.figure(figsize=(10, 8))
    shap.plots.beeswarm(explanation_pos, max_display=15, show=False)
    plt.title(f"SHAP Beeswarm Plot: {disease.capitalize()} Risk Drivers ({track.upper()})", fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()
    beeswarm_path = FIGURES_DIR / f"stage3_shap_beeswarm_{track}_{disease}.png"
    plt.savefig(beeswarm_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved global beeswarm plot: {beeswarm_path.relative_to(BASE_DIR)}")

    # 3. Global Bar Plot
    plt.figure(figsize=(10, 7))
    shap.plots.bar(explanation_pos, max_display=15, show=False)
    plt.title(f"Global Feature Importance: Mean |SHAP| ({disease.capitalize()} - {track.upper()})", fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()
    bar_path = FIGURES_DIR / f"stage3_shap_bar_{track}_{disease}.png"
    plt.savefig(bar_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved global bar plot: {bar_path.relative_to(BASE_DIR)}")

    # 4. Local Waterfall Plots for Representative Test Patients (Low, Medium, High)
    # Predict calibrated probabilities on the full test set
    calibrated_model = bundle["calibrated_model"]
    test_probs = calibrated_model.predict_proba(X_test)[:, 1]

    # Assign risk tiers to find example patients
    with open(CONFIG_DIR / "risk_tiers.yaml", "r") as f:
        tiers_cfg = yaml.safe_load(f)
    disease_key = "diabetes" if disease == "diabetes" else "heart_disease"
    tau_low = tiers_cfg["tracks"][track][disease_key]["low_threshold"]
    tau_high = tiers_cfg["tracks"][track][disease_key]["high_threshold"]

    tier_assignments = [
        assign_risk_tier(p, disease=disease_key, track=track, custom_thresholds=(tau_low, tau_high))["risk_tier"]
        for p in test_probs
    ]
    df_test_assigned = X_test.copy()
    df_test_assigned["pred_prob"] = test_probs
    df_test_assigned["risk_tier"] = tier_assignments

    # Find prototypical patients for each tier (close to median of each tier)
    low_candidates = df_test_assigned[df_test_assigned["risk_tier"] == "Low"]
    med_candidates = df_test_assigned[df_test_assigned["risk_tier"] == "Medium"]
    high_candidates = df_test_assigned[df_test_assigned["risk_tier"] == "High"]

    low_patient_idx = low_candidates["pred_prob"].sub(low_candidates["pred_prob"].median()).abs().idxmin()
    med_patient_idx = med_candidates["pred_prob"].sub(med_candidates["pred_prob"].median()).abs().idxmin()
    high_patient_idx = high_candidates["pred_prob"].sub(high_candidates["pred_prob"].median()).abs().idxmin()

    example_patients = [
        ("Low", low_patient_idx, low_candidates.loc[low_patient_idx, "pred_prob"]),
        ("Medium", med_patient_idx, med_candidates.loc[med_patient_idx, "pred_prob"]),
        ("High", high_patient_idx, high_candidates.loc[high_patient_idx, "pred_prob"]),
    ]

    for tier_label, p_idx, prob in example_patients:
        patient_row = X_test.loc[[p_idx]]
        _, patient_exp, _ = get_shap_explainer_and_values(bundle, X_bg, patient_row)

        plt.figure(figsize=(9, 6))
        shap.plots.waterfall(patient_exp[0], max_display=10, show=False)
        plt.title(
            f"Patient Risk Attribution: {tier_label.upper()} Risk ({disease.capitalize()})\n"
            f"Predicted Probability: {prob:.1%} (Cutoffs: Low<{tau_low:.2f}, High>={tau_high:.2f})",
            fontsize=11, fontweight="bold", pad=10
        )
        plt.tight_layout()
        waterfall_path = FIGURES_DIR / f"stage3_shap_waterfall_{track}_{disease}_{tier_label.lower()}.png"
        plt.savefig(waterfall_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"[OK] Saved {tier_label} risk waterfall plot: {waterfall_path.relative_to(BASE_DIR)}")

    return feat_imp


def main():
    print("=" * 80)
    print("STARTING STAGE 3 SHAP EXPLAINABILITY PIPELINE...")
    print("=" * 80)

    tasks = [
        ("track_a", "diabetes", "Diabetes_binary", "track_a_diabetes_train.csv", "track_a_diabetes_test.csv"),
        ("track_a", "heart", "HeartDiseaseorAttack", "track_a_heart_train.csv", "track_a_heart_test.csv"),
        ("track_b", "diabetes", "Outcome", "track_b_diabetes_train.csv", "track_b_diabetes_test.csv"),
        ("track_b", "heart", "condition", "track_b_heart_train.csv", "track_b_heart_test.csv"),
    ]

    all_feat_imps = []
    for track, disease, target_col, train_csv, test_csv in tasks:
        imp = explain_track_disease(track, disease, target_col, train_csv, test_csv)
        all_feat_imps.append(imp)

    combined_imp_df = pd.concat(all_feat_imps, ignore_index=True)
    table_path = TABLES_DIR / "shap_feature_importance.csv"
    combined_imp_df.to_csv(table_path, index=False)
    print(f"\n[OK] Saved complete SHAP feature importances to: {table_path.relative_to(BASE_DIR)}")

    print("\n" + "=" * 80)
    print("STAGE 3 SHAP EXPLAINABILITY PIPELINE COMPLETED SUCCESSFULLY.")
    print("=" * 80)


if __name__ == "__main__":
    main()
