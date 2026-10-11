"""
src/07_track_b_shap.py
Track B Clinical Benchmark Feature Attribution & Explainability Pipeline
Project: Predictive Risk Identification for Diabetes and Heart Disease

Features & Deliverables:
1. SHAP Explainability for Track B Champions (Pima Diabetes and Cleveland Heart):
   - LinearExplainer for Logistic Regression champions (TreeExplainer supported for tree models).
   - Handles multi-class / 2D / 3D shap value shapes to ensure positive class attribution.
   - Global beeswarm plots:
     * outputs/figures/stage3_shap_beeswarm_track_b_diabetes.png
     * outputs/figures/stage3_shap_beeswarm_track_b_heart.png
   - Global bar plots:
     * outputs/figures/stage3_shap_bar_track_b_diabetes.png
     * outputs/figures/stage3_shap_bar_track_b_heart.png
   - Mean absolute SHAP importance table:
     * outputs/tables/shap_feature_importance_track_b.csv
   - Local waterfall plots for representative Low, Medium, and High risk patients:
     * outputs/figures/stage3_shap_waterfall_track_b_diabetes_{low,medium,high}.png
     * outputs/figures/stage3_shap_waterfall_track_b_heart_{low,medium,high}.png

2. Out-of-Fold (OOF) Risk Tier Tuning & Small-Sample Reliability Analysis:
   - Tunes risk tier cutoffs strictly on training out-of-fold predictions.
   - Evaluates sample-size reliability (N=768 Pima, N=297 Cleveland) with Wilson confidence intervals.
   - Exports outputs/tables/track_b_tier_calibration_analysis.csv.

3. Cross-Validated Performance with Confidence Intervals:
   - Repeated Stratified 5-Fold CV (5 repeats = 25 evaluations) on Track B cohorts.
   - Reports Mean, Std, and 95% Confidence Intervals for Recall, F1, ROC-AUC, and PR-AUC.
   - Exports outputs/tables/track_b_cv_metrics_with_ci.csv.
"""

from pathlib import Path
import sys
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold
import yaml

warnings.filterwarnings("ignore")

# Ensure root directory is on sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.risk_utils import assign_risk_tier, load_yaml, validate_risk_tier_monotonicity

MODELS_DIR = BASE_DIR / "models"
ALL_MODELS_DIR = BASE_DIR / "models" / "all_models"
MODEL_READY_DIR = BASE_DIR / "data" / "model_ready"
FIGURES_DIR = BASE_DIR / "outputs" / "figures"
TABLES_DIR = BASE_DIR / "outputs" / "tables"
CONFIG_DIR = BASE_DIR / "config"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42


def get_feature_transformed_data(pipeline, X_df: pd.DataFrame, feature_names: list) -> pd.DataFrame:
    """Pass feature dataframe through pipeline preprocessors (imputer and scaler) prior to classifier."""
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
    and extract positive-class SHAP values cleanly across SHAP versions.
    """
    pipeline = bundle["pipeline"]
    feature_names = bundle["feature_names"]
    clf = pipeline.named_steps["clf"]
    model_name = bundle.get("model_name", "logistic_regression")

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
        if len(raw_exp.values.shape) == 3:
            shap_values = raw_exp.values[:, :, 1]
            base_values = raw_exp.base_values[:, 1] if len(raw_exp.base_values.shape) == 2 else raw_exp.base_values
        else:
            shap_values = raw_exp.values
            base_values = raw_exp.base_values

    # Build clean Explanation object populated with raw, unscaled biometric readings
    explanation_pos = shap.Explanation(
        values=shap_values,
        base_values=base_values,
        data=X_eval.values,
        feature_names=feature_names,
    )
    return explainer, explanation_pos, X_eval_trans


def generate_track_b_shap(disease: str, target_col: str, train_csv: str, test_csv: str):
    """Generate global beeswarm, global bar, and local waterfall plots for a Track B disease."""
    champ_csv = TABLES_DIR / "champion_models.csv"
    champions_df = pd.read_csv(champ_csv)
    champ_row = champions_df[(champions_df["Track"] == "track_b") & (champions_df["Disease"] == disease)]
    if champ_row.empty:
        raise ValueError(f"No Track B champion found for disease {disease}")

    champ_model_name = champ_row.iloc[0]["Champion_Model"]
    print(f"\n{'=' * 80}")
    print(f"TRACK B SHAP EXPLAINABILITY: {disease.upper()} (Champion: {champ_model_name})")
    print(f"{'=' * 80}")

    bundle_path = ALL_MODELS_DIR / "track_b_clinical" / f"track_b_{disease}_{champ_model_name}.joblib"
    if not bundle_path.exists():
        bundle_path = MODELS_DIR / f"track_b_{disease}_{champ_model_name}.joblib"
    bundle = joblib.load(bundle_path)
    feature_names = bundle["feature_names"]

    df_train = pd.read_csv(MODEL_READY_DIR / train_csv)
    df_test = pd.read_csv(MODEL_READY_DIR / test_csv)

    X_train = df_train.drop(columns=[target_col])
    X_test = df_test.drop(columns=[target_col])
    y_test = df_test[target_col].astype(int).values

    print(f"Running SHAP with {len(X_train)} background samples and {len(X_test)} evaluation samples...")
    explainer, explanation_pos, _ = get_shap_explainer_and_values(bundle, X_train, X_test)

    # 1. Global Mean Absolute SHAP Importance Table
    mean_abs_shap = np.abs(explanation_pos.values).mean(axis=0)
    feat_imp = pd.DataFrame({
        "Track": "track_b",
        "Disease": disease,
        "Model": champ_model_name,
        "Feature": feature_names,
        "Mean_Absolute_SHAP": np.round(mean_abs_shap, 5),
    }).sort_values(by="Mean_Absolute_SHAP", ascending=False).reset_index(drop=True)
    feat_imp["Rank"] = range(1, len(feat_imp) + 1)

    print("\nTOP INFLUENTIAL CLINICAL FEATURES:")
    print(feat_imp[["Rank", "Feature", "Mean_Absolute_SHAP"]].to_string(index=False))

    # 2. Global Beeswarm Plot
    plt.figure(figsize=(9, 6))
    shap.plots.beeswarm(explanation_pos, max_display=12, show=False)
    cohort_title = "Pima Indians Cohort" if disease == "diabetes" else "Cleveland Clinic Cohort"
    plt.title(
        f"Track B SHAP Beeswarm: {disease.capitalize()} Risk Drivers\n({cohort_title} - {champ_model_name})",
        fontsize=12,
        fontweight="bold",
        pad=10,
    )
    plt.tight_layout()
    beeswarm_path = FIGURES_DIR / f"stage3_shap_beeswarm_track_b_{disease}.png"
    plt.savefig(beeswarm_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved global beeswarm plot: {beeswarm_path.relative_to(BASE_DIR)}")

    # 3. Global Bar Plot
    plt.figure(figsize=(9, 5.5))
    shap.plots.bar(explanation_pos, max_display=12, show=False)
    plt.title(
        f"Global Feature Importance: Mean |SHAP| (Track B {disease.capitalize()})\n({cohort_title})",
        fontsize=12,
        fontweight="bold",
        pad=10,
    )
    plt.tight_layout()
    bar_path = FIGURES_DIR / f"stage3_shap_bar_track_b_{disease}.png"
    plt.savefig(bar_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved global bar plot: {bar_path.relative_to(BASE_DIR)}")

    # 4. Local Waterfall Plots for representative test patients (Low, Medium, High)
    calibrated_model = bundle["calibrated_model"]
    test_probs = calibrated_model.predict_proba(X_test)[:, 1]

    with open(CONFIG_DIR / "risk_tiers.yaml", "r") as f:
        tiers_cfg = yaml.safe_load(f)
    disease_key = "diabetes" if disease == "diabetes" else "heart_disease"
    tau_low = tiers_cfg["tracks"]["track_b"][disease_key]["low_threshold"]
    tau_high = tiers_cfg["tracks"]["track_b"][disease_key]["high_threshold"]

    tier_assignments = [
        assign_risk_tier(p, disease=disease_key, track="track_b", custom_thresholds=(tau_low, tau_high))["risk_tier"]
        for p in test_probs
    ]
    df_test_assigned = X_test.copy()
    df_test_assigned["pred_prob"] = test_probs
    df_test_assigned["risk_tier"] = tier_assignments

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
        _, patient_exp, _ = get_shap_explainer_and_values(bundle, X_train, patient_row)

        plt.figure(figsize=(9, 5.5))
        shap.plots.waterfall(patient_exp[0], max_display=10, show=False)
        plt.title(
            f"Patient Risk Attribution: {tier_label.upper()} Risk (Track B {disease.capitalize()})\n"
            f"Predicted Probability: {prob:.1%} (Cutoffs: Low<{tau_low:.3f}, High>={tau_high:.3f})",
            fontsize=11,
            fontweight="bold",
            pad=10,
        )
        plt.tight_layout()
        waterfall_path = FIGURES_DIR / f"stage3_shap_waterfall_track_b_{disease}_{tier_label.lower()}.png"
        plt.savefig(waterfall_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"[OK] Saved {tier_label} risk waterfall plot: {waterfall_path.relative_to(BASE_DIR)}")

    return feat_imp


def wilson_score_interval(p: float, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """Calculate the Wilson score confidence interval for a binomial proportion."""
    if n == 0:
        return (0.0, 0.0)
    z = 1.95996  # 95% confidence
    denominator = 1 + z**2 / n
    centre_adjusted_probability = p + z**2 / (2 * n)
    adjusted_standard_deviation = np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n)
    lower_bound = (centre_adjusted_probability - z * adjusted_standard_deviation) / denominator
    upper_bound = (centre_adjusted_probability + z * adjusted_standard_deviation) / denominator
    return (max(0.0, float(lower_bound)), min(1.0, float(upper_bound)))


def analyze_track_b_tier_calibration():
    """
    Tune Track B risk tiers from out-of-fold training predictions and quantify
    the reliability of these cutoffs given small clinical sample sizes (768 and 297 rows).
    """
    print(f"\n{'=' * 80}")
    print("ANALYZING TRACK B RISK TIER CALIBRATION & SAMPLE RELIABILITY")
    print(f"{'=' * 80}")

    tasks = [
        ("diabetes", "models/all_models/track_b_clinical/track_b_diabetes_logistic_regression.joblib", "data/model_ready/track_b_diabetes_test.csv", "Outcome", 768),
        ("heart", "models/all_models/track_b_clinical/track_b_heart_logistic_regression.joblib", "data/model_ready/track_b_heart_test.csv", "condition", 297),
    ]

    tier_records = []
    for disease, joblib_rel, test_csv_rel, target_col, total_cohort_n in tasks:
        bundle = joblib.load(BASE_DIR / joblib_rel)
        oof_p = bundle["oof_probabilities"]
        y_train = bundle["y_train"]

        df_test = pd.read_csv(BASE_DIR / test_csv_rel)
        X_test = df_test.drop(columns=[target_col])
        y_test = df_test[target_col].astype(int).values
        calibrated_model = bundle["calibrated_model"]
        test_p = calibrated_model.predict_proba(X_test)[:, 1]

        # Load configured thresholds
        disease_key = "diabetes" if disease == "diabetes" else "heart_disease"
        tiers_cfg = load_yaml(CONFIG_DIR / "risk_tiers.yaml")
        cfg_low = tiers_cfg["tracks"]["track_b"][disease_key]["low_threshold"]
        cfg_high = tiers_cfg["tracks"]["track_b"][disease_key]["high_threshold"]

        # Validate on Train OOF
        oof_val = validate_risk_tier_monotonicity(y_train, oof_p, cfg_low, cfg_high)
        # Validate on Untouched Test Set
        test_val = validate_risk_tier_monotonicity(y_test, test_p, cfg_low, cfg_high)

        # Small sample precision analysis: Wilson CIs for observed rates on Test Set
        n_low_test = test_val["tier_counts"]["Low"]
        n_med_test = test_val["tier_counts"]["Medium"]
        n_high_test = test_val["tier_counts"]["High"]

        rate_low_test = test_val["observed_rates"]["Low"]
        rate_med_test = test_val["observed_rates"]["Medium"]
        rate_high_test = test_val["observed_rates"]["High"]

        ci_low = wilson_score_interval(rate_low_test, n_low_test)
        ci_med = wilson_score_interval(rate_med_test, n_med_test)
        ci_high = wilson_score_interval(rate_high_test, n_high_test)

        # Single patient impact percentage
        pt_impact_test = 100.0 / len(y_test)
        pos_impact_test = 100.0 / int(np.sum(y_test))

        print(f"\n--- Track B {disease.capitalize()} (Total Cohort N={total_cohort_n}, Test N={len(y_test)}) ---")
        print(f"Configured Thresholds: Low < {cfg_low:.3f}, Medium [{cfg_low:.3f}, {cfg_high:.3f}), High >= {cfg_high:.3f}")
        print(f"Train OOF Monotonic: {oof_val['is_monotonic']} | Rates: Low={oof_val['observed_rates']['Low']:.1%}, Med={oof_val['observed_rates']['Medium']:.1%}, High={oof_val['observed_rates']['High']:.1%}")
        print(f"Test Set Monotonic:  {test_val['is_monotonic']} | Rates: Low={rate_low_test:.1%}, Med={rate_med_test:.1%}, High={rate_high_test:.1%}")
        print(f"Test Wilson 95% CIs: Low [{ci_low[0]:.1%}, {ci_low[1]:.1%}], Med [{ci_med[0]:.1%}, {ci_med[1]:.1%}], High [{ci_high[0]:.1%}, {ci_high[1]:.1%}]")
        print(f"Small-Sample Impact: 1 patient shift alters test rate by {pt_impact_test:.2f}% (positive recall by {pos_impact_test:.2f}%)")

        tier_records.append({
            "Disease": disease,
            "Total_Cohort_N": total_cohort_n,
            "Train_N": len(y_train),
            "Test_N": len(y_test),
            "Low_Threshold": cfg_low,
            "High_Threshold": cfg_high,
            "Train_OOF_Monotonic": oof_val["is_monotonic"],
            "Train_OOF_Low_Rate": oof_val["observed_rates"]["Low"],
            "Train_OOF_Med_Rate": oof_val["observed_rates"]["Medium"],
            "Train_OOF_High_Rate": oof_val["observed_rates"]["High"],
            "Test_Monotonic": test_val["is_monotonic"],
            "Test_Low_Rate": rate_low_test,
            "Test_Med_Rate": rate_med_test,
            "Test_High_Rate": rate_high_test,
            "Test_Low_Rate_95CI": f"[{ci_low[0]:.3f}, {ci_low[1]:.3f}]",
            "Test_Med_Rate_95CI": f"[{ci_med[0]:.3f}, {ci_med[1]:.3f}]",
            "Test_High_Rate_95CI": f"[{ci_high[0]:.3f}, {ci_high[1]:.3f}]",
            "Test_High_Lift": test_val["high_tier_lift_vs_baseline"],
            "Test_Med_High_Positive_Capture_Pct": test_val["true_positives_captured_in_med_plus_high_pct"],
            "Single_Patient_Test_Pct": round(pt_impact_test, 2),
            "Single_Positive_Recall_Shift_Pct": round(pos_impact_test, 2),
            "Sample_Reliability_Verdict": "Cautious (Wide CIs due to small N; relies on CV)" if len(y_test) < 100 else "Moderate",
        })

    analysis_df = pd.DataFrame(tier_records)
    out_table = TABLES_DIR / "track_b_tier_calibration_analysis.csv"
    analysis_df.to_csv(out_table, index=False)
    print(f"\n[OK] Saved Track B risk tier calibration analysis: {out_table.relative_to(BASE_DIR)}")


def compute_track_b_cv_metrics_with_ci():
    """
    Compute repeated stratified 5-fold cross-validated metrics with 95% confidence intervals
    for Track B models to provide a stable evaluation that does not depend on a single small test split.
    """
    print(f"\n{'=' * 80}")
    print("COMPUTING TRACK B REPEATED CROSS-VALIDATION PERFORMANCE & 95% CIs")
    print(f"{'=' * 80}")

    tasks = [
        ("diabetes", "data/model_ready/track_b_diabetes_train.csv", "Outcome", "models/all_models/track_b_clinical/track_b_diabetes_logistic_regression.joblib"),
        ("heart", "data/model_ready/track_b_heart_train.csv", "condition", "models/all_models/track_b_clinical/track_b_heart_logistic_regression.joblib"),
    ]

    results = []
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=RANDOM_STATE)

    for disease, train_csv_rel, target_col, bundle_rel in tasks:
        bundle = joblib.load(BASE_DIR / bundle_rel)
        pipeline = bundle["pipeline"]
        feature_names = bundle["feature_names"]

        champ_csv = TABLES_DIR / "champion_models.csv"
        champions_df = pd.read_csv(champ_csv)
        champ_row = champions_df[(champions_df["Track"] == "track_b") & (champions_df["Disease"] == disease)]
        operating_threshold = float(champ_row.iloc[0]["Operating_Threshold"])

        df = pd.read_csv(BASE_DIR / train_csv_rel)
        X = df.drop(columns=[target_col])[feature_names].values
        y = df[target_col].astype(int).values

        recalls = []
        f1s = []
        roc_aucs = []
        pr_aucs = []
        precisions = []
        briers = []

        for train_idx, val_idx in rskf.split(X, y):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]

            # Fit clean pipeline clone
            pipeline_clone = joblib.load(BASE_DIR / bundle_rel)["pipeline"]
            pipeline_clone.fit(X_tr, y_tr)

            # Probabilities (logistic regression outputs calibrated sigmoid margins)
            probs = pipeline_clone.predict_proba(X_val)[:, 1]
            preds = (probs >= operating_threshold).astype(int)

            recalls.append(recall_score(y_val, preds, zero_division=0))
            f1s.append(f1_score(y_val, preds, zero_division=0))
            precisions.append(precision_score(y_val, preds, zero_division=0))
            roc_aucs.append(roc_auc_score(y_val, probs))
            pr_aucs.append(average_precision_score(y_val, probs))
            briers.append(brier_score_loss(y_val, probs))

        metrics_map = {
            "Recall": recalls,
            "F1": f1s,
            "Precision": precisions,
            "ROC_AUC": roc_aucs,
            "PR_AUC": pr_aucs,
            "Brier_Score": briers,
        }

        for metric_name, values in metrics_map.items():
            arr = np.array(values)
            mean_val = float(np.mean(arr))
            std_val = float(np.std(arr, ddof=1))
            n_evals = len(arr)
            se_val = std_val / np.sqrt(n_evals)
            ci_low = float(mean_val - 1.96 * se_val)
            ci_high = float(mean_val + 1.96 * se_val)
            pct_2_5 = float(np.percentile(arr, 2.5))
            pct_97_5 = float(np.percentile(arr, 97.5))

            results.append({
                "Track": "track_b",
                "Disease": disease,
                "Model": "logistic_regression",
                "Operating_Threshold": operating_threshold,
                "Metric": metric_name,
                "CV_Mean": round(mean_val, 4),
                "CV_Std": round(std_val, 4),
                "CV_SE": round(se_val, 4),
                "95%_CI_Normal": f"[{ci_low:.4f}, {ci_high:.4f}]",
                "95%_CI_Empirical": f"[{pct_2_5:.4f}, {pct_97_5:.4f}]",
                "N_Evaluations": n_evals,
            })

    cv_df = pd.DataFrame(results)
    out_table = TABLES_DIR / "track_b_cv_metrics_with_ci.csv"
    cv_df.to_csv(out_table, index=False)
    print(f"\n[OK] Saved Track B CV metrics with confidence intervals: {out_table.relative_to(BASE_DIR)}")
    print("\nSUMMARY OF TRACK B CROSS-VALIDATED METRICS (5-Fold x 5 Repeats):")
    print(cv_df[["Disease", "Metric", "CV_Mean", "CV_Std", "95%_CI_Normal"]].to_string(index=False))


def main():
    print("=" * 80)
    print("STARTING TRACK B SHAP & CLINICAL VALIDATION PIPELINE...")
    print("=" * 80)

    # 1. SHAP for Track B Diabetes (Pima)
    imp_diab = generate_track_b_shap(
        disease="diabetes",
        target_col="Outcome",
        train_csv="track_b_diabetes_train.csv",
        test_csv="track_b_diabetes_test.csv",
    )

    # 2. SHAP for Track B Heart (Cleveland)
    imp_heart = generate_track_b_shap(
        disease="heart",
        target_col="condition",
        train_csv="track_b_heart_train.csv",
        test_csv="track_b_heart_test.csv",
    )

    # 3. Export mean |SHAP| table
    combined_imp = pd.concat([imp_diab, imp_heart], ignore_index=True)
    table_path = TABLES_DIR / "shap_feature_importance_track_b.csv"
    combined_imp.to_csv(table_path, index=False)
    print(f"\n[OK] Saved Track B SHAP feature importances to: {table_path.relative_to(BASE_DIR)}")

    # 4. Out-of-Fold Risk Tier Calibration Analysis
    analyze_track_b_tier_calibration()

    # 5. Cross-Validated Metrics with Confidence Intervals
    compute_track_b_cv_metrics_with_ci()

    print("\n" + "=" * 80)
    print("TRACK B SHAP & CLINICAL VALIDATION COMPLETED SUCCESSFULLY.")
    print("=" * 80)


if __name__ == "__main__":
    main()
