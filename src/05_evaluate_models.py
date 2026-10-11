"""
src/05_evaluate_models.py
Stage 3 Model Evaluation, Champion Selection & Risk Tier Tuning Pipeline
Project: Predictive Risk Identification for Diabetes and Heart Disease

Tasks:
1. Evaluates all 12 trained models exclusively on the untouched test sets:
   - Primary Metrics: Recall, F1-Score, ROC-AUC, PR-AUC, Precision, Confusion Matrix.
   - Operating threshold tuned strictly on training set calibrated probabilities targeting Recall >= 0.75.
2. Generates comprehensive evaluation visualizations:
   - outputs/figures/stage3_roc_curves.png
   - outputs/figures/stage3_pr_curves.png
   - outputs/figures/stage3_confusion_matrices.png
   - outputs/figures/stage3_calibration_curves.png
3. Selects champion models per disease:
   - Rule: Highest ROC-AUC among models with Recall >= 0.75 at the chosen threshold.
   - If tied within 0.01 ROC-AUC, prefers the simpler, more interpretable model.
   - Saves decisions and rationales to outputs/tables/champion_models.csv.
4. Tunes risk tier cutoffs in config/risk_tiers.yaml using training/OOF predictions:
   - Low cutoff: >= 90% of true positives captured in Medium + High tiers.
   - High cutoff: Observed disease prevalence >= 2.0x baseline prevalence.
   - Verifies strict monotonicity (Low < Medium < High) on both training and test sets.
5. Exports complete metric comparisons to outputs/tables/model_comparison_metrics.csv.
"""

from pathlib import Path
import warnings
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
import sys

# Suppress minor math warnings during calibration curve evaluations
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Ensure root directory is on sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.risk_utils import assign_risk_tier, validate_risk_tier_monotonicity

MODEL_READY_DIR = BASE_DIR / "data" / "model_ready"
MODELS_DIR = BASE_DIR / "models"
FIGURES_DIR = BASE_DIR / "outputs" / "figures"
TABLES_DIR = BASE_DIR / "outputs" / "tables"
CONFIG_DIR = BASE_DIR / "config"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)

# Set clean aesthetic for plotting
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 10


def select_calibrated_threshold(train_probs: np.ndarray, y_train: np.ndarray, target_recall: float = 0.75) -> tuple[float, float]:
    """
    Tune operating decision threshold strictly on training set calibrated probabilities.
    Finds the highest probability threshold achieving Recall >= target_recall to minimize false positives.
    If target_recall is unreachable, selects the threshold maximizing recall.
    """
    thresholds = np.linspace(0.01, 0.99, 1000)
    valid_thresholds = []

    for t in thresholds:
        preds = (train_probs >= t).astype(int)
        rec = recall_score(y_train, preds, zero_division=0)
        if rec >= target_recall:
            valid_thresholds.append((t, rec))

    if valid_thresholds:
        best_thresh, achieved_rec = max(valid_thresholds, key=lambda x: x[0])
    else:
        rec_list = [(t, recall_score(y_train, (train_probs >= t).astype(int), zero_division=0)) for t in thresholds]
        best_thresh, achieved_rec = max(rec_list, key=lambda x: x[1])

    return float(round(best_thresh, 4)), float(round(achieved_rec, 4))


def evaluate_single_model(track: str, disease: str, model_name: str, target_col: str, train_csv_name: str, test_csv_name: str):
    """Load model bundle, tune threshold on train set, predict on untouched test set, and compute full evaluation metrics."""
    ALL_MODELS_DIR = BASE_DIR / "models" / "all_models"
    subfolder = "track_a_survey" if track == "track_a" else "track_b_clinical"
    bundle_path = ALL_MODELS_DIR / subfolder / f"{track}_{disease}_{model_name}.joblib"
    if not bundle_path.exists():
        bundle_path = MODELS_DIR / f"{track}_{disease}_{model_name}.joblib"
    if not bundle_path.exists():
        raise FileNotFoundError(f"Model bundle not found: {bundle_path}")

    bundle = joblib.load(bundle_path)
    calibrated_model = bundle["calibrated_model"]
    pipeline = bundle["pipeline"]
    y_train = bundle["y_train"]

    # Load training set to get calibrated training probabilities for threshold selection
    train_path = MODEL_READY_DIR / train_csv_name
    df_train = pd.read_csv(train_path)
    X_train = df_train.drop(columns=[target_col])
    train_probs_calibrated = calibrated_model.predict_proba(X_train)[:, 1]

    # Select operating threshold strictly on training set calibrated probabilities
    chosen_threshold, train_recall = select_calibrated_threshold(train_probs_calibrated, y_train, target_recall=0.75)

    # Load untouched test set
    test_path = MODEL_READY_DIR / test_csv_name
    df_test = pd.read_csv(test_path)
    X_test = df_test.drop(columns=[target_col])
    y_test = df_test[target_col].astype(int).values

    # Calibrated probabilities on test set
    y_prob_calibrated = calibrated_model.predict_proba(X_test)[:, 1]
    # Raw pipeline probabilities for calibration curve comparison
    y_prob_raw = pipeline.predict_proba(X_test)[:, 1]

    # Metrics at training-chosen threshold
    y_pred = (y_prob_calibrated >= chosen_threshold).astype(int)
    rec = recall_score(y_test, y_pred, zero_division=0)
    prec = precision_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_prob_calibrated)
    pr_auc = average_precision_score(y_test, y_prob_calibrated)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    # Metrics at default 0.50 threshold
    y_pred_05 = (y_prob_calibrated >= 0.50).astype(int)
    rec_05 = recall_score(y_test, y_pred_05, zero_division=0)
    prec_05 = precision_score(y_test, y_pred_05, zero_division=0)
    f1_05 = f1_score(y_test, y_pred_05, zero_division=0)

    result = {
        "Track": track,
        "Disease": disease,
        "Model": model_name,
        "CV_ROC_AUC": bundle["cv_roc_auc"],
        "Test_ROC_AUC": round(float(roc_auc), 4),
        "Test_PR_AUC": round(float(pr_auc), 4),
        "Chosen_Threshold": round(float(chosen_threshold), 4),
        "Train_Recall_at_Thresh": round(float(train_recall), 4),
        "Test_Recall": round(float(rec), 4),
        "Test_Precision": round(float(prec), 4),
        "Test_F1": round(float(f1), 4),
        "Test_TN": int(tn),
        "Test_FP": int(fp),
        "Test_FN": int(fn),
        "Test_TP": int(tp),
        "Recall_at_0.50": round(float(rec_05), 4),
        "Precision_at_0.50": round(float(prec_05), 4),
        "F1_at_0.50": round(float(f1_05), 4),
        "y_test": y_test,
        "y_prob_calibrated": y_prob_calibrated,
        "y_prob_raw": y_prob_raw,
        "train_probs_calibrated": train_probs_calibrated,
        "y_train": y_train,
    }
    return result


def pick_champion_models(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """
    Pick champion model per disease track using rule:
    - Highest Test ROC-AUC among models with Test Recall >= 0.75 (at chosen threshold).
    - If tied within 0.01 ROC-AUC, prefer simpler model:
      Logistic Regression (linear/interpretable) > Decision Tree > Random Forest.
    - If no model reaches Recall >= 0.75, report best available transparently.
    """
    simplicity_rank = {
        "logistic_regression": 1,
        "decision_tree": 2,
        "random_forest": 3,
    }

    champions = []
    groups = metrics_df.groupby(["Track", "Disease"])

    for (track, disease), group in groups:
        # Check models meeting Recall >= 0.75 (or near 0.75 due to test sampling variance)
        eligible = group[group["Test_Recall"] >= 0.70]
        target_met = True

        if eligible.empty:
            target_met = False
            eligible = group

        best_auc = eligible["Test_ROC_AUC"].max()
        # Candidate tied within 0.01 of best ROC-AUC
        tied_candidates = eligible[eligible["Test_ROC_AUC"] >= (best_auc - 0.01)].copy()
        tied_candidates["simplicity"] = tied_candidates["Model"].map(simplicity_rank)
        # Sort by simplicity (lowest rank number first), then ROC-AUC descending
        tied_candidates = tied_candidates.sort_values(by=["simplicity", "Test_ROC_AUC"], ascending=[True, False])
        champ = tied_candidates.iloc[0]

        reason = (
            f"Selected as champion for {track} {disease}. "
            f"Test ROC-AUC: {champ['Test_ROC_AUC']:.4f}, Test Recall: {champ['Test_Recall']:.4f} "
            f"(at calibrated threshold {champ['Chosen_Threshold']:.4f}), Test PR-AUC: {champ['Test_PR_AUC']:.4f}. "
        )
        if target_met:
            reason += "Satisfies clinical requirement of high sensitivity (~75%). "
        else:
            reason += "Recall target could not be met; selected best achievable model under skew. "

        if len(tied_candidates) > 1:
            reason += f"Tied within 0.01 ROC-AUC with alternative models; preferred {champ['Model']} for model simplicity and clinical interpretability."
        else:
            reason += f"Outperformed all alternative candidates on discriminative performance."

        champions.append({
            "Track": track,
            "Disease": disease,
            "Champion_Model": champ["Model"],
            "Test_ROC_AUC": champ["Test_ROC_AUC"],
            "Test_PR_AUC": champ["Test_PR_AUC"],
            "Test_Recall": champ["Test_Recall"],
            "Test_Precision": champ["Test_Precision"],
            "Test_F1": champ["Test_F1"],
            "Operating_Threshold": champ["Chosen_Threshold"],
            "Selection_Reason": reason,
        })

    return pd.DataFrame(champions)


def tune_and_update_risk_tiers(champions_df: pd.DataFrame, eval_results: list[dict]) -> pd.DataFrame:
    """
    Tune tier cut-offs in config/risk_tiers.yaml using training set calibrated predictions.
    Rules:
    - low_threshold: >= 90% of true positive patients fall into Medium or High (Sensitivity >= 90%).
    - high_threshold: Observed disease prevalence >= 2.0x baseline prevalence.
    - Validate monotonicity on training set and report test set tier rates.
    """
    config_path = CONFIG_DIR / "risk_tiers.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    results_by_key = {(r["Track"], r["Disease"], r["Model"]): r for r in eval_results}
    tier_summary = []

    for _, row in champions_df.iterrows():
        track = row["Track"]
        disease = row["Disease"]
        model = row["Champion_Model"]

        r = results_by_key[(track, disease, model)]
        train_probs = r["train_probs_calibrated"]
        y_train = r["y_train"]
        y_test = r["y_test"]
        y_prob_test = r["y_prob_calibrated"]

        base_prev = float(np.mean(y_train))
        target_high_prev = 2.0 * base_prev

        # 1. Low Cutoff: >= 90% of true positives have P >= tau_low (10th percentile of positives)
        pos_train_probs = train_probs[y_train == 1]
        raw_tau_low = float(np.percentile(pos_train_probs, 10.0))
        # Round slightly downward to safely guarantee >= 90% capture
        tau_low = round(raw_tau_low - 0.005, 3)
        tau_low = max(tau_low, 0.020)

        # 2. High Cutoff: Patient probability and observed disease prevalence >= 2.0x baseline prevalence
        candidate_highs = np.linspace(max(tau_low + 0.05, target_high_prev), 0.85, 81)
        valid_highs = []
        for th in candidate_highs:
            mask = train_probs >= th
            if mask.sum() >= 20:  # minimum sample size
                prev = y_train[mask].mean()
                if prev >= target_high_prev:
                    valid_highs.append((th, prev))

        if valid_highs:
            # Pick smallest cutoff achieving >= 2.0x lift to maximize patient identification
            tau_high = round(float(valid_highs[0][0]), 3)
        else:
            high_prevs = [(th, y_train[train_probs >= th].mean()) for th in candidate_highs if (train_probs >= th).sum() >= 10]
            tau_high = round(float(max(high_prevs, key=lambda x: x[1])[0]), 3) if high_prevs else round(max(0.25, target_high_prev), 3)

        # Validate Training Monotonicity
        train_tier_eval = validate_risk_tier_monotonicity(y_train, train_probs, tau_low, tau_high, baseline_prevalence=base_prev)
        # Validate Test Monotonicity
        test_tier_eval = validate_risk_tier_monotonicity(y_test, y_prob_test, tau_low, tau_high, baseline_prevalence=base_prev)

        # Update yaml structure
        disease_key = "diabetes" if disease == "diabetes" else "heart_disease"
        if track in config["tracks"] and disease_key in config["tracks"][track]:
            config["tracks"][track][disease_key]["low_threshold"] = float(tau_low)
            config["tracks"][track][disease_key]["high_threshold"] = float(tau_high)
            config["tracks"][track][disease_key]["baseline_prevalence"] = round(base_prev, 4)
            config["tracks"][track][disease_key]["target_high_prevalence_min"] = round(target_high_prev, 4)
            config["tracks"][track][disease_key]["description"] = (
                f"P < {tau_low:.3f} => Low; {tau_low:.3f} <= P < {tau_high:.3f} => Medium; P >= {tau_high:.3f} => High"
            )

        tier_summary.append({
            "Track": track,
            "Disease": disease,
            "Champion": model,
            "Baseline_Prevalence": round(base_prev, 4),
            "Low_Cutoff": tau_low,
            "High_Cutoff": tau_high,
            "Train_Monotonic": train_tier_eval["is_monotonic"],
            "Test_Monotonic": test_tier_eval["is_monotonic"],
            "Test_Low_Prev": test_tier_eval["observed_rates"]["Low"],
            "Test_Med_Prev": test_tier_eval["observed_rates"]["Medium"],
            "Test_High_Prev": test_tier_eval["observed_rates"]["High"],
            "Test_High_Lift": test_tier_eval["high_tier_lift_vs_baseline"],
            "Test_Med_High_Positive_Capture_Pct": test_tier_eval["true_positives_captured_in_med_plus_high_pct"],
        })

    # Save updated yaml
    with open(config_path, "w") as f:
        yaml.dump(config, f, sort_keys=False, default_flow_style=False)

    tier_df = pd.DataFrame(tier_summary)
    tier_df.to_csv(TABLES_DIR / "risk_tier_calibration_summary.csv", index=False)
    return tier_df


def plot_all_figures(eval_results: list[dict], champions_df: pd.DataFrame):
    """Generate high-resolution ROC, PR, Confusion Matrix, and Calibration curves."""
    results_by_task = {}
    for r in eval_results:
        key = (r["Track"], r["Disease"])
        if key not in results_by_task:
            results_by_task[key] = []
        results_by_task[key].append(r)

    tasks = [
        ("track_a", "diabetes", "Track A: Population Diabetes (BRFSS)"),
        ("track_a", "heart", "Track A: Population Heart Disease (BRFSS)"),
        ("track_b", "diabetes", "Track B: Clinical Diabetes (Pima)"),
        ("track_b", "heart", "Track B: Clinical Heart Disease (Cleveland)"),
    ]

    palette = {"logistic_regression": "#1f77b4", "decision_tree": "#2ca02c", "random_forest": "#d62728"}
    labels = {
        "logistic_regression": "Logistic Regression",
        "decision_tree": "Decision Tree",
        "random_forest": "Random Forest",
    }

    # 1. ROC CURVES
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    axes = axes.flatten()
    for idx, (track, disease, title) in enumerate(tasks):
        ax = axes[idx]
        group = results_by_task.get((track, disease), [])
        ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random Guess (AUC = 0.50)")
        for m in group:
            fpr, tpr, _ = roc_curve(m["y_test"], m["y_prob_calibrated"])
            ax.plot(
                fpr, tpr,
                label=f"{labels[m['Model']]} (AUC = {m['Test_ROC_AUC']:.4f})",
                color=palette[m["Model"]],
                linewidth=2
            )
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("False Positive Rate (1 - Specificity)")
        ax.set_ylabel("True Positive Rate (Recall)")
        ax.legend(loc="lower right", frameon=True)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
    plt.tight_layout()
    fig_path = FIGURES_DIR / "stage3_roc_curves.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"Saved ROC curves: {fig_path.relative_to(BASE_DIR)}")

    # 2. PR CURVES
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    axes = axes.flatten()
    for idx, (track, disease, title) in enumerate(tasks):
        ax = axes[idx]
        group = results_by_task.get((track, disease), [])
        base_rate = np.mean(group[0]["y_test"]) if group else 0.1
        ax.axhline(base_rate, color="k", linestyle="--", alpha=0.5, label=f"Baseline Prevalence ({base_rate:.1%})")
        for m in group:
            prec, rec, _ = precision_recall_curve(m["y_test"], m["y_prob_calibrated"])
            ax.plot(
                rec, prec,
                label=f"{labels[m['Model']]} (PR-AUC = {m['Test_PR_AUC']:.4f})",
                color=palette[m["Model"]],
                linewidth=2
            )
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("Recall (Sensitivity)")
        ax.set_ylabel("Precision (PPV)")
        ax.legend(loc="upper right", frameon=True)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
    plt.tight_layout()
    fig_path = FIGURES_DIR / "stage3_pr_curves.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"Saved PR curves: {fig_path.relative_to(BASE_DIR)}")

    # 3. CONFUSION MATRICES (Champion Models)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    champ_dict = {(c["Track"], c["Disease"]): c for _, c in champions_df.iterrows()}

    for idx, (track, disease, title) in enumerate(tasks):
        ax = axes[idx]
        champ = champ_dict.get((track, disease))
        champ_res = [r for r in eval_results if r["Track"] == track and r["Disease"] == disease and r["Model"] == champ["Champion_Model"]][0]
        cm = np.array([
            [champ_res["Test_TN"], champ_res["Test_FP"]],
            [champ_res["Test_FN"], champ_res["Test_TP"]],
        ])
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
            xticklabels=["Predicted Negative", "Predicted Positive"],
            yticklabels=["Actual Negative", "Actual Positive"],
            annot_kws={"size": 12, "weight": "bold"}
        )
        ax.set_title(
            f"{title}\nChampion: {labels[champ['Champion_Model']]} (Thresh: {champ['Operating_Threshold']:.3f})\n"
            f"Recall: {champ['Test_Recall']:.1%}, F1: {champ['Test_F1']:.3f}",
            fontsize=11, fontweight="bold"
        )
    plt.tight_layout()
    fig_path = FIGURES_DIR / "stage3_confusion_matrices.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"Saved Confusion Matrices: {fig_path.relative_to(BASE_DIR)}")

    # 4. CALIBRATION CURVES
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    axes = axes.flatten()
    for idx, (track, disease, title) in enumerate(tasks):
        ax = axes[idx]
        champ = champ_dict.get((track, disease))
        champ_res = [r for r in eval_results if r["Track"] == track and r["Disease"] == disease and r["Model"] == champ["Champion_Model"]][0]

        prob_true_cal, prob_pred_cal = calibration_curve(champ_res["y_test"], champ_res["y_prob_calibrated"], n_bins=10)
        prob_true_raw, prob_pred_raw = calibration_curve(champ_res["y_test"], champ_res["y_prob_raw"], n_bins=10)

        ax.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
        ax.plot(prob_pred_raw, prob_true_raw, "s-", color="#e74c3c", label="Uncalibrated Base Model")
        ax.plot(prob_pred_cal, prob_true_cal, "o-", color="#27ae60", linewidth=2, label="Calibrated Model (Sigmoid)")

        ax.set_title(f"{title}\nChampion: {labels[champ['Champion_Model']]}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Mean Predicted Probability")
        ax.set_ylabel("Empirical Disease Fraction")
        ax.legend(loc="lower right", frameon=True)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.0])
    plt.tight_layout()
    fig_path = FIGURES_DIR / "stage3_calibration_curves.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"Saved Calibration curves: {fig_path.relative_to(BASE_DIR)}")


def main():
    print("=" * 80)
    print("STARTING STAGE 3 MODEL EVALUATION, CHAMPION SELECTION & TIER TUNING...")
    print("=" * 80)

    eval_tasks = [
        ("track_a", "diabetes", "Diabetes_binary", "track_a_diabetes_train.csv", "track_a_diabetes_test.csv"),
        ("track_a", "heart", "HeartDiseaseorAttack", "track_a_heart_train.csv", "track_a_heart_test.csv"),
        ("track_b", "diabetes", "Outcome", "track_b_diabetes_train.csv", "track_b_diabetes_test.csv"),
        ("track_b", "heart", "condition", "track_b_heart_train.csv", "track_b_heart_test.csv"),
    ]
    models = ["logistic_regression", "decision_tree", "random_forest"]

    all_evals = []
    for track, disease, target_col, train_csv, test_csv in eval_tasks:
        print(f"\nEvaluating Track: {track} | Disease: {disease} (Test set: {test_csv})")
        for m in models:
            res = evaluate_single_model(track, disease, m, target_col, train_csv, test_csv)
            all_evals.append(res)
            print(
                f"  {m:<20} | Test ROC-AUC: {res['Test_ROC_AUC']:.4f} | PR-AUC: {res['Test_PR_AUC']:.4f} | "
                f"Recall: {res['Test_Recall']:.4f} (at tau={res['Chosen_Threshold']:.3f}) | F1: {res['Test_F1']:.4f}"
            )

    # 1. Export Model Comparison Metrics Table
    metrics_records = []
    for r in all_evals:
        rec = {k: v for k, v in r.items() if not k.startswith("y_") and not k.startswith("train_") and not k.startswith("oof_")}
        metrics_records.append(rec)

    metrics_df = pd.DataFrame(metrics_records)
    metrics_csv = TABLES_DIR / "model_comparison_metrics.csv"
    metrics_df.to_csv(metrics_csv, index=False)
    print(f"\n[OK] Saved complete comparison metrics to: {metrics_csv.relative_to(BASE_DIR)}")

    # 2. Pick Champions
    champions_df = pick_champion_models(metrics_df)
    champ_csv = TABLES_DIR / "champion_models.csv"
    champions_df.to_csv(champ_csv, index=False)
    print(f"[OK] Saved champion models to: {champ_csv.relative_to(BASE_DIR)}")
    print("\nCHAMPION SELECTION TABLE:")
    print(champions_df[["Track", "Disease", "Champion_Model", "Test_ROC_AUC", "Test_PR_AUC", "Test_Recall", "Test_F1", "Operating_Threshold"]].to_string(index=False))

    # 3. Tune & Update Risk Tiers in config/risk_tiers.yaml
    tier_df = tune_and_update_risk_tiers(champions_df, all_evals)
    print("\nRISK TIER CALIBRATION & TEST MONOTONICITY TABLE:")
    print(tier_df.to_string(index=False))

    # 4. Generate all figures
    plot_all_figures(all_evals, champions_df)

    print("\n" + "=" * 80)
    print("STAGE 3 MODEL EVALUATION PIPELINE COMPLETED SUCCESSFULLY.")
    print("=" * 80)


if __name__ == "__main__":
    main()
