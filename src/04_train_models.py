"""
src/04_train_models.py
Stage 3 Model Training & Hyperparameter Tuning Pipeline
Project: Predictive Risk Identification for Diabetes and Heart Disease

Tasks:
1. Loads model-ready training splits from data/model_ready/.
2. For each track and disease, trains three candidate model families inside scikit-learn Pipelines:
   - Logistic Regression (with StandardScaler & L2 regularization)
   - Decision Tree (depth-controlled to avoid overfitting)
   - Random Forest (ensemble with balanced sub-sampling)
3. Performs GridSearchCV with Stratified 5-Fold CV on TRAINING DATA ONLY (scoring='roc_auc').
4. Generates out-of-fold (OOF) cross-validation predicted probabilities on X_train for threshold tuning.
5. Calibrates probabilities via CalibratedClassifierCV for trustworthy risk tier mapping.
6. Serializes model bundles to models/{track}_{disease}_{model}.joblib.
"""

from pathlib import Path
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_READY_DIR = BASE_DIR / "data" / "model_ready"
ALL_MODELS_DIR = BASE_DIR / "models" / "all_models"
(ALL_MODELS_DIR / "track_a_survey").mkdir(parents=True, exist_ok=True)
(ALL_MODELS_DIR / "track_b_clinical").mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42


def get_model_specs(track: str):
    """Return model pipelines and hyperparameter grids tailored to dataset scale."""
    is_small = track.startswith("track_b")

    specs = {
        "logistic_regression": {
            "pipeline": Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    max_iter=1000,
                    solver="lbfgs"
                )),
            ]),
            "param_grid": {
                "clf__C": [0.01, 0.1, 1.0, 10.0],
            },
        },
        "decision_tree": {
            "pipeline": Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("clf", DecisionTreeClassifier(
                    class_weight="balanced",
                    random_state=RANDOM_STATE
                )),
            ]),
            "param_grid": {
                "clf__max_depth": [3, 4, 5] if is_small else [5, 8, 12],
                "clf__min_samples_split": [10, 20] if is_small else [50, 100],
            },
        },
        "random_forest": {
            "pipeline": Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("clf", RandomForestClassifier(
                    n_estimators=100,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    n_jobs=-1
                )),
            ]),
            "param_grid": {
                "clf__max_depth": [3, 5] if is_small else [8, 12],
                "clf__min_samples_split": [10, 20] if is_small else [50, 100],
                "clf__max_features": ["sqrt"],
            },
        },
    }
    return specs


def train_track_disease_models(track: str, disease: str, train_csv_name: str, target_col: str):
    """Train, tune, calibrate, and save all candidate models for a given dataset track and disease."""
    csv_path = MODEL_READY_DIR / train_csv_name
    print(f"\n{'=' * 80}")
    print(f"TRAINING MODELS: {track.upper()} - {disease.upper()} (Target: {target_col})")
    print(f"Source Training File: {csv_path.relative_to(BASE_DIR)}")
    print(f"{'=' * 80}")

    df_train = pd.read_csv(csv_path)
    X_train = df_train.drop(columns=[target_col])
    y_train = df_train[target_col].astype(int)

    feature_names = X_train.columns.tolist()
    n_samples, n_features = X_train.shape
    pos_rate = float(y_train.mean() * 100)
    print(f"Training Matrix: {n_samples:,} rows x {n_features} features | Positive Prevalence: {pos_rate:.2f}%")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    model_specs = get_model_specs(track)
    track_results = []

    for model_name, spec in model_specs.items():
        t0 = time.time()
        print(f"\n--- Tuning {model_name} (5-Fold Stratified CV, scoring='roc_auc') ---")

        grid = GridSearchCV(
            estimator=spec["pipeline"],
            param_grid=spec["param_grid"],
            cv=cv,
            scoring="roc_auc",
            n_jobs=-1,
            refit=True
        )
        grid.fit(X_train, y_train)
        best_pipe = grid.best_estimator_
        best_auc = grid.best_score_
        elapsed = time.time() - t0

        print(f"  Best CV ROC-AUC: {best_auc:.4f} in {elapsed:.2f}s")
        print(f"  Best Parameters: {grid.best_params_}")

        # Out-of-Fold (OOF) predicted probabilities on X_train for risk tier tuning
        t_oof = time.time()
        oof_probs = cross_val_predict(
            best_pipe,
            X_train,
            y_train,
            cv=cv,
            method="predict_proba",
            n_jobs=-1
        )[:, 1]
        print(f"  OOF predictions generated in {time.time() - t_oof:.2f}s")

        # Probability Calibration via CalibratedClassifierCV
        t_cal = time.time()
        calibrated_model = CalibratedClassifierCV(
            estimator=best_pipe,
            cv=3,
            method="sigmoid"
        )
        calibrated_model.fit(X_train, y_train)
        print(f"  Sigmoid calibration fitted in {time.time() - t_cal:.2f}s")

        # Save model bundle
        bundle = {
            "pipeline": best_pipe,
            "calibrated_model": calibrated_model,
            "best_params": grid.best_params_,
            "cv_roc_auc": round(float(best_auc), 4),
            "oof_probabilities": oof_probs,
            "y_train": y_train.values,
            "feature_names": feature_names,
            "target_col": target_col,
            "track": track,
            "disease": disease,
            "model_name": model_name,
        }

        subfolder = "track_a_survey" if track == "track_a" else "track_b_clinical"
        save_path = ALL_MODELS_DIR / subfolder / f"{track}_{disease}_{model_name}.joblib"
        joblib.dump(bundle, save_path)
        print(f"  Saved model bundle: {save_path.relative_to(BASE_DIR)}")

        track_results.append({
            "Track": track,
            "Disease": disease,
            "Model": model_name,
            "CV_ROC_AUC": round(float(best_auc), 4),
            "Tuning_Time_Sec": round(elapsed, 2),
            "Best_Params": str(grid.best_params_),
        })

    return track_results


def main():
    start_total = time.time()
    print("STARTING STAGE 3 COMPREHENSIVE MODEL TRAINING & TUNING...")

    tasks = [
        # Track A: Primary Population Survey Models
        ("track_a", "diabetes", "track_a_diabetes_train.csv", "Diabetes_binary"),
        ("track_a", "heart", "track_a_heart_train.csv", "HeartDiseaseorAttack"),
        # Track B: Clinical Validation Benchmark Models
        ("track_b", "diabetes", "track_b_diabetes_train.csv", "Outcome"),
        ("track_b", "heart", "track_b_heart_train.csv", "condition"),
    ]

    all_results = []
    for track, disease, csv_file, target in tasks:
        res = train_track_disease_models(track, disease, csv_file, target)
        all_results.extend(res)

    summary_df = pd.DataFrame(all_results)
    print("\n" + "=" * 80)
    print("STAGE 3 MODEL TRAINING & 5-FOLD CV ROC-AUC SUMMARY")
    print("=" * 80)
    print(summary_df.to_string(index=False))

    total_time = time.time() - start_total
    print(f"\nALL 12 MODELS TRAINED, TUNED, CALIBRATED, AND SAVED IN {total_time:.2f} SECONDS.")


if __name__ == "__main__":
    main()
