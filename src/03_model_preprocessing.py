"""
src/03_model_preprocessing.py
Stage 3 Feature Matrix & Stratified Split Generation Pipeline
Project: Predictive Risk Identification for Diabetes and Heart Disease

Tasks:
1. Loads clean data for Track A (BRFSS 2015) and Track B (Pima Diabetes & Cleveland Heart).
2. Applies strict clinical leakage rules:
   - Excludes banned features ('high_glucose')
   - Checks feature-target correlations and asserts no |r| > 0.90
   - Enforces default cross-disease target exclusion (Diabetes does not predict Heart Disease, and vice-versa)
3. Enforces biological zero auditing on Pima (impossible zeros converted to NaN; Pregnancies zeros kept).
4. Performs Stratified 80/20 train/test splits with random_state=42.
5. Exports model-ready datasets and split indices to data/model_ready/.
6. Emits comprehensive summary tables of real dataset dimensions and positive class rates.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

# Enforce deterministic reproducibility
RANDOM_STATE = 42
TEST_SIZE = 0.20

BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_DIR = BASE_DIR / "config"
CLEAN_DIR = BASE_DIR / "data" / "clean"
MODEL_READY_DIR = BASE_DIR / "data" / "model_ready"

MODEL_READY_DIR.mkdir(parents=True, exist_ok=True)


def load_config() -> dict:
    """Load model configuration file."""
    cfg_path = CONFIG_DIR / "model_config.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def audit_correlations_and_leakage(df: pd.DataFrame, target_col: str, threshold: float = 0.90) -> pd.Series:
    """
    Compute correlations between all features and the target.
    Raises ValueError if any feature exhibits |r| > threshold (leakage indicator).
    """
    corrs = df.corr(numeric_only=True)[target_col].drop(index=[target_col], errors="ignore")
    leaky_features = corrs[corrs.abs() > threshold]
    if not leaky_features.empty:
        raise ValueError(
            f"CRITICAL LEAKAGE DETECTED: Features {leaky_features.to_dict()} exceed |correlation| > {threshold} with target '{target_col}'!"
        )
    return corrs


def prepare_track_a(config: dict):
    """
    Prepare Track A (CDC BRFSS Population Survey Track).
    Produces two distinct datasets:
    1. Track A Diabetes (Target: Diabetes_binary)
    2. Track A Heart Disease (Target: HeartDiseaseorAttack)
    """
    print("\n" + "=" * 80)
    print("PREPARING TRACK A: CDC BRFSS POPULATION SURVEY TRACK")
    print("=" * 80)

    brfss_path = CLEAN_DIR / "brfss_clean.csv"
    df = pd.read_csv(brfss_path)
    print(f"Loaded raw clean BRFSS: {df.shape} ({len(df):,} rows, {len(df.columns)} columns)")

    # Assert banned feature is absent
    for banned in config["leakage_safeguards"]["banned_features"]:
        assert banned not in df.columns, f"Banned feature '{banned}' found in BRFSS dataset!"

    cross_disease_flag = config["leakage_safeguards"]["include_cross_disease_target"]
    print(f"Leakage Policy: include_cross_disease_target = {cross_disease_flag}")

    # -------------------------------------------------------------
    # Track A - Diabetes Model
    # -------------------------------------------------------------
    target_diab = config["tracks"]["track_a"]["diabetes"]["target_column"]
    exclude_diab = [target_diab]
    if not cross_disease_flag:
        exclude_diab.append("HeartDiseaseorAttack")

    # Audit correlations before dropping
    corrs_diab = audit_correlations_and_leakage(df, target_diab, config["leakage_safeguards"]["correlation_threshold"])
    top_diab = corrs_diab.abs().sort_values(ascending=False).head(5)
    print(f"\nTrack A Diabetes - Top feature correlations with '{target_diab}':")
    for feat, val in top_diab.items():
        print(f"  {feat:<22} r = {corrs_diab[feat]:+.4f}")

    X_diab = df.drop(columns=exclude_diab, errors="ignore")
    y_diab = df[target_diab].astype(int)

    X_diab_train, X_diab_test, y_diab_train, y_diab_test = train_test_split(
        X_diab, y_diab, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_diab
    )

    # Save CSVs
    train_diab = pd.concat([X_diab_train, y_diab_train], axis=1)
    test_diab = pd.concat([X_diab_test, y_diab_test], axis=1)
    train_diab.to_csv(MODEL_READY_DIR / "track_a_diabetes_train.csv", index=False)
    test_diab.to_csv(MODEL_READY_DIR / "track_a_diabetes_test.csv", index=False)

    # -------------------------------------------------------------
    # Track A - Heart Disease Model
    # -------------------------------------------------------------
    target_heart = config["tracks"]["track_a"]["heart_disease"]["target_column"]
    exclude_heart = [target_heart]
    if not cross_disease_flag:
        exclude_heart.append("Diabetes_binary")

    corrs_heart = audit_correlations_and_leakage(df, target_heart, config["leakage_safeguards"]["correlation_threshold"])
    top_heart = corrs_heart.abs().sort_values(ascending=False).head(5)
    print(f"\nTrack A Heart Disease - Top feature correlations with '{target_heart}':")
    for feat, val in top_heart.items():
        print(f"  {feat:<22} r = {corrs_heart[feat]:+.4f}")

    X_heart = df.drop(columns=exclude_heart, errors="ignore")
    y_heart = df[target_heart].astype(int)

    X_heart_train, X_heart_test, y_heart_train, y_heart_test = train_test_split(
        X_heart, y_heart, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_heart
    )

    train_heart = pd.concat([X_heart_train, y_heart_train], axis=1)
    test_heart = pd.concat([X_heart_test, y_heart_test], axis=1)
    train_heart.to_csv(MODEL_READY_DIR / "track_a_heart_train.csv", index=False)
    test_heart.to_csv(MODEL_READY_DIR / "track_a_heart_test.csv", index=False)

    return {
        "track_a_diabetes": {
            "X_train": X_diab_train,
            "X_test": X_diab_test,
            "y_train": y_diab_train,
            "y_test": y_diab_test,
            "train_indices": X_diab_train.index.tolist(),
            "test_indices": X_diab_test.index.tolist(),
        },
        "track_a_heart": {
            "X_train": X_heart_train,
            "X_test": X_heart_test,
            "y_train": y_heart_train,
            "y_test": y_heart_test,
            "train_indices": X_heart_train.index.tolist(),
            "test_indices": X_heart_test.index.tolist(),
        },
    }


def prepare_track_b(config: dict):
    """
    Prepare Track B (Clinical Lab Benchmark Track).
    Produces:
    1. Track B Diabetes (UCI Pima Indians, target: Outcome)
    2. Track B Heart Disease (UCI Cleveland, target: condition)
    """
    print("\n" + "=" * 80)
    print("PREPARING TRACK B: CLINICAL BENCHMARK TRACK")
    print("=" * 80)

    # -------------------------------------------------------------
    # Track B - Pima Indians Diabetes
    # -------------------------------------------------------------
    pima_path = CLEAN_DIR / "pima_clean.csv"
    pima_df = pd.read_csv(pima_path)
    pima_target = config["tracks"]["track_b"]["diabetes"]["target_column"]

    # Verify impossible zeros are NaN and Pregnancies zero is intact
    zero_as_nan = config["tracks"]["track_b"]["diabetes"]["zero_as_nan_features"]
    for col in zero_as_nan:
        zero_cnt = (pima_df[col] == 0).sum()
        if zero_cnt > 0:
            print(f"Warning: Found {zero_cnt} zeros in Pima '{col}'. Replacing with NaN...")
            pima_df[col] = pima_df[col].replace(0, np.nan)

    nulls_pima = pima_df[zero_as_nan].isnull().sum()
    print(f"Pima Indians ({len(pima_df)} rows) - Verified NaN counts in lab features:")
    for col, n in nulls_pima.items():
        print(f"  {col:<18} NaNs: {n:<4} ({n/len(pima_df)*100:5.2f}%)")

    # Nulliparous verification
    nulliparous_zeros = int((pima_df["Pregnancies"] == 0).sum())
    print(f"  Pregnancies (Valid Zeros): {nulliparous_zeros} ({nulliparous_zeros/len(pima_df)*100:.2f}%)")

    corrs_pima = audit_correlations_and_leakage(pima_df, pima_target, config["leakage_safeguards"]["correlation_threshold"])
    X_pima = pima_df.drop(columns=[pima_target])
    y_pima = pima_df[pima_target].astype(int)

    X_pima_train, X_pima_test, y_pima_train, y_pima_test = train_test_split(
        X_pima, y_pima, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_pima
    )

    train_pima = pd.concat([X_pima_train, y_pima_train], axis=1)
    test_pima = pd.concat([X_pima_test, y_pima_test], axis=1)
    train_pima.to_csv(MODEL_READY_DIR / "track_b_diabetes_train.csv", index=False)
    test_pima.to_csv(MODEL_READY_DIR / "track_b_diabetes_test.csv", index=False)

    # -------------------------------------------------------------
    # Track B - Cleveland Heart Disease
    # -------------------------------------------------------------
    clev_path = CLEAN_DIR / "cleveland_clean.csv"
    clev_df = pd.read_csv(clev_path)
    clev_target = config["tracks"]["track_b"]["heart_disease"]["target_column"]

    corrs_clev = audit_correlations_and_leakage(clev_df, clev_target, config["leakage_safeguards"]["correlation_threshold"])
    X_clev = clev_df.drop(columns=[clev_target])
    y_clev = clev_df[clev_target].astype(int)

    X_clev_train, X_clev_test, y_clev_train, y_clev_test = train_test_split(
        X_clev, y_clev, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_clev
    )

    train_clev = pd.concat([X_clev_train, y_clev_train], axis=1)
    test_clev = pd.concat([X_clev_test, y_clev_test], axis=1)
    train_clev.to_csv(MODEL_READY_DIR / "track_b_heart_train.csv", index=False)
    test_clev.to_csv(MODEL_READY_DIR / "track_b_heart_test.csv", index=False)

    return {
        "track_b_diabetes": {
            "X_train": X_pima_train,
            "X_test": X_pima_test,
            "y_train": y_pima_train,
            "y_test": y_pima_test,
            "train_indices": X_pima_train.index.tolist(),
            "test_indices": X_pima_test.index.tolist(),
        },
        "track_b_heart": {
            "X_train": X_clev_train,
            "X_test": X_clev_test,
            "y_train": y_clev_train,
            "y_test": y_clev_test,
            "train_indices": X_clev_train.index.tolist(),
            "test_indices": X_clev_test.index.tolist(),
        },
    }


def verify_and_save_metadata(all_splits: dict):
    """
    Verify integrity of splits and export split index dictionary for 100% reproducibility.
    """
    print("\n" + "=" * 80)
    print("STEP 3: VERIFYING SPLIT INTEGRITY AND EXPORTING INDICES")
    print("=" * 80)

    indices_export = {}
    summary_records = []

    for track_key, split in all_splits.items():
        X_tr = split["X_train"]
        X_te = split["X_test"]
        y_tr = split["y_train"]
        y_te = split["y_test"]

        # Check overlapping index integrity
        overlap = set(X_tr.index).intersection(set(X_te.index))
        assert len(overlap) == 0, f"DATA LEAKAGE: Overlapping indices found in {track_key}!"

        # Check shapes
        assert len(X_tr) == len(y_tr), f"Length mismatch in {track_key} train!"
        assert len(X_te) == len(y_te), f"Length mismatch in {track_key} test!"

        indices_export[track_key] = {
            "train_indices": split["train_indices"],
            "test_indices": split["test_indices"],
        }

        train_pos_rate = float(y_tr.mean() * 100)
        test_pos_rate = float(y_te.mean() * 100)
        total_rows = len(X_tr) + len(X_te)
        n_features = X_tr.shape[1]

        summary_records.append({
            "Dataset_Track": track_key,
            "Total_Rows": total_rows,
            "Features_N": n_features,
            "Train_Rows": len(X_tr),
            "Train_Pos_Rate_Pct": round(train_pos_rate, 2),
            "Test_Rows": len(X_te),
            "Test_Pos_Rate_Pct": round(test_pos_rate, 2),
            "Stratification_Delta_Pct": round(abs(train_pos_rate - test_pos_rate), 4),
        })

    # Save indices to JSON
    indices_file = MODEL_READY_DIR / "split_indices.json"
    with open(indices_file, "w", encoding="utf-8") as f:
        json.dump(indices_export, f)
    print(f"Saved reproducible split index manifests to: {indices_file.relative_to(BASE_DIR)}")

    df_summary = pd.DataFrame(summary_records)
    print("\n" + "=" * 80)
    print("MODEL-READY DATASET AUDIT & STRATIFICATION SUMMARY TABLE")
    print("=" * 80)
    print(df_summary.to_string(index=False))

    summary_csv = MODEL_READY_DIR / "model_ready_summary.csv"
    df_summary.to_csv(summary_csv, index=False)
    print(f"\nSaved summary table to: {summary_csv.relative_to(BASE_DIR)}")

    return df_summary


def main():
    print("Starting Stage 3 Preprocessing & Feature Matrix Build Pipeline...")
    config = load_config()

    track_a_splits = prepare_track_a(config)
    track_b_splits = prepare_track_b(config)

    all_splits = {**track_a_splits, **track_b_splits}
    verify_and_save_metadata(all_splits)

    print("\n[SUCCESS] Stage 3 data preparation completed successfully with zero data leakage.")


if __name__ == "__main__":
    main()
