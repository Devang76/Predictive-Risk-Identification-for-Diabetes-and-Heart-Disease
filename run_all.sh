#!/usr/bin/env bash
# run_all.sh
# End-to-End Execution Pipeline for Predictive Risk Identification for Diabetes and Heart Disease
# Executes the entire modeling, evaluation, explainability, validation, and testing suite in order.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "================================================================================"
echo "PREDICTIVE RISK IDENTIFICATION FOR DIABETES AND HEART DISEASE"
echo "Starting Complete End-to-End Pipeline Execution..."
echo "Timestamp: $(date)"
echo "Python Interpreter: .venv/bin/python"
echo "================================================================================"

# Ensure virtual environment exists
if [ ! -f ".venv/bin/python" ]; then
    echo "[Error] Virtual environment not found at .venv/bin/python. Run: python3 -m venv .venv && .venv/bin/pip install -r requirements.txt"
    exit 1
fi

PYTHON=".venv/bin/python"
PYTEST=".venv/bin/pytest"

# Stage 3 - Step 1: Model Preprocessing & Dataset Splitting
echo ""
echo ">>> STEP 1/7: Splitting Datasets & Verifying Leakage Safeguards..."
$PYTHON src/03_model_preprocessing.py

# Stage 3 - Step 2: Model Training & Probability Calibration
echo ""
echo ">>> STEP 2/7: Training & Calibrating Models (Logistic Regression, Decision Tree, Random Forest)..."
$PYTHON src/04_train_models.py

# Stage 3 - Step 3: Multi-Metric Model Evaluation & Champion Selection
echo ""
echo ">>> STEP 3/7: Evaluating Models, Selecting Champions & Calibrating Tiers..."
$PYTHON src/05_evaluate_models.py

# Stage 3 - Step 4: Track A SHAP Explainability & Global Importance
echo ""
echo ">>> STEP 4/7: Generating Track A Population SHAP Explanations..."
$PYTHON src/06_explainability_shap.py

# Stage 3 - Step 5: Track B Clinical SHAP Explainability & Cross-Validation
echo ""
echo ">>> STEP 5/7: Generating Track B Clinical SHAP Explanations & CV Metrics..."
$PYTHON src/07_track_b_shap.py

# Stage 3 - Step 6: Comprehensive Robustness Validation & Subgroup Auditing
echo ""
echo ">>> STEP 6/7: Executing Bootstrap CIs, Subgroup Fairness, 2022 External Sanity Check & Baselines..."
$PYTHON src/08_validation.py

# Stage 3 - Step 7: Automated Test Suite Execution
echo ""
echo ">>> STEP 7/7: Running Automated Pytest Test Suite..."
$PYTEST tests/ -v

echo ""
echo "================================================================================"
echo "END-TO-END PIPELINE COMPLETED SUCCESSFULLY!"
echo "All models trained, evaluated, explained, validated, and tested."
echo "Timestamp: $(date)"
echo "================================================================================"
