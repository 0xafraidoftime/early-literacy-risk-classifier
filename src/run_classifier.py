"""
run_classifier.py
-----------------
End-to-end runner for the early literacy risk classifier.

Workflow:
  1. Generate / load synthetic ECLS-K data
  2. Feature engineering + risk label construction
  3. Cross-validate all models
  4. Fit best model (Random Forest) on full training set
  5. Evaluate on held-out test set
  6. Compute SHAP values for interpretability
  7. Save all outputs

Usage:
    cd src
    python run_classifier.py
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Allow imports from src/
sys.path.insert(0, str(Path(__file__).parent))

# We reuse the data generator from repo 1 — or generate standalone
from features import (
    create_risk_label, engineer_features, build_preprocessor,
    ENGINEERED_FEATURES, ALL_NUMERIC, CATEGORICAL_FEATURES,
)
from model import (
    split_data, get_models, cross_validate_models,
    evaluate_on_test, compute_shap_values,
)

OUT = Path(__file__).parent.parent / "outputs"
OUT.mkdir(parents=True, exist_ok=True)


def generate_data(n: int = 2000) -> pd.DataFrame:
    """Inline data generator (mirrors quantile-regression-child-outcomes repo)."""
    import numpy as np
    rng = np.random.default_rng(42)
    sex = rng.choice([0, 1], size=n, p=[0.49, 0.51])
    race = rng.choice([1,2,3,4,5], size=n, p=[0.52,0.15,0.23,0.04,0.06])
    ses_quintile = rng.choice([1,2,3,4,5], size=n)
    parent_edu = rng.choice([1,2,3,4], size=n, p=[0.15,0.30,0.35,0.20])
    special_ed_flag = rng.choice([0,1], size=n, p=[0.87,0.13])
    vocab_baseline = np.clip(rng.normal(50,15,n) - (3-ses_quintile)*2.5, 5, 100)
    phonological = np.clip(rng.normal(25,8,n) + (ses_quintile-3)*1.5 + (parent_edu-2)*2, 0, 50)
    working_memory = np.clip(rng.normal(50,12,n), 10, 100)
    teacher_experience_yrs = np.clip(rng.poisson(8,n), 0, 35)
    class_size = rng.integers(12, 30, n)
    school_type = rng.choice([0,1], n, p=[0.75,0.25])

    def reading_score(boost):
        base = 40+boost+0.25*vocab_baseline+0.40*phonological+0.10*working_memory+(ses_quintile-3)*3.5+(parent_edu-2)*2.0-special_ed_flag*8+sex*2
        return np.clip(rng.normal(base, 8), 0, 200).round(1)

    return pd.DataFrame({
        "child_id": np.arange(1, n+1), "sex": sex, "race": race,
        "ses_quintile": ses_quintile, "parent_edu": parent_edu,
        "special_ed_flag": special_ed_flag, "vocab_baseline": vocab_baseline.round(2),
        "phonological": phonological.round(2), "working_memory": working_memory.round(2),
        "teacher_experience_yrs": teacher_experience_yrs, "class_size": class_size,
        "school_type": school_type, "reading_w5": reading_score(49),
    })


def main():
    print("=" * 65)
    print("Early Literacy Risk Classifier — At-Risk Prediction Pipeline")
    print("=" * 65)

    # 1. Data
    print("\n[1/6] Generating synthetic ECLS-K dataset (n=2000)...")
    raw_df = generate_data(n=2000)
    raw_df.to_csv(Path(__file__).parent.parent / "data" / "synthetic_ecls.csv", index=False)

    # 2. Features + label
    print("\n[2/6] Engineering features & constructing risk label...")
    df = engineer_features(raw_df)
    y = create_risk_label(df, outcome_col="reading_w5")
    X = df[ENGINEERED_FEATURES]

    at_risk_pct = y.mean() * 100
    print(f"      At-risk children: {y.sum()} / {len(y)} ({at_risk_pct:.1f}%)")

    # 3. Preprocess
    preprocessor = build_preprocessor()
    X_arr = preprocessor.fit_transform(X)
    feature_names = (
        ALL_NUMERIC +
        CATEGORICAL_FEATURES
    )

    X_train, X_test, y_train, y_test = split_data(X_arr, y)
    print(f"      Train: {len(y_train)} | Test: {len(y_test)}")

    # 4. Cross-validate
    print("\n[3/6] 5-fold cross-validation across all models...")
    models = get_models()
    cv_results = cross_validate_models(models, X_train, y_train)
    cv_results.to_csv(OUT / "cv_results.csv", index=False)
    print(f"\n  CV results saved → {OUT / 'cv_results.csv'}")

    # 5. Fit best model (Random Forest)
    print("\n[4/6] Fitting Random Forest on full training set...")
    rf_model = models["Random Forest"]
    rf_model.fit(X_train, y_train)

    # 6. Test evaluation
    print("\n[5/6] Evaluating on held-out test set...")
    metrics = evaluate_on_test(
        rf_model, X_test, y_test,
        model_name="Random Forest",
        save_path=OUT / "test_evaluation.png"
    )

    # 7. SHAP
    print("\n[6/6] Computing SHAP values for interpretability...")
    shap_exp = compute_shap_values(
        rf_model, X_train, X_test,
        feature_names=feature_names,
        save_dir=OUT
    )

    # Save feature importance table
    import numpy as np
    mean_shap = np.abs(shap_exp.values).mean(axis=0)
    fi_df = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_shap.round(4)
    }).sort_values("mean_abs_shap", ascending=False)
    fi_df.to_csv(OUT / "feature_importance_shap.csv", index=False)

    print(f"\n{'='*65}")
    print(f"✓ Pipeline complete. All outputs saved to {OUT}/")
    print(f"\nTop 5 features by SHAP importance:")
    print(fi_df.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
