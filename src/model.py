"""
model.py
--------
Trains, evaluates, and interprets supervised ML models to identify children
at risk for reading difficulties by Grade 4.

Models:
  - Random Forest (primary)
  - XGBoost (comparison)
  - Logistic Regression (interpretable baseline)

Interpretability:
  - SHAP values (TreeExplainer) for feature importance and individual predictions
  - Beeswarm and waterfall plots for communicating results to education researchers
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import shap
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_val_score
)
from sklearn.metrics import (
    classification_report, roc_auc_score, average_precision_score,
    ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay,
    roc_curve, precision_recall_curve
)
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV

warnings.filterwarnings("ignore")

SEED = 42


# ─────────────────────────────────────────────────────────────────────────────
# 1. Train / test split
# ─────────────────────────────────────────────────────────────────────────────

def split_data(X: pd.DataFrame, y: pd.Series, test_size: float = 0.20):
    return train_test_split(X, y, test_size=test_size,
                            stratify=y, random_state=SEED)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Define models
# ─────────────────────────────────────────────────────────────────────────────

def get_models() -> dict:
    return {
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            min_samples_leaf=10,
            class_weight="balanced",   # handles at-risk class imbalance
            random_state=SEED,
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=4,
            min_samples_leaf=10,
            random_state=SEED,
        ),
        "Logistic Regression": LogisticRegression(
            C=0.1,
            class_weight="balanced",
            max_iter=1000,
            random_state=SEED,
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3. Cross-validated evaluation
# ─────────────────────────────────────────────────────────────────────────────

def cross_validate_models(models: dict, X_train: np.ndarray,
                           y_train: pd.Series) -> pd.DataFrame:
    """
    5-fold stratified CV across all models. Reports ROC-AUC and
    Average Precision (handles class imbalance better than accuracy).
    """
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    rows = []
    for name, model in models.items():
        auc_scores = cross_val_score(model, X_train, y_train,
                                     cv=cv, scoring="roc_auc", n_jobs=-1)
        ap_scores = cross_val_score(model, X_train, y_train,
                                    cv=cv, scoring="average_precision", n_jobs=-1)
        rows.append({
            "Model": name,
            "ROC-AUC (mean)": auc_scores.mean().round(3),
            "ROC-AUC (std)":  auc_scores.std().round(3),
            "Avg Precision (mean)": ap_scores.mean().round(3),
            "Avg Precision (std)":  ap_scores.std().round(3),
        })
        print(f"  {name}: AUC={auc_scores.mean():.3f} ± {auc_scores.std():.3f} | "
              f"AP={ap_scores.mean():.3f} ± {ap_scores.std():.3f}")
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Final evaluation on held-out test set
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_on_test(model, X_test: np.ndarray, y_test: pd.Series,
                     model_name: str = "Random Forest",
                     save_path: Path = None) -> dict:
    """
    Evaluate fitted model on held-out test set.
    Saves ROC curve, PR curve, and confusion matrix.
    """
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    auc = roc_auc_score(y_test, y_prob)
    ap  = average_precision_score(y_test, y_prob)

    print(f"\n  Test ROC-AUC:          {auc:.3f}")
    print(f"  Test Avg Precision:    {ap:.3f}")
    print(f"\n  Classification Report:\n{classification_report(y_test, y_pred)}")

    if save_path:
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        fig.suptitle(f"{model_name} — Test Set Evaluation\n"
                     f"(Predicting at-risk children: bottom 20th percentile reading)",
                     fontsize=13, fontweight="bold")

        RocCurveDisplay.from_predictions(y_test, y_prob, ax=axes[0],
                                         name=f"AUC={auc:.3f}")
        axes[0].set_title("ROC Curve"); axes[0].grid(True, alpha=0.3)

        PrecisionRecallDisplay.from_predictions(y_test, y_prob, ax=axes[1],
                                                name=f"AP={ap:.3f}")
        axes[1].set_title("Precision-Recall Curve\n(Better metric for imbalanced classes)")
        axes[1].grid(True, alpha=0.3)

        ConfusionMatrixDisplay.from_predictions(
            y_test, y_pred,
            display_labels=["Not At-Risk", "At-Risk"],
            ax=axes[2], colorbar=False, cmap="Blues"
        )
        axes[2].set_title("Confusion Matrix")

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  [plot] Saved → {save_path}")
        plt.close()

    return {"auc": auc, "ap": ap}


# ─────────────────────────────────────────────────────────────────────────────
# 5. SHAP interpretability
# ─────────────────────────────────────────────────────────────────────────────

def compute_shap_values(model, X_train_arr: np.ndarray,
                         X_test_arr: np.ndarray,
                         feature_names: list,
                         save_dir: Path = None):
    """
    Compute SHAP values using TreeExplainer and generate:
      1. Beeswarm plot (global feature importance)
      2. Bar plot (mean |SHAP|)
      3. Waterfall plot for a single high-risk child

    Returns
    -------
    shap_values array and Explanation object
    """
    print("\n  Computing SHAP values...")
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer(X_test_arr)

    # Attach feature names
    shap_explanation = shap.Explanation(
        values=shap_vals.values[:, :, 1] if shap_vals.values.ndim == 3
               else shap_vals.values,
        base_values=shap_vals.base_values[:, 1] if shap_vals.base_values.ndim == 2
                    else shap_vals.base_values,
        data=X_test_arr,
        feature_names=feature_names,
    )

    if save_dir:
        # 1. Beeswarm — global summary
        plt.figure(figsize=(10, 7))
        shap.plots.beeswarm(shap_explanation, max_display=15, show=False)
        plt.title("SHAP Beeswarm Plot\nGlobal Feature Impact on At-Risk Prediction",
                  fontsize=12, fontweight="bold", pad=15)
        plt.tight_layout()
        plt.savefig(save_dir / "shap_beeswarm.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  [SHAP] Beeswarm saved → {save_dir / 'shap_beeswarm.png'}")

        # 2. Bar plot — mean |SHAP|
        plt.figure(figsize=(9, 6))
        shap.plots.bar(shap_explanation, max_display=12, show=False)
        plt.title("Mean |SHAP| Feature Importance\n(How much each feature shifts the at-risk prediction)",
                  fontsize=12, fontweight="bold")
        plt.tight_layout()
        plt.savefig(save_dir / "shap_bar.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  [SHAP] Bar plot saved → {save_dir / 'shap_bar.png'}")

        # 3. Waterfall for the highest-risk child in the test set
        risk_scores = shap_explanation.values.sum(axis=1)
        highest_risk_idx = np.argmax(risk_scores)
        plt.figure(figsize=(10, 6))
        shap.plots.waterfall(shap_explanation[highest_risk_idx], show=False)
        plt.title(f"SHAP Waterfall — Highest Risk Child (test set index {highest_risk_idx})\n"
                  "Why this child was flagged as at-risk",
                  fontsize=11, fontweight="bold")
        plt.tight_layout()
        plt.savefig(save_dir / "shap_waterfall_high_risk.png",
                    dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  [SHAP] Waterfall saved → {save_dir / 'shap_waterfall_high_risk.png'}")

    return shap_explanation
