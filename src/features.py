"""
features.py
-----------
Feature engineering pipeline for the early literacy risk classifier.

Transforms raw ECLS-K style data into model-ready features, including:
- Risk label construction (bottom 20th percentile = at-risk)
- Interaction features capturing compounding disadvantage
- Preprocessing pipeline (scaling, encoding)
"""

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer


# ── Risk label: bottom 20th percentile on reading at Grade 4 ─────────────────
RISK_THRESHOLD_PERCENTILE = 20


def create_risk_label(df: pd.DataFrame, outcome_col: str = "reading_w5") -> pd.Series:
    """
    Binary label: 1 if child's Grade 4 reading score is in the
    bottom 20th percentile (at-risk for learning disability), 0 otherwise.
    """
    threshold = df[outcome_col].quantile(RISK_THRESHOLD_PERCENTILE / 100)
    return (df[outcome_col] <= threshold).astype(int)


# ── Feature definitions ───────────────────────────────────────────────────────

BASELINE_FEATURES = [
    "sex",
    "race",
    "ses_quintile",
    "parent_edu",
    "special_ed_flag",
    "vocab_baseline",
    "phonological",
    "working_memory",
    "teacher_experience_yrs",
    "class_size",
    "school_type",
]

NUMERIC_FEATURES = [
    "ses_quintile",
    "parent_edu",
    "vocab_baseline",
    "phonological",
    "working_memory",
    "teacher_experience_yrs",
    "class_size",
]

CATEGORICAL_FEATURES = [
    "sex",
    "race",
    "special_ed_flag",
    "school_type",
]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add engineered features capturing compounding disadvantage and interactions.

    Returns a new DataFrame with additional feature columns.
    """
    df = df.copy()

    # Compound disadvantage index: low SES + low parent edu
    df["disadvantage_index"] = (
        (6 - df["ses_quintile"]) +        # invert: higher = more disadvantaged
        (5 - df["parent_edu"])
    )

    # Cognitive composite (unweighted average of standardized cognitive measures)
    for col in ["vocab_baseline", "phonological", "working_memory"]:
        col_std = f"{col}_z"
        df[col_std] = (df[col] - df[col].mean()) / df[col].std()
    df["cognitive_composite"] = (
        df["vocab_baseline_z"] + df["phonological_z"] + df["working_memory_z"]
    ) / 3

    # SES × Vocabulary interaction (does vocab matter more for low-SES kids?)
    df["ses_x_vocab"] = df["ses_quintile"] * df["vocab_baseline"]

    # Phonological × Special Ed interaction
    df["phonological_x_sped"] = df["phonological"] * df["special_ed_flag"]

    # Large class with low teacher experience (dual risk environment)
    df["class_risk"] = (
        (df["class_size"] > df["class_size"].median()).astype(int) +
        (df["teacher_experience_yrs"] < 3).astype(int)
    )

    return df


ENGINEERED_FEATURES = BASELINE_FEATURES + [
    "disadvantage_index",
    "cognitive_composite",
    "ses_x_vocab",
    "phonological_x_sped",
    "class_risk",
]

ALL_NUMERIC = NUMERIC_FEATURES + [
    "disadvantage_index",
    "cognitive_composite",
    "ses_x_vocab",
    "phonological_x_sped",
    "class_risk",
]


def build_preprocessor() -> ColumnTransformer:
    """
    sklearn ColumnTransformer: scale numeric, pass-through categorical.
    """
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
    ])
    return ColumnTransformer([
        ("num", numeric_transformer, ALL_NUMERIC),
        ("cat", "passthrough", CATEGORICAL_FEATURES),
    ])
