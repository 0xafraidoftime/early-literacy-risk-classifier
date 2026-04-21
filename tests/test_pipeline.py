"""
test_pipeline.py
----------------
Unit tests for the early literacy risk classifier pipeline.
"""
import sys
from pathlib import Path
import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from features import (
    create_risk_label, engineer_features, build_preprocessor,
    ENGINEERED_FEATURES, ALL_NUMERIC, CATEGORICAL_FEATURES
)
from model import get_models, split_data


def make_sample_df(n=200):
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "child_id": range(n),
        "sex": rng.choice([0,1], n),
        "race": rng.choice([1,2,3,4,5], n),
        "ses_quintile": rng.choice([1,2,3,4,5], n),
        "parent_edu": rng.choice([1,2,3,4], n),
        "special_ed_flag": rng.choice([0,1], n, p=[0.87,0.13]),
        "vocab_baseline": rng.uniform(10, 95, n),
        "phonological": rng.uniform(5, 48, n),
        "working_memory": rng.uniform(15, 95, n),
        "teacher_experience_yrs": rng.integers(0, 30, n),
        "class_size": rng.integers(12, 30, n),
        "school_type": rng.choice([0,1], n),
        "reading_w5": rng.uniform(30, 180, n),
    })


@pytest.fixture(scope="module")
def sample_df():
    return make_sample_df()


def test_risk_label_proportion(sample_df):
    y = create_risk_label(sample_df)
    # Should be approximately 20% at-risk
    assert 0.15 <= y.mean() <= 0.25


def test_engineer_features_columns(sample_df):
    df = engineer_features(sample_df)
    assert "disadvantage_index" in df.columns
    assert "cognitive_composite" in df.columns
    assert "ses_x_vocab" in df.columns
    assert "class_risk" in df.columns


def test_preprocessor_output_shape(sample_df):
    df = engineer_features(sample_df)
    X = df[ENGINEERED_FEATURES]
    preprocessor = build_preprocessor()
    X_arr = preprocessor.fit_transform(X)
    assert X_arr.shape[0] == len(df)
    assert X_arr.shape[1] == len(ALL_NUMERIC) + len(CATEGORICAL_FEATURES)


def test_train_test_split(sample_df):
    y = create_risk_label(sample_df)
    df = engineer_features(sample_df)
    X = df[ENGINEERED_FEATURES]
    pre = build_preprocessor()
    X_arr = pre.fit_transform(X)
    X_train, X_test, y_train, y_test = split_data(X_arr, y)
    assert len(X_train) + len(X_test) == len(sample_df)


def test_rf_model_fits(sample_df):
    df = engineer_features(sample_df)
    y = create_risk_label(sample_df)
    X = df[ENGINEERED_FEATURES]
    pre = build_preprocessor()
    X_arr = pre.fit_transform(X)
    X_train, X_test, y_train, y_test = split_data(X_arr, y)
    rf = get_models()["Random Forest"]
    rf.fit(X_train, y_train)
    preds = rf.predict(X_test)
    assert len(preds) == len(y_test)
    assert set(preds).issubset({0, 1})
