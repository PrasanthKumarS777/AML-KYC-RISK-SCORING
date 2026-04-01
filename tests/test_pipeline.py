import pytest
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.core.data_generator import generate_customer_profile
from src.features.risk_features import compute_risk_score


# ── Data Generator Tests ─────────────────────────────────────

def test_generator_returns_dataframe():
    df = generate_customer_profile(n=50)
    assert isinstance(df, pd.DataFrame)


def test_generator_correct_row_count():
    df = generate_customer_profile(n=100)
    assert len(df) == 100


def test_generator_required_columns():
    df = generate_customer_profile(n=20)
    required = [
        "customer_id", "name", "country", "business_type",
        "is_pep", "is_sanctioned", "composite_risk_score"
        if "composite_risk_score" in df.columns else "cash_transaction_ratio"
    ]
    for col in ["customer_id", "name", "country", "business_type",
                "is_pep", "is_sanctioned", "cash_transaction_ratio"]:
        assert col in df.columns, f"Missing column: {col}"


def test_generator_no_null_customer_ids():
    df = generate_customer_profile(n=50)
    assert df["customer_id"].isnull().sum() == 0


def test_generator_unique_customer_ids():
    df = generate_customer_profile(n=100)
    assert df["customer_id"].nunique() == 100


def test_generator_cash_ratio_in_range():
    df = generate_customer_profile(n=100)
    assert df["cash_transaction_ratio"].between(0, 1).all()


def test_generator_kyc_score_in_range():
    df = generate_customer_profile(n=100)
    assert df["kyc_completeness_score"].between(0, 1).all()


# ── Risk Scoring Tests ───────────────────────────────────────

def test_risk_score_returns_dataframe():
    df  = generate_customer_profile(n=50)
    out = compute_risk_score(df)
    assert isinstance(out, pd.DataFrame)


def test_risk_score_column_exists():
    df  = generate_customer_profile(n=50)
    out = compute_risk_score(df)
    assert "composite_risk_score" in out.columns


def test_risk_tier_column_exists():
    df  = generate_customer_profile(n=50)
    out = compute_risk_score(df)
    assert "risk_tier" in out.columns


def test_risk_score_in_valid_range():
    df  = generate_customer_profile(n=100)
    out = compute_risk_score(df)
    assert out["composite_risk_score"].between(0, 100).all()


def test_risk_tier_valid_labels():
    df    = generate_customer_profile(n=200)
    out   = compute_risk_score(df)
    valid = {"LOW", "MEDIUM", "HIGH", "CRITICAL"}
    tiers = set(out["risk_tier"].dropna().unique())
    assert tiers.issubset(valid), f"Unexpected tiers: {tiers - valid}"


def test_sanctioned_customer_gets_high_score():
    df = generate_customer_profile(n=100)
    out = compute_risk_score(df)
    sanctioned = out[out["is_sanctioned"] == True]
    if len(sanctioned) > 0:
        assert sanctioned["composite_risk_score"].mean() > 40


def test_pep_customer_score_higher_than_clean():
    df  = generate_customer_profile(n=200)
    out = compute_risk_score(df)
    pep_avg   = out[out["is_pep"] == True]["composite_risk_score"].mean()
    clean_avg = out[out["is_pep"] == False]["composite_risk_score"].mean()
    if not pd.isna(pep_avg):
        assert pep_avg > clean_avg


def test_row_count_preserved_after_scoring():
    df  = generate_customer_profile(n=150)
    out = compute_risk_score(df)
    assert len(df) == len(out)


# ── Model Tests ──────────────────────────────────────────────

def test_model_artifacts_exist():
    assert os.path.exists("src/models/saved/risk_model.joblib")
    assert os.path.exists("src/models/saved/scaler.joblib")
    assert os.path.exists("src/models/saved/label_encoder.joblib")


def test_model_predicts_valid_tier():
    import joblib
    import numpy as np

    model = joblib.load("src/models/saved/risk_model.joblib")
    le    = joblib.load("src/models/saved/label_encoder.joblib")
    scaler= joblib.load("src/models/saved/scaler.joblib")

    dummy = np.array([[85, 70, 0, 0, 0, 0, 45, 60, 30, 365, 20, 15000, 0]])
    dummy_scaled = scaler.transform(dummy)
    pred  = model.predict(dummy_scaled)
    label = le.inverse_transform(pred)[0]
    assert label in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]