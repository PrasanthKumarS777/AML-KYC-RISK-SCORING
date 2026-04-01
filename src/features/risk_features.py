import pandas as pd
import numpy as np
from src.utils.logger import get_logger
from src.utils.config import LOW_RISK_THRESHOLD, MEDIUM_RISK_THRESHOLD, HIGH_RISK_THRESHOLD

logger = get_logger("risk_features")

COUNTRY_RISK_MAP = {
    "Iran": 100, "North Korea": 100, "Myanmar": 90, "Syria": 95, "Yemen": 85,
    "Nigeria": 65, "Pakistan": 70, "Kenya": 55, "Vietnam": 50, "UAE": 60,
    "India": 30, "USA": 20, "UK": 20, "Germany": 15, "Australia": 15
}

BUSINESS_RISK_MAP = {
    "Crypto Exchange": 90, "Money Transfer": 80, "Import/Export": 70,
    "Real Estate": 65, "NGO": 60, "Consulting": 45, "Retail": 35,
    "IT Services": 30, "Manufacturing": 30
}

WEIGHTS = {
    "country_risk":             0.20,
    "business_risk":            0.15,
    "is_pep":                   0.15,
    "is_sanctioned":            0.20,
    "structuring_flag":         0.10,
    "adverse_media_flag":       0.08,
    "cash_transaction_ratio":   0.05,
    "num_countries_transacted": 0.04,
    "kyc_completeness_score":   0.03,
}


def compute_risk_score(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["country_risk_score"] = df["country"].map(COUNTRY_RISK_MAP).fillna(50)
    df["business_risk_score"] = df["business_type"].map(BUSINESS_RISK_MAP).fillna(50)

    df["pep_score"]             = df["is_pep"].astype(int) * 100
    df["sanctioned_score"]      = df["is_sanctioned"].astype(int) * 100
    df["structuring_score"]     = df["structuring_flag"].astype(int) * 100
    df["adverse_media_score"]   = df["adverse_media_flag"].astype(int) * 100
    df["cash_ratio_score"]      = df["cash_transaction_ratio"] * 100
    df["country_diversity_score"] = (df["num_countries_transacted"] / 15 * 100).clip(0, 100)
    df["kyc_gap_score"]         = (1 - df["kyc_completeness_score"]) * 100

    df["composite_risk_score"] = (
        df["country_risk_score"]      * WEIGHTS["country_risk"] +
        df["business_risk_score"]     * WEIGHTS["business_risk"] +
        df["pep_score"]               * WEIGHTS["is_pep"] +
        df["sanctioned_score"]        * WEIGHTS["is_sanctioned"] +
        df["structuring_score"]       * WEIGHTS["structuring_flag"] +
        df["adverse_media_score"]     * WEIGHTS["adverse_media_flag"] +
        df["cash_ratio_score"]        * WEIGHTS["cash_transaction_ratio"] +
        df["country_diversity_score"] * WEIGHTS["num_countries_transacted"] +
        df["kyc_gap_score"]           * WEIGHTS["kyc_completeness_score"]
    ).round(2)

    df["sar_penalty"] = df["num_sar_filed"] * 5
    df["composite_risk_score"] = (df["composite_risk_score"] + df["sar_penalty"]).clip(0, 100).round(2)

    df["risk_tier"] = pd.cut(
        df["composite_risk_score"],
        bins=[-1, LOW_RISK_THRESHOLD, MEDIUM_RISK_THRESHOLD, HIGH_RISK_THRESHOLD, 100],
        labels=["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    )

    logger.info(f"Risk scores computed for {len(df)} customers")
    logger.info(f"Risk tier distribution:\n{df['risk_tier'].value_counts().to_string()}")
    return df


def save_processed_data(df: pd.DataFrame, path: str = "data/processed/customers_scored.csv"):
    df.to_csv(path, index=False)
    logger.info(f"Processed data saved to {path}")


if __name__ == "__main__":
    raw_df = pd.read_csv("data/raw/customers_raw.csv")
    scored_df = compute_risk_score(raw_df)
    save_processed_data(scored_df)

    print("\n=== Risk Score Summary ===")
    print(scored_df[["name", "country", "business_type", "composite_risk_score", "risk_tier"]].head(10).to_string())
    print(f"\nRisk Tier Distribution:")
    print(scored_df["risk_tier"].value_counts())
    print(f"\nAvg Risk Score : {scored_df['composite_risk_score'].mean():.2f}")
    print(f"Max Risk Score : {scored_df['composite_risk_score'].max():.2f}")
    print(f"Min Risk Score : {scored_df['composite_risk_score'].min():.2f}")