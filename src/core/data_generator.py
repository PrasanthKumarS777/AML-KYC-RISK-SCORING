import pandas as pd
import numpy as np
from faker import Faker
import random
from datetime import datetime, timedelta
from src.utils.logger import get_logger

fake = Faker()
logger = get_logger("data_generator")

COUNTRIES_HIGH_RISK   = ["Iran", "North Korea", "Myanmar", "Syria", "Yemen"]
COUNTRIES_MEDIUM_RISK = ["Nigeria", "Pakistan", "Kenya", "Vietnam", "UAE"]
COUNTRIES_LOW_RISK    = ["India", "USA", "UK", "Germany", "Australia"]

BUSINESS_TYPES      = ["Retail", "Import/Export", "Real Estate", "Crypto Exchange",
                        "Money Transfer", "IT Services", "Manufacturing", "NGO", "Consulting"]
HIGH_RISK_BUSINESS  = ["Crypto Exchange", "Money Transfer", "Import/Export", "Real Estate"]
LOW_RISK_BUSINESS   = ["IT Services", "Manufacturing", "Retail", "Consulting"]


def _make_customer(profile: str) -> dict:
    """
    profile: 'low' | 'medium' | 'high' | 'critical'
    Each profile drives realistic but distinct flag combinations.
    """
    if profile == "critical":
        country       = random.choice(COUNTRIES_HIGH_RISK)
        business_type = random.choice(HIGH_RISK_BUSINESS)
        is_pep        = random.choices([True, False], weights=[0.6, 0.4])[0]
        is_sanctioned = random.choices([True, False], weights=[0.5, 0.5])[0]
        structuring   = random.choices([True, False], weights=[0.7, 0.3])[0]
        adverse_media = random.choices([True, False], weights=[0.6, 0.4])[0]
        cash_ratio    = round(random.uniform(0.6, 1.0), 2)
        num_countries = random.randint(8, 15)
        kyc_score     = round(random.uniform(0.4, 0.65), 2)
        num_sar       = random.choices([1, 2, 3, 4], weights=[0.3, 0.3, 0.25, 0.15])[0]
        num_tx        = random.randint(80, 200)
        avg_amount    = round(random.uniform(100000, 500000), 2)
        age_days      = random.randint(30, 730)

    elif profile == "high":
        country       = random.choices(
                            COUNTRIES_HIGH_RISK + COUNTRIES_MEDIUM_RISK,
                            weights=[3]*5 + [2]*5)[0]
        business_type = random.choice(HIGH_RISK_BUSINESS)
        is_pep        = random.choices([True, False], weights=[0.35, 0.65])[0]
        is_sanctioned = random.choices([True, False], weights=[0.10, 0.90])[0]
        structuring   = random.choices([True, False], weights=[0.45, 0.55])[0]
        adverse_media = random.choices([True, False], weights=[0.30, 0.70])[0]
        cash_ratio    = round(random.uniform(0.4, 0.8), 2)
        num_countries = random.randint(5, 12)
        kyc_score     = round(random.uniform(0.5, 0.75), 2)
        num_sar       = random.choices([0, 1, 2], weights=[0.4, 0.4, 0.2])[0]
        num_tx        = random.randint(50, 150)
        avg_amount    = round(random.uniform(50000, 300000), 2)
        age_days      = random.randint(90, 1825)

    elif profile == "medium":
        country       = random.choices(
                            COUNTRIES_MEDIUM_RISK + COUNTRIES_LOW_RISK,
                            weights=[3]*5 + [1]*5)[0]
        business_type = random.choice(BUSINESS_TYPES)
        is_pep        = random.choices([True, False], weights=[0.10, 0.90])[0]
        is_sanctioned = False
        structuring   = random.choices([True, False], weights=[0.15, 0.85])[0]
        adverse_media = random.choices([True, False], weights=[0.08, 0.92])[0]
        cash_ratio    = round(random.uniform(0.2, 0.55), 2)
        num_countries = random.randint(2, 7)
        kyc_score     = round(random.uniform(0.65, 0.85), 2)
        num_sar       = random.choices([0, 1], weights=[0.85, 0.15])[0]
        num_tx        = random.randint(10, 80)
        avg_amount    = round(random.uniform(5000, 80000), 2)
        age_days      = random.randint(180, 2555)

    else:  # low
        country       = random.choice(COUNTRIES_LOW_RISK)
        business_type = random.choice(LOW_RISK_BUSINESS)
        is_pep        = False
        is_sanctioned = False
        structuring   = False
        adverse_media = False
        cash_ratio    = round(random.uniform(0.0, 0.25), 2)
        num_countries = random.randint(1, 3)
        kyc_score     = round(random.uniform(0.80, 1.0), 2)
        num_sar       = 0
        num_tx        = random.randint(1, 40)
        avg_amount    = round(random.uniform(500, 20000), 2)
        age_days      = random.randint(365, 3650)

    onboarding = datetime.now() - timedelta(days=age_days)
    return {
        "customer_id":              fake.uuid4(),
        "name":                     fake.name(),
        "country":                  country,
        "business_type":            business_type,
        "is_pep":                   is_pep,
        "is_sanctioned":            is_sanctioned,
        "account_age_days":         age_days,
        "num_transactions_monthly": num_tx,
        "avg_transaction_amount":   avg_amount,
        "num_countries_transacted": num_countries,
        "cash_transaction_ratio":   cash_ratio,
        "structuring_flag":         structuring,
        "adverse_media_flag":       adverse_media,
        "kyc_completeness_score":   kyc_score,
        "num_sar_filed":            num_sar,
        "onboarding_date":          onboarding.strftime("%Y-%m-%d"),
        "last_reviewed_date":       (datetime.now() - timedelta(
                                        days=random.randint(0, 365))).strftime("%Y-%m-%d")
    }


def generate_customer_profile(n: int = 500, seed: int = 42) -> pd.DataFrame:
    random.seed(seed)
    np.random.seed(seed)
    Faker.seed(seed)

    # Realistic distribution: LOW 40%, MEDIUM 35%, HIGH 17%, CRITICAL 8%
    counts = {
        "low":      int(n * 0.40),
        "medium":   int(n * 0.35),
        "high":     int(n * 0.17),
        "critical": n - int(n * 0.40) - int(n * 0.35) - int(n * 0.17)
    }

    records = []
    for profile, count in counts.items():
        for _ in range(count):
            records.append(_make_customer(profile))

    random.shuffle(records)
    df = pd.DataFrame(records)
    logger.info(f"Generated {len(df)} synthetic customer profiles")
    logger.info(f"Profile distribution: { {k: v for k, v in counts.items()} }")
    return df


def save_raw_data(df: pd.DataFrame, path: str = "data/raw/customers_raw.csv"):
    df.to_csv(path, index=False)
    logger.info(f"Raw customer data saved to {path}")


if __name__ == "__main__":
    df = generate_customer_profile(n=500)
    save_raw_data(df)
    print(df.head())
    print(f"\nShape: {df.shape}")
    print(f"\nRisk flags summary:")
    print(f"  PEP customers     : {df['is_pep'].sum()}")
    print(f"  Sanctioned        : {df['is_sanctioned'].sum()}")
    print(f"  Structuring flag  : {df['structuring_flag'].sum()}")
    print(f"  Adverse media     : {df['adverse_media_flag'].sum()}")