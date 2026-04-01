import pandas as pd
import numpy as np
from faker import Faker
import random
from datetime import datetime, timedelta
from src.utils.logger import get_logger

fake = Faker()
logger = get_logger("data_generator")

COUNTRIES_HIGH_RISK = ["Iran", "North Korea", "Myanmar", "Syria", "Yemen"]
COUNTRIES_MEDIUM_RISK = ["Nigeria", "Pakistan", "Kenya", "Vietnam", "UAE"]
COUNTRIES_LOW_RISK = ["India", "USA", "UK", "Germany", "Australia"]

BUSINESS_TYPES = ["Retail", "Import/Export", "Real Estate", "Crypto Exchange",
                  "Money Transfer", "IT Services", "Manufacturing", "NGO", "Consulting"]

HIGH_RISK_BUSINESS = ["Crypto Exchange", "Money Transfer", "Import/Export", "Real Estate"]


def generate_customer_profile(n: int = 500, seed: int = 42) -> pd.DataFrame:
    random.seed(seed)
    np.random.seed(seed)
    Faker.seed(seed)

    records = []

    for _ in range(n):
        country = random.choices(
            COUNTRIES_LOW_RISK + COUNTRIES_MEDIUM_RISK + COUNTRIES_HIGH_RISK,
            weights=[6]*len(COUNTRIES_LOW_RISK) + [3]*len(COUNTRIES_MEDIUM_RISK) + [1]*len(COUNTRIES_HIGH_RISK)
        )[0]

        business_type = random.choice(BUSINESS_TYPES)
        is_pep = random.choices([True, False], weights=[0.08, 0.92])[0]
        is_sanctioned = random.choices([True, False], weights=[0.02, 0.98])[0]

        account_age_days = random.randint(30, 3650)
        num_transactions_monthly = random.randint(1, 200)
        avg_transaction_amount = round(random.uniform(500, 500000), 2)
        num_countries_transacted = random.randint(1, 15)
        cash_transaction_ratio = round(random.uniform(0.0, 1.0), 2)
        structuring_flag = random.choices([True, False], weights=[0.1, 0.9])[0]
        adverse_media_flag = random.choices([True, False], weights=[0.05, 0.95])[0]
        kyc_completeness_score = round(random.uniform(0.4, 1.0), 2)
        num_sar_filed = random.choices([0, 1, 2, 3], weights=[0.80, 0.12, 0.05, 0.03])[0]

        onboarding_date = datetime.now() - timedelta(days=account_age_days)

        records.append({
            "customer_id": fake.uuid4(),
            "name": fake.name(),
            "country": country,
            "business_type": business_type,
            "is_pep": is_pep,
            "is_sanctioned": is_sanctioned,
            "account_age_days": account_age_days,
            "num_transactions_monthly": num_transactions_monthly,
            "avg_transaction_amount": avg_transaction_amount,
            "num_countries_transacted": num_countries_transacted,
            "cash_transaction_ratio": cash_transaction_ratio,
            "structuring_flag": structuring_flag,
            "adverse_media_flag": adverse_media_flag,
            "kyc_completeness_score": kyc_completeness_score,
            "num_sar_filed": num_sar_filed,
            "onboarding_date": onboarding_date.strftime("%Y-%m-%d"),
            "last_reviewed_date": (datetime.now() - timedelta(days=random.randint(0, 365))).strftime("%Y-%m-%d")
        })

    df = pd.DataFrame(records)
    logger.info(f"Generated {len(df)} synthetic customer profiles")
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