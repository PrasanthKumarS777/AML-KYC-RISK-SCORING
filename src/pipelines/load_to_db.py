import pandas as pd
import json
from src.core.database import init_db, get_session, Customer, RiskAuditLog, AlertQueue
from src.utils.logger import get_logger
from datetime import datetime

logger = get_logger("load_to_db")


def load_customers(session, df: pd.DataFrame):
    count = 0
    for _, row in df.iterrows():
        existing = session.query(Customer).filter_by(
            customer_id=row["customer_id"]
        ).first()

        if not existing:
            c = Customer(
                customer_id              = row["customer_id"],
                name                     = row["name"],
                country                  = row["country"],
                business_type            = row["business_type"],
                is_pep                   = bool(row["is_pep"]),
                is_sanctioned            = bool(row["is_sanctioned"]),
                account_age_days         = int(row["account_age_days"]),
                num_transactions_monthly = int(row["num_transactions_monthly"]),
                avg_transaction_amount   = float(row["avg_transaction_amount"]),
                num_countries_transacted = int(row["num_countries_transacted"]),
                cash_transaction_ratio   = float(row["cash_transaction_ratio"]),
                structuring_flag         = bool(row["structuring_flag"]),
                adverse_media_flag       = bool(row["adverse_media_flag"]),
                kyc_completeness_score   = float(row["kyc_completeness_score"]),
                num_sar_filed            = int(row["num_sar_filed"]),
                onboarding_date          = row["onboarding_date"],
                last_reviewed_date       = row["last_reviewed_date"],
                composite_risk_score     = float(row["composite_risk_score"]),
                risk_tier                = str(row["risk_tier"])
            )
            session.add(c)
            count += 1
    session.commit()
    logger.info(f"Loaded {count} new customers into database")
    return count


def load_audit_logs(session, df: pd.DataFrame):
    count = 0
    for _, row in df.iterrows():
        log = RiskAuditLog(
            customer_id      = row["customer_id"],
            customer_name    = row["name"],
            previous_tier    = None,
            new_tier         = str(row["risk_tier"]),
            previous_score   = None,
            new_score        = float(row["composite_risk_score"]),
            change_reason    = "Initial onboarding risk assessment",
            top_risk_drivers = str(row.get("structuring_flag", "")),
            reviewed_by      = "system"
        )
        session.add(log)
        count += 1
    session.commit()
    logger.info(f"Loaded {count} audit log entries")
    return count


def load_alerts(session, df: pd.DataFrame):
    high_risk = df[df["risk_tier"].isin(["HIGH", "CRITICAL"])]
    count = 0
    for _, row in high_risk.iterrows():
        alert = AlertQueue(
            customer_id   = row["customer_id"],
            customer_name = row["name"],
            risk_tier     = str(row["risk_tier"]),
            risk_score    = float(row["composite_risk_score"]),
            alert_reason  = f"{row['risk_tier']} risk customer flagged during onboarding assessment",
            status        = "OPEN",
            assigned_to   = "unassigned"
        )
        session.add(alert)
        count += 1
    session.commit()
    logger.info(f"Loaded {count} alerts into alert queue")
    return count


if __name__ == "__main__":
    init_db()
    session = get_session()

    df = pd.read_csv("data/processed/customers_scored.csv")
    df["risk_tier"] = df["risk_tier"].astype(str)

    c = load_customers(session, df)
    a = load_audit_logs(session, df)
    al = load_alerts(session, df)

    print(f"\n✅ Database loaded successfully")
    print(f"   → {c}  customers inserted")
    print(f"   → {a}  audit log entries")
    print(f"   → {al} alerts in queue (HIGH + CRITICAL only)")

    # Quick verify
    total     = session.query(Customer).count()
    open_alerts = session.query(AlertQueue).filter_by(status="OPEN").count()
    print(f"\nDB Verification:")
    print(f"   Total customers in DB : {total}")
    print(f"   Open alerts in queue  : {open_alerts}")
    session.close()