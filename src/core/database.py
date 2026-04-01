from sqlalchemy import create_engine, Column, String, Float, Integer, Boolean, DateTime, Text
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime
from src.utils.config import DATABASE_URL
from src.utils.logger import get_logger

logger = get_logger("database")
Base   = declarative_base()


class Customer(Base):
    __tablename__ = "customers"

    customer_id              = Column(String, primary_key=True)
    name                     = Column(String)
    country                  = Column(String)
    business_type            = Column(String)
    is_pep                   = Column(Boolean)
    is_sanctioned            = Column(Boolean)
    account_age_days         = Column(Integer)
    num_transactions_monthly = Column(Integer)
    avg_transaction_amount   = Column(Float)
    num_countries_transacted = Column(Integer)
    cash_transaction_ratio   = Column(Float)
    structuring_flag         = Column(Boolean)
    adverse_media_flag       = Column(Boolean)
    kyc_completeness_score   = Column(Float)
    num_sar_filed            = Column(Integer)
    onboarding_date          = Column(String)
    last_reviewed_date       = Column(String)
    composite_risk_score     = Column(Float)
    risk_tier                = Column(String)
    created_at               = Column(DateTime, default=datetime.utcnow)
    updated_at               = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class RiskAuditLog(Base):
    __tablename__ = "risk_audit_log"

    id                = Column(Integer, primary_key=True, autoincrement=True)
    customer_id       = Column(String)
    customer_name     = Column(String)
    previous_tier     = Column(String, nullable=True)
    new_tier          = Column(String)
    previous_score    = Column(Float, nullable=True)
    new_score         = Column(Float)
    change_reason     = Column(String)
    top_risk_drivers  = Column(Text)
    reviewed_by       = Column(String, default="system")
    timestamp         = Column(DateTime, default=datetime.utcnow)


class AlertQueue(Base):
    __tablename__ = "alert_queue"

    id              = Column(Integer, primary_key=True, autoincrement=True)
    customer_id     = Column(String)
    customer_name   = Column(String)
    risk_tier       = Column(String)
    risk_score      = Column(Float)
    alert_reason    = Column(String)
    status          = Column(String, default="OPEN")   # OPEN | REVIEWED | ESCALATED | CLOSED
    assigned_to     = Column(String, default="unassigned")
    created_at      = Column(DateTime, default=datetime.utcnow)
    resolved_at     = Column(DateTime, nullable=True)


def get_engine():
    engine = create_engine(DATABASE_URL, echo=False)
    return engine


def init_db():
    engine = get_engine()
    Base.metadata.create_all(engine)
    logger.info("Database tables created: customers, risk_audit_log, alert_queue")
    return engine


def get_session():
    engine = get_engine()
    Session = sessionmaker(bind=engine)
    return Session()


if __name__ == "__main__":
    init_db()
    print("✅ Database initialized — tables created")
    print("   → customers")
    print("   → risk_audit_log")
    print("   → alert_queue")