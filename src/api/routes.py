from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional
from src.core.database import get_session, Customer, AlertQueue, RiskAuditLog
from src.api.schemas import CustomerBase, AlertResponse, AlertStatusUpdate, SummaryStats, RiskExplanation
from src.models.explainer import load_artifacts, explain_customer
from src.features.risk_features import compute_risk_score
from src.utils.logger import get_logger
import pandas as pd

logger  = get_logger("routes")
router  = APIRouter()
model, scaler, le = load_artifacts()


@router.get("/summary", response_model=SummaryStats, tags=["Dashboard"])
def get_summary():
    session = get_session()
    try:
        customers = session.query(Customer).all()
        df = pd.DataFrame([{
            "risk_tier":          c.risk_tier,
            "composite_risk_score": c.composite_risk_score,
            "is_pep":             c.is_pep,
            "is_sanctioned":      c.is_sanctioned
        } for c in customers])

        return SummaryStats(
            total_customers  = len(df),
            critical_count   = int((df["risk_tier"] == "CRITICAL").sum()),
            high_count       = int((df["risk_tier"] == "HIGH").sum()),
            medium_count     = int((df["risk_tier"] == "MEDIUM").sum()),
            low_count        = int((df["risk_tier"] == "LOW").sum()),
            open_alerts      = session.query(AlertQueue).filter_by(status="OPEN").count(),
            avg_risk_score   = round(float(df["composite_risk_score"].mean()), 2),
            pep_count        = int(df["is_pep"].sum()),
            sanctioned_count = int(df["is_sanctioned"].sum())
        )
    finally:
        session.close()


@router.get("/customers", response_model=List[CustomerBase], tags=["Customers"])
def get_customers(
    risk_tier: Optional[str] = Query(None, description="Filter by risk tier: LOW, MEDIUM, HIGH, CRITICAL"),
    limit:     int            = Query(50, le=500),
    offset:    int            = Query(0)
):
    session = get_session()
    try:
        q = session.query(Customer)
        if risk_tier:
            q = q.filter(Customer.risk_tier == risk_tier.upper())
        customers = q.offset(offset).limit(limit).all()
        return customers
    finally:
        session.close()


@router.get("/customers/{customer_id}", response_model=CustomerBase, tags=["Customers"])
def get_customer(customer_id: str):
    session = get_session()
    try:
        c = session.query(Customer).filter_by(customer_id=customer_id).first()
        if not c:
            raise HTTPException(status_code=404, detail="Customer not found")
        return c
    finally:
        session.close()


@router.get("/customers/{customer_id}/explain", response_model=RiskExplanation, tags=["Explainability"])
def explain_customer_risk(customer_id: str):
    session = get_session()
    try:
        c = session.query(Customer).filter_by(customer_id=customer_id).first()
        if not c:
            raise HTTPException(status_code=404, detail="Customer not found")

        row = pd.Series({
            "customer_id":              c.customer_id,
            "name":                     c.name,
            "country_risk_score":       0,
            "business_risk_score":      0,
            "pep_score":                int(c.is_pep) * 100,
            "sanctioned_score":         int(c.is_sanctioned) * 100,
            "structuring_score":        int(c.structuring_flag) * 100,
            "adverse_media_score":      int(c.adverse_media_flag) * 100,
            "cash_ratio_score":         c.cash_transaction_ratio * 100,
            "country_diversity_score":  (c.num_countries_transacted / 15 * 100),
            "kyc_gap_score":            (1 - c.kyc_completeness_score) * 100,
            "account_age_days":         c.account_age_days,
            "num_transactions_monthly": c.num_transactions_monthly,
            "avg_transaction_amount":   c.avg_transaction_amount,
            "num_sar_filed":            c.num_sar_filed,
            "composite_risk_score":     c.composite_risk_score,
            "risk_tier":                c.risk_tier
        })

        from src.features.risk_features import COUNTRY_RISK_MAP, BUSINESS_RISK_MAP
        row["country_risk_score"]  = COUNTRY_RISK_MAP.get(c.country, 50)
        row["business_risk_score"] = BUSINESS_RISK_MAP.get(c.business_type, 50)

        exp = explain_customer(row, model, scaler, le)
        return exp
    finally:
        session.close()


@router.get("/alerts", response_model=List[AlertResponse], tags=["Alerts"])
def get_alerts(
    status: Optional[str] = Query(None, description="OPEN | REVIEWED | ESCALATED | CLOSED"),
    limit:  int            = Query(50, le=200)
):
    session = get_session()
    try:
        q = session.query(AlertQueue)
        if status:
            q = q.filter(AlertQueue.status == status.upper())
        return q.order_by(AlertQueue.created_at.desc()).limit(limit).all()
    finally:
        session.close()


@router.patch("/alerts/{alert_id}", tags=["Alerts"])
def update_alert_status(alert_id: int, update: AlertStatusUpdate):
    session = get_session()
    try:
        alert = session.query(AlertQueue).filter_by(id=alert_id).first()
        if not alert:
            raise HTTPException(status_code=404, detail="Alert not found")
        alert.status      = update.status.upper()
        alert.assigned_to = update.assigned_to
        session.commit()
        return {"message": f"Alert {alert_id} updated to {update.status}", "alert_id": alert_id}
    finally:
        session.close()


@router.get("/audit-log", tags=["Audit"])
def get_audit_log(limit: int = Query(50, le=500)):
    session = get_session()
    try:
        logs = session.query(RiskAuditLog).order_by(
            RiskAuditLog.timestamp.desc()
        ).limit(limit).all()
        return [{
            "id":             l.id,
            "customer_id":    l.customer_id,
            "customer_name":  l.customer_name,
            "previous_tier":  l.previous_tier,
            "new_tier":       l.new_tier,
            "previous_score": l.previous_score,
            "new_score":      l.new_score,
            "change_reason":  l.change_reason,
            "reviewed_by":    l.reviewed_by,
            "timestamp":      str(l.timestamp)
        } for l in logs]
    finally:
        session.close()