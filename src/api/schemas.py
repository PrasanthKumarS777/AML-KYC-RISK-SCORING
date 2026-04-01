from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime


class CustomerBase(BaseModel):
    customer_id:              str
    name:                     str
    country:                  str
    business_type:            str
    is_pep:                   bool
    is_sanctioned:            bool
    account_age_days:         int
    num_transactions_monthly: int
    avg_transaction_amount:   float
    num_countries_transacted: int
    cash_transaction_ratio:   float
    structuring_flag:         bool
    adverse_media_flag:       bool
    kyc_completeness_score:   float
    num_sar_filed:            int
    composite_risk_score:     Optional[float] = None
    risk_tier:                Optional[str]   = None

    class Config:
        from_attributes = True


class RiskDriver(BaseModel):
    feature:    str
    label:      str
    value:      float
    shap_value: float
    direction:  str


class RiskExplanation(BaseModel):
    customer_id:      str
    name:             str
    predicted_tier:   str
    confidence_pct:   float
    composite_score:  float
    top_risk_drivers: List[RiskDriver]
    risk_reasons:     List[str]
    all_class_proba:  dict


class AlertResponse(BaseModel):
    id:            int
    customer_id:   str
    customer_name: str
    risk_tier:     str
    risk_score:    float
    alert_reason:  str
    status:        str
    assigned_to:   str
    created_at:    datetime

    class Config:
        from_attributes = True


class AlertStatusUpdate(BaseModel):
    status:      str
    assigned_to: Optional[str] = "analyst"


class SummaryStats(BaseModel):
    total_customers:   int
    critical_count:    int
    high_count:        int
    medium_count:      int
    low_count:         int
    open_alerts:       int
    avg_risk_score:    float
    pep_count:         int
    sanctioned_count:  int