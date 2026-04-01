# 📄 Project Summary — AML/KYC Customer Risk Re-Scoring System

> **For Recruiters & Hiring Managers — 60-second read**

---

## What I Built

A personal portfolio project that simulates an **Anti-Money Laundering (AML)
compliance tool** — the kind used by banks, FinTechs, and NBFCs to detect
suspicious customers and meet regulatory requirements.

This is **not a tutorial project**. Every design decision maps to a real
compliance problem that cost real banks millions in fines.

---

## The Real Problem I Solved

| Real Incident | Fine | Gap This Project Addresses |
|---|---|---|
| TD Bank (2024) | $3 Billion | No real-time risk monitoring |
| Starling Bank (2024) | £28.9 Million | Static KYC scores post-onboarding |
| Deutsche Bank (2023) | $186 Million | No explainability for SAR filings |

My system solves: **dynamic risk re-scoring + explainable alerts + audit trail**

---

## What the System Does (Non-Technical)

1. **Profiles customers** across 13 AML risk dimensions
   (sanctions, PEP status, transaction patterns, country risk, etc.)

2. **Scores them 0–100** using a weighted formula aligned with
   FATF compliance guidelines

3. **Classifies them** into LOW / MEDIUM / HIGH / CRITICAL risk tiers
   using a machine learning model

4. **Explains why** each customer is flagged — which factors matter most —
   using SHAP values (the same technique used by enterprise AML vendors)

5. **Queues high-risk alerts** for compliance analysts to review,
   escalate, or close

6. **Logs every scoring event** with a full audit trail —
   required by regulators for inspection

---

## Key Numbers

| Metric | Value |
|---|---|
| Customers profiled | 500 |
| ML Model accuracy | 91.80% (5-fold CV) |
| ROC-AUC Score | 0.9865 |
| Open compliance alerts | 60 |
| REST API endpoints | 6 |
| Dashboard pages | 5 |
| Unit tests | 17 / 17 passing |
| Lines of Python code | ~1,200+ |

---

## Skills Demonstrated

### Data Engineering
- Synthetic data generation with realistic AML risk distributions
- Feature engineering pipeline with domain-weighted scoring
- ETL pipeline from CSV → SQLite with 3 normalized tables

### Machine Learning
- XGBoost multiclass classification (4 risk tiers)
- SMOTE for imbalanced AML datasets
- Stratified K-Fold cross-validation
- Model artifact serialization (joblib)

### Explainable AI (XAI)
- SHAP TreeExplainer for per-customer risk breakdowns
- Top-5 risk driver identification
- Human-readable reason generation for SAR support

### Backend Engineering
- FastAPI REST API with Pydantic v2 schema validation
- SQLAlchemy ORM with 3 relational tables
- CORS middleware, health checks, auto Swagger docs
- Structured logging with Loguru

### Frontend / Visualization
- 5-page Streamlit compliance dashboard
- Plotly interactive charts (donut, bar, histogram)
- Live alert status updates from the UI
- CSV export for audit log

### DevOps
- Docker containerization (API + Dashboard services)
- Docker Compose for multi-service orchestration
- .env-based config management
- .dockerignore and build optimization

### Testing
- 17 pytest unit tests
- Data integrity, business logic, and model validation
- 100% pass rate

---

## Project Architecture (One Line Per Layer)
Faker → data_generator.py → risk_features.py → train_model.py
→ explainer.py → database.py → load_to_db.py
→ FastAPI (main.py) → Streamlit (dashboard/app.py)


---

## How to Run It (30 seconds)

```bash
git clone https://github.com/PrasanthKumarS777/AML-KYC-RISK-SCORING.git
cd AML-KYC-RISK-SCORING
python -m venv venv && source venv/Scripts/activate
pip install -r requirements.txt && pip install -e .
python -m src.core.data_generator
python -m src.features.risk_features
python -m src.models.train_model
python -m src.models.explainer
python -m src.core.database
python -m src.pipelines.load_to_db
uvicorn main:app --reload --port 8000        # API → localhost:8000/docs
streamlit run dashboard/app.py               # UI  → localhost:8501
```

---

## What I Learned

- **AML compliance logic** — FATF risk tiers, PEP/sanctions screening,
  structuring patterns, SAR workflows
- **Why ML beats rules** in AML — rules generate 95%+ false positives;
  ML + SHAP gives analysts explainable, actionable alerts
- **Production thinking** — audit trails, alert queues, health checks,
  Docker, env configs — not just "it works on my machine"

---

## Domain Knowledge Acquired

| AML Concept | Applied In |
|---|---|
| FATF High-Risk Jurisdictions | Country risk score mapping |
| Politically Exposed Persons (PEP) | Feature + model weight |
| Structuring / Smurfing | structuring_flag feature |
| Enhanced Due Diligence (EDD) | CRITICAL/HIGH tier alerts |
| Suspicious Activity Report (SAR) | SHAP reason generation |
| Ongoing Monitoring | Dynamic re-scoring design |
| KYC Completeness | kyc_gap_score feature |

---

*Built by Prasanth Kumar Sahu | Personal Portfolio Project | April 2026*
*GitHub: https://github.com/PrasanthKumarS777/AML-KYC-RISK-SCORING*