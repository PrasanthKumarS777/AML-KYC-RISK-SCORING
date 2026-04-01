<div align="center">

<img src="https://img.icons8.com/fluency/120/shield.png" alt="AML Shield" width="100"/>

<h1>🛡️ AML/KYC Customer Risk Re-Scoring System</h1>

<p><strong>An end-to-end Anti-Money Laundering compliance tool powered by Machine Learning, Explainable AI, and Real-Time Analytics</strong></p>

<p>
  <img src="https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/FastAPI-0.111-009688?style=for-the-badge&logo=fastapi&logoColor=white"/>
  <img src="https://img.shields.io/badge/Streamlit-1.35-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white"/>
  <img src="https://img.shields.io/badge/XGBoost-2.0-orange?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/SHAP-Explainable_AI-purple?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Tests-17%2F17_Passing-brightgreen?style=for-the-badge&logo=pytest"/>
  <img src="https://img.shields.io/badge/ROC--AUC-0.9865-success?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge"/>
</p>

<p>
  <a href="#-problem-statement">Problem</a> - 
  <a href="#-system-architecture">Architecture</a> - 
  <a href="#-ml-model-performance">ML Metrics</a> - 
  <a href="#-aml-features-used">AML Features</a> - 
  <a href="#-quick-start">Quick Start</a> - 
  <a href="#-api-endpoints">API Docs</a> - 
  <a href="#-dashboard-pages">Dashboard</a> - 
  <a href="#-tests">Tests</a>
</p>

</div>

***

## 📌 Project Overview

This is a **personal portfolio project** that simulates a real-world AML/KYC compliance system used by banks and FinTechs to continuously monitor customer risk. It is built entirely from scratch using Python, Machine Learning, REST APIs, and a compliance analyst dashboard.

> **Scope:** Fresher-level personal project demonstrating practical knowledge of AML risk logic, ML-based classification, SHAP explainability, REST APIs, and compliance workflows — inspired by real tools like NICE Actimize, Tookitaki, and Flagright.

### 🎯 What Problem Does This Solve?

Banks and FinTechs are legally required by regulators **(FATF, RBI, FCA, FinCEN)** to:

- Screen customers against **PEP lists** (Politically Exposed Persons) and **sanctions databases**
- Continuously monitor customer behavior for **suspicious activity patterns**
- File **Suspicious Activity Reports (SARs)** with regulators when red flags are detected
- Maintain a **full audit trail** of every risk scoring event for regulatory inspection

Traditional rule-based AML systems suffer from:

| Problem | Impact |
|---|---|
| Static risk scores after onboarding | Miss evolving risk behavior |
| Rule-based thresholds | 95%+ false positive rates |
| No explainability | Analysts cannot justify alerts to regulators |
| No audit trail | Non-compliance with FATF Recommendation 10 |

**This project addresses all four gaps** using a modern ML + XAI approach.

***

## 🏗️ System Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                     DATA LAYER                                   │
│                                                                  │
│   Synthetic Customer Profiles (500 records)                      │
│   src/core/data_generator.py                                     │
│   → 4 risk profiles: LOW / MEDIUM / HIGH / CRITICAL              │
│   → 17 attributes: PEP, sanctions, country, business type, etc.  │
└────────────────────────┬─────────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────────────┐
│                  FEATURE ENGINEERING LAYER                       │
│                                                                  │
│   src/features/risk_features.py                                  │
│   → 13 AML risk features computed from raw attributes            │
│   → Weighted composite risk score (0–100)                        │
│   → Country Risk Map (FATF jurisdictions)                        │
│   → Business Type Risk Map (crypto, money transfer, etc.)        │
│   → Risk tier classification: LOW / MEDIUM / HIGH / CRITICAL     │
└────────────────────────┬─────────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────────────┐
│                     ML MODEL LAYER                               │
│                                                                  │
│   src/models/train_model.py                                      │
│   → Algorithm: XGBoost Multiclass Classifier                     │
│   → Class Balancing: SMOTE oversampling                          │
│   → CV Accuracy: 91.80% ± 0.98%                                  │
│   → ROC-AUC: 0.9865 (weighted OvR)                               │
│   → Artifacts saved: model + scaler + label encoder              │
└────────────────────────┬─────────────────────────────────────────┘
                         │
              ┌──────────┴──────────┐
              ▼                     ▼
┌─────────────────────┐   ┌─────────────────────────────────────────┐
│  SHAP EXPLAINER      │   │           DATABASE LAYER                │
│                      │   │                                         │
│  src/models/         │   │  src/core/database.py (SQLite)          │
│  explainer.py        │   │                                         │
│  → TreeExplainer     │   │  Tables:                                │
│  → Top 5 risk        │   │  ├── customers (500 rows)               │
│    drivers per       │   │  ├── risk_audit_log (500 entries)       │
│    customer          │   │  └── alert_queue (60 open alerts)       │
│  → Human-readable    │   │                                         │
│    risk reasons      │   │  src/pipelines/load_to_db.py            │
│  → 500 explanations  │   │  → ETL pipeline: CSV → SQLite           │
│    saved as JSON     │   │                                         │
└──────────┬──────────┘   └──────────────────┬──────────────────────┘
           │                                  │
           └──────────────┬───────────────────┘
                          ▼
┌──────────────────────────────────────────────────────────────────┐
│                    API LAYER (FastAPI)                            │
│                                                                  │
│   main.py + src/api/routes.py + src/api/schemas.py               │
│   → 6 REST endpoints                                             │
│   → Pydantic schema validation                                   │
│   → CORS enabled for dashboard integration                       │
│   → Auto-generated Swagger docs at /docs                         │
│   → Runs on: http://localhost:8000                               │
└────────────────────────┬─────────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────────────┐
│                 DASHBOARD LAYER (Streamlit)                       │
│                                                                  │
│   dashboard/app.py                                               │
│   → Page 1: Dashboard Overview (KPIs, charts)                    │
│   → Page 2: Customer Risk Explorer (filterable table)            │
│   → Page 3: Compliance Alert Queue (update status live)          │
│   → Page 4: SHAP Explainer (per-customer risk breakdown)         │
│   → Page 5: Regulatory Audit Log (CSV download)                  │
│   → Runs on: http://localhost:8501                               │
└──────────────────────────────────────────────────────────────────┘
```

***

## 📊 ML Model Performance

<table>
  <tr>
    <th>Metric</th>
    <th>Value</th>
    <th>Notes</th>
  </tr>
  <tr>
    <td>Algorithm</td>
    <td>XGBoost Multiclass Classifier</td>
    <td>200 estimators, max_depth=6, lr=0.05</td>
  </tr>
  <tr>
    <td>Cross-Validation Accuracy</td>
    <td><strong>91.80% ± 0.98%</strong></td>
    <td>5-fold StratifiedKFold</td>
  </tr>
  <tr>
    <td>ROC-AUC Score</td>
    <td><strong>0.9865</strong></td>
    <td>Weighted One-vs-Rest multiclass</td>
  </tr>
  <tr>
    <td>Class Balancing</td>
    <td>SMOTE</td>
    <td>Synthetic Minority Oversampling</td>
  </tr>
  <tr>
    <td>Test Set Accuracy</td>
    <td>89%</td>
    <td>100-sample holdout</td>
  </tr>
  <tr>
    <td>LOW Risk Precision</td>
    <td>98%</td>
    <td>F1: 0.98</td>
  </tr>
  <tr>
    <td>HIGH Risk Precision</td>
    <td>50–56%</td>
    <td>Expected — borderline cases are genuinely ambiguous in AML</td>
  </tr>
</table>

### Risk Tier Distribution (500 Customers)

| Tier | Count | % of Portfolio | Priority |
|------|-------|----------------|----------|
| 🔴 CRITICAL | 40 | 8% | Immediate escalation required |
| 🟠 HIGH | 85 | 17% | Enhanced Due Diligence (EDD) |
| 🟡 MEDIUM | 175 | 35% | Standard Due Diligence (SDD) |
| 🟢 LOW | 200 | 40% | Simplified Due Diligence (SDD) |

> **Note:** HIGH ↔ MEDIUM confusion is intentional and realistic. In real AML systems, borderline customers require manual analyst review — the model correctly identifies uncertainty at tier boundaries.

***

## 🔬 AML Features Used

| # | Feature | Weight | AML Significance |
|---|---------|--------|-----------------|
| 1 | Country Risk Score | 20% | FATF high-risk and monitored jurisdictions (Iran=100, North Korea=100) |
| 2 | Business Type Risk | 15% | Crypto exchanges, money transfer, real estate = elevated layering risk |
| 3 | PEP Status | 15% | Politically Exposed Persons require Enhanced Due Diligence by law |
| 4 | Sanctions Match | 20% | OFAC, UN, EU sanctions list — highest single risk indicator |
| 5 | Structuring Flag | 10% | Smurfing — breaking large transactions into smaller amounts |
| 6 | Adverse Media Flag | 8% | Negative news screening from public sources |
| 7 | Cash Transaction Ratio | 5% | High cash usage = placement stage indicator |
| 8 | Cross-Border Activity | 4% | Multi-jurisdiction transactions = complex layering |
| 9 | KYC Completeness Gap | 3% | Incomplete documentation = higher due diligence gap |
| 10 | Account Age | — | New accounts with high volume = suspicious |
| 11 | Monthly Transaction Volume | — | Velocity monitoring |
| 12 | Avg Transaction Amount | — | Large transaction monitoring |
| 13 | Prior SARs Filed | +5 penalty | Historical suspicious activity reports filed |

***

## 🔍 SHAP Explainability — Sample Output

```
═══════════════════════════════════════════════════════
Customer  : Ashley Sweeney
Tier      : CRITICAL  |  Score: 100.0  |  Confidence: 99.75%
Class Probabilities: {'CRITICAL': 99.75, 'HIGH': 0.19, 'LOW': 0.02, 'MEDIUM': 0.04}

Top Risk Drivers:
  🔴  Sanctions Match              val=100.0   shap=+2.1804
  🔴  Prior SARs Filed             val=4.0     shap=+0.9353
  🔴  KYC Completeness Gap         val=43.0    shap=+0.5030
  🔴  Monthly Transaction Volume   val=196.0   shap=+0.3548
  🔴  Structuring / Smurfing       val=100.0   shap=+0.2580
═══════════════════════════════════════════════════════
Customer  : Judy Beck
Tier      : HIGH  |  Score: 66.5  |  Confidence: 96.18%

Top Risk Drivers:
  🔴  Prior SARs Filed             val=2.0     shap=+0.6360
  🔴  Monthly Transaction Volume   val=123.0   shap=+0.6021
  🟢  PEP Status                   val=0.0     shap=-0.4801  ← reduces risk
  🔴  Country Risk                 val=85.0    shap=+0.4282
  🔴  Avg Transaction Amount       val=183295  shap=+0.3125
═══════════════════════════════════════════════════════
```

> SHAP values show exactly how much each feature **pushes** the risk score up (🔴) or down (🟢) — this is what compliance officers need to write Suspicious Activity Reports (SARs).

***

## 🗄️ Database Schema

### `customers` table
| Column | Type | Description |
|--------|------|-------------|
| customer_id | String (PK) | UUID |
| name | String | Customer full name |
| country | String | Country of residence/operation |
| business_type | String | Industry type |
| is_pep | Boolean | Politically Exposed Person flag |
| is_sanctioned | Boolean | Sanctions list match |
| composite_risk_score | Float | ML-computed risk score (0–100) |
| risk_tier | String | LOW / MEDIUM / HIGH / CRITICAL |
| created_at | DateTime | Record creation timestamp |

### `risk_audit_log` table
| Column | Type | Description |
|--------|------|-------------|
| id | Integer (PK) | Auto-increment |
| customer_id | String | FK to customers |
| previous_tier | String | Risk tier before change |
| new_tier | String | Risk tier after change |
| change_reason | String | Why the score changed |
| reviewed_by | String | Analyst or system |
| timestamp | DateTime | When the change occurred |

### `alert_queue` table
| Column | Type | Description |
|--------|------|-------------|
| id | Integer (PK) | Auto-increment |
| customer_id | String | FK to customers |
| risk_tier | String | HIGH or CRITICAL only |
| risk_score | Float | Score at time of alert |
| status | String | OPEN / REVIEWED / ESCALATED / CLOSED |
| assigned_to | String | Compliance analyst assigned |
| created_at | DateTime | Alert creation time |
| resolved_at | DateTime | Alert resolution time |

***

## 🌐 API Endpoints

Base URL: `http://localhost:8000/api/v1`

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/summary` | Dashboard KPIs — total customers, tier counts, open alerts, avg score |
| `GET` | `/customers` | List all customers with optional `?risk_tier=HIGH&limit=50` filter |
| `GET` | `/customers/{id}` | Get single customer profile by UUID |
| `GET` | `/customers/{id}/explain` | SHAP explanation — top 5 risk drivers with values |
| `GET` | `/alerts` | Alert queue with optional `?status=OPEN` filter |
| `PATCH` | `/alerts/{id}` | Update alert status (REVIEWED / ESCALATED / CLOSED) |
| `GET` | `/audit-log` | Full regulatory audit trail |
| `GET` | `/health` | Health check endpoint |
| `GET` | `/docs` | Auto-generated Swagger UI |

### Sample API Responses

**GET /api/v1/summary**
```json
{
  "total_customers": 500,
  "critical_count": 16,
  "high_count": 44,
  "medium_count": 112,
  "low_count": 328,
  "open_alerts": 60,
  "avg_risk_score": 28.5,
  "pep_count": 72,
  "sanctioned_count": 28
}
```

**GET /api/v1/customers/{id}/explain**
```json
{
  "customer_id": "abc-123",
  "name": "Ashley Sweeney",
  "predicted_tier": "CRITICAL",
  "confidence_pct": 99.75,
  "composite_score": 100.0,
  "top_risk_drivers": [
    { "label": "Sanctions Match", "value": 100.0, "shap_value": 2.1804, "direction": "increases_risk" },
    { "label": "Prior SARs Filed", "value": 4.0, "shap_value": 0.9353, "direction": "increases_risk" }
  ],
  "risk_reasons": [
    "Sanctions Match (impact: +2.1804)",
    "Prior SARs Filed (impact: +0.9353)"
  ]
}
```

***

## 📺 Dashboard Pages

| Page | Description |
|------|-------------|
| 📊 **Dashboard Overview** | KPI cards, risk tier donut chart, alert coverage bar chart, compliance flag stats |
| 👥 **Customer Risk Explorer** | Filterable/sortable customer table with color-coded risk tiers and score distribution histogram |
| 🚨 **Alert Queue** | Open HIGH/CRITICAL alerts, live status update (assign to analyst, escalate, close) |
| 🔍 **SHAP Explainer** | Select any customer → see exactly why they are flagged with SHAP waterfall bar chart |
| 📋 **Audit Log** | Full regulatory audit trail with CSV export for compliance review |

***

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Git Bash (Windows) or Terminal (Mac/Linux)

### 1. Clone the repository
```bash
git clone https://github.com/PrasanthKumarS777/AML-KYC-RISK-SCORING.git
cd AML-KYC-RISK-SCORING
```

### 2. Create virtual environment and install dependencies
```bash
python -m venv venv
source venv/Scripts/activate      # Windows Git Bash
# source venv/bin/activate         # Mac/Linux

pip install -r requirements.txt
pip install -e .
```

### 3. Generate synthetic data and train model
```bash
# Step 1 — Generate 500 synthetic AML customer profiles
python -m src.core.data_generator

# Step 2 — Compute weighted AML risk scores
python -m src.features.risk_features

# Step 3 — Train XGBoost classifier
python -m src.models.train_model

# Step 4 — Generate SHAP explanations for all 500 customers
python -m src.models.explainer

# Step 5 — Initialize SQLite database
python -m src.core.database

# Step 6 — Load all data into database
python -m src.pipelines.load_to_db
```

### 4. Start the FastAPI backend
```bash
uvicorn main:app --reload --port 8000
```
> API running at: http://localhost:8000
> Swagger docs at: http://localhost:8000/docs

### 5. Start the Streamlit dashboard (new terminal)
```bash
streamlit run dashboard/app.py --server.port 8501
```
> Dashboard running at: http://localhost:8501

***

## 🧪 Tests

```bash
pytest tests/ -v
```

```
PASSED  test_generator_returns_dataframe
PASSED  test_generator_correct_row_count
PASSED  test_generator_required_columns
PASSED  test_generator_no_null_customer_ids
PASSED  test_generator_unique_customer_ids
PASSED  test_generator_cash_ratio_in_range
PASSED  test_generator_kyc_score_in_range
PASSED  test_risk_score_returns_dataframe
PASSED  test_risk_score_column_exists
PASSED  test_risk_tier_column_exists
PASSED  test_risk_score_in_valid_range
PASSED  test_risk_tier_valid_labels
PASSED  test_sanctioned_customer_gets_high_score
PASSED  test_pep_customer_score_higher_than_clean
PASSED  test_row_count_preserved_after_scoring
PASSED  test_model_artifacts_exist
PASSED  test_model_predicts_valid_tier

17 passed in 26.19s ✅
```

**Test Coverage:**
- ✅ Data generator integrity (row count, nulls, UUID uniqueness, value ranges)
- ✅ Risk scoring logic (column presence, score 0–100 range, valid tier labels)
- ✅ Business logic (PEP customers score higher than clean customers)
- ✅ Sanctioned customer threshold validation
- ✅ Model artifact existence and inference validation

***

## 📁 Project Structure

```
aml-kyc-risk-scoring/
│
├── src/
│   ├── core/
│   │   ├── data_generator.py      # Synthetic AML customer profile generator
│   │   └── database.py            # SQLAlchemy ORM — 3 tables
│   │
│   ├── features/
│   │   └── risk_features.py       # Weighted risk feature engineering (13 features)
│   │
│   ├── models/
│   │   ├── train_model.py         # XGBoost training + SMOTE + cross-validation
│   │   ├── explainer.py           # SHAP TreeExplainer engine
│   │   └── saved/                 # Trained artifacts (model, scaler, label encoder)
│   │
│   ├── pipelines/
│   │   └── load_to_db.py          # ETL: CSV → SQLite (customers + alerts + audit)
│   │
│   ├── api/
│   │   ├── routes.py              # FastAPI route handlers (6 endpoints)
│   │   └── schemas.py             # Pydantic request/response schemas
│   │
│   └── utils/
│       ├── config.py              # .env config loader
│       └── logger.py              # Loguru structured logging
│
├── dashboard/
│   └── app.py                     # 5-page Streamlit compliance dashboard
│
├── data/
│   ├── raw/
│   │   └── customers_raw.csv      # Generated: 500 customer profiles
│   ├── processed/
│   │   ├── customers_scored.csv   # Scored: composite risk scores + tiers
│   │   └── explanations.json      # SHAP: 500 customer explanations
│   └── synthetic/
│
├── tests/
│   └── test_pipeline.py           # 17 unit tests (all passing)
│
├── logs/                          # Auto-generated daily rotating logs
├── docs/
│
├── main.py                        # FastAPI application entry point
├── setup.py                       # Package installation config
├── requirements.txt               # All Python dependencies
├── .env                           # Environment variables (local only)
├── .env.example                   # Environment variable template
├── .gitignore                     # Git ignore rules
└── README.md                      # This file
```

***

## 🛠️ Tech Stack

<table>
  <tr>
    <th>Layer</th>
    <th>Technology</th>
    <th>Purpose</th>
  </tr>
  <tr>
    <td>Language</td>
    <td>Python 3.11</td>
    <td>Core development language</td>
  </tr>
  <tr>
    <td>ML Framework</td>
    <td>XGBoost 2.0</td>
    <td>Risk classification model</td>
  </tr>
  <tr>
    <td>Class Balancing</td>
    <td>imbalanced-learn (SMOTE)</td>
    <td>Handle imbalanced AML datasets</td>
  </tr>
  <tr>
    <td>Explainability</td>
    <td>SHAP 0.45</td>
    <td>TreeExplainer for risk drivers</td>
  </tr>
  <tr>
    <td>Anomaly Detection</td>
    <td>PyOD</td>
    <td>Unsupervised outlier detection</td>
  </tr>
  <tr>
    <td>API Framework</td>
    <td>FastAPI 0.111</td>
    <td>REST API with auto Swagger docs</td>
  </tr>
  <tr>
    <td>API Server</td>
    <td>Uvicorn</td>
    <td>ASGI web server</td>
  </tr>
  <tr>
    <td>Data Validation</td>
    <td>Pydantic v2</td>
    <td>Request/response schema validation</td>
  </tr>
  <tr>
    <td>Dashboard</td>
    <td>Streamlit 1.35</td>
    <td>Interactive compliance UI</td>
  </tr>
  <tr>
    <td>Charts</td>
    <td>Plotly 5.22</td>
    <td>Interactive risk visualizations</td>
  </tr>
  <tr>
    <td>Database</td>
    <td>SQLite + SQLAlchemy 2.0</td>
    <td>ORM-based persistent storage</td>
  </tr>
  <tr>
    <td>Data Processing</td>
    <td>Pandas 2.2, NumPy 1.26</td>
    <td>Data manipulation and analysis</td>
  </tr>
  <tr>
    <td>Synthetic Data</td>
    <td>Faker 25.2</td>
    <td>Realistic customer profile generation</td>
  </tr>
  <tr>
    <td>Logging</td>
    <td>Loguru</td>
    <td>Structured daily rotating logs</td>
  </tr>
  <tr>
    <td>Testing</td>
    <td>Pytest 8.2</td>
    <td>17 unit tests, 100% pass rate</td>
  </tr>
  <tr>
    <td>Config</td>
    <td>python-dotenv</td>
    <td>Environment variable management</td>
  </tr>
</table>

***

## 📜 Regulatory Context

This project simulates compliance with the following real-world AML frameworks:

| Framework | Requirement Simulated |
|-----------|----------------------|
| **FATF Recommendation 10** | Customer Due Diligence — ongoing monitoring |
| **FATF Recommendation 12** | Politically Exposed Persons (PEP) screening |
| **FATF Recommendation 6** | Targeted financial sanctions screening |
| **Basel AML Index** | Country risk scoring based on jurisdiction |
| **FinCEN SAR Filing** | Suspicious Activity Report documentation |
| **GDPR / Data Privacy** | Synthetic data used — no real customer PII |

***

## ⚠️ Disclaimer

> This project uses **100% synthetic data** generated by the Faker library. No real customer data, no real financial transactions, and no actual sanctions lists are used. This is a personal portfolio project built for educational and demonstration purposes only. It does not constitute financial or legal advice.

***

## 👤 Author

**Prasanth Kumar Sahu**
- 🔗 GitHub: [@PrasanthKumarS777](https://github.com/PrasanthKumarS777)
- 📍 Cuttack, Odisha, India
- 🎯 Open to: Data Analyst | Business Analyst | ML Engineer roles

***

## 📄 License

This project is licensed under the **MIT License** — feel free to use, modify, and share with attribution.

***

<div align="center">
  <p>⭐ If this project helped you, consider giving it a star!</p>
  <p>Built with 🛡️ for AML compliance awareness</p>
</div>