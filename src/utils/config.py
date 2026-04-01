from dotenv import load_dotenv
import os

load_dotenv()

APP_NAME = os.getenv("APP_NAME", "AML-KYC-Risk-Scoring")
APP_VERSION = os.getenv("APP_VERSION", "1.0.0")
DEBUG = os.getenv("DEBUG", "True") == "True"

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./aml_kyc.db")

LOW_RISK_THRESHOLD = int(os.getenv("LOW_RISK_THRESHOLD", 30))
MEDIUM_RISK_THRESHOLD = int(os.getenv("MEDIUM_RISK_THRESHOLD", 60))
HIGH_RISK_THRESHOLD = int(os.getenv("HIGH_RISK_THRESHOLD", 80))

MODEL_PATH = os.getenv("MODEL_PATH", "src/models/saved/risk_model.joblib")
SCALER_PATH = os.getenv("SCALER_PATH", "src/models/saved/scaler.joblib")

RISK_LABELS = {
    "LOW": "Low Risk",
    "MEDIUM": "Medium Risk",
    "HIGH": "High Risk",
    "CRITICAL": "Critical Risk"
}