import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from src.utils.logger import get_logger

logger = get_logger("train_model")

FEATURE_COLS = [
    "country_risk_score", "business_risk_score", "pep_score",
    "sanctioned_score", "structuring_score", "adverse_media_score",
    "cash_ratio_score", "country_diversity_score", "kyc_gap_score",
    "account_age_days", "num_transactions_monthly", "avg_transaction_amount",
    "num_sar_filed"
]

TARGET_COL = "risk_tier"


def load_data(path: str = "data/processed/customers_scored.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    logger.info(f"Loaded {len(df)} records from {path}")
    return df


def preprocess(df: pd.DataFrame):
    df = df.dropna(subset=[TARGET_COL])
    X = df[FEATURE_COLS]
    y = df[TARGET_COL]

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    logger.info(f"Classes: {le.classes_}")
    logger.info(f"Class distribution: {pd.Series(y_encoded).value_counts().to_dict()}")
    return X_scaled, y_encoded, le, scaler


def train(X, y, le):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    smote = SMOTE(random_state=42, k_neighbors=1)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    logger.info(f"After SMOTE — Train size: {len(X_train_res)}")

    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="mlogloss",
        random_state=42
    )

    model.fit(X_train_res, y_train_res)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    print("\n=== Classification Report ===")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    print("=== Confusion Matrix ===")
    print(confusion_matrix(y_test, y_pred))

    try:
        auc = roc_auc_score(y_test, y_prob, multi_class="ovr", average="weighted")
        print(f"\nROC-AUC Score (weighted OvR): {auc:.4f}")
        logger.info(f"ROC-AUC: {auc:.4f}")
    except Exception as e:
        logger.warning(f"AUC computation skipped: {e}")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
    print(f"\nCross-Validation Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    logger.info(f"CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    return model


def save_artifacts(model, scaler, le):
    os.makedirs("src/models/saved", exist_ok=True)
    joblib.dump(model,  "src/models/saved/risk_model.joblib")
    joblib.dump(scaler, "src/models/saved/scaler.joblib")
    joblib.dump(le,     "src/models/saved/label_encoder.joblib")
    logger.info("Model, scaler, and label encoder saved to src/models/saved/")
    print("\n✅ Model artifacts saved to src/models/saved/")


if __name__ == "__main__":
    df     = load_data()
    X, y, le, scaler = preprocess(df)
    model  = train(X, y, le)
    save_artifacts(model, scaler, le)