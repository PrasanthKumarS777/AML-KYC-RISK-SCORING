import shap
import joblib
import pandas as pd
import numpy as np
import os
import json
from src.utils.logger import get_logger

logger = get_logger("explainer")

FEATURE_COLS = [
    "country_risk_score", "business_risk_score", "pep_score",
    "sanctioned_score", "structuring_score", "adverse_media_score",
    "cash_ratio_score", "country_diversity_score", "kyc_gap_score",
    "account_age_days", "num_transactions_monthly", "avg_transaction_amount",
    "num_sar_filed"
]

FEATURE_LABELS = {
    "country_risk_score":       "Country Risk",
    "business_risk_score":      "Business Type Risk",
    "pep_score":                "PEP Status",
    "sanctioned_score":         "Sanctions Match",
    "structuring_score":        "Structuring / Smurfing",
    "adverse_media_score":      "Adverse Media",
    "cash_ratio_score":         "Cash Transaction Ratio",
    "country_diversity_score":  "Cross-Border Activity",
    "kyc_gap_score":            "KYC Completeness Gap",
    "account_age_days":         "Account Age",
    "num_transactions_monthly": "Monthly Transaction Volume",
    "avg_transaction_amount":   "Avg Transaction Amount",
    "num_sar_filed":            "Prior SARs Filed"
}


def load_artifacts():
    model   = joblib.load("src/models/saved/risk_model.joblib")
    scaler  = joblib.load("src/models/saved/scaler.joblib")
    le      = joblib.load("src/models/saved/label_encoder.joblib")
    logger.info("Model artifacts loaded for SHAP explanation")
    return model, scaler, le


def explain_customer(customer_row: pd.Series, model, scaler, le) -> dict:
    """
    Returns a structured explanation for a single customer.
    """
    X = pd.DataFrame([customer_row[FEATURE_COLS].values], columns=FEATURE_COLS)
    X_scaled = scaler.transform(X)

    pred_class_idx = model.predict(X_scaled)[0]
    pred_proba     = model.predict_proba(X_scaled)[0]
    pred_label     = le.inverse_transform([pred_class_idx])[0]
    confidence     = round(float(pred_proba[pred_class_idx]) * 100, 2)

    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_scaled)

    # XGBoost multiclass returns 3D array: (n_samples, n_features, n_classes)
    # or list of 2D arrays depending on version — handle both
    if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        # shape: (1, n_features, n_classes) → pick predicted class
        class_shap = shap_values[0, :, pred_class_idx]
    elif isinstance(shap_values, list):
        # list of (n_samples, n_features) arrays — one per class
        class_shap = np.array(shap_values[pred_class_idx][0])
    else:
        # binary or flat array
        class_shap = np.array(shap_values[0])

    class_shap = class_shap.flatten()

    feature_impacts = []
    for i, feat in enumerate(FEATURE_COLS):
        feature_impacts.append({
            "feature":    feat,
            "label":      FEATURE_LABELS[feat],
            "value":      round(float(customer_row[feat]), 2),
            "shap_value": round(float(class_shap[i]), 4),
            "direction":  "increases_risk" if class_shap[i] > 0 else "decreases_risk"
        })

    feature_impacts.sort(key=lambda x: abs(x["shap_value"]), reverse=True)
    top_drivers = feature_impacts[:5]

    reasons = []
    for driver in top_drivers:
        if driver["direction"] == "increases_risk":
            reasons.append(f"{driver['label']} (impact: +{abs(driver['shap_value'])})")

    explanation = {
        "customer_id":      customer_row.get("customer_id", "N/A"),
        "name":             customer_row.get("name", "N/A"),
        "predicted_tier":   pred_label,
        "confidence_pct":   confidence,
        "composite_score":  round(float(customer_row.get("composite_risk_score", 0)), 2),
        "top_risk_drivers": top_drivers,
        "risk_reasons":     reasons,
        "all_class_proba":  {
            le.inverse_transform([i])[0]: round(float(p) * 100, 2)
            for i, p in enumerate(pred_proba)
        }
    }

    return explanation


def explain_batch(df: pd.DataFrame, model, scaler, le, save_path: str = "data/processed/explanations.json"):
    results = []
    for _, row in df.iterrows():
        try:
            exp = explain_customer(row, model, scaler, le)
            results.append(exp)
        except Exception as e:
            logger.warning(f"Skipped customer {row.get('customer_id', '?')}: {e}")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Saved {len(results)} SHAP explanations to {save_path}")
    return results


if __name__ == "__main__":
    df    = pd.read_csv("data/processed/customers_scored.csv")
    model, scaler, le = load_artifacts()

    # Show explanation for one HIGH risk and one CRITICAL risk customer
    for tier in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]:
        subset = df[df["risk_tier"] == tier]
        if len(subset) == 0:
            continue
        sample = subset.iloc[0]
        exp    = explain_customer(sample, model, scaler, le)

        print(f"\n{'='*55}")
        print(f"Customer  : {exp['name']}")
        print(f"Tier      : {exp['predicted_tier']}  |  Score: {exp['composite_score']}  |  Confidence: {exp['confidence_pct']}%")
        print(f"Class Probabilities: {exp['all_class_proba']}")
        print(f"\nTop Risk Drivers:")
        for d in exp["top_risk_drivers"]:
            arrow = "🔴" if d["direction"] == "increases_risk" else "🟢"
            print(f"  {arrow}  {d['label']:<30} val={d['value']}   shap={d['shap_value']}")
        print(f"\nRisk Reasons: {exp['risk_reasons']}")

    # Save all explanations
    print("\n\nGenerating explanations for all 500 customers...")
    explain_batch(df, model, scaler, le)
    print("✅ All SHAP explanations saved to data/processed/explanations.json")