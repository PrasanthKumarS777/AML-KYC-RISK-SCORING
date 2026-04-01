import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

API = "http://localhost:8000/api/v1"

st.set_page_config(
    page_title="AML/KYC Risk Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ──────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0f1117; }
    .metric-card {
        background: #1e2130;
        border-radius: 10px;
        padding: 20px;
        border-left: 4px solid;
        margin-bottom: 10px;
    }
    .critical { border-color: #ff4b4b; }
    .high     { border-color: #ff8c00; }
    .medium   { border-color: #ffd700; }
    .low      { border-color: #00cc88; }
    .stDataFrame { font-size: 13px; }
</style>
""", unsafe_allow_html=True)


# ── Sidebar ──────────────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/fluency/96/shield.png", width=60)
st.sidebar.title("🛡️ AML/KYC Monitor")
st.sidebar.markdown("**Risk Scoring & Alert System**")
st.sidebar.divider()
page = st.sidebar.radio("Navigate", [
    "📊 Dashboard Overview",
    "👥 Customer Risk Explorer",
    "🚨 Alert Queue",
    "🔍 SHAP Explainer",
    "📋 Audit Log"
])
st.sidebar.divider()
st.sidebar.caption("v1.0.0 | AML Compliance Tool")


# ── Helper ───────────────────────────────────────────────────
@st.cache_data(ttl=30)
def fetch(endpoint, params=None):
    try:
        r = requests.get(f"{API}{endpoint}", params=params, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"API Error: {e}")
        return None

TIER_COLORS = {
    "CRITICAL": "#ff4b4b",
    "HIGH":     "#ff8c00",
    "MEDIUM":   "#ffd700",
    "LOW":      "#00cc88"
}


# ══════════════════════════════════════════════════════════════
# PAGE 1 — DASHBOARD OVERVIEW
# ══════════════════════════════════════════════════════════════
if page == "📊 Dashboard Overview":
    st.title("📊 AML/KYC Risk Dashboard")
    st.caption("Real-time customer risk monitoring and compliance analytics")
    st.divider()

    data = fetch("/summary")
    if data:
        # KPI Row
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("Total Customers",  data["total_customers"])
        c2.metric("🔴 Critical",      data["critical_count"],  delta="High Priority")
        c3.metric("🟠 High Risk",     data["high_count"])
        c4.metric("🟡 Medium Risk",   data["medium_count"])
        c5.metric("🟢 Low Risk",      data["low_count"])
        c6.metric("⚠️ Open Alerts",   data["open_alerts"])

        st.divider()

        col1, col2, col3 = st.columns(3)

        # Donut chart
        with col1:
            st.subheader("Risk Tier Distribution")
            labels = ["CRITICAL", "HIGH", "MEDIUM", "LOW"]
            values = [data["critical_count"], data["high_count"],
                      data["medium_count"],   data["low_count"]]
            colors = [TIER_COLORS[l] for l in labels]
            fig = go.Figure(go.Pie(
                labels=labels, values=values,
                hole=0.55,
                marker_colors=colors,
                textinfo="percent+label"
            ))
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="white",
                showlegend=False,
                margin=dict(t=10, b=10)
            )
            st.plotly_chart(fig, use_container_width=True)

        # Bar chart — alert vs customers
        with col2:
            st.subheader("Alert Coverage")
            fig2 = go.Figure()
            fig2.add_trace(go.Bar(
                x=["Critical", "High", "Medium", "Low"],
                y=[data["critical_count"], data["high_count"],
                   data["medium_count"],   data["low_count"]],
                marker_color=[TIER_COLORS["CRITICAL"], TIER_COLORS["HIGH"],
                              TIER_COLORS["MEDIUM"],   TIER_COLORS["LOW"]],
                text=[data["critical_count"], data["high_count"],
                      data["medium_count"],   data["low_count"]],
                textposition="outside"
            ))
            fig2.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="white",
                margin=dict(t=10, b=10),
                yaxis=dict(gridcolor="#2a2a3e")
            )
            st.plotly_chart(fig2, use_container_width=True)

        # Stats
        with col3:
            st.subheader("Compliance Flags")
            st.metric("PEP Customers",       data["pep_count"])
            st.metric("Sanctioned Matches",  data["sanctioned_count"])
            st.metric("Avg Risk Score",      f"{data['avg_risk_score']} / 100")
            alert_rate = round(data["open_alerts"] / data["total_customers"] * 100, 1)
            st.metric("Alert Rate",          f"{alert_rate}%")

            risk_rate = round((data["critical_count"] + data["high_count"])
                              / data["total_customers"] * 100, 1)
            st.progress(risk_rate / 100)
            st.caption(f"{risk_rate}% of customers are HIGH or CRITICAL risk")


# ══════════════════════════════════════════════════════════════
# PAGE 2 — CUSTOMER RISK EXPLORER
# ══════════════════════════════════════════════════════════════
elif page == "👥 Customer Risk Explorer":
    st.title("👥 Customer Risk Explorer")
    st.divider()

    tier_filter = st.selectbox("Filter by Risk Tier",
                               ["ALL", "CRITICAL", "HIGH", "MEDIUM", "LOW"])
    params = {"limit": 500}
    if tier_filter != "ALL":
        params["risk_tier"] = tier_filter

    customers = fetch("/customers", params=params)
    if customers:
        df = pd.DataFrame(customers)
        df["risk_score"] = df["composite_risk_score"].round(2)

        # Color-coded risk tier badge
        def color_tier(val):
            colors = {
                "CRITICAL": "background-color: #ff4b4b; color: white",
                "HIGH":     "background-color: #ff8c00; color: white",
                "MEDIUM":   "background-color: #ffd700; color: black",
                "LOW":      "background-color: #00cc88; color: black"
            }
            return colors.get(val, "")

        display_cols = ["name", "country", "business_type", "risk_tier",
                        "risk_score", "is_pep", "is_sanctioned",
                        "structuring_flag", "num_sar_filed"]

        st.dataframe(
            df[display_cols].style.applymap(color_tier, subset=["risk_tier"]),
            use_container_width=True,
            height=500
        )
        st.caption(f"Showing {len(df)} customers")

        # Score distribution
        st.subheader("Risk Score Distribution")
        fig = px.histogram(
            df, x="composite_risk_score", color="risk_tier",
            color_discrete_map=TIER_COLORS,
            nbins=40, template="plotly_dark"
        )
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════
# PAGE 3 — ALERT QUEUE
# ══════════════════════════════════════════════════════════════
elif page == "🚨 Alert Queue":
    st.title("🚨 Compliance Alert Queue")
    st.divider()

    status_filter = st.selectbox("Filter by Status",
                                 ["OPEN", "REVIEWED", "ESCALATED", "CLOSED", "ALL"])
    params = {"limit": 200}
    if status_filter != "ALL":
        params["status"] = status_filter

    alerts = fetch("/alerts", params=params)
    if alerts:
        df = pd.DataFrame(alerts)
        df["created_at"] = pd.to_datetime(df["created_at"]).dt.strftime("%Y-%m-%d %H:%M")

        st.dataframe(
            df[["id", "customer_name", "risk_tier", "risk_score",
                "alert_reason", "status", "assigned_to", "created_at"]],
            use_container_width=True,
            height=400
        )
        st.caption(f"{len(df)} alerts found")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Update Alert Status")
            alert_id    = st.number_input("Alert ID", min_value=1, step=1)
            new_status  = st.selectbox("New Status",
                                       ["REVIEWED", "ESCALATED", "CLOSED"])
            assigned_to = st.text_input("Assign To", value="analyst_1")
            if st.button("✅ Update Alert"):
                try:
                    r = requests.patch(
                        f"{API}/alerts/{int(alert_id)}",
                        json={"status": new_status, "assigned_to": assigned_to}
                    )
                    if r.status_code == 200:
                        st.success(f"Alert {alert_id} updated to {new_status}")
                        st.cache_data.clear()
                    else:
                        st.error(f"Failed: {r.text}")
                except Exception as e:
                    st.error(str(e))

        with col2:
            st.subheader("Alert Tier Breakdown")
            tier_counts = df["risk_tier"].value_counts().reset_index()
            tier_counts.columns = ["Tier", "Count"]
            fig = px.pie(tier_counts, names="Tier", values="Count",
                         color="Tier", color_discrete_map=TIER_COLORS,
                         template="plotly_dark")
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════
# PAGE 4 — SHAP EXPLAINER
# ══════════════════════════════════════════════════════════════
elif page == "🔍 SHAP Explainer":
    st.title("🔍 SHAP Risk Explainer")
    st.caption("Understand WHY a customer is flagged — just like a real compliance analyst would")
    st.divider()

    customers = fetch("/customers", params={"limit": 500})
    if customers:
        df = pd.DataFrame(customers)
        options = df.apply(
            lambda r: f"{r['name']} | {r['risk_tier']} | Score: {r['composite_risk_score']}",
            axis=1
        ).tolist()
        selected = st.selectbox("Select a Customer", options)
        idx      = options.index(selected)
        cust_id  = df.iloc[idx]["customer_id"]

        if st.button("🔍 Explain Risk"):
            with st.spinner("Running SHAP analysis..."):
                exp = fetch(f"/customers/{cust_id}/explain")

            if exp:
                col1, col2 = st.columns(2)
                with col1:
                    tier  = exp["predicted_tier"]
                    color = TIER_COLORS.get(tier, "#888")
                    st.markdown(f"""
                    <div style='background:#1e2130;padding:20px;border-radius:10px;
                                border-left:5px solid {color}'>
                        <h3 style='color:{color}'>{tier} RISK</h3>
                        <p>Score: <b>{exp['composite_score']}</b> / 100</p>
                        <p>Confidence: <b>{exp['confidence_pct']}%</b></p>
                    </div>""", unsafe_allow_html=True)

                    st.subheader("Class Probabilities")
                    prob_df = pd.DataFrame(
                        list(exp["all_class_proba"].items()),
                        columns=["Tier", "Probability %"]
                    )
                    fig = px.bar(prob_df, x="Tier", y="Probability %",
                                 color="Tier", color_discrete_map=TIER_COLORS,
                                 template="plotly_dark")
                    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                                      plot_bgcolor="rgba(0,0,0,0)",
                                      showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)

                with col2:
                    st.subheader("Top Risk Drivers (SHAP)")
                    drivers = exp["top_risk_drivers"]
                    labels  = [d["label"] for d in drivers]
                    values  = [d["shap_value"] for d in drivers]
                    colors  = ["#ff4b4b" if d["direction"] == "increases_risk"
                               else "#00cc88" for d in drivers]
                    fig2 = go.Figure(go.Bar(
                        x=values, y=labels,
                        orientation="h",
                        marker_color=colors,
                        text=[f"{v:+.4f}" for v in values],
                        textposition="outside"
                    ))
                    fig2.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font_color="white",
                        xaxis_title="SHAP Value (impact on risk)",
                        margin=dict(l=10, r=10)
                    )
                    st.plotly_chart(fig2, use_container_width=True)

                st.subheader("📝 Risk Reasons")
                for reason in exp["risk_reasons"]:
                    st.markdown(f"🔴 {reason}")


# ══════════════════════════════════════════════════════════════
# PAGE 5 — AUDIT LOG
# ══════════════════════════════════════════════════════════════
elif page == "📋 Audit Log":
    st.title("📋 Regulatory Audit Log")
    st.caption("Full trail of every risk scoring event — required for compliance audits")
    st.divider()

    logs = fetch("/audit-log", params={"limit": 200})
    if logs:
        df = pd.DataFrame(logs)
        st.dataframe(df, use_container_width=True, height=500)
        st.caption(f"{len(df)} audit entries")
        st.download_button(
            "⬇️ Download Audit Log CSV",
            data=df.to_csv(index=False),
            file_name="aml_audit_log.csv",
            mime="text/csv"
        )