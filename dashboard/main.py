"""
Bhairava AI — Live Monitoring Dashboard
White & Purple Modern Fintech UI for Razorpay AI Buildathon 2026.
"""

import sys
from pathlib import Path

# Set project root to path
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import streamlit as st

from app.services.audit_store import audit_store
from app.services.auto_responder import auto_responder_service
from app.services.fraud_detector import detector_service
from app.schemas.transaction import TransactionPayload

from dashboard.components import (
    action_breakdown_chart,
    risk_distribution_chart,
    shap_attribution_chart,
    render_recent_decisions,
)


# ---------------------------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Bhairava AI | Risk & Fraud Operations Center",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Modern White & Purple Fintech Custom CSS
# ---------------------------------------------------------------------------

st.markdown(
    """
    <style>
    /* Global Typography & Background */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Main container background */
    .stApp {
        background-color: #F8FAFC;
    }
    
    /* Sleek Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #FFFFFF !important;
        border-right: 1px solid #E2E8F0;
    }
    
    /* Top Header Card */
    .hero-header {
        background: linear-gradient(135deg, #6D28D9 0%, #4F46E5 100%);
        padding: 24px 30px;
        border-radius: 16px;
        color: white;
        margin-bottom: 24px;
        box-shadow: 0 10px 25px -5px rgba(79, 70, 229, 0.25);
    }
    
    .hero-header h1 {
        color: white !important;
        font-weight: 800;
        font-size: 1.85rem;
        margin-bottom: 4px;
    }
    
    .hero-header p {
        color: #E0E7FF !important;
        font-size: 0.95rem;
        margin: 0;
    }
    
    /* Metric Cards */
    .kpi-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    
    .kpi-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 16px -4px rgba(124, 58, 237, 0.1);
        border-color: #DDD6FE;
    }
    
    .kpi-title {
        font-size: 0.8rem;
        font-weight: 600;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 6px;
    }
    
    .kpi-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #0F172A;
    }
    
    .kpi-sub {
        font-size: 0.78rem;
        color: #7C3AED;
        font-weight: 500;
        margin-top: 4px;
    }
    
    /* Card Containers */
    .content-box {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
    
    /* Purple Primary Button */
    .stButton>button, .stFormSubmitButton>button {
        background: linear-gradient(135deg, #7C3AED 0%, #6366F1 100%) !important;
        color: white !important;
        font-weight: 600 !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 10px 24px !important;
        box-shadow: 0 4px 12px rgba(124, 58, 237, 0.25) !important;
        transition: all 0.2s ease !important;
    }
    
    .stButton>button:hover, .stFormSubmitButton>button:hover {
        opacity: 0.95 !important;
        transform: scale(1.01) !important;
    }
    
    /* Result Banners */
    .badge-allow {
        background-color: #ECFDF5;
        border: 1px solid #A7F3D0;
        border-left: 5px solid #10B981;
        padding: 16px 20px;
        border-radius: 10px;
        color: #065F46;
    }
    
    .badge-challenge {
        background-color: #FFFBEB;
        border: 1px solid #FDE68A;
        border-left: 5px solid #F59E0B;
        padding: 16px 20px;
        border-radius: 10px;
        color: #92400E;
    }
    
    .badge-decline {
        background-color: #FEF2F2;
        border: 1px solid #FECACA;
        border-left: 5px solid #EF4444;
        padding: 16px 20px;
        border-radius: 10px;
        color: #991B1B;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Sidebar Navigation & System Telemetry
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown(
        """
        <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 12px;">
            <div style="background: #7C3AED; width: 36px; height: 36px; border-radius: 8px; display: flex; align-items: center; justify-content: center; color: white; font-weight: bold; font-size: 20px;">⚡</div>
            <div>
                <h3 style="margin:0; font-size: 1.15rem; font-weight: 700; color: #1E293B;">Bhairava AI</h3>
                <span style="font-size: 0.75rem; color: #7C3AED; font-weight: 600;">Track 2: AI Risk Manager</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.divider()

    model_online = detector_service.model is not None
    if model_online:
        st.markdown(
            """
            <div style="background:#ECFDF5; border:1px solid #A7F3D0; padding:10px 14px; border-radius:8px; display:flex; align-items:center; gap:8px;">
                <span style="color:#10B981; font-size:14px;">●</span>
                <span style="color:#065F46; font-weight:600; font-size:0.85rem;">XGBoost Pipeline: Active</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.write("")
        st.caption(f"**Feature Matrix:** `{len(detector_service.feature_names):,}` dimensions")
        st.caption(f"**Decision Threshold:** `{detector_service.locked_threshold:.2f}` (Validation Locked)")
        st.caption(f"**Dataset Base:** IEEE-CIS (590k samples)")
    else:
        st.error("Model Engine Offline")

    st.divider()
    st.markdown("<p style='font-size:0.8rem; color:#64748B; margin-bottom:12px;'>Operations</p>", unsafe_allow_html=True)
    if st.button("🔄 Refresh Telemetry", use_container_width=True):
        st.rerun()

    st.write("")
    st.caption("Built for **Razorpay AI Buildathon 2026**")


# ---------------------------------------------------------------------------
# Hero Header Banner
# ---------------------------------------------------------------------------

st.markdown(
    """
    <div class="hero-header">
        <h1>Bhairava AI Operations Center</h1>
        <p>Real-time Fraud Probability Scoring, 3-Tier Policy Auto-Responder & Persistent Audit Telemetry</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# KPI Cards Row
# ---------------------------------------------------------------------------

stats = audit_store.get_aggregate_stats()
recent_decisions = audit_store.get_recent_decisions(limit=50)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-title">Transactions Evaluated</div>
            <div class="kpi-value">{stats['total_evaluated']:,}</div>
            <div class="kpi-sub">Total live throughput</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col2:
    avg_score_str = f"{stats['avg_risk_score']:.4f}" if stats["total_evaluated"] > 0 else "0.0000"
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-title">Average Risk Score</div>
            <div class="kpi-value">{avg_score_str}</div>
            <div class="kpi-sub">Population risk index</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col3:
    fraud_rate = stats["fraud_confirmation_rate"]
    rate_str = f"{fraud_rate:.1%}" if fraud_rate is not None else "Pending Data"
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-title">Confirmed Fraud Rate</div>
            <div class="kpi-value">{rate_str}</div>
            <div class="kpi-sub">Merchant verified chargebacks</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col4:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-title">Feedback Ingested</div>
            <div class="kpi-value">{stats['feedback_submitted']}</div>
            <div class="kpi-sub">Audit trail closed loops</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.write("")


# ---------------------------------------------------------------------------
# Visual Analytics (Charts Row)
# ---------------------------------------------------------------------------

col_chart_left, col_chart_right = st.columns([1, 2])

with col_chart_left:
    st.markdown(
        """
        <div class="content-box">
            <h4 style="margin:0 0 12px 0; color:#1E293B; font-weight:700; font-size:1.05rem;">Policy Action Distribution</h4>
        """,
        unsafe_allow_html=True,
    )
    if stats["total_evaluated"] > 0:
        fig_donut = action_breakdown_chart(stats["action_breakdown"])
        st.plotly_chart(fig_donut, use_container_width=True)
    else:
        st.info("No evaluations yet. Test a transaction below.")
    st.markdown("</div>", unsafe_allow_html=True)

with col_chart_right:
    st.markdown(
        """
        <div class="content-box">
            <h4 style="margin:0 0 12px 0; color:#1E293B; font-weight:700; font-size:1.05rem;">Population Risk Distribution</h4>
        """,
        unsafe_allow_html=True,
    )
    if recent_decisions:
        fig_hist = risk_distribution_chart(recent_decisions)
        st.plotly_chart(fig_hist, use_container_width=True)
    else:
        st.info("No risk distribution data recorded yet.")
    st.markdown("</div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Live Transaction Scorer & Decision Simulator
# ---------------------------------------------------------------------------

st.markdown(
    """
    <div class="content-box">
        <h4 style="margin:0 0 6px 0; color:#1E293B; font-weight:700; font-size:1.15rem;">Live Transaction Simulator & Risk Scorer</h4>
        <p style="color:#64748B; font-size:0.88rem; margin-bottom:16px;">
            Simulate an incoming merchant transaction payload to test the end-to-end ML inference, 3-tier policy decisioning, explainability attribution, and automated SQLite persistence.
        </p>
    """,
    unsafe_allow_html=True,
)

with st.form("transaction_scorer", clear_on_submit=False):
    col_a, col_b, col_c = st.columns(3)

    with col_a:
        txn_id = st.text_input("Transaction ID", value="TXN_RZP_4091")
        amount = st.number_input(
            "Transaction Amount ($ / ₹)",
            min_value=0.01,
            max_value=100_000.0,
            value=185.50,
            step=10.0,
            help="High values or abnormal cents distributions trigger risk heuristics.",
        )
        email_domain = st.selectbox(
            "Purchaser Email Domain",
            options=["gmail.com", "yahoo.com", "hotmail.com", "protonmail.com", "tempmail.org", "anonymous.io"],
            help="High-risk and throwaway domains trigger risk spikes.",
        )

    with col_b:
        card1 = st.number_input(
            "Card BIN Hash (card1)",
            min_value=1_000,
            max_value=99_999,
            value=13926,
            step=1,
            help="Issuer identification number hash.",
        )
        card2 = st.number_input(
            "Card Category (card2)",
            min_value=100.0,
            max_value=600.0,
            value=321.0,
            step=10.0,
        )
        card3 = st.number_input(
            "Card Country (card3)",
            min_value=100.0,
            max_value=200.0,
            value=150.0,
            step=5.0,
        )

    with col_c:
        addr1 = st.number_input(
            "Billing Region Prefix (addr1)",
            min_value=100.0,
            max_value=500.0,
            value=299.0,
            step=10.0,
        )
        has_identity = st.radio(
            "Identity / Device Telemetry",
            options=[1, 0],
            format_func=lambda x: "Present (Browser/Device Fingerprint)" if x == 1 else "Missing (High Anonymity)",
        )

    st.write("")
    submitted = st.form_submit_button("⚡ Evaluate Transaction & Execute Auto-Responder", use_container_width=True)

st.markdown("</div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Decision Result Card
# ---------------------------------------------------------------------------

if submitted:
    try:
        payload = TransactionPayload(
            transaction_id=txn_id,
            amount=float(amount),
            card1=int(card1),
            card2=float(card2),
            card3=float(card3),
            addr1=float(addr1),
            P_emaildomain=email_domain,
            has_identity_data=int(has_identity),
        )
        decision = auto_responder_service.evaluate_transaction(payload)
    except Exception as e:
        st.error(f"Execution failed: {str(e)}")
        st.stop()

    action_val = decision.action.value
    tier_val = decision.risk_tier.value

    badge_class = (
        "badge-allow" if action_val == "ALLOW"
        else ("badge-challenge" if action_val == "CHALLENGE_3DS" else "badge-decline")
    )
    badge_icon = "✅" if action_val == "ALLOW" else ("⚠️" if action_val == "CHALLENGE_3DS" else "🚫")

    st.markdown(
        f"""
        <div class="{badge_class}">
            <div style="font-size: 1.4rem; font-weight: 800; display:flex; align-items:center; gap:8px;">
                <span>{badge_icon}</span>
                <span>ACTION: {action_val.replace('_', ' ')}</span>
                <span style="font-size:0.9rem; font-weight:600; opacity:0.85; margin-left:auto;">Risk Tier: {tier_val}</span>
            </div>
            <div style="font-size:0.92rem; margin-top:6px;">
                Transaction <code>{decision.transaction_id}</code> evaluated with <b>{decision.risk_score:.2%}</b> fraud probability.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")

    res_c1, res_c2, res_c3, res_c4 = st.columns(4)
    with res_c1:
        st.metric("Model Probability", f"{decision.risk_score:.4f}")
    with res_c2:
        st.metric("Decision Confidence", f"{decision.confidence:.1%}")
    with res_c3:
        st.metric("3DS Step-Up OTP", "TRIGGERED" if decision.requires_otp_challenge else "BYPASSED")
    with res_c4:
        st.metric("Audit Trail ID", decision.audit_id)

    # --- Explainability & SHAP Attribution ---
    st.write("")
    exp_col1, exp_col2 = st.columns([3, 2])

    with exp_col1:
        st.markdown("<p style='font-size:0.95rem; font-weight:700; color:#1E293B; margin-bottom:6px;'>TreeSHAP Mathematical Feature Contributions</p>", unsafe_allow_html=True)
        if decision.shap_attribution:
            fig_shap = shap_attribution_chart(decision.shap_attribution)
            st.plotly_chart(fig_shap, use_container_width=True)
        else:
            st.info("SHAP attribution calculated at API level.")

    with exp_col2:
        st.markdown("<p style='font-size:0.95rem; font-weight:700; color:#1E293B; margin-bottom:6px;'>Attributed Policy Reason Codes</p>", unsafe_allow_html=True)
        reasons = decision.reasons or ["LOW_RISK_NORMAL_BEHAVIOR"]
        for r in reasons:
            st.markdown(
                f"<div style='background:#F1F5F9; border-left:3px solid #7C3AED; padding:8px 12px; border-radius:6px; margin-bottom:8px; font-size:0.85rem; font-weight:600; color:#334155;'>{r}</div>",
                unsafe_allow_html=True,
            )
        if decision.shap_attribution and decision.shap_attribution.top_features:
            top_f = decision.shap_attribution.top_features[0]
            st.caption(f"💡 **Dominant Driver:** `{top_f.friendly_name}` ({top_f.direction.replace('_', ' ')})")

    with st.expander("📬 Real-Time Merchant Notification Payload"):
        notif = decision.merchant_notification
        n_col1, n_col2 = st.columns([1, 2])
        with n_col1:
            st.markdown(f"**Severity:** `{notif.severity.value}`")
            st.markdown(f"**Dispatch Status:** `{'Active Alert' if notif.enabled else 'Silent Clear'}`")
            st.markdown(f"**Action Required:** `{notif.action_required}`")
        with n_col2:
            st.markdown(f"**{notif.title}**")
            st.write(notif.message)


# ---------------------------------------------------------------------------
# Decision Audit Trail (Table)
# ---------------------------------------------------------------------------

st.markdown(
    """
    <div class="content-box">
        <h4 style="margin:0 0 12px 0; color:#1E293B; font-weight:700; font-size:1.05rem;">Live Persistent Audit Ledger</h4>
    """,
    unsafe_allow_html=True,
)

if recent_decisions:
    render_recent_decisions(recent_decisions[:20])
else:
    st.info("Audit log is currently empty.")

st.markdown("</div>", unsafe_allow_html=True)
