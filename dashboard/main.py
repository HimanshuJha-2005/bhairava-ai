"""
Bhairava AI — Risk & Fraud Operations Center
Production-grade, Razorpay-styled Fintech UI for Track 2: AI Risk Manager.
"""

import sys
from pathlib import Path

# Ensure project root is available on path
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
    page_title="Bhairava AI — Risk & Fraud Operations",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# High-End Fintech Custom CSS (Razorpay / Stripe Aesthetic)
# ---------------------------------------------------------------------------

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .stApp {
        background-color: #F8FAFC;
    }
    
    /* Sleek Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #FFFFFF !important;
        border-right: 1px solid #E2E8F0;
    }
    
    /* Clean Hero Header */
    .hero-banner {
        background: linear-gradient(135deg, #581C87 0%, #7C3AED 50%, #4F46E5 100%);
        border-radius: 14px;
        padding: 22px 28px;
        color: white;
        margin-bottom: 20px;
        box-shadow: 0 4px 14px rgba(124, 58, 237, 0.18);
    }
    
    .hero-banner h1 {
        color: white !important;
        font-weight: 800;
        font-size: 1.65rem;
        margin: 0 0 4px 0;
        letter-spacing: -0.02em;
    }
    
    .hero-banner p {
        color: #E9D5FF !important;
        font-size: 0.88rem;
        margin: 0;
        font-weight: 500;
    }
    
    /* KPI Metric Cards */
    .kpi-container {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 16px 18px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.03);
    }
    
    .kpi-label {
        font-size: 0.74rem;
        font-weight: 700;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 4px;
    }
    
    .kpi-num {
        font-size: 1.55rem;
        font-weight: 800;
        color: #0F172A;
    }
    
    .kpi-desc {
        font-size: 0.76rem;
        color: #7C3AED;
        font-weight: 600;
        margin-top: 2px;
    }
    
    /* Structured Card Boxes */
    .card-panel {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 18px 20px;
        margin-bottom: 18px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.03);
    }
    
    /* Custom Purple Action Button */
    .stButton>button, .stFormSubmitButton>button {
        background: linear-gradient(135deg, #7C3AED 0%, #6366F1 100%) !important;
        color: white !important;
        font-weight: 700 !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 10px 20px !important;
        box-shadow: 0 3px 10px rgba(124, 58, 237, 0.22) !important;
        font-size: 0.9rem !important;
    }
    
    /* Secondary Outline Button */
    .stButton>button:hover, .stFormSubmitButton>button:hover {
        opacity: 0.92 !important;
    }
    
    /* Status Badges */
    .badge-allow {
        background-color: #ECFDF5;
        border: 1px solid #A7F3D0;
        border-left: 5px solid #10B981;
        padding: 14px 18px;
        border-radius: 10px;
        color: #065F46;
        margin-bottom: 16px;
    }
    
    .badge-challenge {
        background-color: #FFFBEB;
        border: 1px solid #FDE68A;
        border-left: 5px solid #F59E0B;
        padding: 14px 18px;
        border-radius: 10px;
        color: #92400E;
        margin-bottom: 16px;
    }
    
    .badge-decline {
        background-color: #FEF2F2;
        border: 1px solid #FECACA;
        border-left: 5px solid #EF4444;
        padding: 14px 18px;
        border-radius: 10px;
        color: #991B1B;
        margin-bottom: 16px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Sidebar Navigation & Telemetry
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown(
        """
        <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 10px;">
            <div style="background: #7C3AED; width: 34px; height: 34px; border-radius: 8px; display: flex; align-items: center; justify-content: center; color: white; font-weight: 800; font-size: 18px;">⚡</div>
            <div>
                <h3 style="margin:0; font-size: 1.1rem; font-weight: 800; color: #0F172A;">Bhairava AI</h3>
                <span style="font-size: 0.72rem; color: #7C3AED; font-weight: 700; text-transform: uppercase;">Track 2: AI Risk Manager</span>
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
            <div style="background:#ECFDF5; border:1px solid #A7F3D0; padding:8px 12px; border-radius:8px; display:flex; align-items:center; gap:8px;">
                <span style="color:#10B981; font-size:12px;">●</span>
                <span style="color:#065F46; font-weight:700; font-size:0.8rem;">XGBoost Inference Active</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.write("")
        st.caption(f"**Feature Matrix:** `{len(detector_service.feature_names):,}` columns")
        st.caption(f"**Locked Threshold:** `{detector_service.locked_threshold:.2f}` (Validation-locked)")
        st.caption(f"**Evaluation Split:** Chronological (0.0% leakage)")
    else:
        st.error("Model Engine Degraded")

    st.divider()
    st.markdown("<p style='font-size:0.75rem; font-weight:700; color:#64748B; text-transform:uppercase;'>Demo Controls</p>", unsafe_allow_html=True)
    
    if st.button("⚡ Seed Gateway Traffic", use_container_width=True, help="Populates 150 realistic payment transactions (92% Allow, 6% 3DS, 2% Decline)"):
        audit_store.seed_realistic_gateway_traffic(total_samples=150)
        st.success("Loaded 150 live gateway transactions!")
        st.rerun()

    if st.button("🗑️ Reset Ledger", use_container_width=True):
        audit_store.reset_store()
        st.info("Audit ledger cleared.")
        st.rerun()

    st.write("")
    st.caption("Built for **Razorpay AI Buildathon 2026**")


# ---------------------------------------------------------------------------
# Hero Header
# ---------------------------------------------------------------------------

st.markdown(
    """
    <div class="hero-banner">
        <h1>Bhairava AI — Risk & Fraud Operations</h1>
        <p>Real-Time Fraud Scoring • 3-Tier Policy Auto-Responder • TreeSHAP Attribution • Closed-Loop Audit Ledger</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Live KPI Metrics
# ---------------------------------------------------------------------------

stats = audit_store.get_aggregate_stats()
recent_decisions = audit_store.get_recent_decisions(limit=50)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(
        f"""
        <div class="kpi-container">
            <div class="kpi-label">Transactions Evaluated</div>
            <div class="kpi-num">{stats['total_evaluated']:,}</div>
            <div class="kpi-desc">Gross processed volume</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col2:
    avg_score_str = f"{stats['avg_risk_score']:.4f}" if stats["total_evaluated"] > 0 else "0.0000"
    st.markdown(
        f"""
        <div class="kpi-container">
            <div class="kpi-label">Average Risk Index</div>
            <div class="kpi-num">{avg_score_str}</div>
            <div class="kpi-desc">Traffic risk baseline</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col3:
    fraud_rate = stats["fraud_confirmation_rate"]
    rate_str = f"{fraud_rate:.1%}" if fraud_rate is not None else "Awaiting Disputes"
    st.markdown(
        f"""
        <div class="kpi-container">
            <div class="kpi-label">Confirmed Chargeback Rate</div>
            <div class="kpi-num">{rate_str}</div>
            <div class="kpi-desc">Merchant verified fraud</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col4:
    st.markdown(
        f"""
        <div class="kpi-container">
            <div class="kpi-label">Feedback Ingested</div>
            <div class="kpi-num">{stats['feedback_submitted']}</div>
            <div class="kpi-desc">Closed-loop audit records</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.write("")


# ---------------------------------------------------------------------------
# Visual Analytics (Charts Row)
# ---------------------------------------------------------------------------

col_chart_left, col_chart_right = st.columns([1, 2])

plotly_config = {"displayModeBar": False}

with col_chart_left:
    st.markdown(
        """
        <div class="card-panel">
            <h4 style="margin:0 0 10px 0; color:#0F172A; font-weight:700; font-size:0.95rem;">Policy Action Breakdown</h4>
        """,
        unsafe_allow_html=True,
    )
    if stats["total_evaluated"] > 0:
        fig_donut = action_breakdown_chart(stats["action_breakdown"])
        st.plotly_chart(fig_donut, use_container_width=True, config=plotly_config)
    else:
        st.info("No transaction data. Click 'Seed Gateway Traffic' in the sidebar.")
    st.markdown("</div>", unsafe_allow_html=True)

with col_chart_right:
    st.markdown(
        """
        <div class="card-panel">
            <h4 style="margin:0 0 10px 0; color:#0F172A; font-weight:700; font-size:0.95rem;">Transaction Risk Distribution</h4>
        """,
        unsafe_allow_html=True,
    )
    if recent_decisions:
        fig_hist = risk_distribution_chart(recent_decisions)
        st.plotly_chart(fig_hist, use_container_width=True, config=plotly_config)
    else:
        st.info("No risk distribution recorded yet.")
    st.markdown("</div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Live Simulator & Transaction Scorer
# ---------------------------------------------------------------------------

st.markdown(
    """
    <div class="card-panel">
        <h4 style="margin:0 0 4px 0; color:#0F172A; font-weight:800; font-size:1.05rem;">⚡ Real-Time Transaction Simulator</h4>
        <p style="color:#64748B; font-size:0.84rem; margin-bottom:14px;">
            Simulate an incoming merchant payment payload to test ML risk scoring, 3-tier policy routing, TreeSHAP explainability, and automated audit logging.
        </p>
    """,
    unsafe_allow_html=True,
)

with st.form("transaction_scorer", clear_on_submit=False):
    col_a, col_b, col_c = st.columns(3)

    with col_a:
        txn_id = st.text_input("Transaction ID", value="pay_rzp_live_9921")
        amount = st.number_input(
            "Transaction Amount (₹ INR)",
            min_value=1.0,
            max_value=500_000.0,
            value=2499.00,
            step=100.0,
            help="High amounts or abnormal cents fractions trigger anomaly signals.",
        )
        email_domain = st.selectbox(
            "Purchaser Email Domain",
            options=["gmail.com", "yahoo.co.in", "outlook.com", "protonmail.com", "tempmail.ninja", "anonymous.io"],
            help="High-risk and throwaway domains trigger risk heuristics.",
        )

    with col_b:
        card1 = st.number_input(
            "Bank BIN Hash (card1)",
            min_value=1_000,
            max_value=99_999,
            value=13926,
            step=1,
            help="Card issuer identification number hash.",
        )
        card2 = st.number_input(
            "Card Category (card2)",
            min_value=100.0,
            max_value=600.0,
            value=321.0,
            step=10.0,
        )
        card3 = st.number_input(
            "Card Country Code (card3)",
            min_value=100.0,
            max_value=200.0,
            value=150.0,
            step=5.0,
        )

    with col_c:
        addr1 = st.number_input(
            "Billing Region PIN Prefix (addr1)",
            min_value=100.0,
            max_value=500.0,
            value=299.0,
            step=10.0,
        )
        has_identity = st.radio(
            "Device / Browser Telemetry",
            options=[1, 0],
            format_func=lambda x: "Present (Verified Device Fingerprint)" if x == 1 else "Missing (Headless / Anonymous)",
        )

    st.write("")
    submitted = st.form_submit_button("⚡ Evaluate Transaction & Execute Policy Auto-Responder", use_container_width=True)

st.markdown("</div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Decision Result Banner & SHAP Visualizer
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
        st.error(f"Inference failed: {str(e)}")
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
            <div style="font-size: 1.3rem; font-weight: 800; display:flex; align-items:center; gap:8px;">
                <span>{badge_icon}</span>
                <span>DECISION: {action_val.replace('_', ' ')}</span>
                <span style="font-size:0.85rem; font-weight:700; opacity:0.85; margin-left:auto;">Risk Tier: {tier_val}</span>
            </div>
            <div style="font-size:0.88rem; margin-top:4px;">
                Transaction <code>{decision.transaction_id}</code> for <b>₹{amount:,.2f}</b> evaluated with <b>{decision.risk_score:.2%}</b> estimated fraud probability.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    res_c1, res_c2, res_c3, res_c4 = st.columns(4)
    with res_c1:
        st.metric("Fraud Probability", f"{decision.risk_score:.4f}")
    with res_c2:
        st.metric("Decision Confidence", f"{decision.confidence:.1%}")
    with res_c3:
        st.metric("3DS Step-Up OTP", "TRIGGERED" if decision.requires_otp_challenge else "BYPASSED")
    with res_c4:
        st.metric("Audit Trail ID", decision.audit_id)

    # --- TreeSHAP Feature Attribution Section ---
    st.write("")
    exp_col1, exp_col2 = st.columns([3, 2])

    with exp_col1:
        st.markdown("<p style='font-size:0.9rem; font-weight:700; color:#0F172A; margin-bottom:4px;'>TreeSHAP Mathematical Feature Contributions</p>", unsafe_allow_html=True)
        if decision.shap_attribution:
            fig_shap = shap_attribution_chart(decision.shap_attribution)
            st.plotly_chart(fig_shap, use_container_width=True, config=plotly_config)
        else:
            st.info("SHAP attribution computed at API tier.")

    with exp_col2:
        st.markdown("<p style='font-size:0.9rem; font-weight:700; color:#0F172A; margin-bottom:4px;'>Attributed Policy Reason Codes</p>", unsafe_allow_html=True)
        reasons = decision.reasons or ["LOW_RISK_NORMAL_BEHAVIOR"]
        for r in reasons:
            st.markdown(
                f"<div style='background:#F1F5F9; border-left:3px solid #7C3AED; padding:7px 11px; border-radius:6px; margin-bottom:6px; font-size:0.8rem; font-weight:600; color:#334155;'>{r}</div>",
                unsafe_allow_html=True,
            )
        if decision.shap_attribution and decision.shap_attribution.top_features:
            top_f = decision.shap_attribution.top_features[0]
            st.caption(f"💡 **Dominant Driver:** `{top_f.friendly_name}` ({top_f.direction.replace('_', ' ')})")

    with st.expander("📬 Real-Time Merchant Webhook Notification Payload"):
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
# Decision Audit Ledger (Table)
# ---------------------------------------------------------------------------

st.markdown(
    """
    <div class="card-panel">
        <h4 style="margin:0 0 10px 0; color:#0F172A; font-weight:700; font-size:0.95rem;">Live Persistent Audit Ledger (SQLite)</h4>
    """,
    unsafe_allow_html=True,
)

if recent_decisions:
    render_recent_decisions(recent_decisions[:20])
else:
    st.info("Audit ledger is empty. Click 'Seed Gateway Traffic' in the sidebar or evaluate a transaction.")

st.markdown("</div>", unsafe_allow_html=True)
