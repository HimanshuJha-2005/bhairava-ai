"""
Bhairava AI — Dashboard Chart Components
Clean, modern white & purple fintech visual components with INR currency formatting.
"""

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import streamlit as st


# ---------------------------------------------------------------------------
# Fintech Color Palette
# ---------------------------------------------------------------------------

ACTION_COLORS = {
    "ALLOW": "#10B981",           # Emerald Green
    "CHALLENGE_3DS": "#F59E0B",   # Warm Amber
    "AUTO_DECLINE": "#EF4444",    # Crimson Red
}

TIER_COLORS = {
    "LOW": "#10B981",
    "MEDIUM": "#F59E0B",
    "HIGH": "#EF4444",
}


def action_breakdown_chart(action_breakdown: dict) -> go.Figure:
    """
    Polished Donut chart with modern clean aesthetic.
    """
    filtered = {k: v for k, v in action_breakdown.items() if v > 0}
    if not filtered:
        fig = go.Figure()
        fig.update_layout(
            annotations=[{"text": "No data yet", "showarrow": False, "font": {"size": 13, "color": "#64748B"}}],
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=250,
        )
        return fig

    labels = list(filtered.keys())
    values = list(filtered.values())
    colors = [ACTION_COLORS.get(k, "#94A3B8") for k in labels]

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.64,
        marker=dict(colors=colors, line=dict(color="#FFFFFF", width=3)),
        textinfo="percent+label",
        textposition="inside",
        hoverinfo="label+value+percent",
        hovertemplate="<b>%{label}</b><br>Count: %{value:,}<br>Share: %{percent}<extra></extra>",
    )])

    fig.update_layout(
        showlegend=False,
        margin=dict(t=10, b=10, l=10, r=10),
        height=250,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def risk_distribution_chart(decisions: list) -> go.Figure:
    """
    Histogram of risk scores with clear 3DS / Auto-decline threshold callouts.
    """
    if not decisions:
        fig = go.Figure()
        fig.update_layout(
            annotations=[{"text": "No data yet", "showarrow": False, "font": {"size": 13, "color": "#64748B"}}],
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=250,
        )
        return fig

    df = pd.DataFrame([
        {"risk_score": d["risk_score"], "tier": d["risk_tier"]}
        for d in decisions
    ])

    fig = px.histogram(
        df,
        x="risk_score",
        color="tier",
        nbins=20,
        barmode="stack",
        color_discrete_map=TIER_COLORS,
        labels={"risk_score": "Risk Score", "count": "Transactions", "tier": "Tier"},
    )

    # Threshold Boundary Lines
    fig.add_vline(
        x=0.35,
        line_dash="dash",
        line_color="#F59E0B",
        line_width=1.5,
        annotation_text="3DS (0.35)",
        annotation_position="top left",
        annotation_font=dict(size=10, color="#D97706"),
    )
    fig.add_vline(
        x=0.65,
        line_dash="dash",
        line_color="#EF4444",
        line_width=1.5,
        annotation_text="Block (0.65)",
        annotation_position="top left",
        annotation_font=dict(size=10, color="#DC2626"),
    )

    fig.update_layout(
        margin=dict(t=20, b=20, l=10, r=10),
        height=250,
        xaxis=dict(
            range=[0, 1],
            title=dict(text="Risk Probability (0.00 - 1.00)", font=dict(size=11, color="#64748B")),
            gridcolor="#F1F5F9",
        ),
        yaxis=dict(
            title=dict(text="Transactions", font=dict(size=11, color="#64748B")),
            gridcolor="#F1F5F9",
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=10, color="#475569"),
            title=None,
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def shap_attribution_chart(shap_block) -> go.Figure:
    """
    Horizontal bar chart showing per-transaction TreeSHAP feature contributions.
    """
    if not shap_block or not shap_block.top_features:
        fig = go.Figure()
        fig.update_layout(
            annotations=[{"text": "SHAP attribution not available", "showarrow": False}],
            height=200,
        )
        return fig

    # Reverse so top feature is at the top of horizontal bar chart
    feats = list(reversed(shap_block.top_features))
    labels = [f.friendly_name for f in feats]
    values = [f.shap_value for f in feats]
    colors = ["#EF4444" if v > 0 else "#10B981" for v in values]

    fig = go.Figure(go.Bar(
        x=values,
        y=labels,
        orientation="h",
        marker=dict(
            color=colors,
            line=dict(color="#FFFFFF", width=1.5),
        ),
        hovertemplate="<b>%{y}</b><br>SHAP Contribution: %{x:+.4f}<extra></extra>",
    ))

    fig.add_vline(x=0, line_width=1.5, line_color="#CBD5E1")

    fig.update_layout(
        margin=dict(t=10, b=25, l=10, r=10),
        height=210,
        xaxis=dict(
            title=dict(text="← Reduces Risk (Legitimate) | Increases Risk (Fraud) →", font=dict(size=10, color="#64748B")),
            gridcolor="#F1F5F9",
        ),
        yaxis=dict(
            tickfont=dict(size=11, color="#1E293B"),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def render_recent_decisions(decisions: list) -> None:
    """
    Render colour-coded table matching the white & purple fintech palette.
    """
    if not decisions:
        st.info("No decisions recorded yet.")
        return

    rows = []
    for d in decisions:
        rows.append({
            "Audit ID": d["audit_id"],
            "Transaction ID": d["transaction_id"],
            "Action": d["action"],
            "Risk Score": f"{d['risk_score']:.4f}",
            "Tier": d["risk_tier"],
            "Confidence": f"{d['confidence']:.1%}",
            "3DS Status": "Step-Up Required" if d["requires_otp"] else "Bypassed",
            "Confirmed Outcome": d.get("outcome") or "Pending Dispute / Feedback",
            "Timestamp (UTC)": d["decided_at"][:19].replace("T", " "),
        })

    df = pd.DataFrame(rows)

    def _color_action(val: str) -> str:
        palette = {
            "ALLOW": "background-color: #ECFDF5; color: #065F46; font-weight: 600; border-radius: 4px;",
            "CHALLENGE_3DS": "background-color: #FFFBEB; color: #92400E; font-weight: 600; border-radius: 4px;",
            "AUTO_DECLINE": "background-color: #FEF2F2; color: #991B1B; font-weight: 600; border-radius: 4px;",
        }
        return palette.get(val, "")

    def _color_tier(val: str) -> str:
        palette = {
            "LOW": "color: #059669; font-weight: 600;",
            "MEDIUM": "color: #D97706; font-weight: 600;",
            "HIGH": "color: #DC2626; font-weight: 600;",
        }
        return palette.get(val, "")

    styled = df.style.map(_color_action, subset=["Action"]).map(_color_tier, subset=["Tier"])
    st.dataframe(styled, use_container_width=True, hide_index=True)
