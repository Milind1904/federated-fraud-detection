import streamlit as st
import plotly.graph_objects as go
import json
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from api_client import get, post
from style import (
    plotly_dark_layout, section_header,
    metric_card, risk_badge, explanation_box
)


def render():
    st.markdown(
        section_header("Transaction scoring"),
        unsafe_allow_html=True
    )

    # Top metrics row
    stats = get("/stats")
    if stats:
        fraud_rate = stats.get("recent_fraud_rate", 0)
        avg_prob   = stats.get("avg_fraud_probability", 0)
        risk_dist  = stats.get("risk_distribution", {})
        high_crit  = (
            risk_dist.get("high", 0) +
            risk_dist.get("critical", 0)
        )
        total = stats.get("total_predictions", 0)

        st.markdown(f"""
        <div class="metric-grid">
            {metric_card("Total scored", f"{total:,}",
                         "all time", "#3b82f6")}
            {metric_card("Fraud rate", f"{fraud_rate:.1%}",
                         "recent window", "#f87171")}
            {metric_card("Avg probability", f"{avg_prob:.3f}",
                         "recent window", "#fbbf24")}
            {metric_card("High / critical", str(high_crit),
                         "active alerts", "#fb923c")}
        </div>
        """, unsafe_allow_html=True)

    st.markdown(
        section_header("Score a transaction"),
        unsafe_allow_html=True
    )

    with st.expander("Open transaction form", expanded=True):
        col1, col2 = st.columns(2)

        with col1:
            amount = st.number_input(
                "Transaction amount ($)",
                min_value=0.01,
                value=500.0,
                step=10.0
            )
            card_type = st.selectbox(
                "Card type",
                ["visa", "mastercard", "discover", "american express"]
            )
            card_category = st.selectbox(
                "Card category",
                ["debit", "credit"]
            )
            device = st.selectbox(
                "Device type",
                ["desktop", "mobile", "unknown"]
            )

        with col2:
            p_email = st.text_input(
                "Sender email domain",
                value="gmail.com"
            )
            r_email = st.text_input(
                "Receiver email domain",
                value="gmail.com"
            )
            c1 = st.slider(
                "C1 — transaction velocity",
                0.0, 20.0, 1.0
            )
            threshold = st.slider(
                "Decision threshold",
                0.1, 0.9, 0.5, 0.05
            )

        col_btn1, col_btn2 = st.columns(2)

        with col_btn1:
            score_clicked = st.button(
                "Score transaction",
                type="primary",
                use_container_width=True
            )

        with col_btn2:
            explain_clicked = st.button(
                "Score + explain",
                use_container_width=True
            )

        if score_clicked:
            payload = {
                "TransactionAmt": amount,
                "card4": card_type,
                "card6": card_category,
                "DeviceType": device,
                "P_emaildomain": p_email,
                "R_emaildomain": r_email,
                "C1": c1,
            }
            result = post(
                "/predict",
                json=payload,
                params={"threshold": threshold}
            )
            if result:
                risk = result.get("risk_level", "low")
                prob = result.get("fraud_probability", 0)
                st.markdown(
                    f"""
                    <div style="margin-top:12px; padding:14px 16px;
                                background:#131720;
                                border:1px solid #1e2433;
                                border-radius:8px;
                                display:flex;
                                align-items:center;
                                gap:16px;">
                        {risk_badge(risk)}
                        <span style="font-family:'JetBrains Mono',
                                     monospace; font-size:13px;
                                     color:#a0aec0;">
                            fraud probability:
                            <span style="color:#e2e8f0;
                                         font-weight:500;">
                                {prob:.4f}
                            </span>
                        </span>
                        <span style="font-family:'JetBrains Mono',
                                     monospace; font-size:11px;
                                     color:#2d3748; margin-left:auto;">
                            threshold: {threshold}
                        </span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        if explain_clicked:
            payload = {
                "TransactionAmt": amount,
                "card4": card_type,
                "card6": card_category,
                "DeviceType": device,
                "P_emaildomain": p_email,
                "R_emaildomain": r_email,
                "C1": c1,
            }
            with st.spinner("Computing SHAP values..."):
                result = post(
                    "/explain",
                    json=payload,
                    params={"threshold": threshold}
                )

            if result:
                risk = result.get("risk_level", "low")
                prob = result.get("fraud_probability", 0)

                st.markdown(
                    f"""
                    <div style="margin-top:12px; padding:14px 16px;
                                background:#131720;
                                border:1px solid #1e2433;
                                border-radius:8px;
                                display:flex;
                                align-items:center;
                                gap:16px;">
                        {risk_badge(risk)}
                        <span style="font-family:'JetBrains Mono',
                                     monospace; font-size:13px;
                                     color:#a0aec0;">
                            fraud probability:
                            <span style="color:#e2e8f0;
                                         font-weight:500;">
                                {prob:.4f}
                            </span>
                        </span>
                        <span style="font-family:'JetBrains Mono',
                                     monospace; font-size:11px;
                                     color:#2d3748; margin-left:auto;">
                            model round: {result.get('model_round')}
                        </span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                eng = result.get("english_explanation", "")
                if eng:
                    st.markdown(
                        explanation_box(eng),
                        unsafe_allow_html=True
                    )

                top_features = result.get("top_features", [])
                if top_features:
                    st.markdown(
                        """<div style="font-size:11px;
                            color:#4a5568;
                            font-family:'JetBrains Mono',monospace;
                            text-transform:uppercase;
                            letter-spacing:0.08em;
                            margin:16px 0 8px 0;">
                            SHAP feature contributions
                        </div>""",
                        unsafe_allow_html=True
                    )

                    features = [
                        f["feature"][-35:]
                        for f in top_features
                    ]
                    values = [
                        f["shap_value"]
                        for f in top_features
                    ]
                    colors = [
                        "#f87171" if v > 0 else "#34d399"
                        for v in values
                    ]

                    fig = go.Figure(go.Bar(
                        x=values,
                        y=features,
                        orientation="h",
                        marker_color=colors,
                        marker_line_width=0,
                    ))
                    fig = plotly_dark_layout(
                        fig,
                        "",
                        380
                    )
                    fig.update_layout(
                        xaxis_title="SHAP value",
                        bargap=0.3,
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    push = result.get("total_fraud_push", 0)
                    pull = result.get("total_fraud_pull", 0)
                    st.markdown(
                        f"""
                        <div style="display:flex; gap:24px;
                                    font-family:'JetBrains Mono',
                                    monospace; font-size:11px;
                                    color:#4a5568; margin-top:4px;">
                            <span>
                                FRAUD PUSH
                                <span style="color:#f87171;">
                                    +{push:.4f}
                                </span>
                            </span>
                            <span>
                                FRAUD PULL
                                <span style="color:#34d399;">
                                    {pull:.4f}
                                </span>
                            </span>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

    st.markdown(
        section_header("Recent predictions"),
        unsafe_allow_html=True
    )

    data        = get("/predictions/recent", params={"limit": 50})
    predictions = data.get("predictions", [])

    if not predictions:
        st.info(
            "No predictions yet. Score some transactions above."
        )
        return

    # Build styled HTML table
    table_html = """
    <table class="tx-table">
        <thead>
            <tr>
                <th>ID</th>
                <th>Time</th>
                <th>Amount</th>
                <th>Probability</th>
                <th>Risk</th>
                <th>Flagged</th>
                <th>Truth</th>
                <th>Correct</th>
            </tr>
        </thead>
        <tbody>
    """

    for p in predictions:
        risk    = p.get("risk_level", "low")
        prob    = p.get("fraud_probability", 0)
        flagged = p.get("is_fraud_predicted", 0)
        truth   = p.get("ground_truth")
        correct = p.get("correct")
        amt     = p.get("transaction_amt", 0)
        ts      = p.get("timestamp", "")[:19].replace("T", " ")

        prob_class = (
            "prob-high" if prob >= 0.6
            else "prob-medium" if prob >= 0.4
            else "prob-low"
        )

        truth_str = (
            "<span style='color:#f87171'>fraud</span>"
            if truth == 1
            else "<span style='color:#34d399'>legit</span>"
            if truth == 0
            else "<span style='color:#2d3748'>—</span>"
        )

        correct_str = (
            "<span style='color:#34d399'>✓</span>"
            if correct == 1
            else "<span style='color:#f87171'>✗</span>"
            if correct == 0
            else "<span style='color:#2d3748'>—</span>"
        )

        flagged_str = (
            "<span style='color:#f87171'>yes</span>"
            if flagged
            else "<span style='color:#34d399'>no</span>"
        )

        table_html += f"""
        <tr>
            <td style="color:#4a5568;">#{p['id']}</td>
            <td style="color:#4a5568;">{ts}</td>
            <td class="amount">${amt:.2f}</td>
            <td class="{prob_class}">{prob:.4f}</td>
            <td>{risk_badge(risk)}</td>
            <td>{flagged_str}</td>
            <td>{truth_str}</td>
            <td>{correct_str}</td>
        </tr>
        """

    table_html += "</tbody></table>"
    st.markdown(table_html, unsafe_allow_html=True)

    # Transaction inspector
    st.markdown(
        section_header("Transaction inspector"),
        unsafe_allow_html=True
    )

    pred_ids    = [p["id"] for p in predictions]
    selected_id = st.selectbox(
        "Select prediction ID to inspect",
        pred_ids
    )
    selected = next(
        (p for p in predictions if p["id"] == selected_id),
        None
    )

    if selected and selected.get("shap_values_json"):
        shap_data = json.loads(selected["shap_values_json"])
        features  = [f["feature"][-35:] for f in shap_data]
        values    = [f["shap_value"] for f in shap_data]
        colors    = [
            "#f87171" if v > 0 else "#34d399"
            for v in values
        ]

        fig = go.Figure(go.Bar(
            x=values,
            y=features,
            orientation="h",
            marker_color=colors,
            marker_line_width=0,
        ))
        fig = plotly_dark_layout(
            fig,
            f"SHAP explanation — prediction #{selected_id}",
            380
        )
        fig.update_layout(
            xaxis_title="SHAP value",
            bargap=0.3,
        )
        st.plotly_chart(fig, use_container_width=True)

        if selected.get("english_explanation"):
            st.markdown(
                explanation_box(selected["english_explanation"]),
                unsafe_allow_html=True
            )

    elif selected:
        st.markdown(
            """<div style="font-size:12px; color:#4a5568;
                font-family:'JetBrains Mono',monospace;
                padding:12px 0;">
                No SHAP data for this prediction.
                Use Score + explain to get explanations.
            </div>""",
            unsafe_allow_html=True
        )