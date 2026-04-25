import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from api_client import get
from style import plotly_dark_layout, section_header, metric_card


def render():
    st.markdown(
        section_header("Federated architecture"),
        unsafe_allow_html=True
    )

    st.markdown("""
    <div style="font-size:13px; color:#718096;
                line-height:1.8; margin-bottom:1rem;
                font-family:'Inter',sans-serif;">
        5 bank clients train collaboratively without sharing raw data.
        Only model weights are exchanged with the central server.
        FedAvg aggregates updates proportional to each client's
        dataset size. Raw transactions never leave the client.
    </div>
    """, unsafe_allow_html=True)

    st.markdown(
        section_header("Client dataset statistics"),
        unsafe_allow_html=True
    )

    client_stats = [
        {"Client": "Client 0", "Transactions": "105,676",
         "Fraud rate": "2.93%", "Fraud share": "15%",
         "Avg amount": "varies"},
        {"Client": "Client 1", "Transactions": "107,742",
         "Fraud rate": "4.79%", "Fraud share": "25%",
         "Avg amount": "varies"},
        {"Client": "Client 2", "Transactions": "104,643",
         "Fraud rate": "1.97%", "Fraud share": "10%",
         "Avg amount": "varies"},
        {"Client": "Client 3", "Transactions": "107,742",
         "Fraud rate": "4.79%", "Fraud share": "25%",
         "Avg amount": "varies"},
        {"Client": "Client 4", "Transactions": "105,676",
         "Fraud rate": "2.93%", "Fraud share": "15%",
         "Avg amount": "varies"},
    ]

    df = pd.DataFrame(client_stats)

    table_html = """
    <table class="tx-table">
        <thead>
            <tr>
                <th>Client</th>
                <th>Transactions</th>
                <th>Fraud rate</th>
                <th>Fraud share</th>
            </tr>
        </thead>
        <tbody>
    """
    colors = [
        "#3b82f6", "#34d399",
        "#fbbf24", "#fb923c", "#a78bfa"
    ]
    for i, row in enumerate(client_stats):
        color = colors[i]
        table_html += f"""
        <tr>
            <td style="color:{color}; font-weight:500;">
                {row['Client']}
            </td>
            <td class="amount">{row['Transactions']}</td>
            <td style="color:#f87171;">{row['Fraud rate']}</td>
            <td style="color:#718096;">{row['Fraud share']}</td>
        </tr>
        """
    table_html += "</tbody></table>"
    st.markdown(table_html, unsafe_allow_html=True)

    st.markdown(
        section_header("Client loss convergence"),
        unsafe_allow_html=True
    )

    history_path = os.path.join(
        os.path.dirname(__file__),
        "../../logs/round_history.json"
    )

    if not os.path.exists(history_path):
        st.info("No training history found.")
        return

    with open(history_path) as f:
        history = json.load(f)

    if not history:
        st.info("Training history is empty.")
        return

    fig = go.Figure()
    colors = [
        "#3b82f6", "#34d399",
        "#fbbf24", "#fb923c", "#a78bfa"
    ]
    for client_id in range(5):
        losses = [
            h["client_losses"][client_id]
            for h in history
            if len(h.get("client_losses", [])) > client_id
        ]
        if losses:
            fig.add_trace(go.Scatter(
                x=list(range(1, len(losses) + 1)),
                y=losses,
                name=f"Client {client_id}",
                mode="lines+markers",
                line=dict(color=colors[client_id], width=2),
                marker=dict(size=5),
            ))
    fig = plotly_dark_layout(
        fig,
        "Local training loss per round",
        380
    )
    fig.update_layout(
        xaxis_title="Round",
        yaxis_title="Loss",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        section_header("FedAvg aggregation"),
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div style="font-size:12px; color:#718096;
                    font-family:'JetBrains Mono',monospace;
                    line-height:2.2;">
            <span style="color:#4a5568;">EACH ROUND</span><br>
            01 &nbsp; Server broadcasts global weights<br>
            02 &nbsp; Each client trains 3 local epochs<br>
            03 &nbsp; Clients send weight updates<br>
            04 &nbsp; Server computes weighted average<br>
            05 &nbsp; Global model evaluated on test set<br>
            06 &nbsp; Checkpoint saved to disk
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div style="font-size:12px; color:#718096;
                    font-family:'JetBrains Mono',monospace;
                    line-height:2.2;">
            <span style="color:#4a5568;">PRIVACY GUARANTEE</span><br>
            ✓ &nbsp; Raw transactions never shared<br>
            ✓ &nbsp; Only float32 tensors exchanged<br>
            ✓ &nbsp; No reconstruction from weights<br>
            ✓ &nbsp; Fraud distributions stay private<br>
            ✓ &nbsp; Each client controls their data<br>
            ✓ &nbsp; GDPR-compatible architecture
        </div>
        """, unsafe_allow_html=True)

    st.markdown(
        section_header("Training summary"),
        unsafe_allow_html=True
    )

    last  = history[-1]
    first = history[0]

    auc_gain    = last["metrics"]["auc"] - first["metrics"]["auc"]
    recall_gain = last["metrics"]["recall"] - first["metrics"]["recall"]

    st.markdown(f"""
    <div class="metric-grid">
        {metric_card(
            "Final AUC",
            f"{last['metrics']['auc']:.4f}",
            f"+{auc_gain:.4f} from round 1",
            "#3b82f6"
        )}
        {metric_card(
            "Final recall",
            f"{last['metrics']['recall']:.4f}",
            f"+{recall_gain:.4f} from round 1",
            "#34d399"
        )}
        {metric_card(
            "Rounds",
            str(len(history)),
            "completed",
            "#a78bfa"
        )}
        {metric_card(
            "Clients",
            "5",
            "bank partitions",
            "#fbbf24"
        )}
    </div>
    """, unsafe_allow_html=True)