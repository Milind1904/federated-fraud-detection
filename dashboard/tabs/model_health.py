import streamlit as st
import plotly.graph_objects as go
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from api_client import get
from style import plotly_dark_layout, section_header, metric_card


def render():
    st.markdown(
        section_header("Current model performance"),
        unsafe_allow_html=True
    )

    info = get("/model-info")
    if info:
        st.markdown(f"""
        <div class="metric-grid">
            {metric_card("AUC", f"{info.get('auc', 0):.4f}",
                         "global test set", "#3b82f6")}
            {metric_card("F1 score", f"{info.get('f1', 0):.4f}",
                         "threshold 0.5", "#1D9E75")}
            {metric_card("Precision", f"{info.get('precision', 0):.4f}",
                         "of flagged txns", "#EF9F27")}
            {metric_card("Recall", f"{info.get('recall', 0):.4f}",
                         "fraud caught", "#f87171")}
        </div>
        """, unsafe_allow_html=True)

        st.markdown(
            f"""<div style="font-size:11px; color:#2d3748;
                font-family:'JetBrains Mono',monospace; margin-bottom:1rem;">
                GLOBAL MODEL &nbsp;·&nbsp; ROUND {info.get('model_round', 0)}
                &nbsp;·&nbsp; {info.get('input_dim', 0)} FEATURES
            </div>""",
            unsafe_allow_html=True
        )

    st.markdown(
        section_header("Federated training history"),
        unsafe_allow_html=True
    )

    history_path = os.path.join(
        os.path.dirname(__file__),
        "../../logs/round_history.json"
    )

    if os.path.exists(history_path):
        with open(history_path) as f:
            history = json.load(f)

        if history:
            rounds     = [h["round"] for h in history]
            aucs       = [h["metrics"]["auc"] for h in history]
            f1s        = [h["metrics"]["f1"] for h in history]
            recalls    = [h["metrics"]["recall"] for h in history]
            precisions = [h["metrics"]["precision"] for h in history]

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=rounds, y=aucs,
                name="AUC",
                mode="lines+markers",
                line=dict(color="#3b82f6", width=2),
                marker=dict(size=5),
            ))
            fig.add_trace(go.Scatter(
                x=rounds, y=recalls,
                name="Recall",
                mode="lines+markers",
                line=dict(color="#34d399", width=2),
                marker=dict(size=5),
            ))
            fig.add_trace(go.Scatter(
                x=rounds, y=f1s,
                name="F1",
                mode="lines+markers",
                line=dict(color="#fbbf24", width=2),
                marker=dict(size=5),
            ))
            fig.add_trace(go.Scatter(
                x=rounds, y=precisions,
                name="Precision",
                mode="lines+markers",
                line=dict(color="#fb923c", width=2),
                marker=dict(size=5),
            ))
            fig = plotly_dark_layout(
                fig,
                "Metrics across federated rounds",
                400
            )
            fig.update_layout(
                xaxis_title="Round",
                yaxis_title="Score",
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1,
                    font=dict(color="#718096", size=10),
                ),
            )
            st.plotly_chart(fig, use_container_width=True)

            st.markdown(
                section_header("Per-client training loss"),
                unsafe_allow_html=True
            )

            fig2 = go.Figure()
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
                    fig2.add_trace(go.Scatter(
                        x=list(range(1, len(losses) + 1)),
                        y=losses,
                        name=f"Client {client_id}",
                        mode="lines+markers",
                        line=dict(color=colors[client_id], width=1.5),
                        marker=dict(size=4),
                    ))
            fig2 = plotly_dark_layout(
                fig2,
                "Client local loss per round",
                320
            )
            fig2.update_layout(
                xaxis_title="Round",
                yaxis_title="Loss",
            )
            st.plotly_chart(fig2, use_container_width=True)

    else:
        st.info("No training history found.")

    st.markdown(
        section_header("Rolling performance windows"),
        unsafe_allow_html=True
    )

    perf    = get("/performance/history")
    windows = perf.get("history", [])

    if windows:
        w_ids  = [w["window_end"] for w in windows]
        w_aucs = [w["auc"] for w in windows if w["auc"]]

        if w_aucs:
            fig3 = go.Figure()
            fig3.add_trace(go.Scatter(
                x=w_ids[:len(w_aucs)],
                y=w_aucs,
                name="Rolling AUC",
                mode="lines+markers",
                line=dict(color="#3b82f6", width=2),
                fill="tozeroy",
                fillcolor="rgba(59,130,246,0.05)",
            ))
            fig3.add_hline(
                y=0.85,
                line_dash="dash",
                line_color="#f87171",
                annotation_text="drift threshold",
                annotation_font_color="#f87171",
                annotation_font_size=10,
            )
            fig3 = plotly_dark_layout(
                fig3,
                "Rolling AUC over prediction stream",
                280
            )
            fig3.update_layout(
                xaxis_title="Prediction ID",
                yaxis_title="AUC",
            )
            st.plotly_chart(fig3, use_container_width=True)
    else:
        st.info(
            "Rolling windows appear after 100+ labeled predictions."
        )