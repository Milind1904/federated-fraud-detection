import streamlit as st
import plotly.graph_objects as go
import json
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from api_client import get, post
from style import (
    plotly_dark_layout, section_header,
    metric_card, drift_card
)


def render():
    st.markdown(
        section_header("Monitor status"),
        unsafe_allow_html=True
    )

    status = get("/drift/status")
    if status:
        running    = status.get("running", False)
        retraining = status.get("retraining_in_progress", False)
        error_rate = status.get("ddm_error_rate", 0)
        last_id    = status.get("last_processed_id", 0)

        st.markdown(f"""
        <div class="metric-grid">
            {metric_card(
                "Monitor",
                "RUNNING" if running else "STOPPED",
                f"every {status.get('monitor_interval_seconds', 30)}s",
                "#34d399" if running else "#f87171"
            )}
            {metric_card(
                "DDM error rate",
                f"{error_rate:.4f}",
                "warning" if status.get('ddm_in_warning') else "nominal",
                "#fbbf24" if status.get('ddm_in_warning') else "#3b82f6"
            )}
            {metric_card(
                "Retraining",
                "ACTIVE" if retraining else "IDLE",
                "federated rounds",
                "#fb923c" if retraining else "#3b82f6"
            )}
            {metric_card(
                "Last processed",
                f"#{last_id}",
                "prediction id",
                "#a78bfa"
            )}
        </div>
        """, unsafe_allow_html=True)

    st.markdown(
        section_header("Manual controls"),
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Simulate drift event"):
            result = post("/drift/simulate")
            if result:
                st.success(
                    f"Drift event created — ID {result.get('event_id')}"
                )
    with col2:
        if st.button("Trigger retraining", type="primary"):
            result = post("/retrain")
            if result:
                st.success(result.get("message", "Retraining started"))
                st.info(
                    "Watch the uvicorn terminal for retraining progress."
                )

    st.markdown(
        section_header("Drift events log"),
        unsafe_allow_html=True
    )

    data   = get("/drift/events", params={"limit": 50})
    events = data.get("events", [])

    if not events:
        st.info(
            "No drift events yet. The monitor logs events here "
            "automatically when drift is detected."
        )
    else:
        for event in events:
            st.markdown(drift_card(event), unsafe_allow_html=True)

    st.markdown(
        section_header("SHAP feature importance timeline"),
        unsafe_allow_html=True
    )

    pred_data   = get("/predictions/recent", params={"limit": 200})
    predictions = pred_data.get("predictions", [])
    shap_preds  = [
        p for p in predictions if p.get("shap_values_json")
    ]

    if len(shap_preds) >= 5:
        feature_timeline = {}

        for p in shap_preds:
            shap_vals = json.loads(p["shap_values_json"])
            for item in shap_vals[:5]:
                fname = item["feature"]
                if fname not in feature_timeline:
                    feature_timeline[fname] = []
                feature_timeline[fname].append(
                    abs(item["shap_value"])
                )

        if feature_timeline:
            fig = go.Figure()
            colors = [
                "#3b82f6", "#34d399",
                "#fbbf24", "#fb923c", "#a78bfa"
            ]
            for i, (fname, values) in enumerate(
                list(feature_timeline.items())[:5]
            ):
                fig.add_trace(go.Scatter(
                    x=list(range(len(values))),
                    y=values,
                    name=fname[-25:],
                    mode="lines",
                    line=dict(
                        color=colors[i % len(colors)],
                        width=1.5
                    ),
                ))
            fig = plotly_dark_layout(
                fig,
                "Top feature SHAP importance over time",
                320
            )
            fig.update_layout(
                xaxis_title="Prediction index",
                yaxis_title="|SHAP value|",
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info(
            f"Have {len(shap_preds)} explained predictions. "
            f"Use Score + explain in Live Feed to populate this chart."
        )