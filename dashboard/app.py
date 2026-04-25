import streamlit as st
import importlib.util
import os
import sys

sys.path.append(os.path.dirname(__file__))

from style import inject_css


def load_tab(name):
    path = os.path.join(os.path.dirname(__file__), "tabs", f"{name}.py")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


live_feed      = load_tab("live_feed")
model_health   = load_tab("model_health")
drift_monitor  = load_tab("drift_monitor")
federated_view = load_tab("federated_view")

st.set_page_config(
    page_title="Federated Fraud Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_css()

st.markdown("""
<div class="page-header">
    <div>
        <h1>🛡️ Federated Fraud Detection</h1>
        <span>IEEE-CIS · PyTorch · SHAP · FedAvg · DDM Drift Monitor</span>
    </div>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("""
    <div style="padding: 8px 0 16px 0;">
        <div style="font-size:18px; font-weight:500;
                    color:#e2e8f0; letter-spacing:0.02em;">
            🛡️ FraudGuard
        </div>
        <div style="font-size:11px; color:#4a5568;
                    font-family:'JetBrains Mono',monospace;
                    margin-top:4px;">
            FEDERATED · EXPLAINABLE · ADAPTIVE
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    from api_client import get
    health = get("/health")

    if health.get("status") == "ok":
        st.markdown(f"""
        <div style="margin-bottom:12px;">
            <span class="status-dot status-online"></span>
            <span style="font-size:13px; color:#34d399;
                         font-family:'JetBrains Mono',monospace;">
                API ONLINE
            </span>
        </div>
        <div style="font-size:11px; color:#4a5568;
                    font-family:'JetBrains Mono',monospace;
                    line-height:1.8;">
            MODEL ROUND &nbsp;·&nbsp;
            <span style="color:#a0aec0;">
                {health.get('model_round', '—')}
            </span><br>
            FEATURES &nbsp;·&nbsp;
            <span style="color:#a0aec0;">
                {health.get('input_dim', '—')}
            </span><br>
            AUC &nbsp;·&nbsp;
            <span style="color:#a0aec0;">0.8805</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div>
            <span class="status-dot status-offline"></span>
            <span style="font-size:13px; color:#f87171;
                         font-family:'JetBrains Mono',monospace;">
                API OFFLINE
            </span>
        </div>
        <div style="font-size:11px; color:#4a5568; margin-top:8px;">
            Run uvicorn on port 8000
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    auto_refresh = st.checkbox("Auto-refresh (5s)", value=False)
    if auto_refresh:
        import time
        time.sleep(5)
        st.rerun()

    st.divider()

    st.markdown("""
    <div style="font-size:11px; color:#2d3748;
                font-family:'JetBrains Mono',monospace;
                line-height:2;">
        BMSIT BANGALORE<br>
        AI/ML · 9.09 CGPA<br>
        IEEE-CIS DATASET<br>
        590,540 TRANSACTIONS
    </div>
    """, unsafe_allow_html=True)


tab1, tab2, tab3, tab4 = st.tabs([
    "LIVE FEED",
    "MODEL HEALTH",
    "DRIFT MONITOR",
    "FEDERATED VIEW",
])

with tab1:
    live_feed.render()

with tab2:
    model_health.render()

with tab3:
    drift_monitor.render()

with tab4:
    federated_view.render()