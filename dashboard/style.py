def inject_css():
    import streamlit as st
    st.markdown("""
    <style>
    /* ── Base ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background-color: #0f1117;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background-color: #080b12 !important;
        border-right: 1px solid #1e2433;
    }

    [data-testid="stSidebar"] * {
        color: #a0aec0 !important;
    }

    [data-testid="stSidebar"] .stMarkdown h1,
    [data-testid="stSidebar"] .stMarkdown strong {
        color: #e2e8f0 !important;
    }

    /* ── Main content ── */
    .main .block-container {
        padding: 1.5rem 2rem;
        max-width: 1400px;
    }

    /* ── Page header ── */
    .page-header {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 1rem 0 1.5rem 0;
        border-bottom: 1px solid #1e2433;
        margin-bottom: 1.5rem;
    }

    .page-header h1 {
        font-size: 20px;
        font-weight: 500;
        color: #e2e8f0;
        margin: 0;
        letter-spacing: 0.01em;
    }

    .page-header span {
        font-size: 12px;
        color: #4a5568;
        font-family: 'JetBrains Mono', monospace;
    }

    /* ── Metric cards ── */
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 12px;
        margin-bottom: 1.5rem;
    }

    .metric-card {
        background: #131720;
        border: 1px solid #1e2433;
        border-radius: 8px;
        padding: 16px 20px;
        position: relative;
        overflow: hidden;
    }

    .metric-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 2px;
        background: var(--accent, #3b82f6);
    }

    .metric-card .label {
        font-size: 11px;
        color: #4a5568;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-family: 'JetBrains Mono', monospace;
        margin-bottom: 8px;
    }

    .metric-card .value {
        font-size: 28px;
        font-weight: 500;
        color: #e2e8f0;
        font-family: 'JetBrains Mono', monospace;
        line-height: 1;
    }

    .metric-card .sub {
        font-size: 11px;
        color: #4a5568;
        margin-top: 6px;
        font-family: 'JetBrains Mono', monospace;
    }

    /* ── Risk badges ── */
    .badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 4px;
        font-size: 11px;
        font-weight: 500;
        font-family: 'JetBrains Mono', monospace;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }

    .badge-low      { background: #0d2b1e; color: #34d399; border: 1px solid #065f46; }
    .badge-medium   { background: #2d1f00; color: #fbbf24; border: 1px solid #78350f; }
    .badge-high     { background: #2d1200; color: #fb923c; border: 1px solid #7c2d12; }
    .badge-critical { background: #2d0000; color: #f87171; border: 1px solid #7f1d1d; }

    /* ── Transaction table ── */
    .tx-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 13px;
        font-family: 'JetBrains Mono', monospace;
    }

    .tx-table th {
        text-align: left;
        padding: 8px 12px;
        color: #4a5568;
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        border-bottom: 1px solid #1e2433;
        font-weight: 400;
    }

    .tx-table td {
        padding: 10px 12px;
        color: #a0aec0;
        border-bottom: 1px solid #0f1421;
    }

    .tx-table tr:hover td {
        background: #131720;
        color: #e2e8f0;
    }

    .tx-table .amount {
        color: #e2e8f0;
        font-weight: 500;
    }

    .tx-table .prob-high   { color: #f87171; }
    .tx-table .prob-medium { color: #fbbf24; }
    .tx-table .prob-low    { color: #34d399; }

    /* ── Drift event cards ── */
    .drift-card {
        background: #131720;
        border: 1px solid #1e2433;
        border-left: 3px solid var(--severity-color, #3b82f6);
        border-radius: 6px;
        padding: 12px 16px;
        margin-bottom: 8px;
    }

    .drift-card .drift-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 4px;
    }

    .drift-card .drift-title {
        font-size: 13px;
        font-weight: 500;
        color: #e2e8f0;
        font-family: 'JetBrains Mono', monospace;
    }

    .drift-card .drift-time {
        font-size: 11px;
        color: #4a5568;
        font-family: 'JetBrains Mono', monospace;
    }

    .drift-card .drift-desc {
        font-size: 12px;
        color: #718096;
        margin-top: 4px;
    }

    /* ── Section headers ── */
    .section-header {
        font-size: 12px;
        font-weight: 500;
        color: #4a5568;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        font-family: 'JetBrains Mono', monospace;
        margin: 1.5rem 0 0.75rem 0;
        padding-bottom: 6px;
        border-bottom: 1px solid #1e2433;
    }

    /* ── Input fields ── */
    .stTextInput input,
    .stNumberInput input,
    .stSelectbox select {
        background-color: #131720 !important;
        border: 1px solid #1e2433 !important;
        color: #e2e8f0 !important;
        border-radius: 6px !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 13px !important;
    }

    .stTextInput input:focus,
    .stNumberInput input:focus {
        border-color: #3b82f6 !important;
        box-shadow: 0 0 0 1px #3b82f6 !important;
    }

    /* ── Buttons ── */
    .stButton > button {
        background: #131720 !important;
        border: 1px solid #1e2433 !important;
        color: #a0aec0 !important;
        border-radius: 6px !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 12px !important;
        font-weight: 500 !important;
        letter-spacing: 0.05em !important;
        padding: 8px 16px !important;
        transition: all 0.15s !important;
    }

    .stButton > button:hover {
        border-color: #3b82f6 !important;
        color: #e2e8f0 !important;
        background: #1a2035 !important;
    }

    .stButton > button[kind="primary"] {
        background: #1a3a6b !important;
        border-color: #2563eb !important;
        color: #93c5fd !important;
    }

    .stButton > button[kind="primary"]:hover {
        background: #1e4080 !important;
        color: #bfdbfe !important;
    }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] {
        background: transparent !important;
        border-bottom: 1px solid #1e2433;
        gap: 0;
    }

    .stTabs [data-baseweb="tab"] {
        background: transparent !important;
        color: #4a5568 !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 12px !important;
        font-weight: 500 !important;
        letter-spacing: 0.05em !important;
        text-transform: uppercase !important;
        padding: 10px 20px !important;
        border: none !important;
    }

    .stTabs [aria-selected="true"] {
        color: #e2e8f0 !important;
        border-bottom: 2px solid #3b82f6 !important;
    }

    /* ── Plotly charts ── */
    .js-plotly-plot {
        border-radius: 8px;
        border: 1px solid #1e2433;
    }

    /* ── Expander ── */
    .streamlit-expanderHeader {
        background: #131720 !important;
        border: 1px solid #1e2433 !important;
        border-radius: 6px !important;
        color: #a0aec0 !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 12px !important;
    }

    /* ── Divider ── */
    hr {
        border-color: #1e2433 !important;
    }

    /* ── Scrollbar ── */
    ::-webkit-scrollbar { width: 4px; height: 4px; }
    ::-webkit-scrollbar-track { background: #0f1117; }
    ::-webkit-scrollbar-thumb { background: #1e2433; border-radius: 2px; }
    ::-webkit-scrollbar-thumb:hover { background: #2d3748; }

    /* ── Status indicators ── */
    .status-dot {
        display: inline-block;
        width: 7px;
        height: 7px;
        border-radius: 50%;
        margin-right: 6px;
    }
    .status-online  { background: #34d399; box-shadow: 0 0 6px #34d399; }
    .status-offline { background: #f87171; box-shadow: 0 0 6px #f87171; }
    .status-warning { background: #fbbf24; box-shadow: 0 0 6px #fbbf24; }

    /* ── SHAP explanation box ── */
    .explanation-box {
        background: #0d1520;
        border: 1px solid #1e3a5f;
        border-left: 3px solid #3b82f6;
        border-radius: 6px;
        padding: 12px 16px;
        font-size: 13px;
        color: #93c5fd;
        font-family: 'Inter', sans-serif;
        line-height: 1.6;
        margin: 8px 0;
    }

    /* ── Metric delta ── */
    [data-testid="stMetricDelta"] {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 12px !important;
    }

    /* ── Success/error/info boxes ── */
    .stSuccess {
        background: #0d2b1e !important;
        border: 1px solid #065f46 !important;
        color: #34d399 !important;
        border-radius: 6px !important;
    }

    .stError {
        background: #2d0000 !important;
        border: 1px solid #7f1d1d !important;
        color: #f87171 !important;
        border-radius: 6px !important;
    }

    .stInfo {
        background: #0d1520 !important;
        border: 1px solid #1e3a5f !important;
        color: #93c5fd !important;
        border-radius: 6px !important;
    }
    </style>
    """, unsafe_allow_html=True)


def metric_card(label: str, value: str,
                sub: str = "", accent: str = "#3b82f6") -> str:
    return f"""
    <div class="metric-card" style="--accent: {accent}">
        <div class="label">{label}</div>
        <div class="value">{value}</div>
        {"<div class='sub'>" + sub + "</div>" if sub else ""}
    </div>
    """


def risk_badge(risk_level: str) -> str:
    return f'<span class="badge badge-{risk_level}">{risk_level}</span>'


def section_header(title: str) -> str:
    return f'<div class="section-header">{title}</div>'


def explanation_box(text: str) -> str:
    return f'<div class="explanation-box">{text}</div>'


def drift_card(event: dict) -> str:
    severity_colors = {
        "critical": "#f87171",
        "high":     "#fb923c",
        "warning":  "#fbbf24",
        "none":     "#34d399",
    }
    color = severity_colors.get(event.get("severity", "none"), "#3b82f6")
    time  = event.get("timestamp", "")[:19].replace("T", " ")
    desc  = event.get("description", "")

    return f"""
    <div class="drift-card" style="--severity-color: {color}">
        <div class="drift-header">
            <span class="drift-title">
                {event.get('drift_type', '').upper()} —
                {event.get('metric_name', '')}
                = {event.get('metric_value', 0):.4f}
            </span>
            <span class="drift-time">{time}</span>
        </div>
        <div class="drift-desc">{desc}</div>
    </div>
    """
def plotly_dark_layout(fig, title: str = "", height: int = 400):
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=13, color="#a0aec0",
                      family="JetBrains Mono"),
        ),
        paper_bgcolor="#0f1117",
        plot_bgcolor="#0d1117",
        font=dict(family="JetBrains Mono", color="#4a5568", size=11),
        xaxis=dict(
            gridcolor="#1e2433",
            linecolor="#1e2433",
            tickcolor="#1e2433",
            tickfont=dict(color="#4a5568", size=10),
        ),
        yaxis=dict(
            gridcolor="#1e2433",
            linecolor="#1e2433",
            tickcolor="#1e2433",
            tickfont=dict(color="#4a5568", size=10),
        ),
        legend=dict(
            bgcolor="#0f1117",
            font=dict(color="#718096", size=10),
        ),
        height=height,
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig