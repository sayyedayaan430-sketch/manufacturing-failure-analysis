"""
app.py
──────
Streamlit web app for the Manufacturing Failure Analysis project.

Run with:
    streamlit run app.py
"""

import streamlit as st
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Manufacturing Failure Analysis",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background-color: #0f1117; }
    h1, h2, h3 { color: #ffffff; }

    .risk-high {
        background: linear-gradient(135deg, #da3633, #b91c1c);
        color: white; padding: 16px 24px; border-radius: 12px;
        font-size: 1.3rem; font-weight: bold; text-align: center;
    }
    .risk-medium {
        background: linear-gradient(135deg, #d29922, #b45309);
        color: white; padding: 16px 24px; border-radius: 12px;
        font-size: 1.3rem; font-weight: bold; text-align: center;
    }
    .risk-low {
        background: linear-gradient(135deg, #238636, #2ea043);
        color: white; padding: 16px 24px; border-radius: 12px;
        font-size: 1.3rem; font-weight: bold; text-align: center;
    }
    .stButton > button {
        background: linear-gradient(135deg, #da3633, #b91c1c);
        color: white; border: none; border-radius: 8px;
        padding: 10px 28px; font-size: 1rem;
        font-weight: 600; width: 100%;
    }
    div[data-testid="metric-container"] {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px; padding: 12px;
    }
    section[data-testid="stSidebar"] {
        background-color: #161b22;
        border-right: 1px solid #30363d;
    }
    .footer {
        text-align: center; color: #6e7681;
        font-size: 0.8rem; margin-top: 40px;
        padding-top: 20px; border-top: 1px solid #30363d;
    }
</style>
""", unsafe_allow_html=True)

# ─── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏭 About")
    st.markdown("---")
    st.markdown("""
    This app predicts **manufacturing equipment failure**
    based on real-time sensor readings using:
    - 🌲 **Random Forest** classifier
    - 📊 Trained on historical sensor data
    - ⚡ Instant risk assessment
    """)
    st.markdown("---")
    st.markdown("### ⚙️ Model Info")
    st.info("Train the model first:\n`python src/train.py`")
    st.markdown("---")
    st.markdown("### 📊 Risk Levels")
    st.markdown("🔴 **HIGH** — > 75% failure probability")
    st.markdown("🟡 **MEDIUM** — 40–75% failure probability")
    st.markdown("🟢 **LOW** — < 40% failure probability")

# ─── Main Page ─────────────────────────────────────────────────────────────────
st.markdown("# 🏭 Manufacturing Failure Analysis")
st.markdown("##### Predict equipment failures before they happen using Machine Learning")
st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 🌡️ Thermal & Mechanical")
    temperature = st.slider("Temperature (°C)", 200.0, 500.0, 300.0, step=1.0)
    pressure    = st.slider("Pressure (PSI)",    50.0, 300.0, 140.0, step=1.0)
    torque      = st.slider("Torque (Nm)",        10.0, 100.0,  40.0, step=0.5)

with col2:
    st.markdown("### ⚙️ Motion & Wear")
    vibration   = st.slider("Vibration Level",   0.0,   2.0,   0.5, step=0.01)
    rpm         = st.slider("Rotational Speed (RPM)", 500.0, 3000.0, 1500.0, step=10.0)
    tool_wear   = st.slider("Tool Wear (minutes)",  0.0, 300.0,  100.0, step=1.0)

st.markdown("---")
predict_clicked = st.button("🔍 Predict Failure Risk", use_container_width=True)

# ─── Prediction ────────────────────────────────────────────────────────────────
if predict_clicked:
    try:
        from predict import predict_failure

        sensor_data = {
            'temperature':      temperature,
            'pressure':         pressure,
            'torque':           torque,
            'vibration':        vibration,
            'rotational_speed': rpm,
            'tool_wear':        tool_wear,
        }

        with st.spinner("Analyzing sensor data..."):
            result = predict_failure(sensor_data)

        st.markdown("---")
        st.markdown("### 🎯 Prediction Result")

        # Risk banner
        risk = result['risk_level']
        prob = result['failure_probability']
        css_class = {'HIGH': 'risk-high', 'MEDIUM': 'risk-medium', 'LOW': 'risk-low'}[risk]
        icon = {'HIGH': '🔴', 'MEDIUM': '🟡', 'LOW': '🟢'}[risk]

        st.markdown(
            f'<div class="{css_class}">{icon} {risk} RISK — {result["prediction"]}<br>'
            f'<span style="font-size:1rem;font-weight:normal">{result["message"]}</span></div>',
            unsafe_allow_html=True
        )

        st.markdown("---")

        # Metrics
        c1, c2, c3 = st.columns(3)
        c1.metric("Failure Probability", f"{prob:.1%}")
        c2.metric("Risk Level", risk)
        c3.metric("Prediction", result['prediction'])

        # Confidence bar
        st.markdown("### 📊 Failure Probability")
        st.progress(prob)
        st.caption(f"{prob:.1%} chance of failure based on current sensor readings")

    except FileNotFoundError as e:
        st.error(f"⚠️ Model not found! Run `python src/train.py` first.\n\n`{e}`")
    except Exception as e:
        st.error(f"An error occurred: {e}")

# ─── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">Built with ❤️ using Python, scikit-learn & Streamlit</div>
""", unsafe_allow_html=True)
