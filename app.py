# ==========================================
# XGBoost - Diabetes Progression Predictor
# ==========================================

import streamlit as st
import numpy as np
import pandas as pd
# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Diabetes Progression Predictor",
    page_icon="🩺",
    layout="wide"
)

# -----------------------------
# Fail-Safe Resource Loading
# -----------------------------
try:
    from utils import load_model_and_metrics, generate_feature_importance_plot
except Exception as e:
    st.error(f"⚠️ Critical Module Failure: {e}")
    st.info("Ensure all dependencies in requirements.txt are installed.")
    st.stop()

# -----------------------------
# Custom Styling "Midnight Glass"
# -----------------------------
st.markdown("""
<style>
.stApp { background: linear-gradient(135deg, #0f172a, #1e293b); color: #e2e8f0; font-family: 'Inter', system-ui, sans-serif; }
h1, h2, h3 { color: #38bdf8 !important; font-weight: 700; }
.glass-panel { background: rgba(255, 255, 255, 0.05); backdrop-filter: blur(10px); -webkit-backdrop-filter: blur(10px); border-radius: 15px; padding: 25px; border: 1px solid rgba(255, 255, 255, 0.1); box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3); margin-bottom: 20px; }
.stNumberInput input, .stSelectbox select, .stTextInput input { background-color: rgba(15, 23, 42, 0.6) !important; color: #f8fafc !important; border: 1px solid rgba(56, 189, 248, 0.3) !important; border-radius: 8px !important; }
.stNumberInput input:focus, .stSelectbox select:focus { border: 1px solid #38bdf8 !important; box-shadow: 0 0 10px rgba(56, 189, 248, 0.5) !important; }
.stButton>button { background: linear-gradient(90deg, #3b82f6, #8b5cf6); color: white; border-radius: 8px; font-weight: 600; padding: 0.6em 1.5em; border: none; transition: all 0.3s ease; width: 100%; margin-top: 15px; }
.stButton>button:hover { background: linear-gradient(90deg, #60a5fa, #a78bfa); box-shadow: 0 0 15px rgba(139, 92, 246, 0.6); transform: translateY(-2px); }
.result-card-success { background: rgba(16, 185, 129, 0.15); border-left: 5px solid #10b981; padding: 20px; border-radius: 10px; margin-top: 20px; }
.result-card-warning { background: rgba(245, 158, 11, 0.15); border-left: 5px solid #f59e0b; padding: 20px; border-radius: 10px; margin-top: 20px; }
.result-card-danger { background: rgba(239, 68, 68, 0.15); border-left: 5px solid #ef4444; padding: 20px; border-radius: 10px; margin-top: 20px; }
.metric-text { font-size: 24px; font-weight: bold; color: #f8fafc; }
</style>
""", unsafe_allow_html=True)


# -----------------------------
# Load Model Details
# -----------------------------
try:
    model, features_list, metrics = load_model_and_metrics()
except Exception as e:
    st.error(f"XGBoost Kernel Error: {e}")
    import traceback
    st.code(traceback.format_exc())
    st.stop()

if model is None:
    st.warning("🧪 XGBoost artifacts missing. Run `python model_training.py` in this directory.")
    st.stop()


# -----------------------------
# Parameter Input Controls
# -----------------------------
st.sidebar.markdown("### 🔬 Patient Baselines")
st.sidebar.caption("Provide Normalized Standard Input.")

inputs = []
for feature in features_list:
    display_name = feature.upper()
    val = st.sidebar.number_input(display_name, value=0.0)
    inputs.append(val)

analyze_btn = st.sidebar.button("Predict Progression Score")


# -----------------------------
# Header Layout
# -----------------------------
st.markdown("<h1>🩺 Diabetes Progression Analytics</h1>", unsafe_allow_html=True)
st.markdown("<p style='font-size: 18px; color: #94a3b8;'>Quantitative Disease Evaluation via XGBoost Regression</p>", unsafe_allow_html=True)
st.markdown("---")


# -----------------------------
# Prediction & UI Tabs
# -----------------------------
tab1, tab2, tab3 = st.tabs(["🔮 Quantitative Output", "📊 Model Evaluation Scores", "📁 Foundational Records"])

with tab1:
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    st.markdown("### 🎯 Expected Progression Results")
    
    if analyze_btn:
        input_data = np.array([inputs])
        prediction = model.predict(input_data)[0]

        if prediction < 100:
            st.markdown(f"""
                <div class="result-card-success">
                    <div class="metric-text">🟢 Low Progression Range (Score: {round(prediction, 1)})</div>
                    <p style="color: #cbd5e1; margin-top: 5px;">Disease expansion potential is statistically subdued.</p>
                </div>
            """, unsafe_allow_html=True)
        elif prediction < 200:
             st.markdown(f"""
                <div class="result-card-warning">
                    <div class="metric-text">🟡 Moderate Progression Range (Score: {round(prediction, 1)})</div>
                    <p style="color: #cbd5e1; margin-top: 5px;">Significant disease activity recorded. Standard containment protocol advised.</p>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div class="result-card-danger">
                    <div class="metric-text">🔴 High Progression Risk (Score: {round(prediction, 1)})</div>
                    <p style="color: #cbd5e1; margin-top: 5px;">Aggressive longitudinal escalation expected. Intensive treatment strongly recommended.</p>
                </div>
            """, unsafe_allow_html=True)
    else:
        st.info("👈 Please map patient parameters via the sidebar controls to initiate inference.")
    st.markdown('</div>', unsafe_allow_html=True)


with tab2:
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    st.markdown("### 🧠 XGBoost Architecture Details")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Mean Squared Error (MSE)", f"{metrics['MSE']}")
    col2.metric("Root Mean Squared Error (RMSE)", f"{metrics['RMSE']}")
    col3.metric("R-Squared Correlation", f"{metrics['R2']}")
    
    st.markdown("---")
    fig = generate_feature_importance_plot(metrics)
    if fig:
        st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)


with tab3:
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    st.markdown("### 📥 Underlying Patient Data (Normalized)")
    try:
        df = pd.read_csv("data/diabetes.csv")
        st.dataframe(df.head(100), use_container_width=True)
        st.caption("Excerpt: Top 100 physiological records (scikit-learn base implementation).")
    except Exception as e:
        st.error(f"Failed to mount local filesystem records: {e}")
    st.markdown('</div>', unsafe_allow_html=True)