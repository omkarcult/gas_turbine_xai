import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="GT Digital Twin | Health Monitor", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- INJECT CUSTOM MODERN CSS (Visual Styling) ---
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background-color: #111111;
        color: #E0E0E0;
    }
    
    /* Header/Text styles */
    h1, h2, h3, p { font-family: 'Helvetica Neue', Arial, sans-serif !important; }
    
    /* Metric Cards Styling */
    div[data-testid="stMetricValue"] {
        font-size: 28px !important;
        font-weight: 600 !important;
        color: #FFFFFF !important;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 14px !important;
        color: #AAAAAA !important;
    }
    
    /* Container/Card styling */
    .st_metric_card {
        background-color: #1A1A1A;
        border: 1px solid #333333;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    
    /* Threshold Badges */
    .status-badge {
        padding: 5px 10px;
        border-radius: 20px;
        font-size: 14px;
        font-weight: 700;
        text-transform: uppercase;
    }
    .badge-green { background-color: #28a745; color: white; }
    .badge-yellow { background-color: #ffc107; color: black; }
    .badge-red { background-color: #dc3545; color: white; }
    
    /* Markdown Text Styling */
    .maintenance-step {
        background-color: #222222;
        border-left: 4px solid #444444;
        color: #CCCCCC;
        padding: 10px 15px;
        margin-bottom: 10px;
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)

# --- CORE LOGIC (unchanged) ---
@st.cache_resource
def load_data_and_model():
    try:
        with open("simple_turbine_model.pkl", "rb") as f:
            artifacts = pickle.load(f)
        df = pd.read_csv("simple_turbine_data.csv")
        return artifacts, df
    except FileNotFoundError:
        st.error("Error: Project files not found. Ensure you ran the generator and trainer scripts.")
        st.stop()

artifacts, df = load_data_and_model()
clf = artifacts["status_model"]
reg_rul = artifacts["rul_model"]
encoder = artifacts["encoder"]
feature_names = artifacts["features"]

# ==========================================
#              LAYOUT STARTS HERE
# ==========================================

# --- APP TOP BANNER ---
t_col1, t_col2 = st.columns([1.5, 3])
with t_col1:
    st.markdown("<h1 style='color: #4CAF50; margin-bottom: 0;'>⚙️ Gas Turbine Digital Twin</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #777777; font-size: 16px; margin-top: -10px;'>Advanced Asset Health & RUL Intelligence</p>", unsafe_allow_html=True)

# --- CONTROLS BOX (Top Right) ---
with t_col2:
    ctl1, ctl2 = st.columns([1, 2])
    with ctl1:
        st.markdown("<div style='margin-top: 25px;'></div>", unsafe_allow_html=True)
        selected_turbine = st.selectbox("Unit Selection", df["turbine_id"].unique())
    with ctl2:
        unit_df = df[df["turbine_id"] == selected_turbine].reset_index(drop=True)
        cycle_idx = st.slider("Historical Baseline Timeline (Operating Cycles)", 0, len(unit_df) - 1, 180)

sample = unit_df.iloc[cycle_idx]
X_sample = unit_df.loc[[cycle_idx], feature_names]

# Predictions (Logic remains the same)
pred_status_code = clf.predict(X_sample)[0]
pred_status_label = encoder.inverse_transform([pred_status_code])[0]
predicted_rul = int(reg_rul.predict(X_sample)[0])
health_val = sample["health_score"]

st.markdown("<hr style='border: 0.5px solid #333;'>", unsafe_allow_html=True)

# --- MAIN DASHBOARD BODY ---
col_left, col_right = st.columns([1, 1.4])

with col_left:
    st.markdown("### 📊 REAL-TIME TELEMETRY")
    
    # 1. Sensor Grid with modern metric cards
    with st.container():
        st.markdown('<div class="st_metric_card">', unsafe_allow_html=True)
        # Dynamic threshold check
        temp_status = "🟡" if sample['temperature'] > 1090 else "🟢"
        press_status = "🔴" if sample['pressure'] < 12.5 else "🟢"
        vib_status = "🟡" if sample['vibration'] > 2.5 else "🟢"
        
        m1, m2, m3 = st.columns(3)
        m1.metric("Temperature", f"{sample['temperature']} °C", f"{temp_status}", delta_color="off")
        m2.metric("Pressure", f"{sample['pressure']} Bar", f"{press_status}", delta_color="off")
        m3.metric("Vibration", f"{sample['vibration']} mm/s", f"{vib_status}", delta_color="off")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<div style='margin-top: 25px;'></div>", unsafe_allow_html=True)
    
    # 2. Maintenance Action Steps (UX Focus)
    st.markdown("### 💡 GUIDED MAINTENANCE WORKFLOW")
    
    if pred_status_label == "Normal Operation":
        st.markdown('<div class="maintenance-step" style="border-left-color: #28a745; color: #81c784; background-color: #121913;">✅ All Parameters Nominal. No intervention required. Routine monitoring active.</div>', unsafe_allow_html=True)
    elif pred_status_label == "Minor Degradation":
        st.markdown('<div class="maintenance-step" style="border-left-color: #ffc107; color: #fff176; background-color: #1e1b12;">⚠️ WARNING: PREDICTIVE ANOMALY DETECTED</div>', unsafe_allow_html=True)
        st.markdown('<div class="maintenance-step">1. Schedule Inspection of Compressors Inlet Filters (Possible Fouling).</div>', unsafe_allow_html=True)
        st.markdown('<div class="maintenance-step">2. Check Sensor Calibration (Temp/Pressure bias check).</div>', unsafe_allow_html=True)
        st.markdown('<div class="maintenance-step">3. Analyze Lube Oil for increased metal particulate matter.</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="maintenance-step" style="border-left-color: #dc3545; color: #ef9a9a; background-color: #241415;">🔴 ALERT: CRITICAL COMPONENT FAILURE IMMINENT</div>', unsafe_allow_html=True)
        st.markdown('<div class="maintenance-step">1. Initiate Phase 1 Controlled Shut-down Procedure.</div>', unsafe_allow_html=True)
        st.markdown('<div class="maintenance-step">2. Inspect Bearing Assemblies & Dynamic Seals (Excessive Vibration).</div>', unsafe_allow_html=True)
        st.markdown('<div class="maintenance-step">3. Replace Hot Section Components (HPT Nozzles / Shrouds).</div>', unsafe_allow_html=True)

with col_right:
    st.markdown("### 📈 HEALTH DEGRADATION PROFILE")
    
    # 3. All-New Modern Interactive Plotly Graph (UX FOCUS)
    fig = go.Figure()

    # Base Health Trend
    fig.add_trace(go.Scatter(
        x=unit_df['cycle'], 
        y=unit_df['health_score'],
        mode='lines',
        name='Health Trend (%)',
        line=dict(color='#2196F3', width=2),
        fill='tozeroy', fillcolor='rgba(33, 150, 243, 0.05)' # Gentle fill
    ))

    # Threshold Zones (Background styling)
    fig.add_hrect(y0=70, y1=100, fillcolor="#223625", opacity=0.4, layer="below", line_width=0, name="Normal Zone")
    fig.add_hrect(y0=35, y1=70, fillcolor="#33331a", opacity=0.4, layer="below", line_width=0, name="Warning Zone")
    fig.add_hrect(y0=0, y1=35, fillcolor="#362222", opacity=0.4, layer="below", line_width=0, name="Fault Zone")

    # The modern glowing marker for current cycle
    fig.add_trace(go.Scatter(
        x=[sample['cycle']], 
        y=[health_val],
        mode='markers+text',
        name='Current Operation Point',
        marker=dict(color='#dc3545', size=16, symbol='circle', line=dict(color='white', width=3)),
        showlegend=False
    ))

    # Layout improvements
    fig.update_layout(
        plot_bgcolor='#111111', paper_bgcolor='#111111',
        margin=dict(l=40, r=30, t=10, b=30),
        xaxis=dict(title="Operating Cycles (Age)", title_font=dict(color='#888888'), gridcolor='#222222', color='#AAAAAA'),
        yaxis=dict(title="Normalized Health Index (%)", range=[0, 105], gridcolor='#222222', color='#AAAAAA'),
        legend=dict(x=0.01, y=0.01, bgcolor='rgba(0,0,0,0.5)', font=dict(color='#AAAAAA')),
        height=380,
    )

    st.plotly_chart(fig, use_container_width=True)

st.markdown("<hr style='border: 0.5px solid #333;'>", unsafe_allow_html=True)

# --- BOTTOM SUMMARY EXEC PANEL (HIGH VISIBILITY UX) ---
summ_c1, summ_c2, summ_c3 = st.columns(3)

with summ_c1:
    st.markdown("##### Machine Status Verdict:")
    if pred_status_label == "Normal Operation":
        st.markdown('<span class="status-badge badge-green">OPTIMAL</span>', unsafe_allow_html=True)
    elif pred_status_label == "Minor Degradation":
        st.markdown('<span class="status-badge badge-yellow">ATTENTION: WEAR PRESENT</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-badge badge-red">DANGER: CRITICAL STATE</span>', unsafe_allow_html=True)

with summ_c2:
    st.metric("Aggregate Health Score", f"{health_val}%")

with summ_c3:
    st.metric("Predicted RUL Count", f"{predicted_rul} Cycles Left", "countdown", delta_color="normal")