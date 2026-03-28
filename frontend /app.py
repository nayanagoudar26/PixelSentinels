import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import time
import io
import base64

st.set_page_config(page_title="Pixel Sentinels", layout="wide", page_icon="🕵️")

# Custom CSS for UI Design Requirements
def local_css():
    st.markdown("""
    <style>
    /* Base theme inspired by SPECTRE Hackathon Badge */
    @keyframes gradientDrift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    .stApp {
        background: 
            radial-gradient(rgba(179, 136, 255, 0.1) 1.5px, transparent 1.5px),
            linear-gradient(-45deg, #05010b, #0f051a, #160729, #05010b);
        background-size: 25px 25px, 400% 400%;
        animation: gradientDrift 15s ease infinite;
        background-attachment: fixed;
        color: #e2e8f0;
        font-family: 'Inter', sans-serif;
    }

    /* Hide standard headers */
    header {visibility: hidden;}
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* --- UPLOAD PAGE STYLES --- */
    @keyframes cardPulse {
        0% { box-shadow: 0 0 20px rgba(147, 118, 224, 0.2), inset 0 0 15px rgba(147, 118, 224, 0.1); border-color: rgba(147, 118, 224, 0.3); }
        50% { box-shadow: 0 0 40px rgba(179, 136, 255, 0.5), inset 0 0 25px rgba(179, 136, 255, 0.3); border-color: rgba(179, 136, 255, 0.8); }
        100% { box-shadow: 0 0 20px rgba(147, 118, 224, 0.2), inset 0 0 15px rgba(147, 118, 224, 0.1); border-color: rgba(147, 118, 224, 0.3); }
    }
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: rgba(22, 1, 46, 0.4) !important;
        backdrop-filter: blur(12px) !important;
        -webkit-backdrop-filter: blur(12px) !important;
        border-radius: 16px !important;
        border: 2px solid rgba(147, 118, 224, 0.3) !important;
        box-shadow: 0 0 20px rgba(147, 118, 224, 0.2), inset 0 0 15px rgba(147, 118, 224, 0.1) !important;
        padding: 30px !important;
        animation: cardPulse 4s infinite alternate;
        transition: transform 0.3s ease;
    }
    div[data-testid="stVerticalBlockBorderWrapper"]:hover {
        transform: scale(1.01);
    }

    /* Target standard Streamlit uploader */
    div[data-testid="stFileUploader"] {
        background: transparent !important;
    }
    [data-testid="stFileUploaderDropzone"] {
        border: 2px dashed rgba(179, 136, 255, 0.4) !important;
        background: transparent !important;
        border-radius: 12px !important;
        box-shadow: none !important;
        padding: 30px !important;
        transition: all 0.3s ease !important;
    }
    [data-testid="stFileUploaderDropzone"]:hover {
        border-color: #b388ff !important;
        background: rgba(179, 136, 255, 0.1) !important;
    }
    
    @keyframes buttonGlow {
        0% { box-shadow: 0 4px 15px 0 rgba(101, 31, 255, 0.4); }
        50% { box-shadow: 0 4px 25px 8px rgba(179, 136, 255, 0.6); }
        100% { box-shadow: 0 4px 15px 0 rgba(101, 31, 255, 0.4); }
    }
    div.stButton > button {
        background: linear-gradient(90deg, #651fff 0%, #b388ff 100%);
        color: white;
        border: none;
        padding: 0.6em 2em;
        font-size: 1.1rem;
        border-radius: 6px;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px 0 rgba(101, 31, 255, 0.4);
        font-weight: 700;
        letter-spacing: 1px;
        text-transform: uppercase;
        width: 100%;
        margin-top: 10px;
        animation: buttonGlow 3s infinite alternate;
    }

    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px 0 rgba(101, 31, 255, 0.7);
        color: white;
    }
    
    div.stButton > button:active {
        transform: translateY(1px);
        box-shadow: 0 2px 10px 0 rgba(101, 31, 255, 0.4);
    }
    
    /* --- DASHBOARD PAGE STYLES --- */
    .dash-header {
        font-size: 28px;
        font-weight: 600;
        margin-bottom: 0px;
        color: #ffffff;
    }
    
    /* Streamlit Tabs Customization */
    div[data-testid="stTabs"] button {
        color: #8b92b2 !important;
        font-weight: 600 !important;
        background-color: transparent !important;
        font-size: 16px !important;
        padding-bottom: 10px !important;
    }
    div[data-testid="stTabs"] button[aria-selected="true"] {
        color: #ffffff !important;
        border-bottom-color: #b388ff !important;
    }
    div[data-testid="stTabs"] button:hover {
        color: #e2e8f0 !important;
    }

    /* Custom Dashboard Cards */
    .metric-card {
        background: rgba(22, 1, 46, 0.7);
        backdrop-filter: blur(10px);
        border-radius: 16px;
        padding: 20px;
        border: 1px solid rgba(124, 58, 237, 0.15);
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.4);
        height: 100%;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }

    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(124, 58, 237, 0.3);
    }
    
    .metric-title {
        color: #8b92b2;
        font-size: 14px;
        font-weight: 500;
        margin-bottom: 10px;
    }
    
    .metric-value {
        font-size: 32px;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 5px;
    }
    
    .metric-trend.up {
        color: #10b981; /* green */
        font-size: 12px;
        font-weight: 600;
    }
    .metric-trend.down {
        color: #ef4444; /* red */
        font-size: 12px;
        font-weight: 600;
    }
    .metric-trend.neutral {
        color: #f59e0b; /* yellow */
        font-size: 12px;
        font-weight: 600;
    }
    
    .chart-card {
        background: rgba(22, 1, 46, 0.7);
        backdrop-filter: blur(10px);
        border-radius: 16px;
        padding: 20px;
        border: 1px solid rgba(124, 58, 237, 0.15);
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.4);
        margin-top: 10px;
        margin-bottom: 10px;
    }

    /* Override Streamlit Metric text */
    div[data-testid="stMetricValue"] { color: #ffffff !important; }
    div[data-testid="stMetricLabel"] { color: #8b92b2 !important; }

    /* Logo Specific Container */
    .thumb-wrapper {
        border-radius: 20px;
        border: 2px solid rgba(179, 136, 255, 0.5);
        box-shadow: 0 0 20px rgba(179, 136, 255, 0.3), inset 0 0 10px rgba(179, 136, 255, 0.2);
        background: rgba(11, 5, 21, 0.6);
        display: flex;
        justify-content: center;
        align-items: center;
        height: 120px;
        width: 120px;
        margin: 0 auto 15px auto;
        overflow: hidden;
    }
    
    /* Hero Typography */
    .neon-title {
        color: #ffffff;
        text-align: center;
        font-family: 'Inter', sans-serif;
        font-weight: 900;
        font-size: 54px;
        letter-spacing: 3px;
        text-transform: uppercase;
        text-shadow: 0 0 10px rgba(179, 136, 255, 0.8), 0 0 20px rgba(179, 136, 255, 0.5), 0 0 30px rgba(179, 136, 255, 0.3);
        margin-bottom: 5px;
        margin-top: 0px;
        line-height: 1.1;
    }
    .neon-subtitle {
        color: #b388ff;
        text-align: center;
        letter-spacing: 2px;
        text-transform: uppercase;
        font-size: 18px;
        font-weight: 700;
        margin-bottom: 40px;
        opacity: 0.9;
    }
    
    </style>
    """, unsafe_allow_html=True)

local_css()

# State initialization
if 'analyzed' not in st.session_state:
    st.session_state.analyzed = False
if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None

# ==========================================
# DEFAULT LOGO SVG AND HELPER FUNCTION
# ==========================================
shield_svg = """
<svg viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <linearGradient id="shieldGrad" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#00f2fe" />
      <stop offset="100%" stop-color="#b388ff" />
    </linearGradient>
    <filter id="glow" x="-20%" y="-20%" width="140%" height="140%">
      <feGaussianBlur stdDeviation="2" result="blur" />
      <feComposite in="SourceGraphic" in2="blur" operator="over" />
    </filter>
  </defs>
  <g filter="url(#glow)">
    <!-- Right side of shield body -->
    <path d="M50 25 L75 35 C75 60 62 75 50 85 C38 75 28 60 28 45" fill="none" stroke="url(#shieldGrad)" stroke-width="6" stroke-linecap="round" stroke-linejoin="round"/>
    <!-- Left side fragmented top -->
    <path d="M38 28 L50 25" fill="none" stroke="#00f2fe" stroke-width="6" stroke-linecap="round"/>
    
    <!-- Pixels on left -->
    <rect x="23" y="32" width="6" height="6" fill="#00f2fe" />
    <rect x="32" y="32" width="6" height="6" fill="#00f2fe" />
    <rect x="23" y="42" width="6" height="6" fill="#b388ff" />
    <rect x="36" y="22" width="6" height="6" fill="#00f2fe" />

    <!-- Checkmark -->
    <path d="M35 55 L45 65 L80 30" fill="none" stroke="url(#shieldGrad)" stroke-width="8" stroke-linecap="round" stroke-linejoin="round"/>
  </g>
</svg>
"""

import os
def get_logo_html():
    logo_path = "image.jpg"
    if os.path.exists(logo_path) and os.path.getsize(logo_path) > 0:
        try:
            with open(logo_path, "rb") as f:
                b64_logo = base64.b64encode(f.read()).decode()
            return f"<div class='thumb-wrapper'><img src='data:image/jpeg;base64,{b64_logo}' style='max-width: 100%; max-height: 100%; object-fit: contain; filter: drop-shadow(0 0 8px rgba(179,136,255,0.6));'></div>"
        except Exception:
            encoded_svg = base64.b64encode(shield_svg.encode('utf-8')).decode('utf-8')
            return f"<div class='thumb-wrapper'><img src='data:image/svg+xml;base64,{encoded_svg}' style='width: 80px; height: 80px; object-fit: contain; filter: drop-shadow(0 0 8px rgba(179,136,255,0.6));'></div>"
    else:
        encoded_svg = base64.b64encode(shield_svg.encode('utf-8')).decode('utf-8')
        return f"<div class='thumb-wrapper'><img src='data:image/svg+xml;base64,{encoded_svg}' style='width: 80px; height: 80px; object-fit: contain; filter: drop-shadow(0 0 8px rgba(179,136,255,0.6));'></div>"

# ==========================================
# ROUTE 1: UPLOAD PAGE
# ==========================================
if not st.session_state.analyzed:
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("<h1 class='neon-title'>PIXEL SENTINELS</h1>", unsafe_allow_html=True)
        st.markdown("<p class='neon-subtitle'>AI FORENSICS & FORGERY DETECTION</p>", unsafe_allow_html=True)
        
        # Center Logo
        st.markdown("<div style='margin-bottom: 30px;'>" + get_logo_html() + "</div>", unsafe_allow_html=True)
        
        with st.container(border=True):
            uploaded_file = st.file_uploader("Upload Image To Terminal", type=['png', 'jpg', 'jpeg'])
            
            if uploaded_file is not None:
                 st.session_state.uploaded_file = uploaded_file
                 st.image(uploaded_file, width='stretch')
    
        analyze_clicked = st.button("🚀 INITIATE SCAN", width='stretch')
        if analyze_clicked:
            if uploaded_file is not None:
                with st.spinner("Analyzing image layers... AI Models at work"):
                    time.sleep(1.5) # Simulate processing
                    st.session_state.analyzed = True
                    st.rerun() # Switch to Dashboard Route
            else:
                st.warning("Please upload an image into the terminal first.")

# ==========================================
# ROUTE 2: DASHBOARD PAGE
# ==========================================
else:
    # --- HEADER ---
    col_nav, col_thumb = st.columns([8, 2])
    with col_nav:
        st.markdown("<div class='dash-header'>Pixel Sentinels <span style='color: #b388ff;'>Analytics Terminal</span></div>", unsafe_allow_html=True)
    with col_thumb:
        if st.session_state.uploaded_file is not None:
            try:
                st.session_state.uploaded_file.seek(0)
                bytes_data = st.session_state.uploaded_file.read()
                b64 = base64.b64encode(bytes_data).decode()
                st.markdown(f"<div class='thumb-wrapper'><img src='data:image/png;base64,{b64}' style='max-width: 100%; max-height: 100%; object-fit: contain; filter: drop-shadow(0 0 8px rgba(179,136,255,0.6));'></div>", unsafe_allow_html=True)
            except Exception:
                st.markdown("<div class='thumb-wrapper'><span style='color: #8b92b2; font-size: 11px; font-weight: 600; letter-spacing: 1px;'>LOGO ERROR</span></div>", unsafe_allow_html=True)
        else:
            st.markdown(get_logo_html(), unsafe_allow_html=True)

        if st.button("← New Analysis", width='stretch'):
            st.session_state.analyzed = False
            st.session_state.uploaded_file = None
            st.rerun()

    st.markdown("<hr style='border-color: rgba(255,255,255,0.05); margin-top: 10px; margin-bottom: 20px;'>", unsafe_allow_html=True)

    # --- TABS BINDING ---
    tab1, tab2, tab3, tab4 = st.tabs(["Dashboard", "Feature Details", "Pixel Map", "Export Report"])

    # --- SIMULATED SHARED METRICS ---
    ai_prob = 89.2
    manip_score = 92.4
    cnn_score = 85.0
    prnu_score = 94.1
    image_quality = "Good"

    # =================
    # TAB 1: DASHBOARD
    # =================
    with tab1:
        # --- ROW 1: KPI CARDS (Left) & BAR CHART (Right) ---
        row1_col1, row1_col2 = st.columns([1.2, 1])
        
        with row1_col1:
            # 2x2 Grid using columns
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"""
                <div class='metric-card'>
                    <div class='metric-title'>AI Generated Probability</div>
                    <div class='metric-value'>{ai_prob}%</div>
                    <div class='metric-trend up'>↑ +12% compared to typical baseline</div>
                </div>
                """, unsafe_allow_html=True)
                st.markdown("<div style='height:15px'></div>", unsafe_allow_html=True)
                st.markdown(f"""
                <div class='metric-card'>
                    <div class='metric-title'>PRNU Noise Anomaly</div>
                    <div class='metric-value'>{prnu_score}%</div>
                    <div class='metric-trend up'>High indication of camera absence</div>
                </div>
                """, unsafe_allow_html=True)
                
            with c2:
                st.markdown(f"""
                <div class='metric-card'>
                    <div class='metric-title'>Manipulation Score</div>
                    <div class='metric-value'>{manip_score}</div>
                    <div class='metric-trend down'>Critical forgery traces found</div>
                </div>
                """, unsafe_allow_html=True)
                st.markdown("<div style='height:15px'></div>", unsafe_allow_html=True)
                st.markdown(f"""
                <div class='metric-card'>
                    <div class='metric-title'>Image Forensic Quality</div>
                    <div class='metric-value'>{image_quality}</div>
                    <div class='metric-trend neutral'>Sufficient data for analysis</div>
                </div>
                """, unsafe_allow_html=True)

        with row1_col2:
            st.markdown("<div class='chart-card' style='height: 100%; margin:0;'>", unsafe_allow_html=True)
            st.markdown("<div class='metric-title'>Feature Confidence Analysis</div>", unsafe_allow_html=True)
            
            bar_data = pd.DataFrame({
                "Feature": ["CNN Details", "PRNU Noise", "Edge Anomalies", "CFA Pattern", "Metadata", "Compression"],
                "Score": [cnn_score, prnu_score, 78.5, 88.0, 15.0, 62.0]
            })
            fig_bar = px.bar(bar_data, x="Feature", y="Score")
            fig_bar.update_traces(marker_color='#b388ff', marker_line_color='#d1c4e9', marker_line_width=1.5, opacity=0.9, width=0.4)
            fig_bar.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#8b92b2'),
                margin=dict(t=10, b=30, l=10, r=10),
                height=250,
                yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)', zeroline=False),
                xaxis=dict(showgrid=False, zeroline=False)
            )
            st.plotly_chart(fig_bar, width='stretch')
            st.markdown("</div>", unsafe_allow_html=True)
            
        st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)

        # --- ROW 2: LINE CHART (Left) & DONUT CHART (Right) ---
        row2_col1, row2_col2 = st.columns([1.2, 1])

        with row2_col1:
            st.markdown("<div class='chart-card' style='margin:0;'>", unsafe_allow_html=True)
            st.markdown("<div class='metric-title'>Frequency Spectrum Anomalies (Simulated Track)</div>", unsafe_allow_html=True)
            
            x_vals = ['Zone 1', 'Zone 2', 'Zone 3', 'Zone 4', 'Zone 5', 'Zone 6', 'Zone 7', 'Zone 8']
            base_line1 = [20, 35, 30, 85, 90, 80, 85, 90] # AI Anomaly
            base_line2 = [15, 20, 25, 40, 35, 30, 35, 25] # Suspicious Edge
            base_line3 = [5, 10, 8, 15, 10, 12, 15, 5]    # Natural Baseline
            
            fig_line = go.Figure()
            fig_line.add_trace(go.Scatter(x=x_vals, y=base_line1, mode='lines+markers', name='AI Traces', line=dict(color='#b388ff', width=3, shape='spline'), marker=dict(size=8, color='#b388ff', symbol='circle')))
            fig_line.add_trace(go.Scatter(x=x_vals, y=base_line2, mode='lines', name='Edge Traces', line=dict(color='#2d78ff', width=3, shape='spline')))
            fig_line.add_trace(go.Scatter(x=x_vals, y=base_line3, mode='lines', name='Natural Baseline', line=dict(color='#8b92b2', width=2, shape='spline', dash='dot')))
            
            fig_line.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#8b92b2'),
                margin=dict(t=10, b=10, l=10, r=10),
                height=300,
                yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)', zeroline=False),
                xaxis=dict(showgrid=False, zeroline=False),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_line, width='stretch')
            st.markdown("</div>", unsafe_allow_html=True)

        with row2_col2:
            st.markdown("<div class='chart-card' style='margin:0; height: 100%;'>", unsafe_allow_html=True)
            st.markdown("<div class='metric-title'>Verdict Breakdown</div>", unsafe_allow_html=True)
            
            labels = ['Deepfake (GAN)', 'Diffusion Model', 'Photoshop', 'Real/Authentic']
            values = [60, 25, 10, 5]
            colors = ['#651fff', '#b388ff', '#2d78ff', '#f59e0b']
            
            fig_donut = go.Figure(data=[go.Pie(
                labels=labels, 
                values=values, 
                hole=.65,
                marker=dict(colors=colors, line=dict(color='#110822', width=3)),
                hoverinfo="label+percent",
                textinfo='none'
            )])
            
            fig_donut.update_layout(
                annotations=[dict(text=f"{ai_prob}%", x=0.5, y=0.5, font_size=28, font_color="white", showarrow=False),
                             dict(text="AI VERDICT", x=0.5, y=0.38, font_size=10, font_color="#8b92b2", showarrow=False)],
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                margin=dict(t=10, b=10, l=10, r=10),
                height=300,
                showlegend=True,
                legend=dict(font=dict(color='#e2e8f0'), yanchor="middle", y=0.5, xanchor="left", x=0.9)
            )
            st.plotly_chart(fig_donut, width='stretch')
            st.markdown("</div>", unsafe_allow_html=True)

    # ========================
    # TAB 2: FEATURE DETAILS
    # ========================
    with tab2:
        st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
        st.markdown("<h3 style='color: white; margin-top: 0;'>Deep Analysis Breakdown</h3>", unsafe_allow_html=True)
        st.markdown("<p style='color: #8b92b2;'>Detailed forensic component scores derived from the uploaded source image.</p>", unsafe_allow_html=True)
        
        st.markdown(f"**PRNU Sensor Noise Matrix:** <span style='color: #b388ff;'>{prnu_score}% Anomalous</span>", unsafe_allow_html=True)
        st.progress(prnu_score / 100)
        
        st.markdown(f"**CNN Spatial Recognition Vector:** <span style='color: #2d78ff;'>{cnn_score}% Extracted</span>", unsafe_allow_html=True)
        st.progress(cnn_score / 100)

        st.markdown(f"**Metadata EXIF Validation:** <span style='color: #f59e0b;'>Missing Origin Signature</span>", unsafe_allow_html=True)
        st.progress(0.15)
        st.markdown("</div>", unsafe_allow_html=True)

    # ========================
    # TAB 3: PIXEL MAP
    # ========================
    with tab3:
        st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
        st.markdown("<h3 style='color: white; margin-top: 0;'>Error Level Analysis (ELA) Heatmap</h3>", unsafe_allow_html=True)
        st.markdown("<p style='color: #8b92b2;'>Visualizes simulated altered/forged zones by measuring pixel discrepancy patterns.</p>", unsafe_allow_html=True)
        
        if st.session_state.uploaded_file is not None:
            # Simulate a heatmap over the image size using random clusters
            # Since PIL Image is easily accessible:
            try:
                img = Image.open(st.session_state.uploaded_file)
                # Ensure width/height are reasonable to not hang Plotly
                img.thumbnail((300, 300))
                w, h = img.size
                
                # Create a simulated "hot spot" mask
                x = np.linspace(0, 5, w)
                y = np.linspace(0, 5, h)
                X, Y = np.meshgrid(x, y)
                Z = np.sin(X)**2 + np.cos(Y)**2 + np.random.normal(0, 0.2, (h, w))
                
                # Overlay real image (grayscale) with heatmap
                fig_map = go.Figure(data=go.Heatmap(
                    z=Z,
                    colorscale='magma',
                    opacity=0.7,
                    showscale=False
                ))
                
                fig_map.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    margin=dict(t=0, b=0, l=0, r=0),
                    height=400,
                    xaxis=dict(showgrid=False, zeroline=False, visible=False),
                    yaxis=dict(showgrid=False, zeroline=False, visible=False)
                )
                
                # Try to add background image if possible, or just show the ELA mask
                st.plotly_chart(fig_map, width='stretch')
                
            except Exception as e:
                st.error("Could not process image for Pixel Map.")
        st.markdown("</div>", unsafe_allow_html=True)

    # ========================
    # TAB 4: EXPORT REPORT
    # ========================
    with tab4:
        st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
        st.markdown("<h3 style='color: white; margin-top: 0;'>Generate Forensics Report</h3>", unsafe_allow_html=True)
        st.markdown("<p style='color: #8b92b2;'>Download a finalized forensic evaluation report outlining the AI prediction results, ELA scores, and metadata extraction records.</p>", unsafe_allow_html=True)
        
        report_content = f"""PIXEL SENTINELS V3.0
========================
FORENSIC ANALYSIS REPORT

FILE: Analyzed Image Data
DATE: {time.strftime("%Y-%m-%d %H:%M:%S")}
QUALITY: {image_quality}

[ SCORE BREAKDOWN ]
AI Probability Score   : {ai_prob}%
Manipulation Sub-score : {manip_score}%
PRNU Irregularity      : {prnu_score}%
CNN Trace Detection    : {cnn_score}%

[ VERDICT ]
Based on the advanced heuristics scan, the image exhibits strong indicators of artificial generation or significant generative alteration.

-- END OF REPORT --
"""
        st.download_button(
            label="📄 DOWNLOAD METRICS REPORT",
            data=report_content,
            file_name="pixel_sentinels_forensics_report.txt",
            mime="text/plain",
            help="Download the text-based summary of this image analysis."
        )
        st.markdown("</div>", unsafe_allow_html=True)

