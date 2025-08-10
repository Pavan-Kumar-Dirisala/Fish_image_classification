# app.py — Beautiful Enhanced Streamlit UI for Fish Image Classification
import streamlit as st
from gradio_client import Client, handle_file
from PIL import Image
import plotly.graph_objects as go
import plotly.express as px
import time, os
import base64

# =========================
# CONFIG (hardcoded values)
# =========================
SPACE_REPO_ID = "PavanKumarD/Fish_Image_Classification"
API_NAME = "/predict"
APP_TITLE = "AquaScan AI - Fish Image Classification"
APP_ICON = "🐟"

# =========================
# EXACT TRAINING MAPPING
# =========================
label_mapping = {
    'animal fish bass': 0,
    'fish sea_food trout': 1,
    'fish sea_food striped_red_mullet': 2,
    'fish sea_food shrimp': 3,
    'fish sea_food red_mullet': 4,
    'fish sea_food red_sea_bream': 5,
    'fish sea_food gilt_head_bream': 6,
    'animal fish': 7,
    'fish sea_food black_sea_sprat': 8,
    'fish sea_food hourse_mackerel': 9,
    'fish sea_food sea_bass': 10
}

# Build an index -> label list in correct order
CLASS_NAMES = [None] * len(label_mapping)
for name, idx in label_mapping.items():
    CLASS_NAMES[idx] = name
NUM_CLASSES = len(CLASS_NAMES)

# =========================
# Enhanced Styling
# =========================
def load_css():
    st.markdown("""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        font-family: 'Inter', sans-serif;
    }
    
    /* Main container */
    .main .block-container {
        max-width: 1200px;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Header styling */
    .main-header {
        text-align: center;
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(20px);
        border-radius: 20px;
        padding: 2rem;
        margin-bottom: 2rem;
        border: 1px solid rgba(255, 255, 255, 0.2);
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
    }
    
    .main-title {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(45deg, #ffffff, #f0f9ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
        text-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .main-subtitle {
        color: rgba(255, 255, 255, 0.8);
        font-size: 1.2rem;
        font-weight: 400;
        margin-bottom: 1rem;
    }
    
    /* Card styles */
    .glass-card {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(20px);
        border-radius: 20px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid rgba(255, 255, 255, 0.2);
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
        transition: all 0.3s ease;
    }
    
    .glass-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px 0 rgba(31, 38, 135, 0.5);
    }
    
    .result-card {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.2), rgba(255, 255, 255, 0.1));
        backdrop-filter: blur(25px);
        border-radius: 25px;
        padding: 2rem;
        margin: 1.5rem 0;
        border: 1px solid rgba(255, 255, 255, 0.3);
        box-shadow: 0 15px 35px rgba(31, 38, 135, 0.3);
    }
    
    .ensemble-card {
        background: linear-gradient(135deg, rgba(34, 197, 94, 0.2), rgba(16, 185, 129, 0.1));
        border: 1px solid rgba(34, 197, 94, 0.3);
    }
    
    .model-card {
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.2), rgba(37, 99, 235, 0.1));
        border: 1px solid rgba(59, 130, 246, 0.3);
        height: 100%;
    }
    
    /* Upload area */
    .upload-section {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(20px);
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
        border: 2px dashed rgba(255, 255, 255, 0.3);
        transition: all 0.3s ease;
    }
    
    .upload-section:hover {
        border-color: rgba(255, 255, 255, 0.6);
        background: rgba(255, 255, 255, 0.15);
    }
    
    /* Button styles */
    .stButton > button {
        background: linear-gradient(45deg, #ff6b6b, #ee5a52);
        color: white;
        border: none;
        border-radius: 15px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1.1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(238, 90, 82, 0.4);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(238, 90, 82, 0.6);
        background: linear-gradient(45deg, #ee5a52, #ff6b6b);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(20px);
    }
    
    /* Progress bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(45deg, #ff6b6b, #ee5a52);
    }
    
    /* Text styling */
    .metric-title {
        color: rgba(255, 255, 255, 0.9);
        font-size: 0.9rem;
        font-weight: 500;
        margin-bottom: 0.5rem;
    }
    
    .metric-value {
        color: white;
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    .section-title {
        color: white;
        font-size: 1.8rem;
        font-weight: 600;
        margin: 2rem 0 1rem 0;
        text-align: center;
    }
    
    .confidence-text {
        color: rgba(255, 255, 255, 0.8);
        font-size: 0.9rem;
        font-style: italic;
    }
    
    /* Animation keyframes */
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes pulse {
        0% {
            transform: scale(1);
        }
        50% {
            transform: scale(1.05);
        }
        100% {
            transform: scale(1);
        }
    }
    
    .animate-fade-in {
        animation: fadeInUp 0.6s ease-out;
    }
    
    .animate-pulse {
        animation: pulse 2s infinite;
    }
    
    /* Custom file uploader */
    .uploadedFile {
        border-radius: 15px;
        overflow: hidden;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);
    }
    
    /* Custom plotly styling */
    .js-plotly-plot {
        border-radius: 15px;
        overflow: hidden;
    }
    </style>
    """, unsafe_allow_html=True)

# =========================
# Enhanced page setup
# =========================
st.set_page_config(
    page_title=APP_TITLE, 
    page_icon=APP_ICON, 
    layout="wide",
    initial_sidebar_state="expanded"
)

load_css()

# Enhanced header
st.markdown(f"""
<div class="main-header animate-fade-in">
    <div class="main-title">{APP_ICON} AquaScan AI</div>
    <div class="main-subtitle">Advanced Fish Species Classification</div>
    <div class="confidence-text">Powered by ResNet18 & MobileNetV2 • Real-time Marine Life Detection</div>
</div>
""", unsafe_allow_html=True)

# =========================
# Client init (cached)
# =========================
@st.cache_resource
def get_client(space_repo_id: str):
    try:
        return Client(space_repo_id), None
    except Exception as e:
        return None, str(e)

client, client_err = get_client(SPACE_REPO_ID)

# Enhanced sidebar
with st.sidebar:
    st.markdown("""
    <div class="glass-card">
        <h3 style="color: white; margin-bottom: 1rem;">🔗 Connection Status</h3>
    </div>
    """, unsafe_allow_html=True)
    
    if client:
        st.success("✅ Connected to AI Models")
        st.markdown(f"""
        <div class="glass-card">
            <div style="color: rgba(255,255,255,0.8); font-size: 0.9rem;">
                <strong>Space:</strong> {SPACE_REPO_ID}<br>
                <strong>Endpoint:</strong> {API_NAME}
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.error("❌ Connection Failed")
        if client_err:
            st.code(str(client_err))
    
    st.markdown("""
    <div class="glass-card">
        <h3 style="color: white; margin-bottom: 1rem;">🐠 Detectable Species</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Display species in a more organized way
    species_display = []
    for name in CLASS_NAMES:
        if name:
            # Clean up the label for display
            # clean_name = name.replace('fish sea_food ', '').replace('animal fish', 'bass').replace('_', ' ').title()
            species_display.append(f"• {name}")
    
    st.markdown(f"""
    <div class="glass-card">
        <div style="color: rgba(255,255,255,0.8); font-size: 0.85rem; line-height: 1.6;">
            {'<br>'.join(species_display)}
        </div>
    </div>
    """, unsafe_allow_html=True)

# =========================
# Enhanced Helpers
# =========================
def idx_to_label(idx: int) -> str:
    if 0 <= idx < NUM_CLASSES:
        return CLASS_NAMES[idx]
    return f"Class-{idx}"

def create_enhanced_confidence_chart(label: str, confidence_pct: float, title: str = "Confidence", color_scheme="blue"):
    """Create a beautiful confidence visualization"""
    
    # Color schemes
    colors = {
        "blue": ["#3B82F6", "#1D4ED8"],
        "green": ["#10B981", "#059669"],
        "orange": ["#F59E0B", "#D97706"],
        "purple": ["#8B5CF6", "#7C3AED"]
    }
    
    primary_color, secondary_color = colors.get(color_scheme, colors["blue"])
    
    # Create the confidence bar
    fig = go.Figure()
    
    # Main confidence bar
    fig.add_trace(go.Bar(
        x=[confidence_pct],
        y=[label],
        orientation='h',
        marker=dict(
            color=primary_color,
            line=dict(color=secondary_color, width=2)
        ),
        text=f"{confidence_pct:.1f}%",
        textposition="inside",
        textfont=dict(color="white", size=14, family="Inter"),
        name="Confidence",
        hovertemplate=f"<b>{label}</b><br>Confidence: {confidence_pct:.1f}%<extra></extra>"
    ))
    
    # Background bar for remaining percentage
    fig.add_trace(go.Bar(
        x=[100 - confidence_pct],
        y=[label],
        orientation='h',
        marker=dict(color="rgba(255,255,255,0.1)"),
        text=f"{100 - confidence_pct:.1f}%",
        textposition="inside",
        textfont=dict(color="rgba(255,255,255,0.6)", size=12, family="Inter"),
        name="Remaining",
        hovertemplate=f"Remaining: {100 - confidence_pct:.1f}%<extra></extra>"
    ))
    
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(color="white", size=16, family="Inter"),
            x=0.5
        ),
        xaxis=dict(
            title=dict(text="Confidence (%)", font=dict(color="white", size=12)),
            tickfont=dict(color="white"),
            gridcolor="rgba(255,255,255,0.1)",
            range=[0, 100]
        ),
        yaxis=dict(
            title=dict(font=dict(color="white", size=12)),
            tickfont=dict(color="white", size=12)
        ),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        height=120,
        margin=dict(l=10, r=10, t=40, b=10),
        showlegend=False,
        barmode='stack'
    )
    
    return fig

def create_model_comparison_chart(resnet_conf: float, mobilenet_conf: float):
    """Create a comparison chart for both models"""
    
    models = ['ResNet18', 'MobileNetV2']
    confidences = [resnet_conf * 100, mobilenet_conf * 100]
    colors = ['#FF6B6B', '#4ECDC4']
    
    fig = go.Figure(data=[
        go.Bar(
            x=models,
            y=confidences,
            marker=dict(
                color=colors,
                line=dict(color='white', width=2)
            ),
            text=[f'{conf:.1f}%' for conf in confidences],
            textposition='auto',
            textfont=dict(color='white', size=14, family='Inter')
        )
    ])
    
    fig.update_layout(
        title=dict(
            text="Model Confidence Comparison",
            font=dict(color="white", size=18, family="Inter"),
            x=0.5
        ),
        xaxis=dict(
            title=dict(font=dict(color="white", size=12)),
            tickfont=dict(color="white", size=12)
        ),
        yaxis=dict(
            title=dict(text="Confidence (%)", font=dict(color="white", size=12)),
            tickfont=dict(color="white"),
            gridcolor="rgba(255,255,255,0.1)",
            range=[0, 100]
        ),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        height=300,
        margin=dict(l=10, r=10, t=40, b=10)
    )
    
    return fig

def parse_space_output(result: dict):
    """Parse the output from the Space"""
    per = {}
    for name in ("ResNet18", "MobileNetV2"):
        data = result.get(name, {})
        idx = int(data.get("predicted_class", -1))
        conf = float(data.get("confidence", 0.0))
        per[name] = {"idx": idx, "label": idx_to_label(idx), "conf": conf}
    
    r18 = per["ResNet18"]
    mv2 = per["MobileNetV2"]
    
    # Majority vote; on tie pick higher-confidence model
    if r18["idx"] == mv2["idx"] and r18["idx"] != -1:
        ens_idx = r18["idx"]
        ens_conf = (r18["conf"] + mv2["conf"]) / 2.0
        note = "Consensus (both models agree)"
    else:
        if r18["conf"] >= mv2["conf"]:
            ens_idx, ens_conf = r18["idx"], r18["conf"]
            note = "Split vote — ResNet18 selected (higher confidence)"
        else:
            ens_idx, ens_conf = mv2["idx"], mv2["conf"]
            note = "Split vote — MobileNetV2 selected (higher confidence)"
    
    return {
        "per_model": per,
        "ensemble": {"idx": ens_idx, "label": idx_to_label(ens_idx), "conf": ens_conf, "note": note}
    }

# =========================
# Enhanced Upload & Inference UI
# =========================
st.markdown('<div class="section-title">📸 Upload Fish Image</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown('<div class="upload-section">', unsafe_allow_html=True)
    file = st.file_uploader(
        "Choose an image file", 
        type=["jpg", "jpeg", "png"], 
        label_visibility="collapsed",
        help="Upload a clear image of a fish for AI classification"
    )
    st.markdown('</div>', unsafe_allow_html=True)

if file:
    # Display uploaded image with enhanced styling
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.markdown("""
        <div class="glass-card animate-fade-in">
            <h3 style="color: white; text-align: center; margin-bottom: 1rem;">📷 Uploaded Image</h3>
        </div>
        """, unsafe_allow_html=True)
        st.image(file, use_container_width=True, caption="Ready for analysis")
    
    with col2:
        st.markdown("""
        <div class="glass-card animate-fade-in">
            <h3 style="color: white; text-align: center; margin-bottom: 1rem;">🚀 Analysis Control</h3>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        if st.button("🔍 Analyze Fish Species", use_container_width=True):
            if not client:
                st.error("🚫 AI models not connected. Please check the connection.")
            else:
                # Enhanced loading experience
                with st.spinner("🧠 AI models are analyzing your image..."):
                    # Save temp file for handle_file()
                    tmp = f"_tmp_{file.name}"
                    with open(tmp, "wb") as f:
                        f.write(file.getbuffer())
                    
                    # Enhanced progress bar
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Progress simulation with status updates
                    progress_steps = [
                        (10, "🔄 Preprocessing image..."),
                        (30, "🧮 Loading ResNet18 model..."),
                        (50, "📱 Loading MobileNetV2 model..."),
                        (70, "🔍 Running inference..."),
                        (90, "📊 Calculating results...")
                    ]
                    
                    for progress, message in progress_steps:
                        status_text.text(message)
                        progress_bar.progress(progress)
                        time.sleep(0.3)
                    
                    try:
                        raw = client.predict(image=handle_file(tmp), api_name=API_NAME)
                        status_text.text("✅ Analysis complete!")
                        progress_bar.progress(100)
                        time.sleep(0.5)
                    except Exception as e:
                        raw = {"__error__": str(e)}
                    finally:
                        try: 
                            os.remove(tmp)
                        except Exception: 
                            pass
                    
                    progress_bar.empty()
                    status_text.empty()
                
                if "__error__" in raw:
                    st.error(f"🚫 Analysis failed: {raw['__error__']}")
                    st.stop()
                
                # Parse & render results with enhanced UI
                try:
                    parsed = parse_space_output(raw)
                except Exception as e:
                    st.error(f"Failed to parse results.\n\nRaw output:\n{raw}\n\nError: {e}")
                    st.stop()
                
                ens = parsed["ensemble"]
                pm = parsed["per_model"]
                
                # Clean up the label for display
                clean_label = ens["label"]
                # ====== Enhanced Ensemble Results ======
                st.markdown('<div class="section-title">🎯 Final Prediction</div>', unsafe_allow_html=True)
                
                st.markdown(f"""
                <div class="result-card ensemble-card animate-fade-in">
                    <div style="text-align: center;">
                        <div class="metric-title">IDENTIFIED SPECIES</div>
                        <div class="metric-value">{clean_label}</div>
                        <div class="confidence-text">{ens['note']}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Confidence visualization
                fig_ensemble = create_enhanced_confidence_chart(
                    clean_label, 
                    ens["conf"] * 100, 
                    "Ensemble Confidence", 
                    "green"
                )
                st.plotly_chart(fig_ensemble, use_container_width=True)
                
                # ====== Model Comparison ======
                st.markdown('<div class="section-title">🧠 Individual Model Results</div>', unsafe_allow_html=True)
                
                # Model comparison chart
                comparison_fig = create_model_comparison_chart(
                    pm['ResNet18']['conf'], 
                    pm['MobileNetV2']['conf']
                )
                st.plotly_chart(comparison_fig, use_container_width=True)
                
                # Individual model cards
                col1, col2 = st.columns(2, gap="large")
                
                with col1:
                    resnet_label = pm['ResNet18']['label']
                    st.markdown(f"""
                    <div class="result-card model-card animate-fade-in">
                        <h3 style="color: white; margin-bottom: 1rem;">🏗️ ResNet18</h3>
                        <div class="metric-title">PREDICTION</div>
                        <div style="color: white; font-size: 1.3rem; font-weight: 600; margin-bottom: 1rem;">{resnet_label}</div>
                        <div class="metric-title">CONFIDENCE</div>
                        <div style="color: white; font-size: 1.5rem; font-weight: 700;">{pm['ResNet18']['conf'] * 100:.1f}%</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    mobilenet_label = pm['MobileNetV2']['label']
                    st.markdown(f"""
                    <div class="result-card model-card animate-fade-in">
                        <h3 style="color: white; margin-bottom: 1rem;">📱 MobileNetV2</h3>
                        <div class="metric-title">PREDICTION</div>
                        <div style="color: white; font-size: 1.3rem; font-weight: 600; margin-bottom: 1rem;">{mobilenet_label}</div>
                        <div class="metric-title">CONFIDENCE</div>
                        <div style="color: white; font-size: 1.5rem; font-weight: 700;">{pm['MobileNetV2']['conf'] * 100:.1f}%</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Additional insights
                if ens["conf"] > 0.8:
                    confidence_level = "Very High"
                    confidence_emoji = "🎯"
                    confidence_color = "#10B981"
                elif ens["conf"] > 0.6:
                    confidence_level = "High"
                    confidence_emoji = "✅"
                    confidence_color = "#3B82F6"
                elif ens["conf"] > 0.4:
                    confidence_level = "Moderate"
                    confidence_emoji = "⚡"
                    confidence_color = "#F59E0B"
                else:
                    confidence_level = "Low"
                    confidence_emoji = "⚠️"
                    confidence_color = "#EF4444"
                
                st.markdown(f"""
                <div class="glass-card animate-fade-in" style="text-align: center; margin-top: 2rem;">
                    <div style="color: {confidence_color}; font-size: 2rem; margin-bottom: 0.5rem;">{confidence_emoji}</div>
                    <div style="color: white; font-size: 1.2rem; font-weight: 600;">
                        Confidence Level: {confidence_level}
                    </div>
                    <div style="color: rgba(255,255,255,0.7); font-size: 0.9rem; margin-top: 0.5rem;">
                        Based on ensemble prediction of {ens['conf']*100:.1f}%
                    </div>
                </div>
                """, unsafe_allow_html=True)

else:
    # Enhanced empty state
    st.markdown("""
    <div class="glass-card animate-fade-in" style="text-align: center; margin: 3rem 0;">
        <div style="font-size: 4rem; margin-bottom: 1rem;">🐠</div>
        <div style="color: white; font-size: 1.3rem; font-weight: 600; margin-bottom: 0.5rem;">
            Ready to Identify Fish Species
        </div>
        <div style="color: rgba(255,255,255,0.7); font-size: 1rem;">
            Upload a clear image of a fish to get started with AI-powered classification
        </div>
    </div>
    """, unsafe_allow_html=True)

# Enhanced footer
st.markdown("""
<div class="glass-card animate-fade-in" style="text-align: center; margin-top: 3rem;">
    <div style="color: rgba(255,255,255,0.6); font-size: 0.9rem;">
        🤖 Powered by Advanced Deep Learning • 🔬 Marine Biology AI Research • 🌊 Ocean Conservation Technology
    </div>
</div>
""", unsafe_allow_html=True)