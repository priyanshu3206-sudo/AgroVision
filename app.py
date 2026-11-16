import hashlib
from io import BytesIO
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image, UnidentifiedImageError

from disease_info import get_disease_info
from src.predict import load_model, predict_image

st.set_page_config(
    page_title="AgroVision | Plant Intelligence",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Visual system: 3D-inspired glass / emerald interface ----------
st.markdown(
    r"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');
:root {
  --bg: #071a13; --panel: #10291e; --panel2: #153727; --line: rgba(180,230,183,.16);
  --text: #eff8ee; --muted: #a5bca9; --lime: #b8f36b; --mint: #64d89a; --deep: #0b2118;
}
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp {
  background: radial-gradient(ellipse at 78% 3%, rgba(45,135,83,.22), transparent 34%),
              radial-gradient(ellipse at 5% 42%, rgba(119,183,79,.10), transparent 30%),
              linear-gradient(135deg,#06150f 0%,#0a2117 52%,#071a13 100%);
  color: var(--text);
}
[data-testid="stHeader"] { background: rgba(5,20,14,.72); }
[data-testid="stAppViewContainer"] { background: transparent; }
[data-testid="stMain"] .block-container { max-width: 1480px; padding-top: 2rem; padding-bottom: 3rem; }
[data-testid="stSidebar"] { background: linear-gradient(180deg,#0d291d,#091b13); border-right:1px solid var(--line); }
[data-testid="stSidebar"] * { color: var(--text) !important; }
.stApp p,.stApp label,.stApp li,.stApp h1,.stApp h2,.stApp h3,.stApp h4 { color: var(--text); }
.stApp small,.stCaption { color: var(--muted) !important; }

/* Hero */
.hero-wrap { perspective: 1100px; margin: .2rem 0 1.5rem; }
.hero {
  position:relative; overflow:hidden; min-height:245px; padding:38px 42px; border-radius:28px;
  background: linear-gradient(112deg,rgba(17,67,43,.97),rgba(24,111,67,.88) 58%,rgba(62,143,78,.78));
  border:1px solid rgba(202,255,188,.24);
  box-shadow: 0 28px 65px rgba(0,0,0,.35), inset 0 1px 0 rgba(255,255,255,.16), inset 0 -18px 35px rgba(0,0,0,.10);
  transform: rotateX(1.2deg); transition: transform .35s ease, box-shadow .35s ease;
}
.hero:hover { transform: rotateX(0deg) translateY(-3px); box-shadow:0 35px 75px rgba(0,0,0,.42),inset 0 1px 0 rgba(255,255,255,.2); }
.hero:before { content:""; position:absolute; width:380px;height:380px;right:5%;top:-190px;border-radius:50%;border:1px solid rgba(221,255,198,.18);box-shadow:0 0 0 28px rgba(221,255,198,.035),0 0 0 58px rgba(221,255,198,.025); }
.hero:after { content:"✦"; position:absolute; right:12%;bottom:25px;font-size:120px;line-height:1;color:rgba(218,255,174,.14);transform:rotate(-18deg); }
.hero-kicker { color:#d0f5b1!important; font-size:11px; font-weight:800; letter-spacing:2.4px; text-transform:uppercase; }
.hero h1 { font-family:'Manrope',sans-serif; font-size:clamp(38px,5vw,62px);font-weight:800;letter-spacing:-2px;color:#fff!important;margin:12px 0 5px; }
.hero p { color:#e0f3e3!important;font-size:16px;max-width:620px;margin:0; }
.hero-chip { display:inline-block;margin-top:23px;padding:8px 13px;border-radius:999px;background:rgba(5,27,17,.28);border:1px solid rgba(226,255,218,.22);color:#e9ffe1;font-size:11px;letter-spacing:.5px; }

/* Metric tiles */
.metric-card { position:relative; overflow:hidden; min-height:126px; padding:19px 20px; border-radius:20px;
  background:linear-gradient(145deg,rgba(24,59,42,.94),rgba(12,37,26,.96)); border:1px solid var(--line);
  box-shadow:0 14px 28px rgba(0,0,0,.22),inset 0 1px 0 rgba(255,255,255,.07); transition:all .25s ease; }
.metric-card:hover { transform:translateY(-6px) rotateX(2deg);box-shadow:0 22px 35px rgba(0,0,0,.32),0 0 25px rgba(112,222,133,.08);border-color:rgba(184,243,107,.35); }
.metric-card:after { content:"";position:absolute;width:105px;height:105px;right:-35px;bottom:-48px;border-radius:50%;background:rgba(184,243,107,.08);filter:blur(1px); }
.metric-label { color:#a9c2ad;font-size:10px;font-weight:800;letter-spacing:1.6px;text-transform:uppercase; }
.metric-value { color:#f3ffed;font-family:'Manrope',sans-serif;font-size:25px;font-weight:800;margin-top:14px; }
.metric-icon { float:right;font-size:23px;filter:drop-shadow(0 4px 8px rgba(0,0,0,.25)); }

/* Section headings and panels */
.section-kicker { color:var(--lime);font-size:10px;font-weight:800;letter-spacing:2px;text-transform:uppercase;margin-bottom:5px; }
.section-title { font-family:'Manrope',sans-serif;color:#f3ffef;font-size:25px;font-weight:800;letter-spacing:-.6px;margin:0 0 4px; }
.section-sub { color:var(--muted);font-size:13px;margin-bottom:15px; }
.glass-panel { background:linear-gradient(145deg,rgba(21,53,37,.88),rgba(10,32,22,.93));border:1px solid var(--line);border-radius:22px;padding:22px;box-shadow:0 18px 35px rgba(0,0,0,.22),inset 0 1px 0 rgba(255,255,255,.06); }

/* Uploader */
[data-testid="stFileUploader"] { background:linear-gradient(145deg,rgba(20,57,39,.92),rgba(9,32,22,.96));border:1px dashed rgba(184,243,107,.48);border-radius:20px;padding:15px;box-shadow:inset 0 0 28px rgba(104,199,118,.045); }
[data-testid="stFileUploader"] section { background:rgba(184,243,107,.08)!important;border:1px solid rgba(184,243,107,.18)!important;border-radius:13px!important; }
[data-testid="stFileUploader"] section * { color:#e9f8e8!important; }
[data-testid="stFileUploaderDropzoneInstructions"] { color:#e9f8e8!important; }

/* Buttons */
.stButton>button { border-radius:13px!important;min-height:47px!important;font-weight:800!important;letter-spacing:.2px!important;transition:all .2s ease!important; }
.stButton>button[kind="primary"] { background:linear-gradient(110deg,#a7e85b,#55c982)!important;color:#082015!important;border:0!important;box-shadow:0 8px 22px rgba(83,201,130,.22),inset 0 1px 0 rgba(255,255,255,.45)!important; }
.stButton>button[kind="primary"]:hover { transform:translateY(-3px);box-shadow:0 13px 28px rgba(83,201,130,.34)!important; }
.stButton>button[kind="secondary"] { background:rgba(218,255,218,.06)!important;color:#eaffea!important;border:1px solid var(--line)!important; }

/* Results and charts */
.result-banner { padding:20px 22px;border-radius:18px;background:linear-gradient(110deg,rgba(73,156,85,.22),rgba(184,243,107,.08));border:1px solid rgba(184,243,107,.26);box-shadow:0 12px 26px rgba(0,0,0,.18); }
.result-label { color:#b9d5bb;font-size:10px;font-weight:800;letter-spacing:1.6px;text-transform:uppercase; }
.result-name { color:#f4ffef;font-family:'Manrope',sans-serif;font-size:27px;font-weight:800;margin-top:7px; }
[data-testid="stMetric"] { background:rgba(20,55,38,.75);padding:15px;border-radius:15px;border:1px solid var(--line); }
[data-testid="stMetricLabel"] p { color:#a9c2ad!important; }
[data-testid="stMetricValue"] { color:#d9ff9d!important; }
[data-testid="stProgressBar"] > div { background:rgba(190,230,180,.12); }
[data-testid="stProgressBar"] > div > div { background:linear-gradient(90deg,#66d58b,#c2f36e); }
[data-testid="stExpander"] { background:rgba(17,48,33,.72);border:1px solid var(--line);border-radius:15px; }
[data-testid="stAlert"] { border-radius:14px; }
[data-testid="stDataFrame"] { border:1px solid var(--line);border-radius:12px;overflow:hidden; }
hr { border-color:var(--line)!important; }
footer { visibility:hidden; }
@media (max-width: 800px) { .hero{padding:27px 24px;min-height:220px}.hero h1{font-size:40px}.metric-value{font-size:21px}.glass-panel{padding:16px} }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Waking up the AgroVision model...")
def get_engine():
    return load_model()


def html_metric(icon, label, value):
    st.markdown(
        f'<div class="metric-card"><span class="metric-icon">{icon}</span>'
        f'<div class="metric-label">{label}</div><div class="metric-value">{value}</div></div>',
        unsafe_allow_html=True,
    )


# ---------- Sidebar ----------
with st.sidebar:
    st.markdown("## 🌿 AgroVision")
    st.caption("PLANT INTELLIGENCE LAB")
    st.divider()
    st.markdown("### The project")
    st.write("A deep-learning prototype for tomato leaf image classification.")
    st.markdown("### Supported crop")
    st.write("🍅 Tomato")
    st.markdown("### Image formats")
    st.write("JPG · JPEG · PNG")
    st.divider()
    st.markdown("**Model**  ·  MobileNetV2")
    st.markdown("**Inference**  ·  Local / PyTorch")
    st.caption("Academic prototype • Not a confirmed diagnosis")

# ---------- Model status ----------
try:
    _, _, classes, checkpoint = get_engine()
    ready = True
    model_error = None
except Exception as exc:
    ready, classes, checkpoint = False, [], {}
    model_error = str(exc)

# ---------- Hero ----------
st.markdown(
    '<div class="hero-wrap"><div class="hero">'
    '<div class="hero-kicker">Plant intelligence · Deep learning</div>'
    '<h1>AgroVision<span style="color:#c8f47b">.</span></h1>'
    '<p>See beyond the leaf. Turn a simple tomato leaf photograph into an AI-assisted insight.</p>'
    '<span class="hero-chip">✦  MOBILENETV2 · 10 TOMATO CLASSES · LOCAL INFERENCE</span>'
    '</div></div>',
    unsafe_allow_html=True,
)

if not ready:
    st.error(f"The CNN model could not be loaded: {model_error}")

m1, m2, m3 = st.columns(3, gap="medium")
with m1:
    html_metric("🍅", "Supported crop", "Tomato")
with m2:
    html_metric("🧬", "Classification classes", str(len(classes)) if ready else "—")
with m3:
    html_metric("◉", "Model status", "Online" if ready else "Offline")

st.markdown("<div style='height:25px'></div>", unsafe_allow_html=True)
st.markdown('<div class="section-kicker">01 / Image lab</div><div class="section-title">Analyse a tomato leaf</div><div class="section-sub">Upload a clear leaf photograph. AgroVision will process it locally using the trained neural network.</div>', unsafe_allow_html=True)

upload = st.file_uploader("Drop a leaf image here or browse files", type=["jpg", "jpeg", "png"], label_visibility="visible")

if upload:
    raw = upload.getvalue()
    fingerprint = hashlib.sha256(raw).hexdigest()
    if st.session_state.get("image_fingerprint") != fingerprint:
        st.session_state.pop("prediction", None)
        st.session_state["image_fingerprint"] = fingerprint
    try:
        img = Image.open(BytesIO(raw)).convert("RGB")
        left, right = st.columns([1.05, 1], gap="large")
        with left:
            st.markdown('<div class="section-kicker">Your specimen</div>', unsafe_allow_html=True)
            st.image(img, caption=f"Uploaded image · {img.width} × {img.height}px", use_container_width=True)
        with right:
            st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
            st.markdown('<div class="section-kicker">Ready for analysis</div><div class="section-title" style="font-size:21px">Image received</div>', unsafe_allow_html=True)
            st.write("Your image is ready to be passed through the MobileNetV2 pipeline.")
            st.caption(f"Resolution: {img.width} × {img.height} pixels")
            if st.button("✦  Analyse leaf", type="primary", use_container_width=True, disabled=not ready):
                with st.spinner("Analysing leaf patterns..."):
                    try:
                        st.session_state["prediction"] = predict_image(img)
                    except Exception as exc:
                        st.error(f"Prediction failed: {exc}")
            st.markdown("</div>", unsafe_allow_html=True)
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        st.error(f"This file could not be read as an image: {exc}")

if upload and "prediction" in st.session_state:
    result = st.session_state["prediction"]
    info = get_disease_info(result["class_name"])
    st.markdown("<div style='height:22px'></div><div class='section-kicker'>02 / Neural network output</div><div class='section-title'>Your analysis</div><div class='section-sub'>The highest-scoring class predicted by the current model.</div>", unsafe_allow_html=True)
    left, right = st.columns([1.25, 1], gap="large")
    with left:
        st.markdown(
            f'<div class="result-banner"><div class="result-label">Predicted class</div>'
            f'<div class="result-name">{info["name"]}</div></div>',
            unsafe_allow_html=True,
        )
        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
        st.markdown("**About this condition**")
        st.write(info["description"])
        st.markdown("**General precautions**")
        st.write(info["precautions"])
    with right:
        st.metric("Model confidence", f'{result["confidence"] * 100:.2f}%')
        st.progress(result["confidence"])
        if result["confidence"] < 0.60:
            st.warning("The model confidence is low. Treat this prediction as uncertain.")
        st.caption("Softmax confidence is not a calibrated probability or a confirmed diagnosis.")
    with st.expander("Explore all 10 class scores", expanded=True):
        df = pd.DataFrame(result["probabilities"])
        chart_df = df.set_index("class_name")["confidence"] * 100
        st.bar_chart(chart_df, height=280)
        display_df = df.copy()
        display_df["confidence"] = (display_df["confidence"] * 100).map(lambda x: f"{x:.2f}%")
        st.dataframe(display_df.rename(columns={"class_name": "Class", "confidence": "Score"}), hide_index=True, use_container_width=True)
    st.info("Educational prototype only. Please consult a qualified agricultural professional before making crop-treatment decisions.")

with st.expander("Model evaluation & training artifacts"):
    st.caption("These are supplied project artifacts. Their metrics have not been independently reproduced by the application.")
    if ready and "test_accuracy" in checkpoint:
        st.metric("Reported test accuracy", f'{checkpoint["test_accuracy"] * 100:.2f}%')
    root = Path(__file__).resolve().parent
    for name, label in [("training_curves.png", "Training curves"), ("confusion_matrix.png", "Confusion matrix")]:
        path = root / "results" / name
        if path.exists():
            st.image(str(path), caption=label, use_container_width=True)

st.markdown("<div style='height:18px'></div><hr>", unsafe_allow_html=True)
st.markdown("<div style='text-align:center;color:#91ad98;font-size:11px;letter-spacing:.5px'>AGROVISION · TOMATO LEAF INTELLIGENCE · BUILT WITH PYTORCH & STREAMLIT</div>", unsafe_allow_html=True)