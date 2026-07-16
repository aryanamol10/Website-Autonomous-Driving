# Running/testing:
# 1. python train.py
# 2. streamlit run app.py

import sys
import subprocess

# Automated environment patch to force Miniconda to install onnxruntime internally
try:
    import onnxruntime as ort
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "onnxruntime"])
    import onnxruntime as ort

import torch
import torchvision.transforms as transforms
from torchvision import models
import torch.nn as nn
from PIL import Image
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import time
from firebase_client import list_demo_datasets, download_image, log_prediction

# ==========================================
# 1. REMEDIES FOR YOUR EXACT CATEGORIES
# ==========================================
OBSTRUCTION_REMEDIES = {
    "clean": "All clear! The camera lens is clean. Continue normal driving.",
    "fog": "Fog detected. Turn on the camera defogger and relying more on radar sensors.",
    "ice": "Ice detected. Turn on the lens heaters to melt the ice.",
    "soiling": "Soiling detected (dirt, mud, and debris). Blast fluid jets and run the wipers to clean the lens.",
    "water": "Rain or water drops detected. Run a quick wiper cycle to clear the view."
}

CLASS_NAMES = sorted(list(OBSTRUCTION_REMEDIES.keys()))

data_transforms = {
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

# ==========================================
# 2. MODEL LOADER & THE INFERENCE FUNCTIONS
# ==========================================
@st.cache_resource
def load_trained_model():
    model = models.resnet18()
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(CLASS_NAMES))

    weights_path = 'best_model_copy.pt'

    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device))
    else:
        st.sidebar.warning("Trained model weights not found. App is running in demo mode with random weights.")

    model = model.to(device)
    model.eval()
    return model

def get_ui_prediction(model, image_obj):
    model.eval()
    img = image_obj.convert('RGB')
    img = data_transforms['val'](img)
    img = img.unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img)
        probabilities = torch.softmax(outputs, dim=1)[0].cpu().numpy()
        _, preds = torch.max(outputs, 1)

    return CLASS_NAMES[preds[0].item()], probabilities

# ==========================================
# OPACITY SEVERITY ONNX LOADER & INFERENCE
# ==========================================
@st.cache_resource
def load_opacity_model():
    model_path = 'soiling_severity_model.onnx'
    if os.path.exists(model_path):
        try:
            session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
            return session
        except Exception as e:
            st.sidebar.warning(f"Failed to load ONNX opacity model: {e}")
            return None
    return None

def get_opacity_prediction(session, pil_image):
    if session is None:
        return None, None

    try:
        # 1. Standardize dimensions
        img = pil_image.convert('RGB').resize((224, 224))
        arr = np.array(img).astype('float32') / 255.0
        arr = np.expand_dims(arr, 0)

        # 2. Dynamic shape alignment check
        input_type = session.get_inputs()[0]
        input_name = input_type.name
        input_shape = input_type.shape

        if len(input_shape) == 4 and (input_shape[1] == 3 or input_shape[3] != 3):
            arr = np.transpose(arr, (0, 3, 1, 2))

        # 3. Run inference session
        preds = session.run(None, {input_name: arr})[0]
        preds = np.array(preds).flatten()

        # 4. Apply manual Softmax to convert raw outputs into clear 0% - 100% confidences
        exp_preds = np.exp(preds - np.max(preds))
        probabilities = exp_preds / np.sum(exp_preds)

        # --- DYNAMIC CALIBRATION: ADVANCED HEURISTICS (BLUR & EXPOSURE) ---
        gray_arr = np.array(pil_image.convert('L'), dtype=np.float32)

        # 1. Measure Blockage via Extreme Darkness/Brightness
        mean_intensity = np.mean(gray_arr)
        exposure_extremity = abs(mean_intensity - 127.5) / 127.5

        # 2. Measure Blurriness via Edge Gradient (Sharpness)
        gy, gx = np.gradient(gray_arr)
        edge_magnitude = np.sqrt(gx**2 + gy**2)
        sharpness = np.mean(edge_magnitude)

        # A clear road has sharp edges (sharpness > 15). Blocked lenses are blurry (sharpness < 5).
        blur_factor = max(0.0, min((15.0 - sharpness) / 15.0, 1.0))

        # 3. Master Degradation Score
        degradation_score = max(blur_factor, exposure_extremity)

        # 4. Dynamic Penalty Execution
        dynamic_severe_penalty = 0.35 * (1.0 - degradation_score)

        probabilities[3] -= dynamic_severe_penalty
        probabilities[2] -= (dynamic_severe_penalty * 0.3) - 0.10
        probabilities[1] += (dynamic_severe_penalty * 1.3)

        # Enforce mathematical boundaries [0.0, 1.0]
        probabilities = np.clip(probabilities, 0.0, 1.0)

        # Re-normalize to ensure the array perfectly sums to 100%
        sum_probs = np.sum(probabilities)
        if sum_probs > 0:
            probabilities = probabilities / sum_probs
        else:
            probabilities = np.array([1.0, 0.0, 0.0, 0.0])

    except Exception:
        return None, None

    idx = int(np.argmax(probabilities))

    # Standard baseline mapping dictionary
    mapping = {
        0: "Clean / None",
        1: "Mild",
        2: "Moderate",
        3: "Severe"
    }
    return mapping.get(idx, f"Class {idx}"), probabilities

# ==========================================
# 3. STREAMLIT PAGE SETUP & SHARED STYLES
# ==========================================
st.set_page_config(page_title="Adaptive Perception Control Unit", layout="wide")

st.markdown(
    """
    <style>
    /* Keyframe Animations */
    @keyframes pulse-highlight {
        0% { box-shadow: 0 0 0 0 rgba(255, 75, 75, 0.3); }
        70% { box-shadow: 0 0 0 8px rgba(255, 75, 75, 0); }
        100% { box-shadow: 0 0 0 0 rgba(255, 75, 75, 0); }
    }
    @keyframes fade-in {
        from { opacity: 0; transform: translateY(8px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .block-container {
        padding-top: 3rem;
        padding-bottom: 1rem;
        padding-left: 2.5rem;
        padding-right: 2.5rem;
        max-width: 100%;
        animation: fade-in 0.5s ease-out;
    }
    div[data-testid="stExpander"] details summary p {
        font-size: 1.05rem !important;
        font-weight: 600 !important;
    }
    .app-header {
        font-size: 1.9rem;
        font-weight: 700;
        letter-spacing: -0.01em;
        margin-bottom: 0.15rem;
        text-align: center;
    }
    .app-subheader {
        font-size: 1rem;
        color: rgba(120, 120, 120, 0.95);
        margin-bottom: 1.4rem;
        text-align: center;
    }
    .section-label {
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: rgba(130, 130, 130, 0.9);
        margin-bottom: 0.6rem;
        margin-top: 0.2rem;
        text-align: center;
    }
    
    /* Interactive Demo Card Hover Effects */
    .demo-card {
        border-radius: 8px;
        transition: transform 0.25s cubic-bezier(0.25, 0.8, 0.25, 1), box-shadow 0.25s ease-in-out;
    }
    .demo-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
    }
    .demo-card p {
        font-size: 0.75rem;
        font-weight: 500;
        text-align: center;
        margin: 0.3rem 0 0.3rem 0;
    }
    
    /* Interactive custom Streamlit buttons (Test triggers) */
    div[data-testid="stButton"] button {
        transition: all 0.2s cubic-bezier(0.25, 0.8, 0.25, 1) !important;
    }
    div[data-testid="stButton"] button:hover {
        transform: translateY(-2px);
        border-color: #ff4b4b !important;
        color: #ff4b4b !important;
        box-shadow: 0 4px 10px rgba(255, 75, 75, 0.1);
    }
    div[data-testid="stButton"] button:active {
        transform: translateY(0);
    }
    
    /* References Card styling */
    .reference-card {
        border: 1px solid rgba(130, 130, 130, 0.2);
        border-radius: 0.5rem;
        padding: 0.9rem 1.1rem;
        margin-bottom: 0.8rem;
        transition: all 0.25s cubic-bezier(0.25, 0.8, 0.25, 1);
    }
    .reference-card:hover {
        border-color: #ff4b4b;
        transform: translateY(-2px);
        box-shadow: 0 6px 14px rgba(255, 75, 75, 0.04);
        background-color: rgba(255, 75, 75, 0.005);
    }
    .reference-card .ref-title {
        font-weight: 600;
        font-size: 0.98rem;
        margin-bottom: 0.2rem;
    }
    .reference-card .ref-desc {
        font-size: 0.88rem;
        color: rgba(120, 120, 120, 0.95);
        margin-bottom: 0.3rem;
    }

    /* Metric Widget Interactivity */
    div[data-testid="stMetricValue"] {
        font-size: 1.1rem !important; 
    }
    div[data-testid="stMetric"] {
        padding: 6px 10px 0.5rem 10px !important;
        background-color: rgba(128, 128, 128, 0.03);
        border-radius: 6px;
        border: 1px solid rgba(128, 128, 128, 0.08);
        transition: all 0.25s ease-in-out;
    }
    div[data-testid="stMetric"]:hover {
        border-color: rgba(255, 75, 75, 0.3);
        background-color: rgba(255, 75, 75, 0.005);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.02);
    }
    div[data-testid="stVerticalBlockBorderWrapper"] {
        height: 100%;
    }

    /* Active Recommended Action alerts with glow pulses */
    .stAlert {
        border-radius: 8px !important;
        transition: all 0.3s ease-in-out;
    }
    div[data-testid="stNotification"] > div:has(.st-ae) {
        animation: pulse-highlight 2.5s infinite;
    }

    /* Laptop / tablet sizing */
    @media (max-width: 1000px) {
        .block-container {
            padding-left: 1.5rem;
            padding-right: 1.5rem;
        }
    }

    /* Mobile adaptations */
    @media (max-width: 640px) {
        .block-container {
            padding-top: 4rem;
            padding-left: 1rem;
            padding-right: 1rem;
        }
        .app-header { font-size: 1.5rem; }
        .app-subheader { font-size: 0.9rem; margin-bottom: 1rem; }
        .section-label { font-size: 0.75rem; margin-top: 0.9rem; }
        div[data-testid="stMetricValue"] { font-size: 1rem !important; }
    }

    /* ==================================================
       TOP NAVIGATION BAR EFFECTS (HOVER UP & SHIFT COLOR)
       ================================================== */
    .stAppHeader { 
        min-height: 4.2rem; 
        height: auto !important; 
        border-bottom: 1px solid rgba(128, 128, 128, 0.05);
    }
    .stAppHeader .rc-overflow { 
        justify-content: center !important; 
        width: 100%; 
    }
    .stAppHeader nav, .stAppHeader ul { 
        justify-content: center !important; 
    }
    .stAppHeader li { 
        padding-left: 0.75rem; 
        padding-right: 0.75rem; 
        transition: transform 0.25s cubic-bezier(0.25, 0.8, 0.25, 1) !important;
    }
    /* Move whole list element up on hover */
    .stAppHeader li:hover {
        transform: translateY(-3px);
    }
    .stAppHeader span { 
        font-size: 1.05rem !important; 
        font-weight: 600 !important; 
        transition: color 0.2s ease-in-out, border-color 0.2s ease-in-out !important;
    }
    /* Target navigation button interactions directly */
    .stAppHeader button {
        border-bottom: 2px solid transparent !important;
        transition: all 0.25s ease-in-out !important;
    }
    .stAppHeader button:hover {
        background-color: transparent !important;
    }
    .stAppHeader button:hover span {
        color: #ff4b4b !important;
    }
    /* Highlight currently active page in navigation */
    .stAppHeader button[aria-selected="true"] span {
        color: #ff4b4b !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<div class="app-header">Adaptive Perception Control Unit</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="app-subheader">Detects camera sensor obstruction and recommends a mitigation step.</div>',
    unsafe_allow_html=True
)

# ==========================================
# 4. PAGE: MODEL
# ==========================================
def render_model_page():
    if 'active_demo_path' not in st.session_state:
        st.session_state.active_demo_path = None
    if 'use_demo' not in st.session_state:
        st.session_state.use_demo = False
    if 'camera_active' not in st.session_state:
        st.session_state.camera_active = False

    demo_files = list_demo_datasets()  # [(label, storage_path), ...] pulled from Firebase Storage

    input_col, results_col = st.columns([0.45, 0.55], gap="large")

    with input_col:
        st.markdown('<div class="section-label">Sample datasets</div>', unsafe_allow_html=True)

        if not demo_files:
            st.caption(
                "No datasets found in Firebase Storage yet. Run upload_demos.py, "
                "or check your st.secrets Firebase configuration."
            )
        else:
            demo_cols = st.columns(min(4, len(demo_files)))

            for idx, (label, storage_path) in enumerate(demo_files):
                with demo_cols[idx % len(demo_cols)]:
                    img_thumb = download_image(storage_path)
                    if img_thumb is not None:
                        st.markdown('<div class="demo-card">', unsafe_allow_html=True)
                        st.image(img_thumb.resize((100, 75)), width='stretch')
                        st.markdown(f"<p>{label}</p>", unsafe_allow_html=True)
                        st.markdown('</div>', unsafe_allow_html=True)

                        # Clicking a dataset auto-feeds it into the model below.
                        if st.button("Test", key=f"demo_btn_{idx}", use_container_width=True):
                            st.session_state.active_demo_path = storage_path
                            st.session_state.use_demo = True
                            st.session_state.camera_active = False
                    else:
                        st.caption(f"Could not load: {label}")    

        st.markdown("<div style='margin-top: 0.5rem;'></div>", unsafe_allow_html=True)
        
        st.markdown('<div class="section-label">Or upload your own frame</div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Upload image (JPG/PNG)", 
            label_visibility="collapsed",
            type=["jpg", "jpeg", "png"]
        )
        if uploaded_file is not None:
            st.session_state.use_demo = False
            st.session_state.camera_active = False

        st.markdown("<div style='margin-top: 0.5rem;'></div>", unsafe_allow_html=True)
        st.markdown('<div class="section-label">Or Live Video Feed</div>', unsafe_allow_html=True)
        
        # Interactive feed toggle
        if not st.session_state.camera_active:
            if st.button("📷 Open Camera Feed", use_container_width=True):
                st.session_state.camera_active = True
                st.session_state.use_demo = False
                st.rerun()
        else:
            if st.button("❌ Close Camera Feed", use_container_width=True):
                st.session_state.camera_active = False
                st.rerun()

        camera_file = None
        if st.session_state.camera_active:
            camera_file = st.camera_input("Capture frame from built-in camera", label_visibility="collapsed")
        else:
            st.info("Live camera is currently offline. Click 'Open Camera Feed' to launch camera hardware.")

    image_to_process = None
    if st.session_state.camera_active and camera_file is not None:
        image_to_process = Image.open(camera_file)
    elif uploaded_file is not None:
        image_to_process = Image.open(uploaded_file)
    elif st.session_state.use_demo and st.session_state.active_demo_path:
        image_to_process = download_image(st.session_state.active_demo_path)

    with results_col:
        if image_to_process is not None:
            preview_col, metrics_col = st.columns([0.35, 0.65], gap="small")

            with preview_col:
                st.image(image_to_process, caption="Current feed", width='stretch')

            with st.spinner("Analyzing frame..."):
                model = load_trained_model()
                prediction_text, prediction_probs = get_ui_prediction(model, image_to_process)
                prediction_confidence = float(np.max(prediction_probs)) * 100

                opacity_session = load_opacity_model()
                severity_label, raw_probs = get_opacity_prediction(opacity_session, image_to_process)

            with metrics_col:
                st.markdown('<div class="section-label">Analysis Results</div>', unsafe_allow_html=True)

                m1, m2, m3 = st.columns(3)
                with m1: 
                    st.metric(label="Obstruction", value=prediction_text.upper())
                with m2: 
                    st.metric(label="Confidence", value=f"{prediction_confidence:.1f}%")
                with m3:
                    if prediction_text == "clean":
                        st.metric(label="Severity", value="N/A")
                    elif severity_label:
                        st.metric(label="Severity", value=str(severity_label).upper())
                    else:
                        st.metric(label="Severity", value="UNAVAILABLE")

                base_action = OBSTRUCTION_REMEDIES.get(prediction_text, "Remedy protocol mismatch.")
                source = (
                    "camera" if st.session_state.camera_active
                    else "upload" if uploaded_file is not None
                    else st.session_state.active_demo_path or "unknown"
                )
                log_prediction(prediction_text, prediction_confidence, severity_label, source)
                if prediction_text == "clean":
                    st.success(f"**System check:** {base_action}")
                elif severity_label == "Severe":
                    st.error(f"**Urgent:** Execute to prevent sensor blinding. {base_action}")
                elif severity_label == "Moderate":
                    st.warning(f"**Moderate priority:** Degradation confirmed. {base_action}")
                elif severity_label == "Mild":
                    st.info(f"**Low priority:** Minor obstruction tracked. {base_action}")
                else:
                    st.warning(f"**Action:** {base_action}")

            st.markdown("<hr style='margin: 0.5rem 0;'>", unsafe_allow_html=True)
            
            # Vertically stacked probabilities sections without truncation
            st.markdown('<div class="section-label">Obstruction Probabilities</div>', unsafe_allow_html=True)
            c1, c2, c3, c4, c5 = st.columns(5)
            for i, class_name in enumerate(CLASS_NAMES):
                col = [c1, c2, c3, c4, c5][i]
                with col: st.metric(class_name.capitalize(), f"{prediction_probs[i]*100:.0f}%")

            if prediction_text != "clean":
                st.markdown('<div class="section-label" style="margin-top: 0.5rem;">Severity Probabilities</div>', unsafe_allow_html=True)
                if raw_probs is not None:
                    s0, s1, s2, s3 = st.columns(4)
                    with s0: st.metric("Clean", f"{raw_probs[0]*100:.0f}%")
                    with s1: st.metric("Mild", f"{raw_probs[1]*100:.0f}%")
                    with s2: st.metric("Moderate", f"{raw_probs[2]*100:.0f}%")
                    with s3: st.metric("Severe", f"{raw_probs[3]*100:.0f}%")
                else:
                    st.caption("Metrics unavailable.")
        else:
            st.info("Select a sample frame, upload your own, or open camera and capture feed to run analysis.")


# ==========================================
# 5. SHARED: MODEL PERFORMANCE GRAPHS
# ==========================================
def render_performance_graphs():
    try:
        metrics_file = 'metrics_copy.json'

        with open(metrics_file, 'r') as f:
            metrics = json.load(f)

        if 'test_acc' in metrics:
            st.success(f"**Final unseen test set accuracy:** {metrics['test_acc']*100:.2f}%")
            st.caption("This metric reflects generalization on completely unseen, held-out data.")
            st.markdown("---")

        st.markdown("<h4 style='text-align: center;'>Training history</h4>", unsafe_allow_html=True)

        epochs = np.arange(1, len(metrics['train_loss']) + 1)
        train_loss = metrics['train_loss']
        val_loss = metrics['val_loss']
        train_acc = metrics['train_acc']
        val_acc = metrics['val_acc']

        fig, ax = plt.subplots(1, 2, figsize=(7, 2.8))

        ax[0].plot(epochs, train_loss, label='Training Loss', color='#1f77b4', linewidth=1.5)
        ax[0].plot(epochs, val_loss, label='Validation Loss', color='#ff7f0e', linestyle='--', linewidth=1.5)
        ax[0].set_title('Loss Convergence', fontsize=10, fontweight='bold')
        ax[0].set_xlabel('Epochs', fontsize=9)
        ax[0].set_ylabel('Loss Metric', fontsize=9)
        ax[0].tick_params(axis='both', which='major', labelsize=8)
        ax[0].grid(True, linestyle=':', alpha=0.6)
        ax[0].legend(fontsize=8)

        ax[1].plot(epochs, train_acc, label='Training Accuracy', color='#2ca02c', linewidth=1.5)
        ax[1].plot(epochs, val_acc, label='Validation Accuracy', color='#d62728', linestyle='--', linewidth=1.5)
        ax[1].set_title('Accuracy Growth', fontsize=10, fontweight='bold')
        ax[1].set_xlabel('Epochs', fontsize=9)
        ax[1].set_ylabel('Accuracy', fontsize=9)
        ax[1].tick_params(axis='both', which='major', labelsize=8)
        ax[1].grid(True, linestyle=':', alpha=0.6)
        ax[1].legend(fontsize=8)

        fig.tight_layout()
        st.pyplot(fig)

        st.markdown("---")

        st.markdown("<h4 style='text-align: center;'>Confusion matrices</h4>", unsafe_allow_html=True)
        st.caption("Select a dataset split to evaluate prediction layouts and spot classification bias.")

        split_sizes = {}
        for split_label, split_key in [
            ("Training Set", "train_confusion_matrix"),
            ("Validation Set", "val_confusion_matrix"),
            ("Unseen Test Set", "test_confusion_matrix"),
        ]:
            if split_key in metrics:
                split_sizes[split_label] = int(np.array(metrics[split_key]).sum())

        if split_sizes:
            size_summary = " · ".join(f"{label}: {count:,} samples" for label, count in split_sizes.items())
            st.caption(size_summary)

        matrix_tab = st.selectbox(
            "Dataset split",
            ["Unseen Test Set", "Validation Set", "Training Set"]
        )

        matrix_mapping = {
            "Unseen Test Set": "test_confusion_matrix",
            "Validation Set": "val_confusion_matrix",
            "Training Set": "train_confusion_matrix"
        }

        target_key = matrix_mapping[matrix_tab]

        # Added dynamic loader wrapper context block
        with st.spinner(f"Generating Confusion Matrix & Metrics..."):
            # Simple UI feel-good load delay to ensure matrix visualization renders smoothly
            time.sleep(0.4) 
            
            if target_key in metrics:
                cm = np.array(metrics[target_key])
                
                col1, col2, col3 = st.columns([0.2, 0.6, 0.2])
                with col2:
                    st.markdown(f"**{matrix_tab}** — {int(cm.sum()):,} total samples")

                    fig2, ax2 = plt.subplots(figsize=(4.5, 3.5))
                    cax = ax2.imshow(cm, cmap='Blues', interpolation='nearest')
                    fig2.colorbar(cax, fraction=0.046, pad=0.04)

                    num_classes = len(CLASS_NAMES)
                    ax2.set_xticks(np.arange(num_classes))
                    ax2.set_yticks(np.arange(num_classes))
                    ax2.set_xticklabels(CLASS_NAMES, rotation=45, ha='right', fontsize=8)
                    ax2.set_yticklabels(CLASS_NAMES, fontsize=8)
                    ax2.set_xlabel('Predicted Label', fontsize=9, fontweight='bold')
                    ax2.set_ylabel('True Label', fontsize=9, fontweight='bold')

                    threshold = np.max(cm) / 2
                    for i in range(num_classes):
                        for j in range(num_classes):
                            ax2.text(j, i, int(cm[i, j]), ha="center", va="center",
                                    color="white" if cm[i, j] > threshold else "black", fontsize=8)

                    fig2.tight_layout()
                    st.pyplot(fig2)

                st.markdown("<h5 style='text-align: center; margin-top: 1rem;'>Per-class F1 score</h5>", unsafe_allow_html=True)
                
                f1_cols = st.columns(len(CLASS_NAMES))

                for i, class_name in enumerate(CLASS_NAMES):
                    tp = cm[i, i]
                    fp = np.sum(cm[:, i]) - tp
                    fn = np.sum(cm[i, :]) - tp

                    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

                    with f1_cols[i]:
                        st.metric(label=f"{class_name.upper()} F1", value=f"{f1_score*100:.1f}%")

            else:
                st.warning(f"Matrix data for '{matrix_tab}' not found in file. Please re-run train.py completely.")

    except FileNotFoundError:
        st.warning("Performance metrics file not found. Run your training script to generate analytical assets.")


# ==========================================
# 6. PAGE: REFERENCES
# ==========================================
def render_references_page():
    st.markdown('<div class="section-label">Data sources and references</div>', unsafe_allow_html=True)

    references = [
        {
            "title": "Deep Residual Learning for Image Recognition",
            "desc": "Seminal paper introducing ResNet architectures, resolving vanishing gradients with skip connections.",
            "url": "https://arxiv.org/abs/1512.03385"
        },
        {
            "title": "WoodScape Dataset",
            "desc": "Fisheye surround-view dataset for autonomous driving used for soiling and obstruction research.",
            "url": "https://woodscape.valeo.com/woodscape/"
        },
        {
            "title": "WoodScape GitHub Repository",
            "desc": "Official code and annotations release for the WoodScape dataset.",
            "url": "https://github.com/valeoai/woodscape"
        },
        {
            "title": "100K Vehicle Dashcam Image Dataset (Kaggle)",
            "desc": "Dashcam image dataset used for additional training and validation samples.",
            "url": "https://www.kaggle.com/datasets/mdfahimbinamin/100k-vehicle-dashcam-image-dataset/data?select=train"
        }
    ]

    for ref in references:
        st.markdown(
            f"""
            <div class="reference-card">
                <div class="ref-title">{ref['title']}</div>
                <div class="ref-desc">{ref['desc']}</div>
                <a href="{ref['url']}" target="_blank">{ref['url']}</a>
            </div>
            """,
            unsafe_allow_html=True
        )


# ==========================================
# 7. PAGE: PAPER
# ==========================================
def render_paper_page():
    st.markdown('<div class="section-label">Abstract</div>', unsafe_allow_html=True)
    st.info("The abstract will be added here once it is written.")
    st.caption("Placeholder — replace with the final abstract text when ready.")

    st.markdown('<div class="section-label">Paper</div>', unsafe_allow_html=True)
    st.info("The full write-up for this project will be linked here once it is published.")
    st.caption("Placeholder — replace with the paper title, authors, and a link or embedded PDF when ready.")

    st.divider()

    st.markdown('<div class="section-label">Supporting results</div>', unsafe_allow_html=True)
    st.caption("Performance graphs referenced in the write-up.")

    render_performance_graphs()


# ==========================================
# 8. NAVIGATION (Reordered Model -> Paper -> References)
# ==========================================
pages = st.navigation([
    st.Page(render_model_page, title="Model", url_path="model", default=True),
    st.Page(render_paper_page, title="Paper", url_path="paper"),
    st.Page(render_references_page, title="References", url_path="references"),
], position="top")
pages.run()
