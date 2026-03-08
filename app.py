import sys
import os
import zipfile
import io
import time
import streamlit as st
import cv2
import numpy as np
import torchvision.transforms.functional as T_F
from types import ModuleType
from PIL import Image

# --- Fix for torchvision dependency ---
if "torchvision.transforms.functional_tensor" not in sys.modules:
    fake_module = ModuleType("torchvision.transforms.functional_tensor")
    fake_module.rgb_to_grayscale = T_F.rgb_to_grayscale
    sys.modules["torchvision.transforms.functional_tensor"] = fake_module

from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Medo's Image Upscaler",
    page_icon="⚡",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────
#  CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@500;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
    background-color: #0d0d12;
    color: #f0f0f5;
}

.stApp { background: #0d0d12; }

#MainMenu, footer, header { visibility: hidden; }
.block-container {
    padding-top: 2.5rem !important;
    max-width: 760px;
}

/* ── TITLE ── */
.page-title {
    font-size: 2.8rem;
    font-weight: 800;
    background: linear-gradient(120deg, #ffffff 30%, #5effa0 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 0.3rem;
    line-height: 1.1;
}

.page-sub {
    font-size: 1.05rem;
    color: #9999bb;
    margin: 0 0 2.2rem;
    font-weight: 500;
}

/* ── UPLOAD AREA ── */
[data-testid="stFileUploader"] {
    border: 2px dashed #2a2a40 !important;
    border-radius: 14px !important;
    background: #13131c !important;
    padding: 0.5rem !important;
}

[data-testid="stFileUploader"] section {
    border: none !important;
    background: transparent !important;
}

[data-testid="stFileUploader"] p,
[data-testid="stFileUploader"] span,
[data-testid="stFileUploader"] small {
    font-size: 0.95rem !important;
    color: #9999bb !important;
}

/* ── BUTTONS ── */
.stButton > button {
    font-family: 'Space Mono', monospace;
    font-size: 0.95rem;
    font-weight: 700;
    letter-spacing: 0.06em;
    background: linear-gradient(120deg, #5effa0, #00d4ff);
    color: #060910;
    border: none;
    border-radius: 12px;
    padding: 0.8rem 2rem;
    width: 100%;
    transition: all 0.18s;
    box-shadow: 0 0 28px #5effa025;
    margin-top: 0.5rem;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 0 42px #5effa045;
}

.stButton > button:disabled {
    background: #22223a !important;
    color: #44445a !important;
    box-shadow: none;
    transform: none;
}

/* ── DOWNLOAD BUTTON ── */
[data-testid="stDownloadButton"] > button {
    font-family: 'Space Mono', monospace;
    font-size: 0.9rem;
    font-weight: 700;
    background: transparent;
    color: #5effa0;
    border: 2px solid #5effa060;
    border-radius: 12px;
    padding: 0.8rem 2rem;
    width: 100%;
    transition: all 0.18s;
    letter-spacing: 0.05em;
    margin-top: 0.4rem;
}

[data-testid="stDownloadButton"] > button:hover {
    background: #5effa012;
    border-color: #5effa0;
    box-shadow: 0 0 24px #5effa020;
}

/* ── PROGRESS ── */
.prog-wrap {
    background: #16161f;
    border: 1px solid #23233a;
    border-radius: 14px;
    padding: 1.4rem 1.6rem 1.2rem;
    margin: 1.4rem 0;
}

.prog-top {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    margin-bottom: 0.8rem;
}

.prog-label {
    font-size: 1rem;
    font-weight: 700;
    color: #aaaacc;
}

.prog-pct {
    font-family: 'Space Mono', monospace;
    font-size: 2.2rem;
    font-weight: 700;
    color: #ffffff;
    line-height: 1;
}

.prog-file {
    font-family: 'Space Mono', monospace;
    font-size: 0.85rem;
    color: #5effa0;
    margin-top: 0.6rem;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}

.stProgress > div > div > div {
    background: linear-gradient(90deg, #5effa0, #00d4ff) !important;
    border-radius: 100px;
    height: 8px !important;
}

.stProgress > div > div {
    background: #23233a !important;
    border-radius: 100px;
    height: 8px !important;
}

/* ── SUCCESS / ERROR ── */
.success-box {
    background: #1a2a1e;
    border: 1px solid #5effa050;
    border-radius: 12px;
    padding: 1rem 1.4rem;
    font-size: 1.05rem;
    color: #5effa0;
    font-weight: 600;
    margin-bottom: 0.8rem;
}

.error-box {
    background: #2a1a1a;
    border: 1px solid #ff6b6b50;
    border-radius: 12px;
    padding: 0.9rem 1.2rem;
    font-size: 0.95rem;
    color: #ffaaaa;
    margin-bottom: 0.8rem;
}

/* ── BEFORE/AFTER ── */
.ba-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.78rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    font-weight: 700;
    padding: 0.3rem 0.75rem;
    border-radius: 6px;
    display: inline-block;
    margin-bottom: 0.5rem;
}

.ba-orig { background: #23233a; color: #ccccee; }
.ba-up   { background: #1a2a1e; color: #5effa0; }

.ba-meta {
    font-family: 'Space Mono', monospace;
    font-size: 0.8rem;
    color: #6666888;
    margin-top: 0.35rem;
}

/* ── RADIO ── */
[data-testid="stRadio"] label {
    font-size: 1rem !important;
    color: #ddddee !important;
    font-weight: 600 !important;
}

/* ── DIVIDER ── */
hr { border-color: #1e1e30 !important; margin: 1.8rem 0 !important; }

/* ── FOOTER ── */
.footer {
    text-align: center;
    padding: 2.5rem 0 1rem;
    font-size: 0.85rem;
    color: #44445a;
}
.footer a {
    color: #5effa0;
    text-decoration: none;
    font-weight: 700;
}
.footer a:hover { text-decoration: underline; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  MODEL
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_upsampler():
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    return RealESRGANer(
        scale=4,
        model_path='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth',
        model=model,
        tile=400,
        half=False
    )


# ─────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────
def bytes_to_cv2(data):
    arr = np.frombuffer(data, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)

def cv2_to_pil(img):
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

def fmt_px(w, h):
    return f"{w} × {h} px"


# ─────────────────────────────────────────────
#  HEADER
# ─────────────────────────────────────────────
st.markdown('<div class="page-title">⚡ Medo\'s Image Upscaler</div>', unsafe_allow_html=True)
st.markdown('<div class="page-sub">AI-powered 4× super-resolution &nbsp;·&nbsp; by <a href="https://github.com/medoyea" target="_blank" style="color:#5effa0;text-decoration:none;font-weight:700;">Mohamed Hassanein</a></div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  MODE SELECTOR
# ─────────────────────────────────────────────
mode = st.radio(
    "Mode",
    options=["Single Image", "Batch (multiple images)"],
    horizontal=True,
    label_visibility="collapsed"
)

st.markdown("<br>", unsafe_allow_html=True)
is_batch = mode == "Batch (multiple images)"

# ─────────────────────────────────────────────
#  FILE UPLOADER
# ─────────────────────────────────────────────
if is_batch:
    uploaded_files = st.file_uploader(
        "Upload images (JPG, PNG)",
        type=['jpg', 'jpeg', 'png'],
        accept_multiple_files=True,
    )
else:
    single = st.file_uploader(
        "Upload an image (JPG, PNG)",
        type=['jpg', 'jpeg', 'png'],
        accept_multiple_files=False,
    )
    uploaded_files = [single] if single else []

# ─────────────────────────────────────────────
#  START BUTTON
# ─────────────────────────────────────────────
start = st.button("⚡  Start Upscaling", disabled=not bool(uploaded_files))

# ─────────────────────────────────────────────
#  PROCESSING
# ─────────────────────────────────────────────
if uploaded_files and start:
    with st.spinner("Loading model…"):
        upsampler = load_upsampler()

    # Progress UI
    prog_container = st.container()
    with prog_container:
        st.markdown('<div class="prog-wrap"><div class="prog-top"><span class="prog-label">Processing</span></div></div>', unsafe_allow_html=True)
        pct_slot  = st.empty()
        bar       = st.progress(0)
        file_slot = st.empty()

    results = []
    errors  = []
    t0 = time.time()
    n  = len(uploaded_files)

    zip_buffer = io.BytesIO()

    with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zf:
        for i, f in enumerate(uploaded_files):
            pct = int(i / n * 100)
            pct_slot.markdown(f'<div class="prog-pct">{pct}%</div>', unsafe_allow_html=True)
            bar.progress(i / n)
            file_slot.markdown(f'<div class="prog-file">→ {f.name} &nbsp;({i+1} of {n})</div>', unsafe_allow_html=True)

            raw = f.read()
            img = bytes_to_cv2(raw)
            if img is None:
                errors.append(f.name)
                continue

            try:
                output, _ = upsampler.enhance(img, outscale=4)
                _, enc = cv2.imencode(".png", output)
                zf.writestr(f"upscaled_{f.name}", enc.tobytes())
                results.append((f.name, img, output))
            except Exception as e:
                errors.append(f"{f.name} ({e})")

    elapsed = time.time() - t0
    pct_slot.markdown('<div class="prog-pct">100%</div>', unsafe_allow_html=True)
    bar.progress(1.0)
    file_slot.empty()

    # ── RESULT ──
    st.markdown(
        f'<div class="success-box"> {len(results)} image{"s" if len(results)!=1 else ""} upscaled in {elapsed:.1f}s</div>',
        unsafe_allow_html=True
    )

    if errors:
        st.markdown(
            f'<div class="error-box"> Failed: {", ".join(errors)}</div>',
            unsafe_allow_html=True
        )

    # ── DOWNLOAD ──
    if not is_batch and results:
        name, _, upscaled = results[0]
        _, enc = cv2.imencode(".png", upscaled)
        st.download_button(
            label="⬇  Download Upscaled Image",
            data=enc.tobytes(),
            file_name=f"upscaled_{name.rsplit('.', 1)[0]}.png",
            mime="image/png"
        )
    elif results:
        st.download_button(
            label="⬇  Download All as ZIP",
            data=zip_buffer.getvalue(),
            file_name="pixelforge_upscaled.zip",
            mime="application/zip"
        )

    # ── BEFORE / AFTER ──
    if results:
        st.markdown("<hr>", unsafe_allow_html=True)

        preview_count = 1 if not is_batch else min(len(results), 4)
        for name, orig, upscaled in results[:preview_count]:
            oh, ow = orig.shape[:2]
            uh, uw = upscaled.shape[:2]

            c1, c2 = st.columns(2, gap="medium")
            with c1:
                st.markdown('<span class="ba-label ba-orig">Original</span>', unsafe_allow_html=True)
                st.image(cv2_to_pil(orig), use_container_width=True)
                st.markdown(f'<div class="ba-meta">{fmt_px(ow, oh)}</div>', unsafe_allow_html=True)
            with c2:
                st.markdown('<span class="ba-label ba-up">Upscaled ×4</span>', unsafe_allow_html=True)
                st.image(cv2_to_pil(upscaled), use_container_width=True)
                st.markdown(f'<div class="ba-meta">{fmt_px(uw, uh)}</div>', unsafe_allow_html=True)

        if is_batch and len(results) > 4:
            st.markdown(
                f'<div style="text-align:center;color:#6666888;font-size:0.9rem;margin-top:0.6rem">'
                f'+ {len(results) - 4} more in the ZIP</div>',
                unsafe_allow_html=True
            )

# ── FOOTER ──
st.markdown(
    '<div class="footer">Built by <a href="https://github.com/medoyea" target="_blank">Mohamed Hassanein</a> · Powered by Real-ESRGAN</div>',
    unsafe_allow_html=True
)