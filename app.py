# app.py — minimal, with optional HF download + auto device
import os
import streamlit as st
from translator import load_translator

# Optional: pull from Hugging Face if local folders are missing
try:
    from huggingface_hub import snapshot_download
except Exception:
    snapshot_download = None

st.set_page_config(page_title="CN→EN Translator", page_icon="🌐", layout="centered")
st.title("CN → EN Translator")

# Where to look locally first
RUN_DIR = os.environ.get("RUN_DIR", ".")  # e.g. "." or an absolute path
HF_REPO = os.environ.get("HF_REPO", "")   # e.g. "ncp93/ai" (set in Cloud)
HF_TOKEN = os.environ.get("HF_TOKEN", "") # set if the repo is private

# Choose backend only (no decoding knobs exposed)
model_type = st.selectbox("Choose Model", ("SMT", "NMT", "Hybrid"), index=1)

FOLDER_MAP = {
    "SMT":    "smt/run",     # best_model/ibm1_s2t.json  (or phrase-based if present)
    "NMT":    "nmt/run",     # best_model/{config.json, weights, tokenizer}
    "Hybrid": "hybrid/run",  # tokenizer/bpe_joint.model, nmt_best.pth, ibm1.pkl
}
PREFER_MAP = {"SMT": "smt", "NMT": "marian", "Hybrid": "hybrid"}

# Fixed decoding settings
FIXED_NUM_BEAMS = 5
FIXED_MAX_NEW   = 128
FIXED_NGRAM     = 3
FIXED_LEN_PEN   = 1.0

# Auto device: use CUDA if available (Marian/Hybrid honor this)
try:
    import torch
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except Exception:
    DEVICE = "cpu"

def _ensure_run_dir(kind: str) -> str:
    """Return a local path containing the model artifacts for the chosen kind."""
    local_run = os.path.join(RUN_DIR, FOLDER_MAP[kind])
    need = local_run if kind == "Hybrid" else os.path.join(local_run, "best_model")
    if os.path.isdir(need):
        return local_run

    # If not found locally, pull just that subtree from HF (optional)
    if snapshot_download and HF_REPO:
        st.info(f"Downloading `{kind}` artifacts from Hugging Face…")
        allow = [f"{FOLDER_MAP[kind]}/**"]
        root = snapshot_download(HF_REPO, allow_patterns=allow, token=HF_TOKEN)
        return os.path.join(root, FOLDER_MAP[kind])

    # Nothing found
    return local_run

run_root = _ensure_run_dir(model_type)
check_path = run_root if model_type == "Hybrid" else os.path.join(run_root, "best_model")

st.caption(f"Selected folder: `{run_root}`")
if os.path.isdir(check_path):
    st.success(f"Found model artifacts in: `{check_path}`")
else:
    st.warning(f"Expected files not found at: `{check_path}`")

@st.cache_resource(show_spinner=True)
def get_translator_cached(run_dir: str, prefer: str, device: str):
    return load_translator(run_dir, prefer=prefer, device=device)

# Load translator
translator = None
try:
    lr = get_translator_cached(run_root, PREFER_MAP[model_type], DEVICE)
    translator = lr.translator
    st.caption(f"Loaded backend: **{lr.backend}** — {lr.info}")
except Exception as e:
    st.error(f"Failed to load translator: {e}")

st.markdown("---")
cn_input = st.text_area("Enter Chinese sentence:", height=120, placeholder="输入中文句子…")

if st.button("Translate", use_container_width=True):
    if translator is None:
        st.error("Translator not loaded. Fix the error above.")
    elif not cn_input.strip():
        st.warning("Please enter a Chinese sentence.")
    else:
        with st.spinner("Translating..."):
            try:
                if model_type == "Hybrid":
                    out, info, table = translator.translate(
                        cn_input,
                        num_beams=FIXED_NUM_BEAMS,
                        max_new_tokens=FIXED_MAX_NEW,
                        no_repeat_ngram_size=FIXED_NGRAM,
                        length_penalty=FIXED_LEN_PEN,
                        return_info=True,
                        return_table=True,
                    )
                    st.subheader("Translation")
                    st.info(out or "(empty)")
                    st.dataframe(table)
                else:
                    out, info = translator.translate(
                        cn_input,
                        num_beams=FIXED_NUM_BEAMS,
                        max_new_tokens=FIXED_MAX_NEW,
                        no_repeat_ngram_size=FIXED_NGRAM,
                        length_penalty=FIXED_LEN_PEN,
                        return_info=True,
                    )
                    st.subheader("Translation")
                    st.info(out or "(empty)")
            except Exception as e:
                st.error(f"Translation failed: {e}")
