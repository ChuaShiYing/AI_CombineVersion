# app.py
import os
import streamlit as st
from huggingface_hub import snapshot_download
from translator import load_translator

st.set_page_config(page_title="CN→EN Translator", page_icon="🌐", layout="centered")
st.title("CN → EN Translator")

RUN_DIR  = os.environ.get("RUN_DIR", ".")
HF_REPO  = os.environ.get("HF_REPO")           # e.g. "Ncp93/zh-en-all"
HF_TOKEN = os.environ.get("HF_TOKEN")          # optional if repo is public

FOLDER_MAP = {"SMT": "smt/run", "NMT": "nmt/run", "Hybrid": "hybrid/run"}
PREFER_MAP = {"SMT": "smt", "NMT": "marian", "Hybrid": "hybrid"}

def ensure_subtree_available(subtree: str):
    """Ensure RUN_DIR/<subtree> exists locally; download from HF if missing."""
    local_path = os.path.join(RUN_DIR, subtree)
    need_download = not os.path.isdir(local_path) or not os.listdir(local_path)
    if need_download:
        if not HF_REPO:
            raise RuntimeError(
                f"'{local_path}' not found locally and HF_REPO is not set. "
                "Either upload model folders to the repo or set HF_REPO/HF_TOKEN."
            )
        snapshot_download(
            repo_id=HF_REPO,
            allow_patterns=[f"{subtree}/**"],   # download only needed branch
            local_dir=RUN_DIR,
            local_dir_use_symlinks=False,
            token=HF_TOKEN,                     # ok if None for public repos
        )
    return local_path

model_type = st.selectbox("Choose Model", ("SMT", "NMT", "Hybrid"), index=0)
subtree = FOLDER_MAP[model_type]

try:
    run_root = ensure_subtree_available(subtree)
    check_path = run_root if model_type == "Hybrid" else os.path.join(run_root, "best_model")
    if os.path.isdir(check_path):
        st.success(f"Found model artifacts in: `{check_path}`")
    else:
        st.warning(f"Expected files not found at: `{check_path}`")
except Exception as e:
    st.error(str(e))
    st.stop()

@st.cache_resource(show_spinner=True)
def get_translator_cached(run_dir: str, prefer: str):
    return load_translator(run_dir, prefer=prefer, device=None)  # auto GPU if available

lr = get_translator_cached(run_root, PREFER_MAP[model_type])
translator = lr.translator

st.markdown("---")
cn_input = st.text_area("Enter Chinese sentence:", height=120, placeholder="输入中文句子…")

if st.button("Translate", use_container_width=True,type="primary"):
    if not cn_input.strip():
        st.warning("Please enter a Chinese sentence.")
    else:
        with st.spinner("Translating..."):
            if model_type == "Hybrid":
                out, info, table = translator.translate(
                    cn_input, num_beams=5, max_new_tokens=128,
                    no_repeat_ngram_size=3, length_penalty=1.0,
                    return_info=True, return_table=True
                )
                st.subheader("Translation")
                st.info(out or "(empty)")
                st.dataframe(table)
            else:
                out, info = translator.translate(
                    cn_input, num_beams=5, max_new_tokens=128,
                    no_repeat_ngram_size=3, length_penalty=1.0,
                    return_info=True
                )
                st.subheader("Translation")
                st.info(out or "(empty)")
