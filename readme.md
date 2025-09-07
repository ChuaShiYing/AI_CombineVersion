# Chinese-English Machine Translation System

This repository provides a complete pipeline for Chinese-to-English translation, including Statistical Machine Translation (SMT), Neural Machine Translation (NMT, Transformer-based), and a Hybrid approach. It includes data preprocessing, training, evaluation, and a Streamlit web app for interactive translation.

---

## Quick Start: Installation & Usage

### 1. Install Python

- Make sure you have **Python 3.8+** installed.
- Download from [python.org](https://www.python.org/downloads/) if needed.


### 2. (Optional) Create a Virtual Environment

```sh
python -m venv venv
venv\Scripts\activate  # On Windows
# Or: source venv/bin/activate  # On Linux/Mac
```

### 3. Install Dependencies

```sh
pip install -r requirements.txt
```
Or manually:
```sh
pip install torch transformers sentencepiece sacrebleu datasets pandas numpy matplotlib tqdm streamlit jieba nltk
```

### 4. Prepare Data

- Place your parallel data in `dataset_CN_EN.txt` (tab-separated: Chinese<TAB>English).

### 5. Train Models

- **NMT:** Run all cells in [`nmt/nmt.ipynb`](nmt/nmt.ipynb).
- **SMT:** Run all cells in [`smt/smt_end_to_end_pipeline.ipynb`](smt/smt_end_to_end_pipeline.ipynb).
- **Hybrid:** Run all cells in [`hybrid/Hybrid.ipynb`](hybrid/Hybrid.ipynb).

### 6. Run the Web App

```sh
streamlit run app.py
```
- This will open a browser window at `http://localhost:8501`.
- Choose the model type (SMT, NMT, Hybrid) in the UI.
- Enter Chinese text and get English translation.

---


## Directory Structure

```
app.py                  # Streamlit web app for translation
main.py                 # (Optional) Main script entry point
translator.py           # Model loading and inference logic
requirements.txt        # Python dependencies
dataset_CN_EN.txt       # Parallel corpus (Chinese <TAB> English)
ReadMe.md               # Project documentation

nmt/
  nmt.ipynb     # Transformer-based NMT notebook (main NMT pipeline)
  combined_corpus.txt   # Corpus for tokenizer training
  run/
    spm.model           # SentencePiece model
    best_model/         # Best NMT model checkpoint, config, tokenizer
    data_splits/        # Train/val/test splits
    eval/               # Evaluation outputs, metrics
    training_checkpoints/
    ...                 # Tokenizer configs, vocab, etc.

smt/
  smt_end_to_end_pipeline.ipynb  # SMT pipeline notebook
  run/
    ...                 # SMT model artifacts

hybrid/
  Hybrid.ipynb          # Hybrid pipeline notebook (combines SMT+NMT)
  needed.txt            # Hybrid requirements and usage notes
  run/
    ibm1.pkl            # SMT model for hybrid
    nmt_best.pth        # NMT model for hybrid
    tokenizer/
      ...               # Tokenizer for hybrid
```

---

## Components

### 1. Neural Machine Translation (NMT)

- **Notebook:** [`nmt/nmt.ipynb`](nmt/nmt.ipynb)
- **Features:** Data cleaning, augmentation, SentencePiece tokenizer, Marian-style Transformer, BLEU/chrF++ evaluation, metrics visualization, interactive translation.
- **Artifacts:** Saved in `nmt/run/best_model/` after training.

### 2. Statistical Machine Translation (SMT)

- **Notebook:** [`smt/smt_end_to_end_pipeline.ipynb`](smt/smt_end_to_end_pipeline.ipynb)
- **Features:** Data loading, cleaning, IBM Model 1 training, phrase extraction, checkpointing.
- **Artifacts:** Saved in `smt/run/`.

### 3. Hybrid Translation

- **Notebook:** [`hybrid/Hybrid.ipynb`](hybrid/Hybrid.ipynb)
- **Features:** Combines SMT and NMT outputs for improved translation.
- **Artifacts:** Saved in `hybrid/run/`.

### 4. Web Application

- **App:** [`app.py`](app.py)
- **Framework:** Streamlit
- **Usage:** Select SMT, NMT, or Hybrid backend and translate Chinese text to English interactively.
- **Model Loading:** Uses [`translator.py`](translator.py) to load models from the appropriate folders.

---

## Setup

### 1. Install Dependencies

```sh
pip install -r requirements.txt
```
Or manually:
```sh
pip install torch transformers sentencepiece sacrebleu datasets pandas numpy matplotlib tqdm streamlit
```

### 2. Prepare Data

- Place your parallel data in `dataset_CN_EN.txt` (tab-separated: Chinese<TAB>English).

### 3. Train Models

- **NMT:** Run all cells in [`nmt/Transformer.ipynb`](nmt/Transformer.ipynb).
- **SMT:** Run all cells in [`smt/smt_end_to_end_pipeline.ipynb`](smt/smt_end_to_end_pipeline.ipynb).
- **Hybrid:** Run all cells in [`hybrid/Hybrid.ipynb`](hybrid/Hybrid.ipynb).

### 4. Run the Web App

```sh
streamlit run app.py
```
- Choose the model type (SMT, NMT, Hybrid) in the UI.
- Enter Chinese text and get English translation.

---

## Notes

- All model artifacts are saved in their respective `run/` folders.
- The app auto-detects available models and backends.
- For custom paths, set the `RUN_DIR` environment variable before launching the app.

---

## References

- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- [SentencePiece](https://github.com/google/sentencepiece)
- [sacreBLEU](https://github.com/mjpost/sacrebleu)
- [Streamlit](https://streamlit.io/)

---

## Citation

If you use this code, please cite the relevant libraries above.

## Local Host
Run the following command on your terminal. 

python -m streamlit run app.py