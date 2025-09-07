# translator.py
# ======================================================================================
# Backends supported:
#   1) SMT (PHRASE-BASED):   phrase_table.json (+ LM + optional weights)
#   2) SMT (SIMPLE IBM1):    ibm1_s2t.json
#   3) HYBRID (Tiny NMT+SMT): nmt_best.pth + ibm1.pkl + tokenizer/bpe_joint.model
#   4) MARIAN/HF Seq2Seq:    best_model/{config.json, model.safetensors|pytorch_model.bin, tokenizer files}
#
# Usage:
#   from translator import load_translator
#   lr = load_translator("hybrid/run")                  # auto-detect
#   lr = load_translator("smt/run", prefer="smt")       # force a family
#   lr = load_translator("nmt/run", prefer="marian")    # force Marian
#   text, info = lr.translator.translate("你好，世界", return_info=True)
# ======================================================================================

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Any, Dict, List
import json, math, warnings, os
from collections import defaultdict, Counter
from functools import lru_cache

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Optional deps
try:
    import sentencepiece as spm
except Exception:
    spm = None
try:
    import jieba
except Exception:
    jieba = None
try:
    import nltk
    nltk.download("punkt", quiet=True)
except Exception:
    nltk = None

# Quiet HF tokenizer rename warning (if you use Marian tokenizer artifacts)
warnings.filterwarnings("ignore", message="Tokenizer 'spm' has been changed to 'flores101'")

# ---------- constants for SMT phrase-based ----------
BOS = "<s>"
EOS = "</s>"
NULL = "<NULL>"

DEFAULT_WEIGHTS = {
    "W_PHRASE": 1.0,
    "W_LEX": 1.0,
    "W_LM": 1.0,
    "W_WORD_PENALTY": -0.1,
    "DIST_PENALTY": -0.2
}
DEFAULT_MAX_SRC_PHRASE_LEN = 8
DEFAULT_MAX_JUMP = 3
DEFAULT_LM_ALPHA = 0.3

# ======================================================================================
# Base interface
# ======================================================================================
class TranslatorBase:
    def translate(
        self,
        text: str,
        num_beams: int = 5,
        max_new_tokens: int = 128,
        no_repeat_ngram_size: int = 3,
        length_penalty: float = 1.0,
        temperature: float = 1.0,
        top_p: float = 1.0,
        return_info: bool = False,
    ) -> Tuple[str, Optional[str]]:
        raise NotImplementedError


# ======================================================================================
# (A) SMT — SIMPLE IBM1 (word-to-word table)
#     expects: <run_dir>/best_model/ibm1_s2t.json  (or <run_dir>/ibm1_s2t.json)
# ======================================================================================
class SmtSimpleTranslator(TranslatorBase):
    def __init__(self, run_dir: Path):
        run = Path(run_dir)
        candidates = [run / "best_model" / "ibm1_s2t.json",
                      run / "best_model" / "ibm1_s2t_final.json",
                      run / "ibm1_s2t.json",
                      run / "ibm1_s2t_final.json"]
        ibm1_path = next((p for p in candidates if p.exists()), None)
        if ibm1_path is None:
            raise FileNotFoundError(f"Could not find IBM1 table at any of: {candidates}")
        with open(ibm1_path, "r", encoding="utf-8") as f:
            table = json.load(f)
        if not isinstance(table, dict):
            raise ValueError("ibm1_s2t.json must be a dict {src_word: {tgt_word: prob, ...}}")
        self.ibm1_table: Dict[str, Dict[str, float]] = table

    def translate(self, text: str, return_info: bool = False, **kwargs):
        src_words = text.strip().split()
        out, missing = [], []
        for w in src_words:
            # choose highest-probability English word
            cand = self.ibm1_table.get(w, {})
            if isinstance(cand, dict) and cand:
                tgt = max(cand.items(), key=lambda kv: kv[1])[0]
                out.append(tgt)
            else:
                missing.append(w)
        hyp = " ".join(out).strip()
        info = ("SMT_SIMPLE | IBM1" + (f" | missing: {', '.join(missing)}" if missing else ""))
        return (hyp, info) if return_info else (hyp, None)


# ======================================================================================
# (B) SMT — PHRASE-BASED DECODER (phrase_table + LM)
#     expects under a "best" directory:
#        - phrase_table.json  (contains {"phi": {...}, "lex": {...}})
#        - lm_trigram_counts.json or lm_counts.json (w/ unigrams, bigrams, trigrams)
#        - (optional) decode_meta.json for weights, max phrase len, max jump
#     The script will accept:
#        <run_dir>/best_model/..., <run_dir>/best/..., or any subfolder that has phrase_table.json
# ======================================================================================
def _tok_zh(s: str) -> List[str]:
    if jieba is not None:
        return [t for t in jieba.lcut(str(s)) if t.strip()]
    # fallback: character tokens
    return [c for c in str(s) if str(c).strip()]

def _load_json(p: Path):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def _find_phrase_best_dir(run_root_or_best: Path) -> Path:
    p = Path(run_root_or_best).resolve()
    # direct
    if p.is_dir() and (p / "phrase_table.json").exists():
        return p
    # common names
    for cand in ("best_model", "best"):
        q = p / cand
        if q.is_dir() and (q / "phrase_table.json").exists():
            return q
    # search downwards
    for root, dirs, files in os.walk(p):
        if "phrase_table.json" in files:
            return Path(root)
    raise FileNotFoundError(f"No phrase_table.json found under {p}")

def _load_phrase_table(best_dir: Path):
    pt = _load_json(best_dir / "phrase_table.json")
    phi_raw, lex_raw = pt["phi"], pt["lex"]
    phrase_table, lex_table = defaultdict(dict), defaultdict(dict)

    for f_str, e_map in phi_raw.items():
        f_tuple = tuple(f_str.split(" ")) if f_str else tuple()
        for e_str, v in e_map.items():
            e_tuple = tuple(e_str.split(" ")) if e_str else tuple()
            phrase_table[f_tuple][e_tuple] = float(v)

    for f_str, e_map in lex_raw.items():
        f_tuple = tuple(f_str.split(" ")) if f_str else tuple()
        for e_str, v in e_map.items():
            e_tuple = tuple(e_str.split(" ")) if e_str else tuple()
            lex_table[f_tuple][e_tuple] = float(v)

    # pre-index: for each source phrase f, keep candidate (e, log(phi), log(lex))
    src_phrase_index: Dict[Tuple[str, ...], List[Tuple[Tuple[str, ...], float, float]]] = {}
    for f, e_dict in phrase_table.items():
        cands = []
        for e, phi in e_dict.items():
            lp = math.log(max(phi, 1e-12))
            ll = math.log(max(lex_table[f].get(e, 1e-12), 1e-12))
            cands.append((e, lp, ll))
        src_phrase_index[f] = cands
    return phrase_table, lex_table, src_phrase_index

def _load_ibm1_for_phrase(best_dir: Path):
    for name in ("ibm1_s2t.json", "ibm1_s2t_final.json"):
        p = best_dir / name
        if p.exists():
            return _load_json(p)
    raise FileNotFoundError("Missing ibm1_s2t.json (or ibm1_s2t_final.json) for backoff.")

def _load_lm(best_dir: Path):
    # accept either
    lm_path = None
    for name in ("lm_trigram_counts.json", "lm_counts.json"):
        p = best_dir / name
        if p.exists():
            lm_path = p
            break
    if not lm_path:
        raise FileNotFoundError("Missing lm_trigram_counts.json / lm_counts.json")

    lm = _load_json(lm_path)
    unigrams = Counter()
    bigrams  = Counter()
    trigrams = Counter()

    if isinstance(lm.get("unigrams"), dict):
        unigrams.update({k: int(v) for k, v in lm["unigrams"].items()})
        if "bigrams" in lm:
            for k, v in lm["bigrams"].items():
                pair = tuple(k.split(" ")) if isinstance(k, str) else tuple(k)
                bigrams[pair] = int(v)
        if "trigrams" in lm:
            for k, v in lm["trigrams"].items():
                tri = tuple(k.split(" ")) if isinstance(k, str) else tuple(k)
                trigrams[tri] = int(v)
    else:
        raise ValueError("LM JSON structure unexpected")

    total_unigrams = sum(unigrams.values())
    V = len(unigrams)
    LM_ALPHA = DEFAULT_LM_ALPHA

    def lm_logprob(nextw, w1, w2):
        tri = trigrams.get((w1, w2, nextw), 0)
        if tri > 0:
            denom = bigrams.get((w1, w2), 1)
            return math.log(tri / denom)
        bi = bigrams.get((w2, nextw), 0)
        if bi > 0:
            denom = unigrams.get(w2, 1)
            return math.log(LM_ALPHA * bi / denom)
        uni = unigrams.get(nextw, 0)
        return math.log(LM_ALPHA * LM_ALPHA * (uni + 1) / (total_unigrams + V + 1))

    return lm_logprob

def _load_decode_meta(best_dir: Path):
    meta_path = best_dir / "decode_meta.json"
    W = DEFAULT_WEIGHTS.copy()
    max_len = DEFAULT_MAX_SRC_PHRASE_LEN
    max_jump = DEFAULT_MAX_JUMP
    if meta_path.exists():
        meta = _load_json(meta_path)
        W.update(meta.get("weights", {}))
        max_len = int(meta.get("max_src_phrase_len", max_len))
        max_jump = int(meta.get("max_jump", max_jump))
    return W, max_len, max_jump

def _make_phrase_decoder(src_phrase_index, ibm1, lm_logprob, W, MAX_SRC_PHRASE_LEN, MAX_JUMP):
    W_PHRASE=W["W_PHRASE"]; W_LEX=W["W_LEX"]; W_LM=W["W_LM"]
    W_WP=W["W_WORD_PENALTY"]; DIST=W["DIST_PENALTY"]

    def decode(s_words: List[str]):
        N = len(s_words)
        span_options = defaultdict(list)
        for i in range(N):
            for L in range(1, min(MAX_SRC_PHRASE_LEN, N - i) + 1):
                f = tuple(s_words[i:i+L])
                if f in src_phrase_index:
                    span_options[(i, L)] = src_phrase_index[f]

        @lru_cache(maxsize=None)
        def search(mask: int, w1: str, w2: str):
            if mask == (1<<N) - 1:
                return ([], W_LM * lm_logprob(EOS, w1, w2))
            best_hyp, best_score = [], -1e9

            pos = 0
            while pos < N and ((mask >> pos) & 1):
                pos += 1
            advanced = False

            # normal extend
            for L in range(1, min(MAX_SRC_PHRASE_LEN, N - pos) + 1):
                if any(((mask>>k)&1) for k in range(pos, pos+L)): 
                    continue
                if (pos, L) not in span_options:
                    continue
                advanced = True
                new_mask = mask | sum(1<<k for k in range(pos, pos+L))
                for e_tokens, lp, ll in span_options[(pos, L)]:
                    lm_s = 0.0; ww1, ww2 = w1, w2
                    for tok in e_tokens:
                        lm_s += lm_logprob(tok, ww1, ww2)
                        ww1, ww2 = ww2, tok
                    sub_hyp, sub_score = search(new_mask, ww1, ww2)
                    score = sub_score + W_PHRASE*lp + W_LEX*ll + W_LM*lm_s + W_WP*len(e_tokens)
                    if score > best_score:
                        best_score = score; best_hyp = list(e_tokens) + sub_hyp

            # limited jump
            for jump in range(1, MAX_JUMP+1):
                jpos = pos + jump
                if jpos >= N or ((mask>>jpos)&1): break
                for L in range(1, min(MAX_SRC_PHRASE_LEN, N - jpos) + 1):
                    if any(((mask>>k)&1) for k in range(jpos, jpos+L)): 
                        continue
                    if (jpos, L) not in span_options: 
                        continue
                    advanced = True
                    new_mask = mask | sum(1<<k for k in range(jpos, jpos+L))
                    for e_tokens, lp, ll in span_options[(jpos, L)]:
                        lm_s = 0.0; ww1, ww2 = w1, w2
                        for tok in e_tokens:
                            lm_s += lm_logprob(tok, ww1, ww2)
                            ww1, ww2 = ww2, tok
                        sub_hyp, sub_score = search(new_mask, ww1, ww2)
                        score = sub_score + W_PHRASE*lp + W_LEX*ll + W_LM*lm_s + W_WP*len(e_tokens) + DIST*jump
                        if score > best_score:
                            best_score = score; best_hyp = list(e_tokens) + sub_hyp

            # backoff: single-word IBM1 at current pos
            if not advanced:
                s = s_words[pos]
                cands = ibm1.get(s, {})
                best = None
                for t, p in sorted(cands.items(), key=lambda kv: kv[1], reverse=True):
                    if t != NULL:
                        best = (t, p); break
                if best:
                    tok = best[0]
                    lm_s = lm_logprob(tok, w1, w2)
                    sub_hyp, sub_score = search(mask | (1<<pos), w2, tok)
                    lp = math.log(max(best[1], 1e-12))
                    ll = lp
                    score = sub_score + W_PHRASE*lp + W_LEX*ll + W_LM*lm_s + W_WP
                    if score > best_score:
                        best_score = score; best_hyp = [tok] + sub_hyp
            return (best_hyp, best_score)

        hyp, _ = search(0, BOS, BOS)
        return hyp
    return decode

class SmtPhraseTranslator(TranslatorBase):
    def __init__(self, run_dir: Path):
        # figure out where phrase_table.json lives
        best_dir = _find_phrase_best_dir(run_dir)
        self.best_dir = best_dir
        self.phrase_table, self.lex_table, self.src_phrase_index = _load_phrase_table(best_dir)
        self.ibm1 = _load_ibm1_for_phrase(best_dir)
        self.lm_logprob = _load_lm(best_dir)
        self.W, self.MAX_SRC_PHRASE_LEN, self.MAX_JUMP = _load_decode_meta(best_dir)
        self.decode = _make_phrase_decoder(self.src_phrase_index,
                                           self.ibm1, self.lm_logprob,
                                           self.W, self.MAX_SRC_PHRASE_LEN, self.MAX_JUMP)

    def translate(self, text: str, return_info: bool = False, **kwargs):
        s_words = _tok_zh(text)
        hyp = self.decode(s_words)
        out = " ".join(hyp)
        info = f"SMT_PHRASE | tokens={s_words} | best_dir={self.best_dir}"
        return (out, info) if return_info else (out, None)



# ======================================================================================
# (C) HYBRID — Tiny Transformer NMT + SMT reranking
# ======================================================================================

def tok_en(s: str):
    if nltk is not None:
        try:
            return nltk.word_tokenize(s)
        except Exception:
            return s.split()
    return s.split()


def tok_cn(s: str):
    if jieba is not None:
        return list(jieba.cut(s, cut_all=False))
    return [c for c in s if c.strip()]


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0)/d_model))
        pe[:,0::2] = torch.sin(position * div_term)
        pe[:,1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)
    def forward(self, x):
        return x + self.pe[:,:x.size(1),:]


class TinyTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=4, num_layers=3, dim_feedforward=512, max_len=80, pad_id=0):
        super().__init__()
        self.vocab_size, self.d_model, self.pad_id = vocab_size, d_model, pad_id
        self.src_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.tgt_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos = PositionalEncoding(d_model, max_len=max_len)
        enc_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout=0.1, batch_first=True)
        dec_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout=0.1, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=num_layers)
        self.out = nn.Linear(d_model, vocab_size)

    def forward(self, src_ids, tgt_ids):
        tgt_in = tgt_ids[:, :-1]
        src_mask = (src_ids == self.pad_id)
        tgt_key_mask = (tgt_in == self.pad_id)
        src_emb = self.pos(self.src_emb(src_ids) * math.sqrt(self.d_model))
        memory = self.encoder(src_emb, src_key_padding_mask=src_mask)
        tgt_emb = self.pos(self.tgt_emb(tgt_in) * math.sqrt(self.d_model))
        sz = tgt_emb.size(1)
        tgt_mask = torch.triu(torch.ones(sz, sz, device=src_ids.device), diagonal=1).bool()
        out = self.decoder(tgt_emb, memory, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_mask, memory_key_padding_mask=src_mask)
        logits = self.out(out)
        return logits

    @torch.no_grad()
    def greedy_decode(self, src_ids, max_len=80, no_repeat_ngram_size=3):
        self.eval()
        src_mask = (src_ids == self.pad_id)
        src_emb = self.pos(self.src_emb(src_ids) * math.sqrt(self.d_model))
        memory = self.encoder(src_emb, src_key_padding_mask=src_mask)
        ys = torch.full((src_ids.size(0), 1), 1, dtype=torch.long, device=src_ids.device)
        for _ in range(max_len-1):
            tgt_emb = self.pos(self.tgt_emb(ys) * math.sqrt(self.d_model))
            L = tgt_emb.size(1)
            tgt_mask = torch.triu(torch.ones(L, L, device=src_ids.device), diagonal=1).bool()
            dec = self.decoder(tgt_emb, memory, tgt_mask=tgt_mask, memory_key_padding_mask=src_mask)
            logits = self.out(dec)[:, -1, :]
            if no_repeat_ngram_size and no_repeat_ngram_size > 0 and ys.size(1) >= no_repeat_ngram_size:
                seq = ys[0].tolist()
                n = no_repeat_ngram_size
                seen = set(tuple(seq[i:i+n]) for i in range(len(seq)-n+1))
                prefix = tuple(seq[-(n-1):]) if n-1>0 else tuple()
                forbid = [ng[-1] for ng in seen if ng[:-1]==prefix]
                if forbid:
                    logits[0, forbid] = -1e9
            next_tok = logits.argmax(-1, keepdim=True)
            ys = torch.cat([ys, next_tok], dim=1)
            if (next_tok == 2).all():
                break
        return ys


class HybridTranslator:
    def __init__(self, run_dir: Path, device: Optional[str] = None):
        import pickle
        run = Path(run_dir)
        sp_path = run / "tokenizer" / "bpe_joint.model"
        nmt_ckpt = run / "nmt_best.pth"
        ibm_pkl  = run / "ibm1.pkl"
        if spm is None:
            raise ModuleNotFoundError("sentencepiece required for Hybrid")
        self.device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sp = spm.SentencePieceProcessor(model_file=str(sp_path))
        vocab_size = self.sp.get_piece_size()
        self.nmt = TinyTransformer(vocab_size=vocab_size, max_len=80, pad_id=0).to(self.device)
        ckpt = torch.load(nmt_ckpt, map_location=self.device)
        if isinstance(ckpt, dict) and 'model_state' in ckpt:
            self.nmt.load_state_dict(ckpt['model_state'])
        else:
            self.nmt.load_state_dict(ckpt)
        self.nmt.eval()
        with open(ibm_pkl, "rb") as f:
            smt_saved = pickle.load(f)
        self.translation_table = smt_saved.get("translation_table", {})
        self.src2tgt = smt_saved.get("src2tgt", {})
        self.counts = defaultdict(Counter, smt_saved.get("bigram_counts", {}))
        self.unigrams = Counter(smt_saved.get("unigrams", {}))
        self.vocab_size_lm = max(1, len(self.unigrams))

    def encode_with_sp(self, text: str, max_len: int = 80):
        ids = self.sp.encode(str(text), out_type=int)
        ids = [1] + ids[:max_len-2] + [2]
        if len(ids) < max_len:
            ids += [0] * (max_len - len(ids))
        return ids

    @torch.no_grad()
    def nmt_greedy(self, src_text: str):
        ids = self.encode_with_sp(src_text)
        src = torch.tensor([ids], dtype=torch.long, device=self.device)
        out_ids = self.nmt.greedy_decode(src, max_len=80, no_repeat_ngram_size=3)[0].tolist()
        seq = [i for i in out_ids if i not in (0,1,2)]
        return " ".join(tok_en(self.sp.decode(seq)))

    @torch.no_grad()
    def nmt_logprob(self, src_text: str, tgt_text: str):
        src_ids = torch.tensor([self.encode_with_sp(src_text)], dtype=torch.long, device=self.device)
        tgt_ids = torch.tensor([self.encode_with_sp(tgt_text)], dtype=torch.long, device=self.device)
        logits = self.nmt(src_ids, tgt_ids)
        logp = F.log_softmax(logits, dim=-1)
        total, count = 0.0, 0
        for t in range(logp.size(1)):
            token = int(tgt_ids[0, t+1].item())
            total += float(logp[0, t, token].item())
            count += 1
        return total, total/max(1,count), count

    def compute_smt_score_for_candidate(self, src_sentence: str, cand_text: str):
        src_toks = list(jieba.cut(src_sentence, cut_all=False)) if jieba else list(src_sentence)
        tgt_toks = cand_text.split() if " " in cand_text else list(cand_text)
        l = max(1, len(src_toks)); denom = float(l + 1)
        total_logp = 0.0
        for tj in tgt_toks:
            srcmap = self.translation_table.get(tj, {})
            ssum = 0.0
            for si in src_toks:
                ssum += srcmap.get(si, 0.0)
            ssum += srcmap.get("", 0.0)
            if ssum <= 0:
                total_logp += math.log(1e-12 / denom)
            else:
                total_logp += math.log(ssum / denom)
        vocab_lm = max(1, len(self.unigrams))
        lm_logp = 0.0
        prev = "<S>"
        for tok in tgt_toks:
            num = self.counts[prev].get(tok, 0) + 1
            denom2 = self.unigrams.get(prev, 0) + vocab_lm
            if denom2 <= 0:
                denom2 = vocab_lm
            lm_logp += math.log(num / denom2)
            prev = tok
        num = self.counts[prev].get("</S>", 0) + 1
        denom2 = self.unigrams.get(prev, 0) + vocab_lm
        if denom2 <= 0:
            denom2 = vocab_lm
        lm_logp += math.log(num / denom2)
        return float(total_logp + lm_logp)

    def smt_decode(self, src_sentence: str, beam_size=6, n_best=5):
        src_tokens = tok_cn(src_sentence)
        beams = [(0.0, [])]
        for s_tok in src_tokens:
            cands = self.src2tgt.get(s_tok, [])
            if not cands:
                cands = [(" ".join(tok_en(s_tok)), 1e-6)]
            new_beams = []
            for prev_score, seq in beams:
                prev_tok = seq[-1] if seq else "<S>"
                for tgt_tok, prob in cands:
                    num = self.counts[prev_tok].get(tgt_tok, 0) + 1
                    denom = self.unigrams.get(prev_tok, 0) + self.vocab_size_lm
                    bigram_lp = math.log(num / max(1, denom))
                    sc = prev_score + math.log(max(prob,1e-12)) + bigram_lp
                    new_beams.append((sc, seq + [tgt_tok]))
            new_beams.sort(key=lambda x: x[0], reverse=True)
            beams = new_beams[:beam_size]
        final = [(sc, " ".join(tok_en(" ".join(seq)))) for sc, seq in beams]
        final.sort(key=lambda x: x[0], reverse=True)
        return final[:n_best]

    def _normalize(self, arr):
        a = np.array(arr, dtype=float)
        if np.isnan(a).all():
            return np.full_like(a, 0.5, dtype=float)
        mn, mx = float(np.nanmin(a)), float(np.nanmax(a))
        if abs(mx - mn) < 1e-12:
            return np.full_like(a, 0.5, dtype=float)
        return (a - mn) / (mx - mn)

    def translate(self, text: str, num_beams=5, return_info=False, return_table=False, **kwargs):
        text = (text or "").strip()
        if not text:
            return ("", "Empty input") if return_info else ("", None)

        nbest = self.smt_decode(text, beam_size=num_beams, n_best=5)
        cands = [cand for _, cand in nbest]
        try:
            nmt_g = self.nmt_greedy(text)
            if nmt_g and nmt_g not in cands:
                cands.append(nmt_g)
        except Exception:
            pass
        if not cands:
            cands = [self.nmt_greedy(text)]

        rows = []
        src_len = max(1, len(tok_cn(text)))
        for cand in cands:
            smt_sc = self.compute_smt_score_for_candidate(text, cand)
            try:
                _, nmt_avg, _ = self.nmt_logprob(text, cand)
            except Exception:
                nmt_avg = -1e12
            tgt_len = max(1, len(cand.split()))
            len_penalty = -abs(tgt_len - src_len) / src_len
            rows.append((cand, smt_sc, nmt_avg, len_penalty,
                         "SMT_nbest" if cand in [t for _, t in nbest] else "NMT_greedy"))

        smt_norm = self._normalize([r[1] for r in rows])
        nmt_norm = self._normalize([r[2] for r in rows])
        len_norm = self._normalize([r[3] for r in rows])
        alpha, len_w = 0.6, 0.15
        blended = (1.0 - len_w) * (alpha * nmt_norm + (1.0 - alpha) * smt_norm) + len_w * len_norm

        best_idx = int(np.argmax(blended))
        best = rows[best_idx][0]

        info = f"HYBRID | device={self.device} | nbest={len(rows)}"

        if return_table:
            df = pd.DataFrame([{
                "candidate": r[0],
                "source": r[4],
                "blended": round(float(b), 2)
            } for r, b in zip(rows, blended)])
            df = df.sort_values("blended", ascending=False).reset_index(drop=True)
            return (best, info, df)

        return (best, info) if return_info else (best, None)





# ======================================================================================
# (D) MARIAN / HF Seq2Seq (CPU-safe by default; pass device='cuda' to use GPU)
#     expects: <run_dir>/best_model/{config.json, model.safetensors|pytorch_model.bin, tokenizer files}
# ======================================================================================
class MarianTranslator(TranslatorBase):
    def __init__(self, run_dir: Path, device: Optional[str] = None):
        from transformers import AutoTokenizer, AutoConfig, AutoModelForSeq2SeqLM
        safeload = None
        try:
            from safetensors.torch import load_file as _lf
            safeload = _lf
        except Exception:
            pass

        best = Path(run_dir) / "best_model"
        if not (best / "config.json").exists():
            raise FileNotFoundError(f"config.json not found in {best}")
        has_sft = (best / "model.safetensors").exists()
        has_bin = (best / "pytorch_model.bin").exists()

        # Default to CPU unless device specified (GPU ok if you pass device='cuda')
        self.device = torch.device(device) if device else torch.device("cpu")

        # tokenizer
        self.tok = AutoTokenizer.from_pretrained(str(best), local_files_only=True)

        # model
        if has_sft and safeload is not None:
            cfg = AutoConfig.from_pretrained(str(best), local_files_only=True)
            self.model = AutoModelForSeq2SeqLM.from_config(cfg).to("cpu")
            sd = safeload(str(best / "model.safetensors"))
            missing, unexpected = self.model.load_state_dict(sd, strict=False)
            if unexpected: print("Unexpected keys:", unexpected)
            if missing:    print("Missing keys:", missing)
            self.model.to(self.device).eval()
        else:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(str(best), local_files_only=True).to(self.device).eval()

        # generation special ids (important for Marian)
        pad_id = self.tok.pad_token_id
        eos_id = self.tok.eos_token_id
        bos_id = getattr(self.tok, "bos_token_id", None)
        self.model.config.pad_token_id = pad_id
        self.model.config.eos_token_id = eos_id
        if bos_id is not None:
            self.model.config.bos_token_id = bos_id
        self.model.config.decoder_start_token_id = pad_id  # Marian-style

    @torch.no_grad()
    def translate(
        self,
        text: str,
        num_beams: int = 5,
        max_new_tokens: int = 128,
        no_repeat_ngram_size: int = 3,
        length_penalty: float = 1.0,
        temperature: float = 1.0,
        top_p: float = 1.0,
        return_info: bool = False,
    ):
        text = (text or "").strip()
        if not text:
            return ("", "Empty input.") if return_info else ("", None)

        enc = self.tok([text], return_tensors="pt", padding=True, truncation=True, max_length=128).to(self.device)

        # sampling vs beam search
        do_sample = (temperature is not None and abs(float(temperature) - 1.0) > 1e-6) or \
                    (top_p is not None and float(top_p) < 1.0)

        gen = self.model.generate(
            **enc,
            num_beams=int(num_beams if not do_sample else max(1, num_beams)),
            max_new_tokens=int(max_new_tokens),
            no_repeat_ngram_size=int(no_repeat_ngram_size),
            length_penalty=float(length_penalty),
            do_sample=bool(do_sample),
            temperature=float(temperature),
            top_p=float(top_p),
            decoder_start_token_id=self.model.config.decoder_start_token_id,
            pad_token_id=self.model.config.pad_token_id,
            eos_token_id=self.model.config.eos_token_id,
        )
        out = self.tok.batch_decode(gen, skip_special_tokens=True)[0]
        info = f"MARIAN | device={self.device} | beams={num_beams} | sample={do_sample}"
        return (out, info) if return_info else (out, None)


# ======================================================================================
# Loader / factory
# ======================================================================================
@dataclass
class LoadResult:
    translator: TranslatorBase
    backend: str
    info: str

def _detect_backend(run_dir: Path) -> str:
    run = Path(run_dir)

    # MARIAN?
    if (run / "best_model" / "config.json").exists() and (
        (run / "best_model" / "model.safetensors").exists() or
        (run / "best_model" / "pytorch_model.bin").exists()
    ):
        return "marian"

    # HYBRID?
    if (run / "nmt_best.pth").exists() and (run / "ibm1.pkl").exists() and (run / "tokenizer" / "bpe_joint.model").exists():
        return "hybrid"

    # SMT_PHRASE?
    # (phrase_table.json anywhere under run/ or best/ or best_model/)
    try:
        _find_phrase_best_dir(run)  # will raise if not found
        return "smt_phrase"
    except Exception:
        pass

    # SMT_SIMPLE?
    for p in (
        run / "best_model" / "ibm1_s2t.json",
        run / "best_model" / "ibm1_s2t_final.json",
        run / "ibm1_s2t.json",
        run / "ibm1_s2t_final.json",
    ):
        if p.exists():
            return "smt_simple"

    return "unknown"

def load_translator(run_dir: str | Path, prefer: Optional[str] = None, device: Optional[str] = None) -> LoadResult:
    """
    Auto-detects backend from files in run_dir.
    You can force with prefer in {"marian","hybrid","smt","smt_phrase","smt_simple"}.
    - If prefer="smt", will pick phrase-based if available, else simple IBM1.
    """
    run = Path(run_dir)

    pref = (prefer or "").lower().strip()
    if pref and pref not in ("marian", "hybrid", "smt", "smt_phrase", "smt_simple"):
        raise ValueError("prefer must be one of: None, 'marian', 'hybrid', 'smt', 'smt_phrase', 'smt_simple'")

    if pref == "smt":
        # prefer phrase → simple
        try:
            _find_phrase_best_dir(run)
            pref = "smt_phrase"
        except Exception:
            pref = "smt_simple"

    backend = pref or _detect_backend(run)

    if backend == "marian":
        t = MarianTranslator(run, device=device)
        return LoadResult(t, "marian", f"Loaded Marian from {run.resolve()}")

    if backend == "hybrid":
        t = HybridTranslator(run, device=device)
        return LoadResult(t, "hybrid", f"Loaded Hybrid Tiny from {run.resolve()}")

    if backend == "smt_phrase":
        t = SmtPhraseTranslator(run)
        return LoadResult(t, "smt_phrase", f"Loaded SMT (phrase-based) from {run.resolve()}")

    if backend == "smt_simple":
        t = SmtSimpleTranslator(run)
        return LoadResult(t, "smt_simple", f"Loaded SMT (IBM1 simple) from {run.resolve()}")

    raise FileNotFoundError(
        "No supported backend detected under:\n"
        f"  {run}\n"
        "Expected one of:\n"
        "  - Marian: best_model/{config.json, model.safetensors|pytorch_model.bin}\n"
        "  - Hybrid: nmt_best.pth, ibm1.pkl, tokenizer/bpe_joint.model\n"
        "  - SMT phrase: phrase_table.json (+ lm_*counts.json)\n"
        "  - SMT simple: ibm1_s2t.json"
    )

# ---------- optional CLI ----------
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("run_dir", type=str, help="Path to run directory (root containing artifacts)")
    p.add_argument("--prefer", type=str, default=None,
                   help="Force backend: marian | hybrid | smt | smt_phrase | smt_simple")
    p.add_argument("--device", type=str, default=None, help="e.g. cuda or cpu (Marian/Hybrid only)")
    p.add_argument("--text", type=str, default="你好，世界")
    p.add_argument("--beams", type=int, default=5)
    p.add_argument("--max_new", type=int, default=128)
    p.add_argument("--temp", type=float, default=1.0)
    p.add_argument("--topp", type=float, default=1.0)
    args = p.parse_args()

    lr = load_translator(args.run_dir, prefer=args.prefer, device=args.device)
    out, info = lr.translator.translate(
        args.text,
        num_beams=args.beams,
        max_new_tokens=args.max_new,
        temperature=args.temp,
        top_p=args.topp,
        return_info=True,
    )
    print(f"[{lr.backend}] {info}\n→ {out}")
