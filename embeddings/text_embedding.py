import json
import numpy as np
import re

# ==== TOKEN CLEANING FUNCTION (same as words_to_embed) ====
def clean_token(tok: str) -> str:
    tok = tok.strip().lower()
    if not tok:
        return ""
    if tok.replace(".", "", 1).isdigit():
        return "<NUM>"
    if all(ch in ".,;:!?-_=+*/\\'\"`~" for ch in tok):
        return ""
    if tok in ["$", "€", "£", "%"]:
        return tok
    return tok

class TextEmbeddingLookup:
    def __init__(self, emb_path, vocab_path):
        self.emb = np.load(emb_path)
        with open(vocab_path, "r", encoding="utf-8") as f:
            self.token2id = json.load(f)
        self.dim = self.emb.shape[1]

    def get(self, token: str) -> np.ndarray:
        token = clean_token(token)
        if token in self.token2id:
            return self.emb[self.token2id[token]]
        return np.zeros(self.dim, dtype=np.float32)