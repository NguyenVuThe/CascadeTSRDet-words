# IN EXPERIMENTAL USE ONLY
# This script precomputes BERT embeddings for words extracted from OCR data.
# It processes words in batches and saves the embeddings along with their bounding boxes and text.

import os
import json
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

# ==== CONFIG ====
INPUT_DIR = "D:\MyWorking\dataset\FinTabNet.c\FinTabNet.c-Structure\words"
OUTPUT_DIR = "D:\MyWorking\dataset\FinTabNet.c\FinTabNet.c-Structure\embeds"
MODEL_NAME = "distilbert-base-uncased"
EMB_DIM = 768   # process this many words at once

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load BERT model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME).eval()
if torch.cuda.is_available():
    model = model.cuda()

# Step 1: Collect tokens
all_tokens = set()
print("🔍 Collecting tokens...")
for fname in os.listdir(INPUT_DIR):
    if fname.endswith(".json"):
        with open(os.path.join(INPUT_DIR, fname), "r", encoding="utf-8") as f:
            data = json.load(f)
        for w in data:
            tok = w["text"].strip().lower()
            if tok:
                all_tokens.add(tok)

print(f"✅ Collected {len(all_tokens)} unique tokens.")

# Step 2: Build vocab
token2id = {tok: i for i, tok in enumerate(sorted(all_tokens))}
with open(os.path.join(OUTPUT_DIR, "token2id.json"), "w", encoding="utf-8") as f:
    json.dump(token2id, f, ensure_ascii=False, indent=2)

# Step 3: Encode embeddings
embeddings = np.zeros((len(token2id), EMB_DIM), dtype=np.float32)

print("⚡ Encoding tokens with BERT...")
for tok, idx in tqdm(token2id.items()):
    inputs = tokenizer(tok, return_tensors="pt", truncation=True, max_length=16)
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    vec = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()  # [CLS] embedding
    embeddings[idx] = vec

# Step 4: Save embeddings
np.save(os.path.join(OUTPUT_DIR, "embeddings.npy"), embeddings)
print(f"✅ Saved embeddings to {OUTPUT_DIR}/embeddings.npy")
print(f"✅ Saved vocab to {OUTPUT_DIR}/token2id.json")