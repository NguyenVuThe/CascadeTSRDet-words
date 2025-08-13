# IN EXPERIMENTAL USE ONLY
# This script precomputes BERT embeddings for words extracted from OCR data.
# It processes words in batches and saves the embeddings along with their bounding boxes and text.

import os
import json
import torch
from tqdm import tqdm
from transformers import BertTokenizer, BertModel

# ==== CONFIG ====
WORDS_DIR = "D:\MyWorking\dataset\FinTabNet.c\FinTabNet.c-Structure\words"
OUTPUT_DIR = "D:\MyWorking\dataset\FinTabNet.c\FinTabNet.c-Structure\embeds"
BERT_NAME = "prosusai/finbert"
BATCH_SIZE = 128   # process this many words at once

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==== Device ====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==== Load BERT ====
tokenizer = BertTokenizer.from_pretrained(BERT_NAME)
bert_model = BertModel.from_pretrained(BERT_NAME).to(device)
bert_model.eval()

print(f"🔹 Using device: {device}")
print(f"🔹 Precomputing embeddings from {WORDS_DIR} → {OUTPUT_DIR}")

with torch.no_grad():
    for json_file in tqdm(os.listdir(WORDS_DIR)):
        if not json_file.endswith(".json"):
            continue

        # Load OCR word data
        json_path = os.path.join(WORDS_DIR, json_file)
        with open(json_path, "r", encoding="utf-8") as f:
            words_data = json.load(f)

        texts = [w.get("text", "").strip() or "[PAD]" for w in words_data]
        bboxes = [w.get("bbox", []) for w in words_data]

        embeddings = []
        # Process in batches
        for i in range(0, len(texts), BATCH_SIZE):
            batch_texts = texts[i:i + BATCH_SIZE]
            inputs = tokenizer(batch_texts, return_tensors="pt", truncation=True,
                               max_length=32, padding=True).to(device)
            outputs = bert_model(**inputs)
            batch_emb = outputs.last_hidden_state.mean(dim=1)  # [batch_size, 768]
            embeddings.extend(batch_emb.cpu())

        # Combine bbox + text + embedding
        page_data = [
            {"bbox": bbox, "text": text, "embedding": emb}
            for bbox, text, emb in zip(bboxes, texts, embeddings)
        ]

        torch.save(page_data, os.path.join(OUTPUT_DIR, json_file.replace(".json", ".pt")))

print("✅ All pages saved with bbox + text + embedding (GPU accelerated)")
