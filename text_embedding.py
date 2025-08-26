import numpy as np
import json

class TextEmbeddingLookup:
    def __init__(self, emb_path, vocab_path):
        self.emb = np.load(emb_path)   # (V, 768)
        with open(vocab_path, "r", encoding="utf-8") as f:
            self.token2id = json.load(f)
        self.dim = self.emb.shape[1]

    def get(self, token):
        token = token.lower().strip()
        if token in self.token2id:
            return self.emb[self.token2id[token]]
        return np.zeros(self.dim, dtype=np.float32)

# def get_box_embedding(proposal_bbox, words, lookup: TextEmbeddingLookup):
#     """
#     proposal_bbox: [x1,y1,x2,y2]
#     words: list of dict {text, bbox: [x1,y1,x2,y2]}
#     lookup: TextEmbeddingLookup
#     """
#     embs = []
#     for w in words:
#         x1,y1,x2,y2 = w["bbox"]
#         if (x1 >= proposal_bbox[0] and y1 >= proposal_bbox[1] and
#             x2 <= proposal_bbox[2] and y2 <= proposal_bbox[3]):
#             embs.append(lookup.get(w["text"]))
#     if not embs:
#         return np.zeros(lookup.dim, dtype=np.float32)
#     return np.mean(embs, axis=0)