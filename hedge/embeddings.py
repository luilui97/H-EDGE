"""Corpus-aligned NV-Embed-v2 document and occurrence similarities.

The encoder is supplied by the runtime environment. The same adapter must expose
encode_query(text), encode_passages(texts), and model_id='nvidia/NV-Embed-v2'.
It owns the model's retrieval instruction and normalization configuration.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


def texts_hash(texts):
    return hashlib.sha256(json.dumps(texts, ensure_ascii=False).encode("utf-8")).hexdigest()


def unit(matrix):
    matrix = np.asarray(matrix, dtype=np.float32)
    if not np.isfinite(matrix).all():
        raise ValueError("non-finite embedding")
    norms = np.linalg.norm(matrix, axis=-1, keepdims=True)
    if (norms <= 1e-12).any():
        raise ValueError("zero embedding")
    return matrix / norms


class NVEmbedRetrieval:
    def __init__(self, memory, folder, encoder):
        folder = Path(folder)
        manifest = json.loads((folder / "passages.manifest.json").read_text(encoding="utf-8"))
        self.ids = sorted(memory.documents)
        texts = [memory.documents[d]["title"] + "\n" + memory.documents[d]["text"] for d in self.ids]
        config = manifest["configuration"]
        if config["texts_sha256"] != texts_hash(texts):
            raise ValueError("passage embeddings do not match source text/order")
        if (getattr(encoder, "model_id", None) != "nvidia/NV-Embed-v2"
                or config["encoder"] != encoder.model_id):
            raise ValueError("H-EDGE requires matching NV-Embed-v2 passage/query identities")
        self.passages = unit(np.load(folder / "passages.npy", allow_pickle=False))
        if self.passages.ndim != 2 or len(self.passages) != len(self.ids):
            raise ValueError("passage rows must match source documents")
        self.encoder, self.fact_vectors, self.query_vectors = encoder, {}, {}

    def query(self, text):
        if text not in self.query_vectors:
            vector = unit(self.encoder.encode_query(text))
            if vector.shape != (self.passages.shape[1],):
                raise ValueError("query/passages embedding dimensions differ")
            self.query_vectors[text] = vector
        return self.query_vectors[text]

    def scores(self, text):
        return {d: float(s) for d, s in zip(self.ids, self.passages @ self.query(text))}

    def score_facts(self, query, facts):
        # Atomic and recursive facts expose the same relational representation.
        texts = {f["id"]: json.dumps({k: f[k] for k in ("predicate", "roles", "qualifiers")},
                                     sort_keys=True, ensure_ascii=False) for f in facts}
        missing = list(dict.fromkeys(t for t in texts.values() if t not in self.fact_vectors))
        if missing:
            vectors = unit(self.encoder.encode_passages(missing))
            if vectors.shape != (len(missing), self.passages.shape[1]):
                raise ValueError("occurrence embedding dimensions differ")
            self.fact_vectors.update(zip(missing, vectors))
        q = self.query(query)
        return {fid: float(self.fact_vectors[text] @ q) for fid, text in texts.items()}
