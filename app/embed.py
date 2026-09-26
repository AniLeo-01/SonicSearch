"""Local text embeddings (sentence-transformers, BGE): normalised vectors for passages and queries."""

import threading
import numpy as np
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "  # BGE: prefix queries only

class Embedder:
  def __init__(self, name: str) -> None:
    from sentence_transformers import SentenceTransformer
    self.model = SentenceTransformer(name, device="cpu")
    self._lock = (
        threading.Lock()
    )  # the tokenizer is not safe to call from several API threads at once

  def docs(self, texts: list[str]) -> np.ndarray:
    with self._lock:
      return self.model.encode(
        texts, batch_size=32, normalize_embeddings=True, convert_to_numpy=True
      )

  def query(self, text: str) -> np.ndarray:
    return self.docs([QUERY_PREFIX + text])[0]

  def lexicon(self):
    """Whole words in the model's WordPiece vocabulary (~20k): a free list of common English words."""
    vocab = self.model.tokenizer.get_vocab()
    return frozenset(
      t for t in vocab if t.isalpha() and t.islower() and len(t) >= 3
    )


class ColBERT:
  """Late interaction: one normalised vector per token, scored by MaxSim. Reproduces PyLate's encoding for
  lightonai/GTE-ModernColBERT-v1 without PyLate, which pins older sentence-transformers/transformers."""

  def __init__(self, name: str) -> None:
    import json
    from pathlib import Path

    import torch
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    from transformers import AutoModel, AutoTokenizer

    cfg = json.loads(Path(hf_hub_download(name, "config_sentence_transformers.json")).read_text())
    self.tok = AutoTokenizer.from_pretrained(name)
    self.model = AutoModel.from_pretrained(name).eval()
    self.proj = load_file(hf_hub_download(name, "1_Dense/model.safetensors"))["linear.weight"]  # 128 x hidden, no bias
    self.max_len = {True: cfg["query_length"], False: cfg["document_length"]}
    self.prefix = {True: self.tok.convert_tokens_to_ids(cfg["query_prefix"]), False: self.tok.convert_tokens_to_ids(cfg["document_prefix"])}
    self.skip = torch.tensor([self.tok.convert_tokens_to_ids(w) for w in cfg["skiplist_words"]])  # punctuation
    self._lock = threading.Lock()  # the tokenizer is not safe to call from several API threads at once

  def encode(self, texts: list[str], is_query: bool) -> list[np.ndarray]:
    if not texts:
      return []
    import torch

    with self._lock:
      b = self.tok([t.strip() for t in texts], padding=True, truncation=True, max_length=self.max_len[is_query] - 1, return_tensors="pt")
    n = len(texts)
    # [CLS] [Q]/[D] tokens...: the prefix token goes right after the first token
    ids = torch.cat([b["input_ids"][:, :1], torch.full((n, 1), self.prefix[is_query]), b["input_ids"][:, 1:]], 1)
    att = torch.cat([b["attention_mask"][:, :1], torch.ones((n, 1), dtype=b["attention_mask"].dtype), b["attention_mask"][:, 1:]], 1)
    with torch.inference_mode():
      h = self.model(input_ids=ids, attention_mask=att).last_hidden_state @ self.proj.T
    keep = att.bool() if is_query else att.bool() & ~torch.isin(ids, self.skip)  # passages also drop punctuation
    return [torch.nn.functional.normalize(h[i][keep[i]], dim=1).numpy() for i in range(n)]


def maxsim(q: np.ndarray, doc: np.ndarray) -> float:
  """Each query token's best match among the passage's tokens, summed."""
  return float((doc.astype(np.float32) @ q.T).max(0).sum())
