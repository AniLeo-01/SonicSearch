"""Score search against the golden queries (see docs/EVALUATION.md).

uv run python -m app.evaluate                 # dev split: tuning allowed
uv run python -m app.evaluate --split test    # held-out split: run once per decision, never tune on it
"""
import argparse
import time
from collections import Counter, defaultdict

import yaml
from sentence_transformers import CrossEncoder

from app.config import settings
from app.db import make_pool
from app.embed import ColBERT, Embedder
from app.search import Searcher

KS = (1, 3, 5, 10)  # recall@k cut-offs; searches return 10 results


def hits_label(h, label: dict, tol: float) -> bool:
  """A hit (the moment's sentence) counts if it overlaps a labelled interval of the same file, give or take `tol`."""
  return label["file"] == h.file_id and h.start <= label["end"] + tol and h.end >= label["start"] - tol


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--split", choices=["dev", "test", "all"], default="dev")
  args = ap.parse_args()
  ev = yaml.safe_load((settings.data_dir / "eval" / "queries.yaml").read_text(encoding="utf-8"))
  tol, queries = ev["tolerance_sec"], [q for q in ev["queries"] if args.split in ("all", q["split"])]
  by: defaultdict[str, Counter] = defaultdict(Counter)
  with make_pool() as pool:
    emb = Embedder(settings.embed_model)
    s = Searcher(pool, emb, CrossEncoder(settings.rerank_model, device="cpu"), ColBERT(settings.colbert_model))
    s.search("warm up")  # first call pays for lazy initialisation
    for q in queries:
      t = time.perf_counter()
      hits = s.search(q["query"], k=10, role=q.get("role"))
      ms = 1000 * (time.perf_counter() - t)
      ok = [any(hits_label(h, r, tol) for r in q["relevant"]) for h in hits]
      for c in (by["ALL"], by[q["category"]]):
        c["queries"] += 1
        c["ms"] += ms
        for h, good in zip(hits, ok, strict=True):
          c["related" if h.related else "match"] += 1
          c["related_ok" if h.related else "match_ok"] += good
        for k in KS:  # a label is found at k if one of the first k results covers it
          c[f"found@{k}"] += sum(any(hits_label(h, r, tol) for h in hits[:k]) for r in q["relevant"])
          c[f"best@{k}"] += min(k, len(q["relevant"]))
        c["labels"] += len(q["relevant"])
        c["rr"] += next((1 / h.rank for h, good in zip(hits, ok) if good), 0.0)
  ratio = lambda a, b: f"{a / b:.2f}" if b else "  - "  # noqa: E731
  print(f"{args.split} split, {len(queries)} queries")
  print(f"{'':12s} {'queries':>7} {'match P':>7} {'matches':>7} {'related P':>9}" + "".join(f" {f'R@{k}':>5}" for k in KS)
        + f" {'MRR@10':>6} {'ms/query':>8}")
  for name, c in by.items():
    print(
      f"{name:12s} {c['queries']:7d} {ratio(c['match_ok'], c['match']):>7} {c['match']:7d} {ratio(c['related_ok'], c['related']):>9}"
      + "".join(f" {ratio(c[f'found@{k}'], c['labels']):>5}" for k in KS)
      + f" {c['rr'] / c['queries']:6.2f} {c['ms'] / c['queries']:8.0f}"
    )
  a = by["ALL"]  # one result can cover one label, so recall@k is capped when a query has more than k labels
  print("best possible: " + "  ".join(f"R@{k} {ratio(a[f'best@{k}'], a['labels'])}" for k in KS))


if __name__ == "__main__":
  main()
