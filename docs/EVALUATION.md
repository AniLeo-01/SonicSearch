# Evaluation

How SonicSearch's search quality is measured, the rules for tuning against the query set, the current numbers, and the experiments behind each design decision. The dataset itself is described in [DATASET.md](DATASET.md).

## What is evaluated

The full `/api/search` path (`app/search.py`), with the models loaded as in production:

1. **Candidates:** keyword search (BM25 over Postgres lexemes, plus sounds-like expansion for typos and ASR spellings) and meaning search (BGE embeddings in pgvector), merged by reciprocal rank fusion. The top 30 go on.
2. **Matches:** the cross-encoder `gte-reranker-modernbert-base` scores each candidate against the query. Scores ≥ `rerank_min` (0.83), or hits containing every query word, are **matches**, ordered by that score.
3. **Related moments:** everything else is kept as context, ordered by ColBERT (`GTE-ModernColBERT-v1`) similarity to the query. For short keyword queries (at most 3 words, at most 2 of them content words) with a match, it uses similarity to the best match's passage instead.
4. **Output:** each passage is snapped to its best sentence, moments within 10 s of a better one in the same file are dropped, and the top 10 are returned. Each result carries `related: true/false`.

## Query set

`data/eval/queries.yaml`: 81 queries over the 6 recordings (56 min), with 154 labelled time intervals (145 directly relevant, 9 partially relevant).

| Category | Dev | Test | What it probes |
|---|---|---|---|
| `keyword` | 6 | 12 | Rare terms said verbatim (`cesium`, `WB-57`) |
| `phrase` | 4 | 6 | Quoted exact phrases (`"firing room one"`) |
| `paraphrase` | 5 | 9 | Same meaning, different words |
| `question` | 5 | 10 | Natural-language questions |
| `cross_file` | 3 | 4 | Topics that span several recordings |
| `misspelled` | 4 | 7 | Typos and ASR-mangled terms (`apheresis` → ASR "ismoresis") |
| `speaker` | 2 | 4 | Host- or guest-only search |
| **Total** | **29** | **52** | |

Queries are authored in `data/eval/queries.src.yaml` as utterance ranges and converted to time intervals in `queries.yaml`, so labels don't depend on chunking. See [DATASET.md](DATASET.md#file-formats) for both formats.

## Protocol

**How the labels were made** (fixed; don't change without re-labelling):

- Queries were written by reading the transcripts **before** any retrieval tuning, then frozen.
- Relevance is **exhaustive**: every moment in the corpus that answers a query is labelled.
- Speaker-scoped queries use the **true** speaker from NASA's transcript, so diarization mistakes count as retrieval misses.
- Split: within each category, every third query is `dev` and the rest are `test`.

**Rules for experiments:**

1. Tune only on `dev`: model choice, thresholds, weights, prompts.
2. Make the decision on dev first, then run `test` **once** to confirm it.
3. If a test result changes a decision, say so in the decision log. Test is then no longer independent for that decision.
4. Record every result that changed or confirmed a decision in the [decision log](#decision-log).

## Metrics

A result is a moment: one sentence with a start and end time. It **counts as relevant** if it overlaps a labelled interval in the same file, with `tolerance_sec` (5 s) of slack on either side. Partially relevant labels count as relevant; grades aren't weighted.

| Metric | Meaning |
|---|---|
| **Match precision** | Share of results marked as matches that are relevant. The number that matters for the top of the page. |
| **Matches** | How many results were marked as matches, across all queries. At equal precision, more is better. |
| **Related precision** | The same for the related section. It undercounts: labels mark direct answers, not topical context, so check related sections by eye too. |
| **Recall@k** | Share of labelled intervals covered by at least one of the top *k* results (k = 1, 3, 5, 10), matches and related together. One result covers one moment, so recall@k can't reach 1.0 when a query has more than *k* labels. The evaluator prints that ceiling (dev: 0.50 at k = 1, 0.90 at 3, 0.98 at 5, 1.00 at 10). |
| **MRR@10** | Mean of 1 / (rank of the first relevant result); 0 if none. |
| **ms/query** | Wall time per search, on CPU (Apple Silicon Mac), after warm-up. |

Precision over all 10 results isn't reported. Queries have about 2 labelled moments each, so a full list of 10 is at least ~80% "irrelevant" by construction. That's why results are split into matches and related moments.

## Running it

Prerequisites: Postgres with pgvector running (see `.env.example`), transcripts built (`python -m app.ingest`), and the index built. The first run downloads the BGE, gte and ColBERT models (about 1.6 GB).

```sh
uv run python -m app.index                   # after changing chunking, embeddings or models
uv run python -m app.evaluate                # dev split (about 1 min)
uv run python -m app.evaluate --split test   # test split, once per decision (about 1.5 min)
```

It prints overall and per-category numbers, plus the best possible recall@k for the split. The query labels only cover the six dataset recordings, so results are comparable only when the library is exactly those six: uploading or removing recordings in the web app changes the numbers.

## Current results

2026-09-26, current configuration (`rerank_min=0.83`; related moments ordered by the best match for keyword queries, by the query otherwise).

**Dev (29 queries)**

| | Match P | Matches | Related P | R@1 | R@3 | R@5 | R@10 | MRR@10 | ms/query |
|---|---|---|---|---|---|---|---|---|---|
| **All** | **0.45** | 134 | 0.08 | **0.45** | **0.74** | **0.88** | **0.97** | **0.94** | 1155 |
| keyword | 0.42 | 19 | 0.02 | 0.86 | 1.00 | 1.00 | 1.00 | 1.00 | 1448 |
| phrase | 1.00 | 12 | – | 0.33 | 0.83 | 1.00 | 1.00 | 1.00 | 270 |
| paraphrase | 0.22 | 27 | 0.13 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1271 |
| question | 0.31 | 29 | 0.05 | 0.60 | 1.00 | 1.00 | 1.00 | 0.77 | 1253 |
| cross_file | 0.29 | 24 | 0.00 | 0.25 | 0.38 | 0.62 | 0.88 | 0.78 | 1272 |
| misspelled | 0.75 | 12 | 0.18 | 0.33 | 0.58 | 0.83 | 0.92 | 1.00 | 1189 |
| speaker | 0.82 | 11 | 0.00 | 0.22 | 0.67 | 0.78 | 1.00 | 1.00 | 1271 |
| *best possible* | | | | *0.50* | *0.90* | *0.98* | *1.00* | | |

**Test (52 queries)**

| | Match P | Matches | Related P | R@1 | R@3 | R@5 | R@10 | MRR@10 | ms/query |
|---|---|---|---|---|---|---|---|---|---|
| **All** | **0.53** | 195 | 0.11 | **0.46** | **0.69** | **0.80** | **0.96** | **0.88** | 1206 |
| keyword | 0.40 | 53 | 0.01 | 0.56 | 0.83 | 0.94 | 1.00 | 0.87 | 1282 |
| phrase | 1.00 | 12 | – | 0.50 | 0.92 | 1.00 | 1.00 | 1.00 | 177 |
| paraphrase | 0.45 | 29 | 0.08 | 0.60 | 0.80 | 0.90 | 1.00 | 0.74 | 1317 |
| question | 0.44 | 45 | 0.16 | 0.47 | 0.59 | 0.76 | 0.88 | 0.86 | 1352 |
| cross_file | 0.50 | 20 | 0.50 | 0.25 | 0.38 | 0.50 | 0.94 | 1.00 | 1408 |
| misspelled | 0.93 | 15 | 0.09 | 0.50 | 0.75 | 0.83 | 1.00 | 0.89 | 1358 |
| speaker | 0.62 | 21 | 0.05 | 0.36 | 0.64 | 0.73 | 0.91 | 1.00 | 1436 |
| *best possible* | | | | *0.54* | *0.88* | *0.98* | *1.00* | | |

**Reading recall@k.** Against its ceiling, recall is 90% of the best possible at k = 1 on dev (0.45 of 0.50) and 85% on test (0.46 of 0.54). By k = 5 it's 0.88 of 0.98 on dev and 0.80 of 0.98 on test. The labelled moments that are found mostly appear in the first five results. The exception is cross-file topics: their moments are spread over several recordings, so recall@5 is only 0.50–0.62 and they need the whole list of 10 (0.88–0.94). Questions and speaker queries also gain a lot between 5 and 10 on test.

Phrase queries are fast (about 0.2 s) because the exact-phrase filter leaves few candidates to rerank. Everything else spends most of its 1.2–1.4 s in the gte reranker; encoding the query with ColBERT adds about 40 ms.

## Decision log

All experiments ran on 2026-09-26, on the splits above. Numbers are overall unless a category is named, and "recall" means recall@10 (recall@1/3/5 were added to the evaluator after these runs). "Hide" means results under the cutoff were removed; later versions keep them as related moments.

| # | Change | Dev | Test | Decision |
|---|---|---|---|---|
| 1 | **Baseline:** fused keyword + meaning search, no reranker | precision@10 0.26, recall 0.91, MRR 0.93, 59 ms | precision@10 0.27, recall 0.92, MRR 0.93 | Starting point. Recall was already high; precision was the problem. |
| 2 | Drop meaning-only hits below a cosine cutoff | precision 0.26 → 0.34, losing 3 of 67 relevant hits | – | Rejected: weak separation (median cosine 0.52 relevant vs 0.40 irrelevant). |
| 3 | For keyword queries, drop meaning-only hits | keyword queries: 23/100 → 19/22 relevant, losing 4 of 23 | – | Not built: it hides meaning-based context. |
| 4 | Reranker `ms-marco-MiniLM-L6-v2` | MRR 0.87 (worse than no reranker), 200 ms | – | Rejected. |
| 5 | Reranker `gte-reranker-modernbert-base`, hide below 0.83 | MRR 0.94, 1.1 s; precision 0.45, recall 0.88 | precision 0.53, recall 0.81, 4 empty lists; `cross_file` 0.81 → 0.56, `question` 0.82 → 0.71, `misspelled` 0.92 → 0.75 | Kept as the judge of matches. |
| 6 | Always keep hits containing every query word | recall 0.88 → 0.90, no empty lists (`cesium` had returned nothing) | unchanged | Kept. |
| 7 | Give the reranker the transcript's spelling of typos (appended or replaced) | recall 0.90 → 0.83 / 0.81; non-dictionary words only: 0.84, `misspelled` 0.58 → 0.33 | – | Rejected. Keeping hits with ≥ 50–75% keyword coverage changed nothing either. |
| 8 | Laya decision model as the reranker | 7.1 s, MRR 0.93; at 131 kept hits precision 0.43 / recall 0.83 vs gte 0.45 / 0.88 | – | Rejected: never better than gte at any cutoff, about 6× slower. |
| 9 | Show under-cutoff hits as related (gte order) | match precision 0.45, recall 0.91 | – | Kept (demote, don't hide). |
| 10 | Related moments in pre-reranker (fused) order | recall 0.91 → 0.95, related precision 0.05 → 0.08 | recall 0.93, MRR 0.90 | Superseded by #11. |
| 11 | Related moments ordered by ColBERT similarity to the query | recall 0.97 | recall 0.96 (`cross_file` 0.81 → 0.94, `question` 0.82 → 0.88), related precision 0.12, MRR 0.88 | Kept, then changed in #13. |
| 12 | ColBERT as the only reranker (cutoff 11.78, from dev) | 105 matches at precision 0.43, MRR 0.89 | 127 matches at precision 0.52, MRR 0.94 | Rejected: finds fewer matches than gte at the same precision; MRR mixed. |
| 13 | Related moments ordered by ColBERT similarity to the **best match** | unchanged: 0.45 / 0.09 / 0.97 / 0.94 | recall 0.96 → **0.93** (`cross_file` 0.94 → 0.75) | Fixed one-word queries (below) but hurt cross-file topics: narrowed in #14. |
| 14 | Best-match order only for keyword queries (≤ 3 words, ≤ 2 content words); query order otherwise (current) | unchanged: 0.45 / 0.08 / 0.97 / 0.94 | recall 0.93 → **0.96** (`cross_file` 0.75 → 0.94), related precision 0.11, MRR 0.88 | Kept. **Chosen after seeing #13's test result**, so test isn't independent for this decision; dev was checked first and didn't change. |

Example for #13, still how keyword queries behave under #14: `cesium`'s related moments went from an off-topic job title ("Chief Science Data Officer") at #1 to atomic-clock and timekeeping moments. `Guppy` and `WB-57` went from mostly other recordings to the runway episode.

ColBERT is run by hand-written code in `app/embed.py`, because PyLate pins older `sentence-transformers` and `transformers`. It matches PyLate's scores to within 0.0004.

## Open issues

- **Cross-file topics rank late:** their moments are spread across recordings, so recall@5 is 0.50–0.62 against 0.88–0.94 at 10. More results per query, or diversifying the top of the list across recordings, would help.
- **Misspelled queries:** the reranker reads the typo, not the transcript's spelling, so some correct hits fall below the cutoff and only appear as related moments (dev `misspelled` related precision 0.18; #7 found no fix).
- **Questions that refer to a person indirectly** ("How long has *she* worked at Kennedy Space Center?"): the reranker can't tell who "she" is and scores the right passages below the cutoff. With results hidden (#5), two such test queries returned nothing.
- **Latency:** about 1.2 s per query on CPU, mostly the reranker.
