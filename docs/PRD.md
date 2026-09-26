# PRD — AudioSearch: Conversation-Aware Hybrid Search for Two-Speaker Audio

> **Status:** this document describes the original *AudioSearch* design that SonicSearch started from (`audiosearch` CLI, `src/audiosearch/`, local faster-whisper ASR, `reports/`). The product goals and requirements still apply; the commands, paths and tooling it names don't exist in this repo. The implemented system is described in [ARCHITECTURE.md](../ARCHITECTURE.md), and its current numbers are in [EVALUATION.md](EVALUATION.md).


| | |
|---|---|
| **Problem** | HackerEarth 2026, Problem Statement 1: *Effective retrieval from audio transcripts* |
| **Document** | Product Requirements Document, v1.0 |
| **Status** | Implemented; results in [EVALUATION.md](EVALUATION.md) and [SUBMISSION.md](../SUBMISSION.md) |
| **Companion docs** | [TDD.md](TDD.md) (technical design) · [DATASET.md](DATASET.md) (golden dataset card) · [AGENT_COLLABORATION.md](AGENT_COLLABORATION.md) |

---

## 1. Summary

AudioSearch lets a user search a collection of two-person conversations (interviews, podcasts) by
**exact words** *or* **meaning**. Every result lands on **the moment**: the file, the timestamp
where the relevant sentence starts, and **who** said it (anonymous speaker label, inferred
host/guest role, and a display name when metadata provides one). The matched words are highlighted,
and a player jumps straight to that second.

All embedding and indexing run locally (PostgreSQL + pgvector). Transcription runs locally as well
(faster-whisper). The brief would also allow a hosted service for that step. Retrieval quality is proven by an
automated Recall@K test suite over a labelled golden set, which also gates CI.

## 2. Problem & opportunity

Audio is opaque to text search. What someone said is spread over long recordings, split between two
voices, and filtered through imperfect speech recognition. A useful search system has to solve four
things at once:

1. **Vocabulary mismatch.** Users paraphrase ("people who get lost in their own neighborhood") while
   speakers use jargon ("developmental topographical disorientation"). → *semantic retrieval*
2. **Exactness.** Names, numbers and technical terms must be matched literally ("WB-57", "cesium"),
   and embeddings are unreliable on them. → *lexical retrieval*
3. **Recognition errors.** ASR mangles exactly the rare words people search for ("apheresis" →
   "aphoresis"; "Abba Zubair" → "Abizubair"). Neither lexical nor semantic search recovers these on its own.
   → *sounds-like retrieval* (novel here)
4. **Provenance.** A hit is only useful if it says *where* and *who*, precisely enough to press play.
   → *moment localisation + diarization + role inference*

## 3. Users and jobs-to-be-done

| Persona | Job | Typical query |
|---|---|---|
| Researcher / journalist | "Find where the guest explains X, so I can quote it." | `why do clocks run faster on Mars?` · `role:guest` |
| Producer / editor | "Find every mention of a term for a clip reel." | `"firing room one"` · `WB-57` |
| Analyst | "Which conversations touch topic Y?" | `mentors who shaped their career` |
| Anyone half-remembering | "I think the name sounded like…" | `Zubaire`, `vestibuler system` |
| Operator | "Add recordings and keep the index correct." | `audiosearch ingest && audiosearch index` |

### User stories

* **Search.** Search an exact term and get every utterance containing it. Search conceptually and get
  paraphrases. See file, timestamp and speaker. Restrict to what the host or the guest said. Recover
  from my own misspellings and the ASR's. Click a result and hear that moment.
* **Ingestion.** Add a recording and get searchable, speaker-attributed segments automatically,
  locally. Re-running is idempotent and costs nothing if nothing changed.
* **Evaluation.** Measure Recall@K automatically on a labelled set. Compare lexical, dense and hybrid.
  Measure every design choice (chunking, embedding model, ASR model, diarization smoothing) on the
  same labels, and see upstream quality (WER, speaker attribution) next to retrieval quality.

## 4. Goals and non-goals

### Goals (all required, all measured)

| ID | Goal | Evidence |
|---|---|---|
| G1 | Golden dataset: 5–6 recordings, 8–10 min, 2 speakers each, unique pairs | `tests/unit/test_metrics_and_golden.py::test_dataset_contract` |
| G2 | Timestamped, speaker-attributed transcripts generated from the audio | `data/transcripts/*`, `audiosearch eval stages` |
| G3 | Hybrid lexical + semantic search with file, timestamp, speaker and text in every result | API/CLI/UI, `tests/integration` |
| G4 | Local embedding generation and vector indexing | bge / e5 via sentence-transformers, pgvector HNSW |
| G5 | Automated Recall@K tests on a labelled query set | `tests/eval/test_retrieval_quality.py` (CI gate) |
| G6 | The differentiators of §5, each shown to help (or reported honestly if not) | ablation tables in EVALUATION.md |
| G7 | Production readiness: one-command deploy, observability, safe operations | §7, `docker-compose.yml`, `/metrics` |

### Non-goals

* Training ASR, diarization or embedding models from scratch (we compose strong open models).
* Speaker **identity** recognition by voice. Names come only from metadata; the system infers roles, not identities.
* Overlapping-speech separation (each word gets exactly one speaker).
* Multilingual search (English only; the pipeline is language-agnostic by configuration).
* Thousands of hours in this submission. The scale path is designed and documented ([TDD §10](TDD.md#10-scaling-plan)), not benchmarked.

## 5. Differentiators (the novelty)

Standard retrieval over transcripts (BM25 + embeddings + rank fusion) meets the brief but does not set
a system apart. The six pillars below exploit what is *specific* to two-speaker audio: speech
recognition errors, conversational structure, and the need to land on an exact second. Each has a
falsifiable hypothesis and an acceptance metric. The evaluation reports all of them, including any
that fail.

| ID | Pillar | Hypothesis | Acceptance metric |
|---|---|---|---|
| **N1** | **Sounds-like channel.** Phonetic (Metaphone / Double Metaphone) + orthographic (trigram, Levenshtein, substring) expansion against the *spoken vocabulary*. A lexicon-aware policy avoids touching ordinary words. | Rare names and jargon, the terms people search for most, are where ASR and typing fail; exact BM25 misses them and embeddings don't model them. | Recall@5 on `misspelled` queries: +≥10 pts vs the same system without expansion. No loss on `keyword`/`phrase`. |
| **N2** | **IDF-coverage-weighted fusion.** Scale each passage's lexical RRF contribution by the share of the query's information content (IDF mass) it matched. | Plain RRF lets BM25 hits on filler words ("people", "world") outrank the right semantic hit. Coverage is a calibrated, per-document query-performance signal. | Hybrid ≥ best single channel on Recall@10 and MRR. Paraphrase Recall@5 recovers to dense-only levels. |
| **N3** | **Moment localisation.** Snap every passage hit to its best single-speaker utterance using lexical + semantic evidence. Return the exact time of the first matched word. Suppress near-duplicates in time (temporal NMS). | Users need the second to press play and the speaker who said it. Passage-level hits are imprecise and redundant. | Recall@5 with snapping > without (passage start). Median start offset ≤ 5 s. |
| **N4** | **Role-aware conversation model.** Infer host/guest from question rate, talk share and turn-taking. Filter by role (`role:guest`). Display metadata names when role confidence is high. | Two-party conversations have a strong structural signature; roles make results readable and filterable without voice identification. | Role accuracy 6/6 vs human transcripts. Role-scoped queries answerable. |
| **N5** | **Linguistically-aware diarization.** ECAPA embeddings + 2-speaker spectral clustering + Viterbi smoothing whose switch cost drops at sentence ends, ASR segment ends and pauses. Fully local, no gated models. | Turns change at sentence boundaries. Using ASR punctuation removes speaker "flicker" that window voting produces. | Word-level speaker attribution ≥ 95% vs human transcripts, and better than raw window voting. |
| **N6** | **Chunking-invariant, stage-aware evaluation.** Time-interval labels, dev/test split, per-category analysis, CIs and significance tests. ASR WER and speaker accuracy against NASA's human transcripts. | Retrieval quality depends on every upstream stage; a good evaluation must isolate each one and survive design changes. | Every design choice in this PRD is backed by an ablation on the same labels. |

**Tested but not adopted.** *Dialogue-context augmentation* prepends the interviewer's question to
answer passages before embedding. It sounds compelling for Q&A audio. On our dev split it did not
help, because guests tend to restate the topic, so it is off by default and kept as an option
(`AUDIOSEARCH_CHUNK_CONTEXT=question`). We report the result either way.

## 6. Functional requirements

| ID | Requirement | Acceptance criteria |
|---|---|---|
| FR-01 | Golden audio corpus | 5–6 files; 8–10 min each; exactly two speakers each; no speaker in two files; redistributable licence; provenance and offsets recorded; rebuildable from source (`scripts/build_dataset.py`). |
| FR-02 | Audio ingestion | ffprobe validation (decodable, non-empty); stable `file_id`; metadata: duration, sample rate, channels, codec, sha256, provenance. |
| FR-03 | ASR with timestamps | Word-level timestamps: every indexed word has text and start/end times. Model and parameters recorded with the output; content-addressed cache keyed by audio hash and model. |
| FR-04 | Diarization | Every word gets a speaker label and a confidence; labels consistent within a file; method and failure modes documented; accuracy measured against human transcripts. |
| FR-05 | Transcript/speaker alignment | Canonical transcript = words → single-speaker utterances → turns, with timestamps mapping back to the audio. |
| FR-06 | Retrieval chunking | Configurable windows of whole utterances with overlap. Chunk timestamps map back to the audio. Chunk size chosen by experiment on the dev split. |
| FR-07 | Lexical search | Exact words and phrases retrievable with real BM25 (IDF + length normalisation), not `ts_rank`. Quoted phrases are strict. |
| FR-08 | Semantic search | Local embedding model; local vector index (pgvector HNSW); query and document prefixes per model family. |
| FR-09 | Hybrid fusion | Both channels run for every query and fuse deterministically with IDF-coverage-weighted RRF (N2); `k`, depth and weights configurable; convex combination available for ablation. |
| FR-10 | Result presentation | Each result has file, timestamp (moment start), speaker label, role and name, matched text with highlights, surrounding passage, per-channel ranks ("why this result"), and a playable audio URL. |
| FR-11 | Golden query set | Relevance as audio-time intervals + verbatim quotes, exhaustive per query, graded (2/1). ≥ 60 queries over ≥ 6 categories. Dev/test split. Frozen before tuning. |
| FR-12 | Recall@K evaluation | Recall@1/3/5/10 plus MRR, nDCG@10, Precision@K, Success@K. Bootstrap CIs, paired permutation tests, per-category breakdown. |
| FR-13 | Retrieval ablation | Lexical-only vs dense-only vs hybrid. Each differentiator on/off; index-time variants (chunk size, context, embedding model, ASR model). |
| FR-14 | Reproducibility | Clone → `docker compose up` → working UI; or `make setup && make build && make eval`. Committed transcripts mean retrieval results reproduce without ASR hardware. |
| FR-15 | Coding-agent disclosure | How the agent was directed, what it produced, and how its output was verified ([AGENT_COLLABORATION.md](AGENT_COLLABORATION.md)). |
| FR-16 | Sounds-like expansion | Out-of-vocabulary names and typos expand to phonetically or orthographically similar spoken terms. Common words are expanded only on strong phonetic identity. Expansions are shown to the user. |
| FR-17 | Role-aware search | Host/guest inferred per file with a confidence. `role:` and `speaker:` filters in both the query syntax and the API. |
| FR-18 | Moment localisation | Result start = best matching utterance; `match_time` = first matched word; temporal NMS removes near-duplicates. |
| FR-19 | Incremental, idempotent indexing | Re-indexing an unchanged file is a no-op. Add, replace or delete of one file costs O(file). BM25 statistics stay exactly consistent (tested invariant). |
| FR-20 | Interfaces | CLI (`audiosearch search`), HTTP API (`/api/search`, OpenAPI docs), and a web UI with synchronized transcript playback. |

## 7. Non-functional requirements

The brief asks what would matter if the system were brought to production. These requirements
answer that in the implementation, not only in prose.

| ID | Area | Requirement |
|---|---|---|
| NFR-01 | Local execution | Embedding, indexing and search run on a laptop-class CPU. ASR is local too (GPU optional). |
| NFR-02 | Latency | p95 search latency ≤ 300 ms on 4 CPU cores without reranking (the reranker adds ~250 ms; it is opt-in). Warm p50 target ≤ 60 ms. |
| NFR-03 | Determinism | Same index + model + query → same ranking (stable tie-breaks; no sampling). |
| NFR-04 | Reproducibility | Pinned model names in config and recorded in outputs. The migration refuses to reuse an index built with a different embedding model or dimension. |
| NFR-05 | Idempotency & consistency | Ingestion and indexing can be re-run safely. Per-file transactions plus an advisory lock mean readers never see half-indexed files. |
| NFR-06 | Observability | Structured JSON logs with request IDs. Prometheus metrics (per-stage latency, zero-result rate, expansions, HTTP). `/healthz` and `/readyz`. |
| NFR-07 | Security | Parameterised SQL only. Input validation (length, enums). No path traversal on media. CSP and security headers on the UI. Non-root container. No secrets in the repo. |
| NFR-08 | Portability | Standard Postgres extensions only (`vector`, `pg_trgm`, `fuzzystrmatch`), all available on managed Postgres (RDS, Cloud SQL, Azure, Supabase, Neon). |
| NFR-09 | Extensibility | ASR, diarizer, embedder and reranker are replaceable behind small interfaces. Index-time variants live in isolated schemas (blue/green re-indexing). |
| NFR-10 | Privacy | No voice identification. Names only from operator-supplied metadata. |

## 8. Golden dataset requirements

* Public-domain / redistributable two-speaker interviews with a human reference transcript. The
  reference makes stage-level measurement possible.
* Distinct speaker pairs. Topical diversity with cross-file confusers. Realistic conversational audio.
* The query set covers exact terms, phrases, paraphrases, questions, multi-file topics, misspellings
  (including real ASR errors) and role-scoped queries.
* See [DATASET.md](DATASET.md) for the implemented dataset card.

## 9. Evaluation strategy and success criteria

**Offline protocol.** Queries were frozen before tuning. All tuning happened on the dev split
(29 queries). Final numbers are reported once on the held-out test split (52 queries). A result
*hits* a labelled moment if it is in the same file and its time span overlaps the labelled interval
±5 s. Each labelled moment is credited once. Labels are audio-time intervals rather than segment or
chunk IDs: IDs are an artefact of one chunking run, so changing the chunk size, ASR model or
diarization would silently point them at the wrong text. Time intervals stay valid, so one golden
set compares *any* pipeline configuration fairly. Together with ablations, confidence intervals and
significance tests, this protocol is what makes claims such as "hybrid beats either channel" credible.

| ID | Success criterion (test split unless stated) | Target |
|---|---|---|
| SC-1 | Recall@5 / Recall@10 of the default system | ≥ 0.80 / ≥ 0.90 |
| SC-2 | Hybrid vs best single channel (Recall@10 and MRR) | hybrid ≥ both |
| SC-3 | Exact terms: Recall@5 on `keyword` + `phrase` | ≥ 0.90 |
| SC-4 | Meaning: Recall@10 on `paraphrase` | ≥ 0.80 and ≥ BM25 + 0.2 |
| SC-5 | N1 sounds-like: Recall@5 gain on `misspelled` | ≥ +0.10 |
| SC-6 | N3 moments: Recall@5 gain from snapping; median start offset | > 0; ≤ 5 s |
| SC-7 | N4/N5: word-level speaker attribution; role accuracy | ≥ 95%; 6/6 |
| SC-8 | ASR WER vs human reference (clean verbatim, so an upper bound) | ≤ 10% |
| SC-9 | Latency (NFR-02): p95 without reranking, 4-core CPU | ≤ 300 ms |
| SC-10 | Reproducibility: CI runs unit, integration and Recall@K gates from a clean clone | green |

**Metrics we would add in production** (online; see TDD §9.4): zero-result rate, click-through and
*play-through* at rank (did the user keep listening after the jump?), query reformulation rate, time to
first play, abandonment, p50/p95/p99 latency per stage, ingestion lag (audio-hours per hour, freshness),
index consistency checks, cost per audio hour. Plus interleaved A/B tests for ranking changes, and a
periodically refreshed human-judged sample (optionally LLM-assisted and then human-verified) so the golden
set grows with real queries.

## 10. UX

**Result anatomy:** `▶ 06:17.8` · file title · `Guest · Abba Zubair` → the utterance with highlighted
matches → collapsible passage context → "why" badges (`keyword #2 · semantic #1 · match @ 06:19.2`).
If the query was expanded: *"Sounds-like: also matched 'zubaire' → 'zubair'"*.

**Flows:**
1. *Search → play.* Type a query and choose mode (Hybrid / Keyword / Semantic), role and file. Click ▶:
   the audio loads, jumps to the moment, and the transcript panel follows playback with speaker colours.
2. *Operator.* `audiosearch ingest` (ASR + diarization, cached) → `audiosearch index` (idempotent) →
   `audiosearch eval run` (report) → deploy with `docker compose up`.

## 11. Risks, failure modes and mitigations

| Failure mode | Mitigation in this design | Residual risk |
|---|---|---|
| ASR errors break exact search | Sounds-like expansion (N1); dense channel; strong ASR model | Errors that are neither phonetically nor semantically close ("cod blood") |
| Speaker overlap | One label per word; boundary-aware smoothing | Overlapping speech is attributed to one speaker |
| Similar voices | Objective difficulty metric (inter-speaker cosine) recorded per file | Accuracy drops when voices are very similar |
| Very short segments | Back-channels are embedded but down-weighted for semantic snapping | Short turns can be mis-attributed (1 known case) |
| Chunk-boundary loss | 50% overlapping windows; moment snapping picks the best utterance | Answers longer than a window are split |
| Duplicate results | Temporal NMS | — |
| Exact-vs-semantic conflict | Coverage-weighted fusion (N2); strict quoted phrases | Name queries can still pull weak semantic neighbours at low ranks |
| Metadata drift | Content-addressed caches; index signatures; schema/model check at start-up | — |
| Embedding-model change | Migration refuses dimension/model mismatch; blue/green schemas | Re-embedding cost at scale |

## 12. Milestones (status)

| Milestone | Status |
|---|---|
| M1 Dataset & audio pipeline | ✅ 6 recordings, ASR (2 models), diarization, reference transcripts |
| M2 Unified transcript store | ✅ canonical transcript JSON, Postgres schema + migrations |
| M3 Retrieval MVP | ✅ BM25 in SQL, pgvector HNSW, CLI/API |
| M4 Hybrid retrieval | ✅ coverage-weighted RRF, sounds-like expansion, moments, NMS, optional reranker |
| M5 Evaluation | ✅ 81 labelled queries, Recall@K gates, ablations, stage metrics |
| M6 Demo & submission | ✅ web UI, Docker, CI, docs, agent disclosure |

## 13. Traceability: problem statement → requirement → evidence

| Problem statement asks for… | PRD | Where it is satisfied |
|---|---|---|
| 5–6 audio files, 8–10 min, unique two-speaker pairs | FR-01 | `data/audio`, `data/manifest.yaml`, dataset contract test |
| Hybrid search: specific words *and* semantically similar terms | FR-07–09, FR-16 | `src/audiosearch/search/*` |
| Diarization, database, embeddings, indexing strategies for "ideal" quality at scale | N1–N6, TDD §4–§10 | TDD; ablations in EVALUATION.md |
| Transcripts generated from each file (hosted or local) | FR-03 | local faster-whisper; `data/transcripts` |
| Results show file, timestamp, speaker | FR-10, FR-18 | API/CLI/UI; integration tests |
| Local embedding generation and indexing | FR-08, NFR-01 | sentence-transformers + pgvector |
| Automated tests measuring Recall@K on a labelled query set | FR-11–12 | `tests/eval/test_retrieval_quality.py`, CI |
| Metrics for production; what an effective evaluation looks like | §9, TDD §9.4 | this document + EVALUATION.md |
| Design explanation, rationale, success criteria, achievement, limitations | — | [SUBMISSION.md](../SUBMISSION.md) |
| Coding-agent disclosure | FR-15 | [AGENT_COLLABORATION.md](AGENT_COLLABORATION.md) |

## 14. Future extensions

Streaming or near-real-time ingestion. GPU batch ASR with word-level forced alignment. Overlap-aware
diarization (e.g. pyannote 3.x when a licensed token is available: the diarizer interface already
isolates it). Learned fusion weights once real click logs exist. Query-by-example audio (voice or
sound-event retrieval). Speaker enrolment with explicit consent. Multilingual models. BM25 via
`pg_search` or OpenSearch past ~10M passages. Binary or half-precision vector quantisation
with re-scoring. Topic segmentation and chaptering.
