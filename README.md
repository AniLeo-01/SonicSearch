# SonicSearch

[![CI/CD](https://github.com/AniLeo-01/SonicSearch/actions/workflows/ci.yml/badge.svg)](https://github.com/AniLeo-01/SonicSearch/actions/workflows/ci.yml)

Search what was said in audio recordings. Type a word, a phrase or a question, and SonicSearch returns the **moments** that match: which recording, the time to play from, who said it (host or guest, by name when known), and the sentence with the matching words highlighted. Click a result and the player jumps there.

It combines keyword search (BM25, with sounds-like matching for typos and misheard names) with meaning search (sentence embeddings) in one PostgreSQL database. A cross-encoder reranker then decides what counts as a confident **match**. Everything else it finds is still shown, as **related moments** ordered by ColBERT late interaction.

On the golden query set (81 queries over six NASA podcast interviews), the first result alone finds 85–90% of what it could, and the top 10 find 96–97% of all labelled moments. See [Evaluation](#evaluation).

## Quick start with Docker

Needs Docker with at least 4 GB of memory.

```sh
cp .env.example .env   # set GROQ_API_KEY to upload or transcribe new recordings; search works without it
docker compose up --build
```

Open <http://localhost:8000>.

- **First start:** downloads about 1.7 GB of models into a volume and indexes `data/` (about 30 s). Later starts skip both unless recordings changed outside the app. After editing transcripts on the host, run `docker compose exec app python -m app.index`.
- **Another port:** set `PORT` in `.env`.
- **Stop:** `docker compose down`. Add `-v` to also delete the model cache and the database, which is rebuilt from `data/` on the next start.

The app is only reachable from this machine: there's no login, so don't expose it publicly as is.

**CI/CD** ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)):
- **Every pull request and push:** runs the tests, then builds the image and smoke-tests it (CPU-only torch, `torchaudio` loads, the app imports).
- **Pushes to `main`:** also publish the image to `ghcr.io/anileo-01/sonicsearch`, tagged `latest` and with the commit SHA.

## Run locally

Needs Python 3.13, [uv](https://docs.astral.sh/uv/), ffmpeg, and PostgreSQL with `pgvector` (`pg_trgm` and `fuzzystrmatch` ship with Postgres).

```sh
cp .env.example .env                 # set DB_URL (and GROQ_API_KEY for new recordings)
uv sync
uv run python -m app.index           # build the search index from data/transcripts
uv run uvicorn app.main:app          # http://127.0.0.1:8000
```

Transcripts for the six dataset recordings are in `data/transcripts/`. If they're missing, run `uv run python -m app.ingest` first; it transcribes with Groq, labels the speakers and caches everything, so it only calls Groq once per file.

## Using it

**In the browser**
- **Search box:** plain words, `"a quoted phrase"` for an exact phrase, or a question.
- **Mode:** Hybrid (default), Keywords only, or Meaning only.
- **Speaker:** anyone, host or guest. You can also type `role:host` or `role:guest` in the query.
- **Results:** confident matches come first, then **Related moments**. ▶ plays from the start of that sentence.
- **Recordings (collapsible):** upload an audio or video file, or remove a recording. Each change transcribes as needed and then re-indexes; the list shows progress. Removed recordings are moved to `data/removed/<id>/`, not deleted.

**HTTP API** (OpenAPI docs at `/docs`)

| Method and path | What it does |
|---|---|
| `GET /api/search?q=…&k=10&role=host\|guest&mode=hybrid\|lexical\|semantic` | Moments for a query. Each has `file_id`, `title`, `start`, `end`, `match_time`, `timestamp`, `speaker`, `role`, `name`, `text`, `highlights`, `score` and `related`. |
| `GET /api/files` | Recordings, with duration and status (`ready`, `transcribing`, `indexing`, `removing`, `failed: …`). |
| `PUT /api/files?name=…&title=…` | Upload: the raw audio is the request body. Returns `202` and the new `file_id`; processing runs in the background. |
| `DELETE /api/files/{file_id}` | Remove a recording (`202`), then re-index. |
| `GET /audio/{file_id}.mp3` | The audio, with Range support for seeking. |
| `GET /healthz` | Database check. |

**Command line**

| Command | What it does |
|---|---|
| `uv run python -m app.ingest [--only ID …] [--force]` | Transcribe (Groq, cached) and label speakers for every recording in `data/manifest.yaml`. |
| `uv run python -m app.index` | Rebuild the whole search index. Run it after anything that changes transcripts, chunking or models. |
| `uv run python -m app.evaluate [--split dev\|test]` | Score search on the golden queries. |
| `uv run --with beautifulsoup4 --with lxml python scripts/build_dataset.py` | Rebuild the dataset audio from NASA's public sources. |
| `uv run python scripts/label_helper.py` | Turn `data/eval/queries.src.yaml` into `queries.yaml` (existing labels are kept). |
| `uv run python -m pytest` | Tests. |

## How it works

1. **Ingest:** Groq Whisper large-v3-turbo transcribes each recording with word timestamps. A local ECAPA + spectral clustering + Viterbi pipeline labels each word's speaker (two speakers per recording). The words are grouped into sentence-level utterances, and host and guest are told apart by who asks the questions.
2. **Index:** utterances are grouped into overlapping ~50-word passages. PostgreSQL stores each passage's BGE embedding, its BM25 postings, its ColBERT token vectors, and a phonetic vocabulary of every spoken word.
3. **Search:**
   - **Candidates:** keyword and meaning search each return 50 candidates, merged by rank fusion.
   - **Moments:** the top 30 are snapped to their best sentence.
   - **Matches:** the gte cross-encoder picks the matches.
   - **Related:** ColBERT orders the rest.
   - **Output:** near-duplicates are dropped and the top 10 moments are returned.

The full design, schema diagram and the reasons behind each choice are in [ARCHITECTURE.md](ARCHITECTURE.md).

## Evaluation

81 hand-labelled queries in 7 categories (keywords, exact phrases, paraphrases, questions, cross-recording topics, misspellings, speaker-only), split into dev (for tuning) and test (for confirmation):

| Split | Match precision | Recall@1 | Recall@5 | Recall@10 | MRR@10 | Time per query (CPU) |
|---|---|---|---|---|---|---|
| Dev (29) | 0.45 | 0.45 (max 0.50) | 0.88 | 0.97 | 0.94 | 1.2 s |
| Test (52) | 0.53 | 0.46 (max 0.54) | 0.80 | 0.96 | 0.88 | 1.2 s |

Recall@1 can't reach 1.0 because most queries have more than one labelled moment. [docs/EVALUATION.md](docs/EVALUATION.md) has the protocol, per-category results, and a log of the 14 experiments behind the current design, including what was tried and rejected.

## Project layout

```
app/
  main.py          HTTP API and page      search.py    the search pipeline
  library.py       upload/remove jobs     index.py     index builder
  ingest.py        transcripts            asr.py       Groq ASR + word alignment
  diarize.py       speaker per word       transcript.py utterances + host/guest roles
  chunking.py      passages               text.py      tokenizer for the vocabulary
  embed.py         BGE + ColBERT          evaluate.py  golden-set scoring
  config.py        settings from .env     schema.sql   database schema
  static/index.html  the web page
data/              manifest, audio, transcripts, eval queries (see docs/DATASET.md)
scripts/           dataset rebuild and label tools
tests/             unit tests
docs/              DATASET.md, EVALUATION.md; PRD.md and TDD.md (the original design)
```

## Configuration

All settings are environment variables, read from `.env`. [.env.example](.env.example) lists every one with its default, and [ARCHITECTURE.md](ARCHITECTURE.md#configuration) explains them. Only `GROQ_API_KEY` has to be set, and only to add recordings.

## Limitations

- **Two speakers assumed:** diarization expects exactly two speakers per recording, so solo recordings and panels are mislabelled.
- **Latency:** searches take about 1.2 s on CPU, mostly the reranker.
- **Re-indexing:** every upload or removal rebuilds the whole index (about 30 s now), which grows with the library.
- **Long uploads:** recordings over about an hour may exceed Groq's upload size limit.
- **No login:** keep it on localhost.

The upgrade path for each is in [ARCHITECTURE.md](ARCHITECTURE.md#operations).

## Documentation

- [ARCHITECTURE.md](ARCHITECTURE.md): system design, diagrams, schema, the 18 architectural decisions
- [docs/EVALUATION.md](docs/EVALUATION.md): protocol, metrics, current results, decision log
- [docs/DATASET.md](docs/DATASET.md): the six recordings, the query set, how to rebuild both
- [docs/PRD.md](docs/PRD.md), [docs/TDD.md](docs/TDD.md): the original AudioSearch requirements and design this project started from
- [AI_USAGE.md](AI_USAGE.md): how AI tools were used
