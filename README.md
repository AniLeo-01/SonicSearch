# SonicSearch

[![CI/CD](https://github.com/AniLeo-01/SonicSearch/actions/workflows/ci.yml/badge.svg)](https://github.com/AniLeo-01/SonicSearch/actions/workflows/ci.yml)

Search what was said in audio recordings. Type a word, a phrase or a question, and SonicSearch returns the **moments** that match: which recording, the time to play from, who said it (host or guest, by name when known), and the sentence with the matching words highlighted. Click a result and the player jumps there.

It combines keyword search (BM25, with sounds-like matching for typos and misheard names) with meaning search (sentence embeddings) in one PostgreSQL database. A cross-encoder reranker then decides what counts as a confident **match**. Everything else it finds is still shown, as **related moments** ordered by ColBERT late interaction.

On the golden query set (81 queries over six NASA podcast interviews), the first result alone finds 85–90% of what it could, and the top 10 find 96–97% of all labelled moments. See [Evaluation](#evaluation).

## Getting started

The repository includes six sample recordings (NASA podcast interviews) with their transcripts, so you can search as soon as the app is up. A [Groq API key](https://console.groq.com/keys) (`GROQ_API_KEY` in `.env`) is only needed to upload or transcribe new recordings.

### With Docker Compose

Needs Docker with at least 4 GB of memory. Compose builds the image from source and starts it with a PostgreSQL database.

```sh
git clone https://github.com/AniLeo-01/SonicSearch.git
cd SonicSearch
cp .env.example .env    # optional: set GROQ_API_KEY to add recordings
docker compose up --build
```

Open <http://localhost:8000>.

- **First start:** downloads about 1.7 GB of models into a Docker volume and indexes the recordings in `data/` (about 30 s). Later starts reuse both.
- **Another port:** set `PORT` in `.env`.
- **Stop:** `docker compose down`. Add `-v` to also delete the model cache and the database; the database is rebuilt from `data/` on the next start.

### With the pre-built Docker image

Needs Docker with at least 4 GB of memory. The image is published as `ghcr.io/anileo-01/sonicsearch:latest`, so there is nothing to build. It still needs a clone of this repository for the recordings in `data/`, and a PostgreSQL database with pgvector, which runs in a second container:

```sh
git clone https://github.com/AniLeo-01/SonicSearch.git
cd SonicSearch
cp .env.example .env    # read by --env-file below; set GROQ_API_KEY to add recordings

# PostgreSQL with pgvector, on a private network
docker network create sonicsearch
docker run -d --name sonicsearch-db --network sonicsearch \
  -e POSTGRES_USER=sonicsearch -e POSTGRES_PASSWORD=sonicsearch -e POSTGRES_DB=sonicsearch \
  -v sonicsearch-pgdata:/var/lib/postgresql/data \
  pgvector/pgvector:pg17

# the app (--restart on-failure retries until the database is ready)
docker run -d --name sonicsearch-app --network sonicsearch --restart on-failure \
  -p 127.0.0.1:8000:8000 --env-file .env \
  -e DB_URL=postgresql://sonicsearch:sonicsearch@sonicsearch-db:5432/sonicsearch \
  -v "$PWD/data:/app/data" -v sonicsearch-models:/home/app/.cache \
  ghcr.io/anileo-01/sonicsearch:latest

docker logs -f sonicsearch-app    # Ctrl+C once it prints "Application startup complete"
```

Open <http://localhost:8000>. As with Compose, the first start downloads the models and indexes `data/`.

- **Stop:** `docker stop sonicsearch-app sonicsearch-db`. Start again with `docker start sonicsearch-db sonicsearch-app`.
- **Remove:** `docker rm -f sonicsearch-app sonicsearch-db && docker network rm sonicsearch`. To also delete the database and the model cache: `docker volume rm sonicsearch-pgdata sonicsearch-models`.
- **ARM machines:** the image is built for linux/amd64. On ARM, such as Apple Silicon Macs, use Docker Compose, which builds the image for your machine.

### Without Docker

Needs Python 3.13, [uv](https://docs.astral.sh/uv/), ffmpeg, and PostgreSQL with `pgvector` (`pg_trgm` and `fuzzystrmatch` ship with Postgres).

```sh
git clone https://github.com/AniLeo-01/SonicSearch.git
cd SonicSearch
cp .env.example .env                 # point DB_URL at your database (and set GROQ_API_KEY to add recordings)
uv sync
uv run python -m app.index           # build the search index from data/transcripts
uv run uvicorn app.main:app          # http://127.0.0.1:8000
```

No PostgreSQL with pgvector? Start one in Docker. It matches the default `DB_URL` in `.env.example`, so `.env` needs no changes:

```sh
docker run -d --name sonicsearch-postgres -p 127.0.0.1:5432:5432 \
  -e POSTGRES_USER=sonicsearch -e POSTGRES_PASSWORD=password -e POSTGRES_DB=sonicsearch \
  pgvector/pgvector:pg17
```

## Using it

**In the browser**
- **Search box:** plain words, `"a quoted phrase"` for an exact phrase, or a question.
- **Mode:** Hybrid (default), Keywords only, or Meaning only.
- **Speaker:** anyone, host or guest. You can also type `role:host` or `role:guest` in the query.
- **Results:** confident matches come first, then **Related moments**. ▶ plays from the start of that sentence.
- **Recordings (collapsible):** upload an audio or video file (needs `GROQ_API_KEY`), or remove a recording. Each change transcribes as needed and then re-indexes; the list shows progress. Removed recordings are moved to `data/removed/<id>/`, not deleted.

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
| `uv run python -m app.evaluate [--split dev\|test\|all]` | Score search on the golden queries. |
| `uv run python -m pytest` | Tests. Set `TEST_DB_URL` to a throwaway PostgreSQL with pgvector to include the database tests, which rebuild its tables. |

With Docker, run the `app` commands inside the app container: `docker compose exec app python -m app.index` with Compose, or `docker exec sonicsearch-app python -m app.index` with the pre-built image.

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

All settings are environment variables, read from `.env`. [.env.example](.env.example) lists every one with its default, and [ARCHITECTURE.md](ARCHITECTURE.md#configuration) explains them. Only `GROQ_API_KEY` has to be set, and only to add recordings. With Docker Compose, `DB_URL` and `DATA_DIR` are set for you, and `POSTGRES_PASSWORD` and `PORT` set the database password and the host port.

## Limitations

- **Two speakers assumed:** diarization expects exactly two speakers per recording, so solo recordings and panels are mislabelled.
- **Latency:** searches take about 1.2 s on CPU, mostly the reranker.
- **Re-indexing:** every upload or removal rebuilds the whole index (about 30 s with the six sample recordings), so it takes longer as the library grows.
- **Long uploads:** recordings over about an hour may exceed Groq's upload size limit.

## Documentation

- [ARCHITECTURE.md](ARCHITECTURE.md): system design, diagrams, schema, the 18 architectural decisions
- [docs/EVALUATION.md](docs/EVALUATION.md): protocol, metrics, current results, decision log
- [docs/DATASET.md](docs/DATASET.md): the six recordings, the query set, how to rebuild both
- [docs/PRD.md](docs/PRD.md), [docs/TDD.md](docs/TDD.md): the original AudioSearch requirements and design this project started from
- [AI_USAGE.md](AI_USAGE.md): how AI tools were used
