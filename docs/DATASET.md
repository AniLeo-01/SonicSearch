# Dataset

`nasa-hwhap-two-speaker-v1`: a small golden corpus for evaluating SonicSearch. It has six two-speaker interview excerpts from NASA's *Houston We Have a Podcast* (HWHAP) and 81 labelled search queries.

| | |
|---|---|
| Audio files | 6 MP3s, 32 MB |
| Total audio | 3,360 s (56.0 min), ~9.3 min per file |
| Format | MP3, mono, 44.1 kHz, ~80 kbps |
| Speakers | 12 distinct people (6 hosts, 6 guests), exactly 2 per file |
| Queries | 81 (29 dev / 52 test), 7 categories |
| Relevance labels | 154 audio-time intervals (145 grade 2, 9 grade 1) |

## Layout

```
data/
├── manifest.yaml          # one entry per audio file: source, excerpt offsets, speakers, topic
├── audio/<file_id>.mp3    # the excerpts
└── eval/
    ├── queries.src.yaml   # human-authored queries, labels as utterance ranges (edit this)
    └── queries.yaml       # GENERATED: labels as time intervals + verbatim quotes (evaluator reads this)
```

`app/dataset.py:load_manifest()` reads the manifest. It resolves each audio path as `DATA_DIR/audio/<file_id>.mp3` and rejects duplicate `file_id`s.

Recordings uploaded in the web app are added to `manifest.yaml` as well (`source: upload`), with their audio converted to mono 48 kbps MP3. Removing a recording takes it out of the manifest and moves its audio, transcripts and manifest entry to `data/removed/<file_id>/`. Saving the manifest keeps its header comments but rewrites the YAML layout.

## Audio files

| file_id | Ep. | Published | Host | Guest | Topic | Excerpt (s) | Duration |
|---|---|---|---|---|---|---|---|
| `ai_at_nasa` | 424 | 2026-05-29 | Nilufar Ramji | Kevin Murphy | AI, ML and data science at NASA | 107.78–670.18 | 562.4 s |
| `stem_cells` | 432 | 2026-08-07 | Gary Jordan | Abba Zubair | Stem cell research on the ISS, regenerative medicine | 142.88–707.16 | 564.3 s |
| `wayfinding` | 427 | 2026-06-26 | Leah Cheshier | Giuseppe Iaria | Spatial orientation and cognitive maps in astronauts | 115.18–668.90 | 553.7 s |
| `telling_time` | 419 | 2026-04-24 | Dane Turner | Kevin Coggins | Precision timekeeping and navigation on the Moon and Mars | 87.76–649.06 | 561.3 s |
| `runway` | 389 | 2025-06-13 | Courtney Beasley | David Johnson | Ellington Field history, NASA aircraft ops | 76.72–635.34 | 558.6 s |
| `artemis_launch` | 401 | 2025-09-12 | Joseph Zakrzewski | Charlie Blackwell-Thompson | Career of the Artemis launch director | 96.00–655.76 | 559.8 s |

Excerpt offsets are in seconds into the full episode at `source_audio_url`. Episode pages and source URLs are listed in `data/manifest.yaml`.

## Selection rationale

- **Exactly two speakers per excerpt (host + guest)**, checked against NASA's human-written transcript. This makes role-scoped search (`host`/`guest`) well defined and keeps diarization simple enough to score.
- **No speaker appears in more than one file.** A search that matches on voice or name can't pick up hits from another recording.
- **Six distinct topics, with deliberate confusers across files.** For example, brain navigation (`wayfinding`) vs. clock-based navigation (`telling_time`), and gravity in stem-cell, neuroscience and timekeeping research. Lexical overlap between files makes precision meaningful.
- **Clean boundaries.** Each excerpt starts at the first conversational turn after the intro music and ends on a sentence boundary. This removes music and ad reads and leaves ~9.3 min of speech per file.
- **Short enough to label exhaustively.** At 56 minutes, every relevant moment for every query can be listed by hand, so recall numbers are real.

## Evaluation queries

### Categories

| Category | Count (dev/test) | What it probes | Example |
|---|---|---|---|
| `keyword` | 18 (6/12) | Rare terms said verbatim | `cesium`, `WB-57` |
| `phrase` | 10 (4/6) | Quoted exact phrases | `"firing room one"` |
| `paraphrase` | 14 (5/9) | Same meaning, different words | "people who get lost even in their own neighborhood" |
| `question` | 15 (5/10) | Natural-language questions | "Why do clocks on Mars run faster?" |
| `cross_file` | 7 (3/4) | Topics that span several recordings | "military service before joining NASA" |
| `misspelled` | 11 (4/7) | Typos and ASR-mangled terms | `apheresis` (ASR wrote "ismoresis"), `El Passo` |
| `speaker` | 6 (2/4) | Role-scoped search (`role: host` or `guest`) | "thanks for having me", guest only |

### Labelling protocol

- Queries were written by reading the transcripts **before** any retrieval tuning, then frozen.
- Relevance is **exhaustive**: every moment in the corpus that answers a query is listed.
- Grades: `2` = directly relevant (default), `1` = partially relevant (`@1` suffix in the source file).
- Speaker-scoped queries use the **true** speaker from NASA's transcript, even where our diarization is wrong, so diarization errors count as retrieval misses (e.g. `s04`: `runway:2` is the guest, diarization says host).
- Split: within each category, every third query is `dev` (tuning allowed) and the rest are `test`. **Never tune on `test`.**

### File formats

`queries.src.yaml` (source of truth, hand-edited) references utterance ranges as `"file_id:first-last"` (inclusive utterance indices into the transcript):

```yaml
- {id: m03, query: "cord blood", category: misspelled, split: test, notes: "ASR wrote 'cod blood'",
   relevant: ["stem_cells:51"]}
```

`queries.yaml` (generated, do not edit) resolves those to audio time, so labels don't depend on chunking or re-transcription:

```yaml
- id: k01
  query: cesium
  category: keyword
  split: dev
  relevant:
  - file: telling_time
    start: 395.66     # seconds into data/audio/telling_time.mp3
    end: 412.52
    grade: 2
    quote: that's right the vibrations of an atom. It could be rubidium. It could be cesium. ...
```

Optional query fields: `role` (`host`/`guest`, speaker category only) and `notes`.

A hit matches a labelled interval if it overlaps it within `tolerance_sec: 5.0`.

## Regenerating

**Audio:** `scripts/build_dataset.py` downloads each full episode from `source_audio_url` and cuts `excerpt.start`–`excerpt.end` into `DATA_DIR/audio/<file_id>.mp3` (mono, 44.1 kHz, 80 kbps). It also scrapes NASA's human transcript of each episode into `DATA_DIR/reference/<file_id>.json`. Those reference transcripts are for checking speakers and ASR quality; search never reads them. Existing files are skipped unless you pass `--force`.

```sh
uv run --with beautifulsoup4 --with lxml python scripts/build_dataset.py [--only FILE_ID ...] [--force]
```

A rebuilt `runway.mp3` matched the committed file: same duration and format, waveform correlation 0.9998.

**Labels:** `scripts/label_helper.py` turns `queries.src.yaml` into `queries.yaml`.

```sh
uv run python scripts/label_helper.py
```

Utterance ranges only mean something against the transcripts they were written for. The existing 81 queries were written against an earlier transcript set that isn't in this repo; this repo's transcripts (`DATA_DIR/transcripts/`) number utterances differently, and only about half of the old ranges land on the labelled moment. So the script keeps every query already in `queries.yaml` as it is, updating only its text, category, split and notes from the source. It resolves only new queries, against the current transcripts. `--force` re-resolves everything, which is only valid with the original transcripts.

To add a query, write its ranges against the current transcripts' utterance numbers and run the script. To change an existing query's labels, delete it from `queries.yaml`, fix its ranges to current numbering, and run the script.
