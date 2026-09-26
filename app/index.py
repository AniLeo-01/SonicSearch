from collections import Counter

import numpy as np
import psycopg
from psycopg.types.json import Jsonb

from app.chunking import build_chunks
from app.config import settings
from app.dataset import load_manifest
from app.db import connect, init_db
from app.embed import ColBERT, Embedder
from app.models import Transcript
from app.text import content_tokens


def index_all(conn: psycopg.Connection, transcripts: list[Transcript], titles: dict[str, str], emb: Embedder, colbert: ColBERT):
    vocab: Counter[str] = Counter()  # term -> number of passages containing it
    # run the models first: the TRUNCATE below blocks searches until commit, so keep the transaction to the writes
    prepared = []
    for t in transcripts:
      chunks = build_chunks(t, settings.chunk_words, settings.chunk_stride)
      texts = [c.text for c in chunks]
      prepared.append((t, chunks, emb.docs(texts), colbert.encode(texts, is_query=False), emb.docs([u.text for u in t.utterances])))
      for c in chunks:
        vocab.update(set(content_tokens(c.text)))
    with conn.transaction():
      conn.execute("TRUNCATE audio_files, vocabulary CASCADE")  # cascades to every other table
      for t, chunks, chunk_vecs, chunk_toks, utt_vecs in prepared:
        conn.execute("INSERT INTO audio_files VALUES (%s, %s, %s)", (t.file_id, titles[t.file_id], t.duration))
        with conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO speakers VALUES (%s, %s, %s, %s)",
                [(t.file_id, s.label, s.role, s.name) for s in t.speakers],
            )
            cur.executemany(
                """INSERT INTO utterances (file_id, idx, speaker, start_sec, end_sec, text, words, embedding)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)""",
                [
                    (t.file_id, u.idx, u.speaker, u.start, u.end, u.text, Jsonb(u.words), v)
                    for u, v in zip(t.utterances, utt_vecs, strict=True)
                ],
            )
            cur.executemany(
                """INSERT INTO chunks (id, file_id, utt_start, utt_end, start_sec, end_sec, speakers, text,
                                        embedding, colbert)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                [
                    (c.id, c.file_id, c.utt_start, c.utt_end, c.start, c.end, c.speakers, c.text, v, tv.astype(np.float16).tobytes())
                    for c, v, tv in zip(chunks, chunk_vecs, chunk_toks, strict=True)
                ],
            )
      # BM25 postings straight from the generated tsvector: one row per (lexeme, passage) with its count
      conn.execute(
          """INSERT INTO chunk_terms (lexeme, chunk_id, tf)
             SELECT t.lexeme, c.id, coalesce(array_length(t.positions, 1), 1) FROM chunks c, unnest(c.tsv) t"""
      )
      conn.execute(
          """UPDATE chunks c SET doc_len = x.dl
             FROM (SELECT chunk_id, sum(tf) AS dl FROM chunk_terms GROUP BY chunk_id) x WHERE c.id = x.chunk_id"""
      )
      terms = sorted(vocab)
      conn.execute(
          """INSERT INTO vocabulary
             SELECT x.t, x.d, metaphone(x.t, 12), dmetaphone(x.t), dmetaphone_alt(x.t)
             FROM unnest(%s::text[], %s::int[]) AS x(t, d)""",
          (terms, [vocab[t] for t in terms]),
      )


def reindex(emb: Embedder, colbert: ColBERT) -> str:
    """Rebuild the whole index from the manifest's transcripts."""
    # ponytail: full rebuild on every change (about 1 min for the six dataset files); go incremental if the library grows
    entries = load_manifest()
    transcripts = [Transcript.load(settings.transcripts_dir / f"{e.file_id}.json") for e in entries]
    with connect() as conn:
        index_all(conn, transcripts, {e.file_id: e.title for e in entries}, emb, colbert)
        n = conn.execute(
            "SELECT (SELECT count(*) FROM chunks), (SELECT count(*) FROM utterances),"
            " (SELECT count(*) FROM chunk_terms), (SELECT count(*) FROM vocabulary)"
        ).fetchone()
    return f"indexed {len(transcripts)} files: {n[0]} chunks, {n[1]} utterances, {n[2]} postings, {n[3]} vocabulary terms"


def main() -> None:
    init_db()  # safe to repeat; makes sure the tables exist even if the API has never started
    print(reindex(Embedder(settings.embed_model), ColBERT(settings.colbert_model)))


if __name__ == "__main__":
    main()