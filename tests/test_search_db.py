"""The search path end to end against a real PostgreSQL: schema, index_all(), BM25 and sounds-like SQL, filters,
snapping, reranking and related ordering. The models are small deterministic stand-ins, so nothing is downloaded.

Needs TEST_DB_URL: a throwaway PostgreSQL with pgvector. These tests DROP and rebuild its tables.
"""
import os
import zlib
from pathlib import Path

import numpy as np
import psycopg
import pytest
from pgvector.psycopg import register_vector
from psycopg_pool import ConnectionPool

from app.config import settings
from app.index import index_all
from app.models import Speaker, Transcript, Utterance
from app.search import Searcher
from app.text import tokens

URL = os.environ.get("TEST_DB_URL")
pytestmark = pytest.mark.skipif(not URL or URL == settings.db_url,
                                reason="set TEST_DB_URL to a throwaway PostgreSQL with pgvector (not the app's database)")


def hashed(features, dim):
  v = np.zeros(dim, np.float32)
  for f in features:
    v[zlib.crc32(f.encode()) % dim] += 1
  return v / (np.linalg.norm(v) or 1)


class Embedder:  # bag of words in 768 dimensions, like BGE's vectors
  def docs(self, texts):
    return np.array([hashed(tokens(t), 768) for t in texts])

  def query(self, text):
    return self.docs([text])[0]

  def lexicon(self):
    return frozenset({"water", "warm", "team"})


class ColBERT:  # one vector per word, from its character trigrams: "zirconum" resembles "zirconium"
  proj = np.zeros((256, 1))

  def encode(self, texts, is_query):
    return [np.array([hashed([w[i:i + 3] for i in range(max(1, len(w) - 2))], 256) for w in tokens(t)] or [np.zeros(256)])
            for t in texts]


class Reranker:  # share of the query's words found in the passage
  def predict(self, pairs):
    return np.array([np.mean([w in tokens(p) for w in tokens(q)]) if tokens(q) else 0.0 for q, p in pairs])


def transcript(file_id, lines):
  utts, t = [], 0.0
  for i, (speaker, text) in enumerate(lines):
    words = [[w, t + j * 0.4, t + j * 0.4 + 0.3] for j, w in enumerate(text.split())]
    utts.append(Utterance(i, speaker, t, words[-1][2], text, words))
    t = words[-1][2] + 0.5
  return Transcript(file_id, t, utts, [Speaker("SPEAKER_00", "host", "Hana Host"), Speaker("SPEAKER_01", "guest", "Gus Guest")])


H, G = "SPEAKER_00", "SPEAKER_01"
CORPUS = [
  transcript("reactors", [
    (H, "Welcome back to the show."), (H, "What is zirconium used for?"),
    (G, "Zirconium coats the fuel rods in nuclear reactors."), (G, "It barely absorbs neutrons, which is why engineers love it."),
    (H, "And what about firing room one?"), (G, "Firing room one is where the launch team sits.")]),
  transcript("garden", [
    (H, "Today we talk about tomatoes and compost."), (G, "Tomatoes love warm soil and plenty of sunlight."),
    (H, "How often should I water them?"), (G, "Water the tomatoes deeply twice a week.")]),
]
TITLES = {"reactors": "Nuclear Reactors", "garden": "Garden Hour"}


@pytest.fixture(scope="module")
def searcher():
  with psycopg.connect(URL, autocommit=True) as conn:
    conn.execute("DROP TABLE IF EXISTS chunk_terms, chunks, utterances, speakers, audio_files, vocabulary CASCADE")
    conn.execute((Path(__file__).parents[1] / "app" / "schema.sql").read_text())
  with pytest.MonkeyPatch.context() as mp, psycopg.connect(URL) as conn:
    mp.setattr(settings, "chunk_words", 12)  # several passages per recording
    mp.setattr(settings, "chunk_stride", 6)
    register_vector(conn)
    index_all(conn, CORPUS, TITLES, Embedder(), ColBERT())
  pool = ConnectionPool(URL, min_size=1, max_size=2, configure=register_vector, open=True)
  yield Searcher(pool, Embedder(), Reranker(), ColBERT())
  pool.close()


def test_index_fills_every_table(searcher):
  with searcher.pool.connection() as conn:
    n = conn.execute("SELECT (SELECT count(*) FROM audio_files), (SELECT count(*) FROM speakers), (SELECT count(*) FROM utterances),"
                     " (SELECT count(*) FROM chunks), (SELECT count(*) FROM chunks WHERE colbert IS NULL),"
                     " (SELECT count(*) FROM chunk_terms), (SELECT count(*) FROM vocabulary WHERE term = 'zirconium')").fetchone()
  assert n[:3] == (2, 4, 10) and n[3] > 2 and n[4] == 0 and n[5] > 0 and n[6] == 1


def test_a_keyword_finds_its_moment_with_file_time_speaker_and_highlight(searcher):
  hit = searcher.search("zirconium")[0]
  assert (hit.file_id, hit.title, hit.related) == ("reactors", "Nuclear Reactors", False)
  assert "zirconium" in hit.text.lower() and hit.highlights and hit.start <= hit.match_time <= hit.end
  assert (hit.role, hit.name) in {("host", "Hana Host"), ("guest", "Gus Guest")}


def test_the_speaker_filter(searcher):
  assert {h.role for h in searcher.search("zirconium", role="guest")} == {"guest"}
  assert searcher.search("zirconium", role="guest")[0].text.startswith("Zirconium coats")
  assert {h.role for h in searcher.search("zirconium role:host")} == {"host"}


def test_a_quoted_phrase_only_matches_passages_containing_it(searcher):
  hits = searcher.search('"firing room one"')
  assert hits and {h.file_id for h in hits} == {"reactors"} and "firing room one" in hits[0].text.lower()


def test_a_misspelling_finds_the_spoken_word_through_sounds_like_expansion(searcher):
  hit = searcher.search("zirconum")[0]
  assert hit.file_id == "reactors" and "zirconium" in hit.text.lower()
  assert hit.related  # the reranker reads the typo literally, so it's related rather than a match


def test_modes_use_only_their_channel(searcher):
  assert all(set(h.channels) == {"lexical"} for h in searcher.search("tomatoes", mode="lexical"))
  assert all(set(h.channels) == {"dense"} for h in searcher.search("tomatoes", mode="semantic"))
  assert searcher.search("tomatoes")[0].file_id == "garden"


def test_unmatched_and_empty_queries(searcher):
  assert all(h.related for h in searcher.search("xylophone"))  # nothing matches: everything is only related
  assert searcher.search('""') == []
