from types import SimpleNamespace as NS

import pytest

from app.models import Chunk
from app.search import (Utt, first_match_time, idf_coverage, mmss, overlap, parse, parse_headline, rrf, snap,
                        substring_ratio, temporal_nms, tsquery, where_clause)


def test_parse_finds_phrases_role_and_intent():
  q = parse('"firing room one" role:guest')
  assert (q.text, q.phrases, q.role, q.intent) == ("firing room one", ["firing room one"], "guest", "phrase")
  assert parse("Why do clocks on Mars run faster?").intent == "question"
  assert parse("cesium").intent == "keyword" and parse("WB-57").intent == "keyword"
  assert parse("people who get lost even in their own neighborhood").intent == "topic"
  assert parse("cesium", role="host").role == "host"


def test_user_text_only_reaches_sql_as_bound_parameters():
  where, params = where_clause(parse('"x\'); DROP TABLE chunks; --" role:host'))
  assert "DROP" not in where and "%(ph0)s" in where and params == {"ph0": "x'); DROP TABLE chunks; --", "role": "host"}


def test_tsquery_quotes_lexemes_and_drops_repeats():
  assert tsquery(["zirconium", "o'neil", "zirconium"]) == "'zirconium' | 'o''neil'"


def test_headline_spans_and_the_time_of_the_first_highlighted_word():
  spans = parse_headline("It could be \x02cesium\x03.")
  assert spans == [(12, 18)]
  u = Utt("f", 0, "A", 3.0, 4.0, "It could be cesium.", [["It", 3.0, 3.2], ["could", 3.2, 3.4], ["be", 3.4, 3.5], ["cesium.", 3.5, 4.0]],
          None, [], None)
  assert first_match_time(u, spans) == 3.5 and first_match_time(u, []) == 3.0


def test_snap_picks_the_sentence_with_the_query_words_and_respects_the_speaker_filter():
  c = Chunk("f_c0000", "f", 0, 3, 0.0, 9.0, ["A", "B"], "")
  utts = {
    ("f", 0): Utt("f", 0, "A", 0, 3, "Hello there friend.", [], 0.9, [], None),
    ("f", 1): Utt("f", 1, "B", 3, 6, "It could be cesium.", [["It", 3, 3.2], ["could", 3.2, 3.4], ["be", 3.4, 3.5], ["cesium.", 3.5, 4]],
                  0.2, ["cesium"], "It could be \x02cesium\x03."),
    ("f", 2): Utt("f", 2, "A", 6, 9, "Nice.", [], 0.5, [], None),
  }
  m = snap(c, utts, {"cesium": 2.0}, lex_w=0.8, allowed=None)
  assert (m.utt.idx, m.highlights, m.match_time) == (1, [(12, 18)], 3.5)
  assert snap(c, utts, {"cesium": 2.0}, 0.8, allowed={"A"}).utt.speaker == "A"
  assert snap(c, utts, {}, 0.8, allowed=set()) is None


def test_rrf_sums_reciprocal_ranks_with_multipliers():
  order, score, ranks = rrf({"lexical": ["a", "b"], "dense": ["b", "c"]}, k=10, mult={"lexical": {"a": 0.5}})
  assert order == ["b", "c", "a"]
  assert score["b"] == pytest.approx(1 / 12 + 1 / 11) and score["a"] == pytest.approx(0.5 / 11)
  assert ranks["b"] == {"lexical": 2, "dense": 1}


def test_idf_coverage_gives_sounds_like_stand_ins_partial_credit():
  lexemes = [("cesium", 1.0, "cesium"), ("clock", 1.0, "clock"), ("caesium", 0.72, "cesium")]
  hits = [("c1", 5.0, ["cesium", "clock"]), ("c2", 2.0, ["clock"]), ("c3", 1.0, ["caesium"])]
  cov = idf_coverage({"cesium": 3.0, "clock": 1.0}, 100, lexemes, hits)
  assert cov == pytest.approx({"c1": 1.0, "c2": 0.25, "c3": 0.72 * 0.75})


def test_temporal_nms_drops_moments_near_a_better_one_in_the_same_file():
  def m(file, idx, start):
    return NS(chunk=NS(file_id=file, start=start, end=start + 20), utt=NS(idx=idx, start=start))
  ms = [m("a", 5, 100), m("a", 6, 105), m("a", 9, 200), m("b", 5, 100)]
  assert temporal_nms(ms, gap=10) == [ms[0], ms[2], ms[3]]
  assert overlap(0, 10, 5, 15) == 0.5 and overlap(0, 10, 20, 30) == 0


def test_substring_ratio_catches_asr_merged_words():
  assert substring_ratio("zubaire", "abizubair") >= 0.85


def test_mmss_rounds_before_splitting_minutes():
  assert (mmss(399.81), mmss(59.96), mmss(0)) == ("06:39.8", "01:00.0", "00:00.0")
