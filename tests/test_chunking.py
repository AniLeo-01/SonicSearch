import pytest

from app.chunking import build_chunks
from app.models import Transcript, Utterance


def transcript(word_counts):
  utts, t = [], 0.0
  for i, n in enumerate(word_counts):
    utts.append(Utterance(i, "AB"[i % 2], t, t + n * 0.4, " ".join(f"w{i}_{j}" for j in range(n))))
    t += n * 0.4
  return Transcript("f", t, utts, [])


def test_passages_are_overlapping_windows_of_whole_utterances():
  chunks = build_chunks(transcript([20] * 10), target=50, stride=25)
  assert [(c.utt_start, c.utt_end) for c in chunks] == [(0, 3), (2, 5), (4, 7), (6, 9), (8, 10)]
  assert chunks[0].id == "f_c0000" and chunks[0].speakers == ["A", "B"]
  assert chunks[-1].end == pytest.approx(80.0)


def test_an_utterance_longer_than_a_passage_still_moves_forward():
  assert [(c.utt_start, c.utt_end) for c in build_chunks(transcript([120, 5]), 50, 25)] == [(0, 1), (1, 2)]
