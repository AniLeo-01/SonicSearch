import numpy as np
import torch

from app.diarize import SR, boundaries, cluster_two, diarize, speech_windows, viterbi
from app.models import Word


def test_speech_windows_merge_small_gaps_skip_blips_and_slide():
  words = [Word("a", 0.0, 0.3), Word("b", 0.6, 3.0), Word("c", 10.0, 10.2)]  # "c" alone is too short to judge a voice
  assert speech_windows(words) == [(0.0, 1.5), (0.75, 2.25), (1.5, 3.0)]


def test_boundaries_at_sentence_end_segment_end_and_pause():
  words = [Word("Hi.", 0, 0.3), Word("so", 0.4, 0.6), Word("and", 0.7, 0.9), Word("then", 2.0, 2.2), Word("more", 2.3, 2.5)]
  assert boundaries(words, seg_ends={1}).tolist() == [True, True, True, True, False]


def test_viterbi_ignores_a_weak_flip_but_switches_at_a_boundary():
  flip = np.log([[0.9, 0.1]] * 3 + [[0.4, 0.6]] + [[0.9, 0.1]] * 3)
  assert viterbi(flip, np.zeros(7, bool)).tolist() == [0] * 7  # one unsure word isn't worth two switches
  turn = np.log([[0.9, 0.1]] * 3 + [[0.1, 0.9]] * 3)
  assert viterbi(turn, np.array([True, False, False, True, False, False])).tolist() == [0, 0, 0, 1, 1, 1]
  assert viterbi(np.zeros((0, 2)), np.zeros(0, bool)).tolist() == []


def test_cluster_two_separates_two_voices():
  rng = np.random.default_rng(0)
  a, b = rng.normal(size=192), rng.normal(size=192)
  emb = np.vstack([a + 0.1 * rng.normal(size=(20, 192)), b + 0.1 * rng.normal(size=(20, 192))])
  x, cents = cluster_two(emb)
  labels = (x @ cents.T).argmax(1)
  assert len(set(labels[:20])) == 1 and len(set(labels[20:])) == 1 and labels[0] != labels[20]


class SignEncoder:
  """Stand-in for ECAPA: a window's "voice" is the sign of its audio (host +0.5, guest -0.5)."""

  def encode_batch(self, wav, lens):
    m = wav.mean(1, keepdim=True)
    return torch.cat([m.clamp(min=0), (-m).clamp(min=0), torch.full_like(m, 0.01)], 1)[:, None, :]


def test_diarize_labels_each_word_and_switches_at_the_sentence_end():
  audio = np.concatenate([np.full(6 * SR, 0.5), np.full(6 * SR, -0.5)]).astype(np.float32)
  words = [Word("host?" if i == 11 else "host", i * 0.5, i * 0.5 + 0.4) for i in range(12)]
  words += [Word("guest." if i == 23 else "guest", i * 0.5, i * 0.5 + 0.4) for i in range(12, 24)]
  diarize(SignEncoder(), audio, words, seg_ends=set())
  assert [w.speaker for w in words] == ["SPEAKER_00"] * 12 + ["SPEAKER_01"] * 12
