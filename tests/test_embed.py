import numpy as np
import pytest

from app.embed import maxsim


def test_maxsim_sums_each_query_tokens_best_match():
  q = np.array([[1, 0], [0, 1]], np.float32)
  doc = np.array([[1, 0], [0.6, 0.8]], np.float16)  # stored as float16, like the index
  assert maxsim(q, doc) == pytest.approx(1.0 + 0.8, abs=1e-3)
