from types import SimpleNamespace as NS

from app.evaluate import hits_label


def test_a_hit_counts_within_tolerance_of_a_label_in_the_same_file():
  label = {"file": "f", "start": 100.0, "end": 110.0}
  assert hits_label(NS(file_id="f", start=112.0, end=115.0), label, 5.0)
  assert not hits_label(NS(file_id="f", start=116.0, end=120.0), label, 5.0)
  assert not hits_label(NS(file_id="g", start=100.0, end=110.0), label, 5.0)
