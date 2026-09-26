import pytest

from app.config import settings
from app.dataset import load_manifest

ENTRIES = load_manifest()

def test_num_files():
  assert 5 <= len(ENTRIES) <= 6

def test_every_speaker_appears_in_one_file():
  names = [name for e in ENTRIES for name in (e.host, e.guest)]
  assert None not in names, "every file needs a host and a guest"
  assert len(set(names)) == len(names) # a speaker appear more than 1 file


def test_duplicate_file_ids_are_rejected(tmp_path, monkeypatch):
  monkeypatch.setattr(settings, "data_dir", tmp_path)
  (tmp_path / "manifest.yaml").write_text("files:\n  - {file_id: a}\n  - {file_id: a}\n")
  with pytest.raises(ValueError, match="duplicate"):
    load_manifest()
