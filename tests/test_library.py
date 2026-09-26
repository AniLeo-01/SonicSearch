import shutil

import pytest

from app import library
from app.config import settings
from app.dataset import load_manifest
from app.library import new_file_id


def test_new_file_id_is_path_safe_and_unused():
  assert new_file_id("../../etc/Pass Wd.MP3", set()) == "pass_wd"  # no directories, no dots, lowercase
  assert new_file_id("Ep 1!.wav", {"ep_1"}) == "ep_1_2"
  assert new_file_id("Ep 1!.wav", {"ep_1", "ep_1_2"}) == "ep_1_3"
  assert new_file_id("....mp3", set()) == "recording"


@pytest.fixture
def data(tmp_path, monkeypatch):
  monkeypatch.setattr(settings, "data_dir", tmp_path)
  (tmp_path / "manifest.yaml").write_text("# header line\n# second\ndataset: x\nfiles:\n  - {file_id: a, title: A}\n  - {file_id: b, title: B}\n")
  return tmp_path


def test_saving_the_manifest_keeps_its_header_comments(data):
  header, raw = library._manifest()
  raw["files"].append({"file_id": "c", "title": "C", "source": "upload"})
  library._save_manifest(header, raw)
  assert (data / "manifest.yaml").read_text().startswith("# header line\n# second\n")
  assert [e.file_id for e in load_manifest()] == ["a", "b", "c"]


class Rows:  # the one query listing() runs: file_id and duration of indexed recordings
  def __init__(self, rows):
    self.rows = rows

  def execute(self, *_):
    return self

  def fetchall(self):
    return self.rows


def test_listing_merges_manifest_index_and_running_jobs(data, monkeypatch):
  monkeypatch.setitem(library.jobs, "b", {"title": "B", "status": "indexing"})
  monkeypatch.setitem(library.jobs, "new", {"title": "New", "status": "transcribing"})
  files = library.listing(Rows([("a", 60.0)]))
  assert [(f["file_id"], f["status"], f["duration"]) for f in files] == [
    ("a", "ready", 60.0), ("b", "indexing", None), ("new", "transcribing", None)]


def test_remove_rejects_unknown_and_busy_recordings_and_dismisses_failed_uploads(data, monkeypatch):
  with pytest.raises(LookupError):
    library.remove("nope", None, None)
  monkeypatch.setitem(library.jobs, "a", {"title": "A", "status": "indexing"})
  with pytest.raises(RuntimeError):
    library.remove("a", None, None)
  monkeypatch.setitem(library.jobs, "x", {"title": "X", "status": "failed: boom"})
  library.remove("x", None, None)
  assert "x" not in library.jobs


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="needs ffmpeg")
def test_a_non_audio_upload_fails_cleanly(data, monkeypatch):
  (data / "audio").mkdir()
  tmp = data / "upload.tmp"
  tmp.write_text("not audio")
  fid = library.add(tmp, "notes.txt", None, None, None)
  library._worker.submit(lambda: None).result()  # one worker: this waits for the upload job
  assert library.jobs[fid]["status"] == "failed: not an audio file ffmpeg can read"
  assert not tmp.exists() and not any((data / "audio").iterdir()) and [e.file_id for e in load_manifest()] == ["a", "b"]
  monkeypatch.delitem(library.jobs, fid)
