from types import SimpleNamespace as NS

from fastapi.testclient import TestClient

from app import library
from app.main import app

client = TestClient(app)  # not used as a context manager, so the lifespan (database, models) doesn't run


def test_search_validates_its_parameters():
  assert client.get("/api/search").status_code == 422
  for bad in ({"role": "boss"}, {"mode": "fuzzy"}, {"k": 0}, {"k": 51}):
    assert client.get("/api/search", params={"q": "x", **bad}).status_code == 422


def test_upload_rejects_empty_and_oversized_bodies(monkeypatch):
  assert client.put("/api/files", params={"name": "a.mp3"}, content=b"").status_code == 400
  monkeypatch.setattr(library, "MAX_UPLOAD", 10)
  assert client.put("/api/files", params={"name": "a.mp3"}, content=b"x" * 11).status_code == 413


def test_removing_an_unknown_recording_is_404(monkeypatch):
  monkeypatch.setattr(app.state, "searcher", NS(emb=None, colbert=None), raising=False)
  assert client.delete("/api/files/nope").status_code == 404


def test_the_page_and_seekable_audio_are_served():
  assert "<title>SonicSearch</title>" in client.get("/").text
  r = client.get("/audio/runway.mp3", headers={"Range": "bytes=0-99"})
  assert r.status_code == 206 and len(r.content) == 100
