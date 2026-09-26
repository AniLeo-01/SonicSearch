import importlib.util
import sys
from pathlib import Path

import pytest
import yaml

from app.models import Transcript, Utterance

SCRIPTS = Path(__file__).parents[1] / "scripts"


def load(name):
  spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


def test_label_references_parse_and_close_moments_merge():
  lh = load("label_helper")
  m = lh.REF_RE.match("runway:70-72@1")
  assert (m["file"], m["a"], m["b"], m["grade"]) == ("runway", "70", "72", "1")
  assert lh.REF_RE.match("ep_2:5")["file"] == "ep_2" and not lh.REF_RE.match("Runway:1")
  rel = [{"file": "f", "start": 0, "end": 5, "grade": 2, "quote": "a"}, {"file": "f", "start": 12, "end": 15, "grade": 1, "quote": "b"},
         {"file": "f", "start": 40, "end": 45, "grade": 2, "quote": "c"}]
  assert [(r["start"], r["end"], r["grade"]) for r in lh.merge_close(rel, gap=10)] == [(0, 15, 2), (40, 45, 2)]


def test_label_helper_keeps_existing_labels_and_resolves_new_queries(tmp_path, monkeypatch):
  lh = load("label_helper")
  Transcript("f", 10.0, [Utterance(0, "A", 1.0, 2.0, "Hello."), Utterance(1, "B", 2.5, 4.0, "Hi there.")], []).save(tmp_path / "t" / "f.json")
  (tmp_path / "src.yaml").write_text(yaml.safe_dump({"tolerance_sec": 5.0, "queries": [
    {"id": "q1", "query": "hello", "category": "keyword", "split": "dev", "relevant": ["f:0"]},
    {"id": "q2", "query": "hi", "category": "keyword", "split": "test", "relevant": ["f:1"]}]}))
  (tmp_path / "out.yaml").write_text(yaml.safe_dump({"queries": [
    {"id": "q1", "relevant": [{"file": "f", "start": 99.0, "end": 100.0, "grade": 2, "quote": "frozen"}]}]}))
  monkeypatch.setattr(sys, "argv", ["label_helper", "--transcripts", str(tmp_path / "t"), "--src", str(tmp_path / "src.yaml"),
                                    "--out", str(tmp_path / "out.yaml")])
  assert lh.main() == 0
  out = {q["id"]: q["relevant"][0] for q in yaml.safe_load((tmp_path / "out.yaml").read_text())["queries"]}
  assert out["q1"]["quote"] == "frozen"  # already labelled: kept as it was
  assert (out["q2"]["start"], out["q2"]["end"], out["q2"]["quote"]) == (2.5, 4.0, "Hi there.")


def test_nasa_transcript_pages_parse_into_speaker_turns():
  pytest.importorskip("bs4")
  pytest.importorskip("lxml")
  html = """<div><p><strong>Transcript</strong></p>
    <p><strong>Gary Jordan (Host)</strong></p><p>Welcome to the show.</p><p>&lt;Intro Music&gt;</p>
    <p><strong>Abba Zubair</strong></p><p>Thanks for having me.</p><p>It's great to be here.</p></div>"""
  assert load("build_dataset").parse_nasa_transcript(html) == [
    {"speaker": "Gary Jordan", "text": "Welcome to the show."},
    {"speaker": "Abba Zubair", "text": "Thanks for having me. It's great to be here."}]
