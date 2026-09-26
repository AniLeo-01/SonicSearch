"""Add and remove recordings from the web app. Every change ends in a full re-index.

Changes run one at a time on a background worker; `jobs` holds the state of the ones not finished yet.
Removed recordings are moved to data/removed/<file_id>/ (audio, transcripts, manifest entry), not deleted.
"""
import re
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from itertools import takewhile
from pathlib import Path

import httpx
import yaml

from app.config import settings
from app.dataset import ManifestEntry, load_manifest
from app.diarize import load_encoder
from app.index import reindex
from app.ingest import build_transcript

MAX_UPLOAD = 200 * 1024 * 1024  # bytes
jobs: dict[str, dict] = {}  # file_id -> {"title", "status"}: queued | transcribing | indexing | removing | failed: ...
_worker = ThreadPoolExecutor(max_workers=1)  # one change at a time: each ends in a full re-index
_encoder = None  # the speaker model, loaded on the first upload


def new_file_id(filename: str, taken: set[str]) -> str:
  """A safe, unused id from an uploaded file's name: lowercase letters, digits and underscores only."""
  base = re.sub(r"[^a-z0-9]+", "_", Path(filename).stem.lower()).strip("_")[:48] or "recording"
  fid, n = base, 2
  while fid in taken:
    fid, n = f"{base}_{n}", n + 1
  return fid


def _manifest() -> tuple[str, dict]:
  """The manifest's leading comment block (kept verbatim on save) and its data."""
  text = (settings.data_dir / "manifest.yaml").read_text(encoding="utf-8")
  return "".join(takewhile(lambda line: line.startswith("#"), text.splitlines(keepends=True))), yaml.safe_load(text)


def _save_manifest(header: str, raw: dict) -> None:
  body = yaml.safe_dump(raw, sort_keys=False, allow_unicode=True, width=120)
  (settings.data_dir / "manifest.yaml").write_text(header + body, encoding="utf-8")


def listing(conn) -> list[dict]:
  durations = dict(conn.execute("SELECT file_id, duration_sec FROM audio_files").fetchall())
  files = [
    {"file_id": e.file_id, "title": e.title, "duration": durations.get(e.file_id),
     "status": jobs.get(e.file_id, {}).get("status") or ("ready" if e.file_id in durations else "not indexed")}
    for e in load_manifest()
  ]
  known = {f["file_id"] for f in files}
  return files + [{"file_id": fid, "title": j["title"], "duration": None, "status": j["status"]} for fid, j in jobs.items() if fid not in known]


def add(tmp: Path, filename: str, title: str | None, emb, colbert) -> str:
  fid = new_file_id(filename, {e.file_id for e in load_manifest()} | set(jobs))
  jobs[fid] = {"title": title or Path(filename).stem, "status": "queued"}
  _worker.submit(_add, tmp, fid, emb, colbert)
  return fid


def _add(tmp: Path, fid: str, emb, colbert) -> None:
  global _encoder
  title, audio = jobs[fid]["title"], settings.data_dir / "audio" / f"{fid}.mp3"
  try:
    jobs[fid]["status"] = "transcribing"
    # mono 48 kbps MP3: one format for the player and the pipeline, and about 1 hour fits Groq's upload limit
    cmd = ["ffmpeg", "-v", "error", "-y", "-i", str(tmp), "-vn", "-ac", "1", "-b:a", "48k", str(audio)]
    if subprocess.run(cmd, capture_output=True).returncode:
      raise ValueError("not an audio file ffmpeg can read")
    _encoder = _encoder or load_encoder()
    with httpx.Client(timeout=settings.asr_timeout_sec) as client:
      t = build_transcript(ManifestEntry(fid, title, audio), client, _encoder)
    if not t.utterances:
      raise ValueError("no speech found")
    header, raw = _manifest()  # only now: a manifest entry without a transcript would break every re-index
    raw["files"].append({"file_id": fid, "title": title, "source": "upload"})
    _save_manifest(header, raw)
    jobs[fid]["status"] = "indexing"
    print(reindex(emb, colbert))
    del jobs[fid]
  except Exception as exc:  # noqa: BLE001 - surfaced to the UI as the job's status
    jobs[fid]["status"] = f"failed: {exc}"
    if not any(e.file_id == fid for e in load_manifest()):
      for p in (audio, settings.transcripts_dir / f"{fid}.json", settings.transcripts_dir / f"{fid}.groq.json"):
        p.unlink(missing_ok=True)
  finally:
    tmp.unlink(missing_ok=True)


def remove(fid: str, emb, colbert) -> None:
  """Raises LookupError for an unknown id, RuntimeError while the recording has a change in progress."""
  if jobs.get(fid, {}).get("status", "").startswith("failed"):
    del jobs[fid]  # a failed upload never reached the manifest: just forget it
    return
  if fid in jobs:
    raise RuntimeError(f"{fid} is busy: {jobs[fid]['status']}")
  entry = next((e for e in load_manifest() if e.file_id == fid), None)
  if entry is None:
    raise LookupError(fid)
  jobs[fid] = {"title": entry.title, "status": "removing"}
  _worker.submit(_remove, fid, emb, colbert)


def _remove(fid: str, emb, colbert) -> None:
  try:
    header, raw = _manifest()
    entry = next(f for f in raw["files"] if f["file_id"] == fid)
    raw["files"].remove(entry)
    _save_manifest(header, raw)
    trash = settings.data_dir / "removed" / fid
    trash.mkdir(parents=True, exist_ok=True)
    (trash / "manifest_entry.yaml").write_text(yaml.safe_dump(entry, sort_keys=False, allow_unicode=True), encoding="utf-8")
    for p in (settings.data_dir / "audio" / f"{fid}.mp3", settings.transcripts_dir / f"{fid}.json", settings.transcripts_dir / f"{fid}.groq.json"):
      if p.exists():
        shutil.move(p, trash / p.name)
    print(reindex(emb, colbert))
    del jobs[fid]
  except Exception as exc:  # noqa: BLE001 - surfaced to the UI as the job's status
    jobs[fid]["status"] = f"failed: {exc}"
