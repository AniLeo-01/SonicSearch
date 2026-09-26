#!/usr/bin/env python
"""Rebuild the golden audio dataset from its public sources.

For every entry in data/manifest.yaml this script
  1. downloads the full NASA podcast episode (US Government work),
  2. cuts the two-speaker excerpt at the recorded offsets (mono, 44.1 kHz, 80 kbps MP3),
  3. scrapes NASA's human transcript of the episode into data/reference/<file_id>.json
     (used to measure ASR word error rate and diarization accuracy - never used for retrieval).

Usage:  uv run --with beautifulsoup4 --with lxml python scripts/build_dataset.py [--force] [--only FILE_ID ...]
Requires ffmpeg. The HTML parsers are pulled in per run (only this script needs them); data goes to DATA_DIR.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import requests
import yaml
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config import settings  # noqa: E402

DATA = ROOT / settings.data_dir  # DATA_DIR may be relative (to the repo) or absolute
UA = {"User-Agent": "sonicsearch-dataset-builder/1.0 (research; contact via repository)"}


def parse_nasa_transcript(html: str) -> list[dict[str, str]]:
    """Parse the transcript section of a nasa.gov podcast page into speaker turns.

    NASA pages mark speakers as a paragraph containing only ``<strong>Name</strong>``; the following
    paragraphs are that speaker's words. Stage directions such as ``<Intro Music>`` are dropped.
    """
    soup = BeautifulSoup(html, "lxml")
    start = next(
        (s.find_parent("p") for s in soup.find_all("strong") if s.get_text(strip=True).lower() == "transcript"),
        None,
    )
    if start is None:
        raise ValueError("no transcript section found")
    turns: list[dict[str, list[str] | str]] = []
    for p in start.find_next_siblings():
        if p.name not in ("p", "h2", "h3", "h4"):
            continue
        text = p.get_text(" ", strip=True).replace("\xa0", " ").strip()
        if not text:
            continue
        strongs = p.find_all("strong")
        label = " ".join(s.get_text(" ", strip=True) for s in strongs).strip()
        if strongs and label == text and len(text.split()) <= 6 and not text.endswith((".", "?", "!")):
            speaker = re.sub(r"\s+", " ", text).strip().rstrip(":")
            speaker = re.sub(r"\s*\(host\)\s*$", "", speaker, flags=re.I)
            turns.append({"speaker": speaker, "paras": []})
            continue
        if re.fullmatch(r"[<\[][^>\]]*[>\]]", text):  # stage direction, e.g. <Intro Music>
            continue
        if not turns:
            continue
        turns[-1]["paras"].append(text)  # type: ignore[union-attr]
    out = []
    for t in turns:
        if str(t["speaker"]).startswith("<"):
            continue
        body = " ".join(t["paras"])  # type: ignore[arg-type]
        if body:
            if out and out[-1]["speaker"] == t["speaker"]:
                out[-1]["text"] += " " + body
            else:
                out.append({"speaker": str(t["speaker"]), "text": body})
    return out


def cut_excerpt(src: Path, dst: Path, start: float, end: float, file_id: str) -> None:
    dur = end - start
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-y",
        "-ss",
        f"{start}",
        "-t",
        f"{dur:.2f}",
        "-i",
        str(src),
        "-ac",
        "1",
        "-ar",
        "44100",
        "-b:a",
        "80k",
        "-af",
        f"afade=t=in:d=0.08,afade=t=out:st={dur - 0.3:.2f}:d=0.3",
        "-map_metadata",
        "-1",
        "-id3v2_version",
        "3",
        "-metadata",
        f"title={file_id}",
        "-metadata",
        "comment=Excerpt of a NASA Houston We Have a Podcast episode (US Government work)",
        str(dst),
    ]
    subprocess.run(cmd, check=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true", help="re-download and overwrite existing files")
    ap.add_argument("--only", nargs="*", help="restrict to these file_ids")
    args = ap.parse_args()

    manifest = yaml.safe_load((DATA / "manifest.yaml").read_text())
    (DATA / "audio").mkdir(parents=True, exist_ok=True)
    (DATA / "reference").mkdir(parents=True, exist_ok=True)
    for item in manifest["files"]:
        fid = item["file_id"]
        if args.only and fid not in args.only:
            continue
        audio_out = DATA / "audio" / f"{fid}.mp3"
        ref_out = DATA / "reference" / f"{fid}.json"
        if args.force or not audio_out.exists():
            with tempfile.TemporaryDirectory() as tmp:
                src = Path(tmp) / "episode.mp3"
                print(f"[{fid}] downloading {item['source_audio_url']}")
                with requests.get(item["source_audio_url"], headers=UA, stream=True, timeout=120) as r:
                    r.raise_for_status()
                    with src.open("wb") as f:
                        for block in r.iter_content(1 << 16):
                            f.write(block)
                cut_excerpt(src, audio_out, item["excerpt"]["start"], item["excerpt"]["end"], fid)
            print(f"[{fid}] wrote {audio_out}")
        if args.force or not ref_out.exists():
            html = requests.get(item["page_url"], headers=UA, timeout=60).text
            turns = parse_nasa_transcript(html)
            ref = {
                "file_id": fid,
                "source": item["page_url"],
                "scope": "full episode (the audio excerpt covers only part of it)",
                "style": "clean verbatim, human-edited by NASA",
                "turns": turns,
            }
            ref_out.write_text(json.dumps(ref, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
            print(f"[{fid}] wrote {ref_out} ({len(turns)} turns)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
