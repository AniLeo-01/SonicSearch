"""
to run a single file: uv run python -m app.ingest --only runway
to run all the files: uv run python -m app.ingest
uv run python -m app.ingest --asr-only       # only make sure every file is transcribed
uv run python -m app.ingest --force          # redo speaker labelling even if a transcript exists
"""
import argparse

import httpx

from app.asr import to_words, transcribe
from app.config import settings
from app.dataset import ManifestEntry, load_manifest
from app.diarize import SR, diarize, load_audio, load_encoder
from app.models import Transcript
from app.transcript import build_utterances, infer_roles


def build_transcript(entry: ManifestEntry, client: httpx.Client, encoder) -> Transcript:
  """ASR (cached) -> speaker per word -> utterances -> host/guest roles, saved next to the other transcripts."""
  words, seg_ends = to_words(transcribe(entry, client))
  audio = load_audio(entry.audio_path)
  diarize(encoder, audio, words, seg_ends)
  utts = build_utterances(words)
  t = Transcript(entry.file_id, round(len(audio) / SR, 3), utts, infer_roles(utts, entry.host, entry.guest))
  t.save(settings.transcripts_dir / f"{entry.file_id}.json")
  return t


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--only", nargs="*", help="file_ids to process (default: all)")
  ap.add_argument("--asr-only", action="store_true", help="stop after transcription")
  ap.add_argument("--force", action="store_true", help="rebuild transcripts that already exist")
  args = ap.parse_args()
  entries = [e for e in load_manifest() if not args.only or e.file_id in args.only]
  encoder = None
  with httpx.Client(timeout=settings.asr_timeout_sec) as client:
    for entry in entries:
      out = settings.transcripts_dir / f"{entry.file_id}.json"
      if args.asr_only or (out.exists() and not args.force):
        words, _ = to_words(transcribe(entry, client))  # on an error, files done so far stay cached
        print(f"{entry.file_id}: {len(words)} words" + (" (transcript already built)" if out.exists() else ""))
        continue
      if encoder is None:
        encoder = load_encoder()  # load the speaker model once, and only if we need it
      t = build_transcript(entry, client, encoder)
      roles = ", ".join(f"{s.label}={s.role} ({s.name})" for s in t.speakers)
      print(f"{entry.file_id}: {sum(len(u.words) for u in t.utterances)} words, {len(t.utterances)} utterances; {roles}")

if __name__ == "__main__":
    main()