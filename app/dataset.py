from dataclasses import dataclass
from app.config import settings
from pathlib import Path
import yaml

@dataclass(frozen=True, slots=True)
class ManifestEntry:
  file_id: str
  title: str
  audio_path: Path
  host: str | None = None
  guest: str | None = None

def load_manifest():
  raw = yaml.safe_load((settings.data_dir / "manifest.yaml").read_text(encoding="utf-8")) or {}
  entries: list[ManifestEntry] = []
  seen: set[str] = set()
  for item in raw.get("files", []):
    fid = str(item["file_id"])
    if fid in seen:
      raise ValueError(f"duplicate file_id in manifest: {fid}")
    seen.add(fid)
    speakers = item.get("speakers") or {}
    entries.append(
      ManifestEntry(
        file_id=fid,
        title=str(item.get("title", fid)),
        audio_path=settings.data_dir / "audio" / f"{fid}.mp3",
        host=speakers.get("host"),
        guest=speakers.get("guest"),
      )
    )
  return entries
