import json
from app.models import Word
from app.dataset import ManifestEntry
import httpx
from app.config import settings

TRAILING_PUNC = '?./,;:"])'

def transcribe(entry: ManifestEntry, client: httpx.Client):
  cache = settings.transcripts_dir / f"{entry.file_id}.groq.json"
  if cache.exists():
    return json.loads(cache.read_text())

  api_key = settings.groq_api_key.get_secret_value()
  if not api_key:
    raise RuntimeError("Groq API KEY not set")
  with entry.audio_path.open('rb') as f:
    r = client.post(
      settings.groq_url,
      headers={"Authorization": f"Bearer {api_key}"},
      files={"file": (entry.audio_path.name, f, 'audio/mpeg')},
      data = {
        "model": settings.asr_model,
        "temperature": "0",
        "response_format":"verbose_json",
        "language": settings.asr_language,
        "timestamp_granularities[]": ["word", "segment"]
      }
    )
  if r.is_error:
    raise RuntimeError(f"Groq returned {r.status_code}: {r.text}")
  resp = r.json()
  cache.parent.mkdir(parents=True, exist_ok=True)
  cache.write_text(json.dumps(resp))
  print(f"{entry.file_id} is sent to Groq; requests left {r.headers.get('x-ratelimit-remaining-requests')}")
  return resp
    
def to_words(response: dict):
  """Align Groq's timestamped words with the punctuated transcript text.

  Each word is located in `response['text']` (searching forward from the last match) and takes
  its trailing punctuation from there (`tons` -> `tons.`). Sub-word pieces glued in the text
  (`don` + `'t`, `WB` + `-57`) are merged into one Word. Starts are forced non-decreasing and
  every word lasts at least 20 ms. Words not found in the text are kept as Groq sent them.

  Returns (words, seg_ends): the Word list, and the indices of words that end a Whisper segment.
  """
  text = response.get('text', '')
  words: list[Word] = []
  cursor = 0
  for w in response.get('words', []):
    tok = w['word'].strip()
    if not tok:
      continue
    start,end = float(w['start']), float(w['end'])
    pos = text.find(tok, cursor)
    if pos>=0:
      stop = j = pos + len(tok)
      while j< len(text) and text[j] in TRAILING_PUNC:
        j+=1
      if j == len(text) or text[j].isspace():
        stop = j
      glued = bool(words) and pos == cursor and pos > 0 and not text[pos-1].isspace()
      tok, cursor = text[pos:stop], stop
      if glued:
        prev = words[-1]
        words[-1] = Word(prev.text + tok, prev.start, max(prev.end, end))
        continue
    start = max(start, words[-1].start if words else 0.0)
    words.append(Word(tok, start, max(end, start+0.02)))
  seg_ends: set[int] = set()
  i = 0
  for seg in response.get("segments", []):
    while i < len(words) and words[i].start < seg['end']:
      i+=1
    if i:
      seg_ends.add(i-1)
  return words, seg_ends
    