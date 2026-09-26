from app.models import Chunk, Transcript

def build_chunks(t: Transcript, target: int = 50, stride: int = 25):
  """Windows of target words that never cut a sentence, starting every stride words.
  A window may span both speakers, so a question and the start of its answer stay together. Search
  later "snaps" each hit to a single utterance, so the speaker shown for a result is never ambiguous.
  """
  utts = t.utterances
  counts = [len(u.text.split()) for u in utts]
  chunks: list[Chunk] = []
  i, n = 0, len(utts)
  while i < n:
    j, words = i, 0
    while j < n and words < target:
      words += counts[j]
      j += 1
    win = utts[i:j]
    chunks.append(
      Chunk(
        id=f"{t.file_id}_c{len(chunks):04d}",
        file_id=t.file_id,
        utt_start=i,
        utt_end=j,
        start=win[0].start,
        end=win[-1].end,
        speakers=list(dict.fromkeys(u.speaker for u in win)),
        text=" ".join(u.text for u in win),
      )
    )
    if j >= n:
      break
    k, acc = i, 0
    while k < j and acc < stride:
      acc += counts[k]
      k += 1
    i = max(k, i + 1)
  return chunks