import subprocess
from pathlib import Path

import numpy as np
from app.models import Word

SR, FRAME = 16000, 0.1, # sample rate; posterior frame length in seconds

def load_audio(path: Path)-> np.ndarray:
  # Decode the audio to ndarray (16KHz, mono float32, [-1,1])
  cmd = ["ffmpeg", "-v", "error", "-i", str(path), "-f", "s16le", "-ac", "1", "-ar", str(SR), "-"]
  raw = subprocess.run(cmd, capture_output = True, check=True).stdout
  return np.frombuffer(raw, np.int16).astype(np.float32)/32768

def load_encoder():
  #load the ECAPA encoder for spectral clustering
  from speechbrain.inference.speaker import EncoderClassifier
  # cache the model
  savedir = Path.home() / ".cache" / "ecapa"
  return EncoderClassifier.from_hparams(source='speechbrain/spkrec-ecapa-voxceleb', savedir = str(savedir))

def speech_windows(words: list[Word], win: float = 1.5, hop: float = 0.75):
  # sliding windows over the regions where words were spoken (gaps <= 0.5 seconds merge into one region)
  regions = []
  for w in words:
    # check if there are regions existing and the selected word to the second region gap is less than 0.5
    if regions and w.start - regions[-1][1] <= 0.5:
      regions[-1][1] = max(regions[-1][1], w.end)
    else:
      regions.append([w.start, w.end])

  out= []

  for s, e in regions:
    # too short for voice (0.4s)
    if e-s < 0.4: 
      continue
    if e-s <= win:
      out.append((s,e))
      continue
    t = s
    while t + win < e:
      out.append((t, t+win))
      t += hop
    out.append((max(s,e-win), e)) # last window ends at region end
  return out

def embed_windows(enc, audio: np.ndarray, windows, win: float=1.5, batch: int = 64):
  import torch
  n = int(win*SR)
  out = []
  for b in range(0, len(windows), batch):
    part = windows[b : b + batch]
    wav = np.zeros((len(part), n), np.float32)
    lens = np.zeros(len(part), np.float32)
    for j, (s, e) in enumerate(part):
      seg = audio[int(s * SR) : int(e * SR)][:n]
      wav[j, : len(seg)] = seg
      lens[j] = max(len(seg), 1) / n  # relative length, so padding is ignored
    # NNPACK kicks in at batch >= 16 and is ~60x slower than the default conv on Apple M1
    with torch.inference_mode(), torch.backends.nnpack.flags(enabled=False):
      out.append(enc.encode_batch(torch.from_numpy(wav), torch.from_numpy(lens)).squeeze(1).numpy())
  return np.concatenate(out)

def cluster_two(emb: np.ndarray, prune: float = 0.3) -> tuple[np.ndarray, np.ndarray]:
    """Spectral clustering on a pruned cosine-similarity graph, then a few k-means refinement steps.
    Returns the L2-normalised embeddings and the two cluster centroids."""
    from sklearn.cluster import SpectralClustering

    x = emb / np.linalg.norm(emb, axis=1, keepdims=True).clip(1e-8)
    sim = x @ x.T
    n = len(x)
    keep = max(2, int(np.ceil(prune * n)))  # keep each window's 30 % most similar neighbours
    thr = np.partition(sim, n - keep, axis=1)[:, n - keep][:, None]
    aff = np.where(sim >= thr, sim.clip(0), 0.0)
    aff = (aff + aff.T) / 2
    np.fill_diagonal(aff, 1.0)
    labels = SpectralClustering(2, affinity="precomputed", assign_labels="cluster_qr", random_state=0).fit_predict(aff)
    for _ in range(5):
        cents = np.stack([x[labels == k].mean(0) for k in (0, 1)])
        cents /= np.linalg.norm(cents, axis=1, keepdims=True)
        new = (x @ cents.T).argmax(1)
        if (new == labels).all():
            break
        labels = new
    return x, cents


def word_loglik(words: list[Word], windows, win_post: np.ndarray, duration: float) -> np.ndarray:
    """Average window posteriors onto 0.1 s frames, then average the frames under each word."""
    nf = int(np.ceil(duration / FRAME)) + 1
    acc, cnt = np.zeros((nf, 2)), np.zeros(nf)
    for (s, e), p in zip(windows, win_post, strict=True):
      a = int(s / FRAME)
      b = min(nf, max(a + 1, int(np.ceil(e / FRAME))))  # ASR timestamps can run past the audio end
      # triangle weights: a window speaks mostly for its centre, so windows straddling a turn change
      # don't drown out the one centred on a short reply ("Thanks for having me.")
      wt = 1 - np.abs((np.arange(a, b) + 0.5) * FRAME - (s + e) / 2) / ((e - s) / 2 + FRAME)
      acc[a:b] += wt[:, None] * p
      cnt[a:b] += wt
    cov = cnt > 0
    fp = np.full((nf, 2), 0.5)
    fp[cov] = acc[cov] / cnt[cov, None]
    idx = np.flatnonzero(cov)
    out = np.zeros((len(words), 2))
    for i, w in enumerate(words):
        a = int(w.start / FRAME)
        b = max(a + 1, int(np.ceil(w.end / FRAME)))
        seg = fp[a:b][cov[a:b]]
        if not len(seg) and len(idx):  # a word no window covered: use the nearest covered frame
            j = idx[np.argmin(np.abs(idx - (a + b) // 2))]
            seg = fp[j : j + 1]
        out[i] = np.log(np.clip(seg.mean(0) if len(seg) else [0.5, 0.5], 1e-4, 1.0))
    return out


def boundaries(words: list[Word], seg_ends: set[int], pause: float = 0.6) -> np.ndarray:
  """True where a speaker change before word i is natural: sentence end, ASR segment end or a pause."""
  return np.array(
    [
      i == 0
      or words[i - 1].text.endswith((".", "?", "!"))
      or (i - 1) in seg_ends
      or words[i].start - words[i - 1].end >= pause
      for i in range(len(words))
    ]
  )


def viterbi(loglik: np.ndarray, bound: np.ndarray, switch: float = 4.0, discount: float = 0.15) -> np.ndarray:
  """Most likely speaker per word; changing speaker costs `switch`, or `switch * discount` at a boundary."""
  n, k = loglik.shape
  if n == 0:
    return np.zeros(0, int)
  score = loglik[0].copy()
  back = np.zeros((n, k), int)
  for i in range(1, n):
    cost = switch * (discount if bound[i] else 1.0)
    cand = score[:, None] - cost * (1 - np.eye(k))  # cand[previous speaker, current speaker]
    back[i] = cand.argmax(0)
    score = cand[back[i], np.arange(k)] + loglik[i]
  path = np.zeros(n, int)
  path[-1] = score.argmax()
  for i in range(n - 1, 0, -1):
    path[i - 1] = back[i, path[i]]
  return path


def diarize(enc, audio: np.ndarray, words: list[Word], seg_ends: set[int]) -> None:
  """Set word.speaker to SPEAKER_00 (whoever speaks first) or SPEAKER_01, in place."""
  windows = speech_windows(words)
  x, cents = cluster_two(embed_windows(enc, audio, windows))
  logits = 10.0 * (x @ cents.T)  # softmax(10 x cosine) = how sure each window is about each speaker
  post = np.exp(logits - logits.max(1, keepdims=True))
  post /= post.sum(1, keepdims=True)
  path = viterbi(word_loglik(words, windows, post, len(audio) / SR), boundaries(words, seg_ends))
  for w, c in zip(words, path, strict=True):
    w.speaker = f"SPEAKER_{int(c != path[0]):02d}"
