import math
import re

from app.models import Speaker, Utterance, Word

ABBREVIATIONS = {"dr.", "mr.", "mrs.", "ms.", "st.", "jr.", "sr.", "prof.", "vs.", "etc.", "e.g.", "i.e.", "u.s.",
                 "u.k.", "a.m.", "p.m.", "no.", "approx.", "inc.", "co.", "lt.", "col.", "gen.", "sgt."}  # fmt: skip
SENTENCE_END = re.compile(r"[.?!][\"')\]]*$")


def ends_sentence(token: str) -> bool:
    t = token.lower()
    if t in ABBREVIATIONS or re.fullmatch(r"(?:[a-z]\.){2,}", t):  # "Dr." and "U.S." don't end sentences
        return False
    return bool(SENTENCE_END.search(token))


def build_utterances(words: list[Word], max_words: int = 50, max_gap: float = 1.5) -> list[Utterance]:
    """Start a new utterance on a speaker change, a sentence end, a pause >= max_gap, or after max_words
    (split at the last comma in the second half, so long monologues still break at a natural point)."""
    utts: list[Utterance] = []
    cur: list[Word] = []

    def flush() -> None:
        if cur:
            text = " ".join(w.text for w in cur)
            timings = [[w.text, w.start, w.end] for w in cur]
            utts.append(Utterance(len(utts), cur[0].speaker, cur[0].start, cur[-1].end, text, timings))
            cur.clear()

    for w in words:
        if cur and (w.speaker != cur[-1].speaker or w.start - cur[-1].end >= max_gap):
            flush()
        cur.append(w)
        if ends_sentence(w.text):
            flush()
        elif len(cur) >= max_words:
            k = next((k for k in range(len(cur) - 1, len(cur) // 2, -1) if cur[k].text.endswith(",")), None)
            if k is None:
                flush()
            else:
                tail = cur[k + 1 :]
                del cur[k + 1 :]
                flush()
                cur.extend(tail)
    flush()
    return utts


def infer_roles(utts: list[Utterance], host: str | None = None, guest: str | None = None) -> list[Speaker]:
    """The host asks more questions, talks less and usually speaks first. Names from the manifest are
    attached only when the decision is confident."""
    total = sum(len(u.text.split()) for u in utts) or 1
    scored = []
    for label in dict.fromkeys(u.speaker for u in utts):
        mine = [u for u in utts if u.speaker == label]
        n_words = sum(len(u.text.split()) for u in mine)
        q_rate = sum(u.is_question for u in mine) / len(mine)
        score = 3.0 * q_rate + 1.5 * (0.5 - n_words / total) + 0.5 * (label == utts[0].speaker)
        scored.append((score, Speaker(label, n_words=n_words, question_rate=round(q_rate, 4))))
    scored.sort(key=lambda t: -t[0])
    margin = scored[0][0] - scored[1][0] if len(scored) > 1 else 1.0
    confidence = 1 / (1 + math.exp(-4 * margin))
    for i, (_, s) in enumerate(scored):
        s.role, s.confidence = ("host" if i == 0 else "guest"), round(confidence, 4)
        if confidence >= 0.75:
            s.name = host if i == 0 else guest
    return [s for _, s in scored]