import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

@dataclass
class Word:
  text: str
  start: float
  end: float
  speaker: str = ""

@dataclass
class Utterance:
  idx: int
  speaker: str
  start: float
  end: float
  text: str
  words: list[list] = field(default_factory=list)
  
  @property
  def is_question(self)-> bool:
    return self.text.rstrip().endswith("?")

@dataclass
class Speaker:
  label: str
  role: str = "unknown"
  name: str | None = None
  confidence: float = 0.0
  n_words: int = 0
  question_rate: float = 0.0

@dataclass
class Chunk:
  id: str
  file_id: str
  utt_start: int 
  utt_end: int #exclusive
  start: float
  end: float
  speakers: list[str]
  text: str
  
@dataclass
class Transcript:
  file_id: str
  duration: float
  utterances: list[Utterance]
  speakers: list[Speaker]

  def save(self, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(self), ensure_ascii=False))

  @classmethod
  def load(cls, path: Path):
    d = json.loads(path.read_text())
    utterances = [Utterance(**u) for u in d['utterances']]
    return cls(d['file_id'], d['duration'], utterances, [Speaker(**s) for s in d['speakers']])