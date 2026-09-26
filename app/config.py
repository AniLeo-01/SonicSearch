from pydantic.types import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path

class Settings(BaseSettings):
  model_config = SettingsConfigDict(
    env_file=".env", extra='ignore'
  )
  # storage
  db_url: str = "postgresql://sonicsearch:password@localhost:5432/sonicsearch"
  db_pool_min: int = 1
  db_pool_max: int = 8

  # data layout
  data_dir: Path = "data"

  #ASR
  asr_model: str = "whisper-large-v3-turbo"
  asr_language: str = "en"
  asr_timeout_sec: float = 300

  # groq
  groq_api_key: SecretStr = SecretStr("")
  groq_url: str = "https://api.groq.com/openai/v1/audio/transcriptions"
  # retrieval (values tuned on the dev split in the reference build)
  embed_model: str = "BAAI/bge-base-en-v1.5"  # 768-d: must match vector(768) in schema.sql
  chunk_words: int = 50  # passage size, in words
  chunk_stride: int = 25  # a new passage starts every ~25 words
  depth: int = 50  # candidates per retrieval channel
  rrf_k: int = 10  # rank-fusion constant
  nms_gap_sec: float = 10.0  # results closer than this in the same file are duplicates
  rerank_model: str = "Alibaba-NLP/gte-reranker-modernbert-base"  # cross-encoder over the fused candidates
  rerank_min: float = 0.83  # drop moments the reranker scores below this (tuned on the dev split)
  colbert_model: str = "lightonai/GTE-ModernColBERT-v1"  # late interaction: orders the moments below the cutoff
  @property
  def transcripts_dir(self):
    return self.data_dir / "transcripts"

settings = Settings()