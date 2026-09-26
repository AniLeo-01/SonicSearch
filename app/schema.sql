CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE EXTENSION IF NOT EXISTS fuzzystrmatch;

CREATE TABLE IF NOT EXISTS audio_files (
  file_id text PRIMARY KEY,
  title text NOT NULL,
  duration_sec float8 NOT NULL
);

CREATE TABLE IF NOT EXISTS speakers (
  file_id text REFERENCES audio_files ON DELETE CASCADE,
  label text,
  role text NOT NULL,
  name text,
  PRIMARY KEY (file_id, label)
);

CREATE TABLE IF NOT EXISTS utterances (
  file_id text REFERENCES audio_files ON DELETE CASCADE,
  idx int,
  speaker text NOT NULL,
  start_sec float8 NOT NULL,
  end_sec float8 NOT NULL,
  text text NOT NULL,
  words jsonb NOT NULL, --[[token, start, end], ...]
  tsv tsvector GENERATED ALWAYS AS (to_tsvector('english',text)) STORED,
  embedding vector(768),
  PRIMARY KEY (file_id, idx) 
);

-- unit we retrieve (50 word groups of utterances) 
CREATE TABLE IF NOT EXISTS chunks (
  id text PRIMARY KEY,
  file_id text REFERENCES audio_files ON DELETE CASCADE,
  utt_start int NOT NULL,
  utt_end int NOT NULL,
  start_sec float8 NOT NULL,
  end_sec float8 NOT NULL,
  speakers text[] NOT NULL,
  text text NOT NULL,
  tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', text)) STORED,
  doc_len int NOT NULL DEFAULT 0,
  embedding vector(768)
);

-- ColBERT token vectors (float16, 128 per token) that order the "related" moments; ALTER so existing databases get it
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS colbert bytea;

--BM25 postings
CREATE TABLE IF NOT EXISTS chunk_terms(
  lexeme text,
  chunk_id text REFERENCES chunks ON DELETE CASCADE,
  tf int NOT NULL,
  PRIMARY KEY (lexeme, chunk_id)
);

-- spoken words, for sounds-like expansion
CREATE TABLE IF NOT EXISTS vocabulary(
  term text PRIMARY KEY,
  df int NOT NULL,
  metaphone text NOT NULL,
  dmetaphone text NOT NULL,
  dmetaphone_alt text NOT NULL
);

CREATE INDEX IF NOT EXISTS chunks_tsv ON chunks USING gin (tsv);
CREATE INDEX IF NOT EXISTS vocabulary_trgm ON vocabulary USING gin (term gin_trgm_ops);
CREATE INDEX IF NOT EXISTS vocabulary_metaphone ON vocabulary (metaphone);