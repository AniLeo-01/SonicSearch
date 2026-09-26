from pathlib import Path

import psycopg
from pgvector.psycopg import register_vector
from psycopg_pool import ConnectionPool

from app.config import settings

def init_db():
  with psycopg.connect(settings.db_url, autocommit=True) as conn:
    conn.execute(Path(__file__).with_name("schema.sql").read_text())

def make_pool():
  return ConnectionPool(settings.db_url, min_size=settings.db_pool_min, max_size=settings.db_pool_max, configure=register_vector, open=True)

def connect():
  conn = psycopg.connect(settings.db_url)
  register_vector(conn)
  return conn