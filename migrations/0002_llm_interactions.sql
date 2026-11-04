-- migrations/0002_llm_interactions.sql

CREATE TABLE IF NOT EXISTS llm_interactions (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  source TEXT NOT NULL,
  question TEXT NOT NULL,
  answer TEXT NOT NULL,
  suggested_action TEXT,
  confidence REAL NOT NULL DEFAULT 1.0,
  model TEXT NOT NULL,
  latency_ms INTEGER,
  metadata_json TEXT,
  created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_llm_interactions_created_at
  ON llm_interactions(created_at);
