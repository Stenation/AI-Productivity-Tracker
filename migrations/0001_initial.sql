-- migrations/0001_initial.sql

PRAGMA journal_mode = WAL;

CREATE TABLE IF NOT EXISTS schema_meta (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS activity_spans (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  activity TEXT NOT NULL,
  source TEXT NOT NULL,
  confidence REAL NOT NULL DEFAULT 1.0,
  reason TEXT NOT NULL,
  started_at TEXT NOT NULL,
  ended_at TEXT,
  metadata_json TEXT
);

CREATE INDEX IF NOT EXISTS idx_activity_started
  ON activity_spans(started_at);

CREATE TABLE IF NOT EXISTS window_spans (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  app_class TEXT NOT NULL,
  window_title TEXT NOT NULL,
  category TEXT NOT NULL,
  classifier TEXT NOT NULL,
  calendar_event_id TEXT,
  started_at TEXT NOT NULL,
  ended_at TEXT,
  metadata_json TEXT
);

CREATE INDEX IF NOT EXISTS idx_window_started
  ON window_spans(started_at);

CREATE TABLE IF NOT EXISTS calendar_cache (
  event_id TEXT PRIMARY KEY,
  calendar_id TEXT NOT NULL,
  title TEXT NOT NULL,
  description TEXT,
  color_id TEXT,
  task_type TEXT NOT NULL DEFAULT 'unknown',
  suitable_for_walking INTEGER,
  recommended_pomodoro TEXT,
  start_time TEXT NOT NULL,
  end_time TEXT NOT NULL,
  raw_json TEXT,
  updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_calendar_cache_times
  ON calendar_cache(start_time, end_time);

CREATE TABLE IF NOT EXISTS timer_spans (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  kind TEXT NOT NULL,
  calendar_event_id TEXT,
  started_at TEXT NOT NULL,
  ended_at TEXT,
  planned_duration_seconds INTEGER,
  actual_duration_seconds INTEGER,
  status TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_timer_spans_status
  ON timer_spans(status, ended_at);

CREATE TABLE IF NOT EXISTS recommendations (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  policy TEXT NOT NULL,
  message TEXT NOT NULL,
  suggested_action TEXT,
  action_payload_json TEXT,
  dedupe_key TEXT,
  status TEXT NOT NULL DEFAULT 'pending',
  created_at TEXT NOT NULL,
  resolved_at TEXT,
  resolution_metadata_json TEXT
);

CREATE INDEX IF NOT EXISTS idx_recommendations_dedupe_created
  ON recommendations(dedupe_key, created_at);

CREATE TABLE IF NOT EXISTS calendar_drafts (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  operation TEXT NOT NULL,
  event_id TEXT,
  proposed_changes_json TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'pending',
  created_at TEXT NOT NULL,
  decided_at TEXT,
  decision_metadata_json TEXT
);

CREATE TABLE IF NOT EXISTS module_health (
  module TEXT PRIMARY KEY,
  status TEXT NOT NULL,
  last_event_at TEXT,
  last_error TEXT,
  updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS settings (
  key TEXT PRIMARY KEY,
  value_json TEXT NOT NULL,
  updated_at TEXT NOT NULL
);

