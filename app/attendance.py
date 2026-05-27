"""
Attendance logger.

Source of truth is the `events` table (one row per confirmed sighting per
camera, debounced by a per-(name, role) cooldown). The legacy `attendance`
table is preserved for back-compat and historical rows but is no longer
written to; reports derive first-entry / last-exit from `events`.

Storage layer abstracted behind a `--db_url` so SQLite now can be swapped
for Postgres later by changing the URL and installing psycopg.
"""

import logging as log
import os
import sqlite3
import time
from datetime import datetime
from urllib.parse import urlparse

import cv2

SCHEMA_SQLITE = [
    """CREATE TABLE IF NOT EXISTS attendance (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        marked_date TEXT NOT NULL,
        marked_time TEXT NOT NULL,
        confidence REAL,
        snapshot_path TEXT,
        UNIQUE (name, marked_date)
    )""",
    """CREATE TABLE IF NOT EXISTS spoof_attempts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        occurred_date TEXT NOT NULL,
        occurred_time TEXT NOT NULL,
        snapshot_path TEXT
    )""",
    """CREATE TABLE IF NOT EXISTS events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        camera_role TEXT NOT NULL,
        event_date TEXT NOT NULL,
        event_time TEXT NOT NULL,
        event_ts TEXT NOT NULL,
        confidence REAL,
        snapshot_path TEXT
    )""",
    "CREATE INDEX IF NOT EXISTS idx_events_name_date ON events(name, event_date)",
    "CREATE INDEX IF NOT EXISTS idx_events_ts ON events(event_ts)",
]


class AttendanceLogger:
    def __init__(self, db_url, snapshot_dir, camera_role='entry',
                 event_cooldown_s=30.0, spoof_rate_limit_s=5.0):
        if camera_role not in ('entry', 'exit'):
            raise ValueError("camera_role must be 'entry' or 'exit'")
        self.camera_role = camera_role
        self.event_cooldown_s = event_cooldown_s
        self.snapshot_dir = os.path.join(snapshot_dir, camera_role)
        self.spoof_dir = os.path.join(snapshot_dir, 'spoof')
        os.makedirs(self.snapshot_dir, exist_ok=True)
        os.makedirs(self.spoof_dir, exist_ok=True)

        parsed = urlparse(db_url)
        if parsed.scheme in ('sqlite', '') and (parsed.path or db_url):
            db_path = parsed.path.lstrip('/') if parsed.scheme == 'sqlite' else db_url
            self.conn = sqlite3.connect(db_path, check_same_thread=False)
            self.placeholder = '?'
            self._schema = SCHEMA_SQLITE
        else:
            raise NotImplementedError(
                'Only sqlite is wired up. For postgres, install psycopg and '
                'extend AttendanceLogger.__init__.')
        for stmt in self._schema:
            self.conn.execute(stmt)
        self.conn.commit()

        # Per-name debounce, scoped to this camera_role. Keyed by name.
        self._last_event_at = {}
        self._last_spoof_at = 0.0
        self._spoof_rate_limit_s = spoof_rate_limit_s

    def _crop_face(self, frame, roi):
        x1 = max(int(roi.position[0]), 0)
        y1 = max(int(roi.position[1]), 0)
        x2 = min(int(roi.position[0] + roi.size[0]), frame.shape[1])
        y2 = min(int(roi.position[1] + roi.size[1]), frame.shape[0])
        if x2 <= x1 or y2 <= y1:
            return None
        return frame[y1:y2, x1:x2]

    def mark(self, name, confidence, frame, roi):
        """Log an event for this camera's role, debounced per-name by
        event_cooldown_s. Returns (logged, time_str)."""
        now_mono = time.monotonic()
        last = self._last_event_at.get(name, 0.0)
        if now_mono - last < self.event_cooldown_s:
            return False, None
        self._last_event_at[name] = now_mono

        now = datetime.now()
        date_str = now.strftime('%Y-%m-%d')
        time_str = now.strftime('%H:%M:%S')
        ts_str = now.isoformat(timespec='seconds')
        snapshot_path = os.path.join(
            self.snapshot_dir,
            f'{name}_{date_str}_{time_str.replace(":", "-")}.jpg')

        crop = self._crop_face(frame, roi)
        if crop is not None:
            cv2.imwrite(snapshot_path, crop)

        self.conn.execute(
            f'INSERT INTO events (name, camera_role, event_date, event_time, '
            f'event_ts, confidence, snapshot_path) VALUES '
            f'({self.placeholder}, {self.placeholder}, {self.placeholder}, '
            f'{self.placeholder}, {self.placeholder}, {self.placeholder}, '
            f'{self.placeholder})',
            (name, self.camera_role, date_str, time_str, ts_str,
             float(confidence), snapshot_path))
        self.conn.commit()
        log.info('Event: %s %s at %s %s', name, self.camera_role, date_str, time_str)
        return True, time_str

    def log_spoof(self, frame, roi):
        """Log a spoof attempt, rate-limited so a held-up phone doesn't spam."""
        now_mono = time.monotonic()
        if now_mono - self._last_spoof_at < self._spoof_rate_limit_s:
            return False
        self._last_spoof_at = now_mono

        now = datetime.now()
        date_str = now.strftime('%Y-%m-%d')
        time_str = now.strftime('%H:%M:%S')
        snapshot_path = os.path.join(
            self.spoof_dir, f'spoof_{date_str}_{time_str.replace(":", "-")}.jpg')
        crop = self._crop_face(frame, roi)
        if crop is not None:
            cv2.imwrite(snapshot_path, crop)
        self.conn.execute(
            f'INSERT INTO spoof_attempts (occurred_date, occurred_time, snapshot_path) '
            f'VALUES ({self.placeholder}, {self.placeholder}, {self.placeholder})',
            (date_str, time_str, snapshot_path))
        self.conn.commit()
        log.warning('Spoof attempt logged at %s %s', date_str, time_str)
        return True

    def close(self):
        self.conn.close()
