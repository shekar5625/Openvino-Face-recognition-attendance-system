"""
Attendance logger.

Storage layer abstracted behind a `--db_url` so SQLite now can be swapped for
Postgres later by changing the URL and installing psycopg.
"""

import logging as log
import os
import sqlite3
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
]


class AttendanceLogger:
    def __init__(self, db_url, snapshot_dir, spoof_rate_limit_s=5.0):
        self.snapshot_dir = snapshot_dir
        self.spoof_dir = os.path.join(snapshot_dir, 'spoof')
        os.makedirs(self.snapshot_dir, exist_ok=True)
        os.makedirs(self.spoof_dir, exist_ok=True)

        parsed = urlparse(db_url)
        if parsed.scheme in ('sqlite', '') and (parsed.path or db_url):
            # Accept "sqlite:///abs/path.db" or bare "path.db".
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
        """Insert one attendance row per (name, date). Returns True if the row
        was new (i.e. this is the first sighting today), False if already marked."""
        now = datetime.now()
        date_str = now.strftime('%Y-%m-%d')
        time_str = now.strftime('%H:%M:%S')
        snapshot_path = os.path.join(
            self.snapshot_dir, f'{name}_{date_str}_{time_str.replace(":", "-")}.jpg')

        cur = self.conn.execute(
            f'INSERT INTO attendance (name, marked_date, marked_time, confidence, snapshot_path) '
            f'VALUES ({self.placeholder}, {self.placeholder}, {self.placeholder}, {self.placeholder}, {self.placeholder}) '
            f'ON CONFLICT (name, marked_date) DO NOTHING',
            (name, date_str, time_str, float(confidence), snapshot_path))
        self.conn.commit()

        if cur.rowcount > 0:
            crop = self._crop_face(frame, roi)
            if crop is not None:
                cv2.imwrite(snapshot_path, crop)
            log.info('Attendance: %s marked at %s %s', name, date_str, time_str)
            return True, time_str
        return False, None

    def log_spoof(self, frame, roi):
        """Log a spoof attempt, rate-limited so a held-up phone doesn't spam."""
        import time
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
