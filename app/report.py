"""Daily attendance report derived from the events table.

Usage:
    python app/report.py [--db attendance.db] [--date YYYY-MM-DD]

Defaults to today's date and ./attendance.db.
"""

import argparse
import sqlite3
from datetime import date


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--db', default='attendance.db')
    ap.add_argument('--date', default=date.today().isoformat())
    args = ap.parse_args()

    conn = sqlite3.connect(args.db)
    rows = conn.execute(
        """
        SELECT name,
               MIN(CASE WHEN camera_role='entry' THEN event_time END) AS first_entry,
               MAX(CASE WHEN camera_role='exit'  THEN event_time END) AS last_exit,
               SUM(CASE WHEN camera_role='entry' THEN 1 ELSE 0 END) AS n_entries,
               SUM(CASE WHEN camera_role='exit'  THEN 1 ELSE 0 END) AS n_exits
        FROM events
        WHERE event_date = ?
        GROUP BY name
        ORDER BY first_entry IS NULL, first_entry
        """,
        (args.date,)).fetchall()

    if not rows:
        print(f'No events for {args.date}')
        return

    print(f'Attendance report — {args.date}')
    print(f'{"Name":<20} {"First entry":<12} {"Last exit":<12} {"#in":>4} {"#out":>5}  Status')
    print('-' * 70)
    for name, first_entry, last_exit, n_in, n_out in rows:
        if first_entry and not last_exit:
            status = 'INSIDE'
        elif last_exit and not first_entry:
            status = 'EXIT-WITHOUT-ENTRY'
        elif n_in > n_out:
            status = 'INSIDE'
        else:
            status = 'left'
        print(f'{name:<20} {first_entry or "-":<12} {last_exit or "-":<12} '
              f'{n_in:>4} {n_out:>5}  {status}')


if __name__ == '__main__':
    main()
