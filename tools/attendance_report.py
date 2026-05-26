"""
Print an attendance roster for a given date (default: today).

Usage:
    python tools/attendance_report.py             # today
    python tools/attendance_report.py 2026-05-26  # specific date
    python tools/attendance_report.py --spoof     # show spoof attempts for today
"""
import argparse
import sqlite3
import sys
from datetime import date
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
DEFAULT_DB = PROJECT / 'attendance.db'

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('day', nargs='?', default=date.today().isoformat(),
               help='Date in YYYY-MM-DD format. Defaults to today.')
p.add_argument('--db', default=str(DEFAULT_DB), help='SQLite DB path.')
p.add_argument('--spoof', action='store_true',
               help='Show spoof attempts instead of attendance.')
p.add_argument('--reset', choices=('attendance', 'spoof', 'all'),
               help='Wipe data and exit. "all" also deletes face snapshots.')
p.add_argument('--yes', action='store_true', help='Skip confirmation prompt for --reset.')
args = p.parse_args()

if not Path(args.db).exists():
    sys.exit(f'No attendance DB at {args.db}')

conn = sqlite3.connect(args.db)

if args.reset:
    targets = {
        'attendance': ['attendance'],
        'spoof': ['spoof_attempts'],
        'all': ['attendance', 'spoof_attempts'],
    }[args.reset]
    if not args.yes:
        print(f'About to wipe: {", ".join(targets)}'
              + (' + snapshots in logs/' if args.reset == 'all' else ''))
        if input('Type "yes" to confirm: ').strip().lower() != 'yes':
            sys.exit('Aborted.')
    for table in targets:
        conn.execute(f'DELETE FROM {table}')
        conn.execute(f'DELETE FROM sqlite_sequence WHERE name = ?', (table,))
    conn.commit()
    if args.reset == 'all':
        import shutil
        snap_dir = PROJECT / 'logs'
        if snap_dir.exists():
            shutil.rmtree(snap_dir)
            print(f'Removed {snap_dir}')
    print(f'Cleared: {", ".join(targets)}')
    sys.exit(0)

if args.spoof:
    rows = conn.execute(
        'SELECT occurred_time, snapshot_path FROM spoof_attempts '
        'WHERE occurred_date = ? ORDER BY occurred_time', (args.day,)).fetchall()
    print(f'Spoof attempts on {args.day}: {len(rows)}')
    print('-' * 60)
    for t, path in rows:
        print(f'  {t}   {path}')
else:
    rows = conn.execute(
        'SELECT name, marked_time, confidence FROM attendance '
        'WHERE marked_date = ? ORDER BY marked_time', (args.day,)).fetchall()
    print(f'Attendance on {args.day}: {len(rows)} present')
    print('-' * 60)
    print(f'{"Name":<25} {"Time":<10} {"Confidence":>10}')
    print('-' * 60)
    for name, t, conf in rows:
        print(f'{name:<25} {t:<10} {conf:>9.1%}')
