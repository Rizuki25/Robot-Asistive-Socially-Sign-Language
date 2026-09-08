#!/usr/bin/env python3
"""Read-only inspection: does not import servo drivers or move the robot."""
import ast
from pathlib import Path
import sqlite3


root = Path('/home/ubuntu')
manager = root / 'ros_ws/src/ainex_driver/ainex_kinematics/src/ainex_kinematics/motion_manager.py'
print('=== MotionManager source ===')
if manager.is_file():
    source = manager.read_text(encoding='utf-8')
    tree = ast.parse(source)
    lines = source.splitlines()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in (
                '__init__', 'run_action', 'set_servos_position'):
            print('\n'.join(lines[node.lineno - 1:node.end_lineno]))
else:
    print('NOT FOUND:', manager)

folder = root / 'software/ainex_controller/ActionGroups'
print('\n=== Available actions ===')
print('\n'.join(p.name for p in sorted(folder.glob('*.d6a'))))
greet = folder / 'greet.d6a'
if greet.is_file():
    print('\n=== greet database (read-only) ===')
    try:
        with sqlite3.connect(greet.as_uri() + '?mode=ro', uri=True) as db:
            tables = db.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
            print('Tables:', tables)
            if ('ActionGroup',) in tables:
                print('Columns:', db.execute('PRAGMA table_info(ActionGroup)').fetchall())
                print('Frames:', db.execute('SELECT COUNT(*) FROM ActionGroup').fetchone())
                print('First frames:', db.execute('SELECT * FROM ActionGroup LIMIT 3').fetchall())
    except sqlite3.Error as error:
        print('Cannot inspect action:', error)

print('\n=== Head configuration references ===')
examples = root / 'ros_ws/src/ainex_example'
shown = 0
for path in sorted(examples.rglob('*.py')):
    for number, line in enumerate(path.read_text(encoding='utf-8', errors='replace').splitlines(), 1):
        if any(key in line for key in ('head_tilt_init =', 'head_tilt_range =',
                                       'head_tilt_init=', 'head_tilt_range=')):
            print('{}:{}: {}'.format(path, number, line.strip()))
            shown += 1
            if shown >= 15:
                break
    if shown >= 15:
        break
