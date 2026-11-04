# -*- coding: utf-8 -*-
"""Статистика по слоям, форматам и типам разметки в train-expds-3."""
import os
import re
import sys
from collections import Counter

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
root = r'G:\projects\train-expds-3'

exts = Counter()
layer_kinds = Counter()
marks = Counter()
for dp, _dn, fns in os.walk(root):
    for n in fns:
        rel = os.path.relpath(os.path.join(dp, n), root)
        rel = rel.replace(os.sep, '/')
        exts[os.path.splitext(n)[1].lower()] += 1
        parts = rel.split('/')
        if len(parts) == 3:  # <место>/<слой>/<файл>
            m = re.match(r'\d+_(.+)', parts[1])
            if m:
                layer_kinds[m.group(1)] += 1
        if n.endswith('.geojson') and 'разметка' in dp:
            marks[re.sub(r'^[^_]+_', '', n.replace('.geojson', ''))] += 1

print('Расширения файлов:', dict(exts.most_common()))
print()
print('Типы слоёв (по второму уровню папок):')
for k, v in sorted(layer_kinds.items()):
    print(f'  {k}: {v}')
print()
print('Типы разметки geojson:')
for k, v in sorted(marks.items()):
    print(f'  {k}: {v}')
