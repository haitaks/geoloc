# -*- coding: utf-8 -*-
"""Покрытие разметкой: у всех ли мест/источников есть .geojson с объектами."""
import json
import os
import re
import sys
from collections import Counter, defaultdict

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

ROOT = r'G:\projects\train-expds-3'
SRC_RE = re.compile(r'_(La|Li|Ae|SpOr|Sp|Or)_')
HOMOGLYPHS = str.maketrans({'А': 'A', 'Е': 'E', 'О': 'O', 'С': 'C', 'Р': 'P',
                            'а': 'a', 'е': 'e', 'о': 'o', 'с': 'c', 'р': 'p'})


def norm(s: str) -> str:
    """Кириллические омофоны (Ае, Сп, Ор...) -> латиница, чтобы не терять разметку."""
    return s.translate(HOMOGLYPHS)

sites = {}  # место -> {'sources': set, 'markup_srcs': set, 'files': int, 'features': Counter}
for top in sorted(os.listdir(ROOT)):
    top_path = os.path.join(ROOT, top)
    if not os.path.isdir(top_path):
        continue
    info = {'sources': set(), 'markup_srcs': set(), 'features': Counter()}
    for layer in os.listdir(top_path):
        layer_path = os.path.join(top_path, layer)
        if not os.path.isdir(layer_path):
            continue
        m = SRC_RE.search(norm(layer) + '_')
        if 'разметка' in layer:
            for fn in os.listdir(layer_path):
                if fn.endswith('.geojson'):
                    ms = SRC_RE.search(norm(fn))
                    if ms:
                        info['markup_srcs'].add(ms.group(1))
                    try:
                        g = json.load(open(os.path.join(layer_path, fn), encoding='utf-8'))
                        cls = re.sub(r'^.*?_(курганы|городища|фортификации|архитектуры)', r'\1', fn.replace('.geojson', ''))
                        cls_full = norm(fn.replace('.geojson', '')).split('_', 1)[1]
                        info['features'][cls_full] += len(g.get('features', []))
                    except Exception as e:
                        info['features'][f'ОШИБКА ЧТЕНИЯ {fn}'] += 0
        elif m:
            info['sources'].add(m.group(1))
    sites[top] = info

no_markup = [t for t, i in sites.items() if not i['markup_srcs']]
partial = {t: i for t, i in sites.items()
           if i['markup_srcs'] and (i['sources'] - i['markup_srcs'])}
orphan = {t: i for t, i in sites.items() if i['markup_srcs'] - i['sources']}

print(f'Всего мест: {len(sites)}')
print(f'Без разметки вообще: {len(no_markup)}')
for t in no_markup:
    print(f'  {t}  (источники: {", ".join(sorted(sites[t]["sources"])) or "нет"})')

print(f'\nС разметкой не на все источники: {len(partial)}')
miss_counter = Counter()
for t, i in sorted(partial.items()):
    for s in i['sources'] - i['markup_srcs']:
        miss_counter[s] += 1
print('  чаще всего без разметки источник:', dict(miss_counter))
for t, i in sorted(partial.items())[:15]:
    print(f'  {t}: есть данные {sorted(i["sources"])}, размечено {sorted(i["markup_srcs"])}')

print(f'\nРазметка без данных (источник размечен, слоя нет): {len(orphan)}')
for t, i in sorted(orphan.items())[:15]:
    print(f'  {t}: размечено {sorted(i["markup_srcs"])}, данные {sorted(i["sources"])}')

total_feats = sum(sum(i['features'].values()) for i in sites.values())
print(f'\nВсего объектов в разметке: {total_feats}')
cls_counter = Counter()
for i in sites.values():
    for k, v in i['features'].items():
        key = re.sub(r'^.*?_(La|Li|Ae|SpOr|Sp|Or)_', r'\1: ', k)
        cls_counter[key] += v
for k, v in sorted(cls_counter.items(), key=lambda x: -x[1]):
    print(f'  {k}: {v}')

# есть ли пустые geojson (0 объектов)?
empty = []
for top, i in sites.items():
    pass
print()
n_files = 0
n_empty = 0
for dp, _dn, fns in os.walk(ROOT):
    for fn in fns:
        if fn.endswith('.geojson'):
            n_files += 1
            try:
                g = json.load(open(os.path.join(dp, fn), encoding='utf-8'))
                if not g.get('features'):
                    n_empty += 1
                    if n_empty <= 10:
                        print(f'  ПУСТОЙ: {os.path.relpath(os.path.join(dp, fn), ROOT)}')
            except Exception:
                pass
print(f'GeoJSON файлов: {n_files}, из них пустых (0 объектов): {n_empty}')
