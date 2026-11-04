# -*- coding: utf-8 -*-
"""Объёмы по типам данных и общие параметры источников train-expds-3."""
import json
import os
import re
import sys
from collections import Counter, defaultdict

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import laspy
import rasterio

ROOT = r'G:\projects\train-expds-3'
GiB = 2**30


def source_kind(key: str) -> str:
    """Источник данных по ключу '<место>/<слой>/<файл>'."""
    parts = key.split('/')
    if len(parts) < 3:
        return 'UTM.json'
    layer = parts[1]
    m = re.match(r'\d+_(.+)', layer)
    name = m.group(1) if m else layer
    m2 = re.match(r'.*?_(La|Li|Ae|SpOr|Sp|Or)_[а-яё]+$', name)
    return m2.group(1) if m2 else name


# --- объёмы из сохранённого листинга ---
data = json.load(open('listings.json', encoding='utf-8'))
b3 = data['expds3']
vol = Counter()
cnt = Counter()
for key, size in b3.items():
    k = source_kind(key)
    vol[k] += size
    cnt[k] += 1

print('=== Объём и количество по источникам ===')
for k in sorted(vol, key=lambda x: -vol[x]):
    print(f'  {k:8} {cnt[k]:5} файлов  {vol[k]/GiB:8.2f} GiB')

# --- параметры растров ---
print('\n=== Параметры GeoTIFF по источникам (все 457 файлов) ===')
raster_stats = defaultdict(Counter)
for dp, _dn, fns in os.walk(ROOT):
    for n in fns:
        if not n.lower().endswith(('.tif', '.tiff')):
            continue
        rel = os.path.relpath(os.path.join(dp, n), ROOT).replace(os.sep, '/')
        kind = source_kind(rel)
        try:
            with rasterio.open(os.path.join(dp, n)) as src:
                raster_stats[kind][(src.width == src.height, src.count,
                                    str(src.dtypes[0]),
                                    round(src.res[0], 3),
                                    src.crs.to_string() if src.crs else None)] += 1
        except Exception as e:
            raster_stats[kind][f'ОШИБКА {type(e).__name__}'] += 1

for kind in sorted(raster_stats):
    print(f'  {kind}:')
    for params, c in raster_stats[kind].most_common():
        print(f'    {c:4} шт: каналов={params[1]}, {params[2]}, '
              f'разрешение={params[3]} м/пикс, СК={params[4]}')

# --- параметры LAS ---
print('\n=== Параметры облаков точек .las (все 56) ===')
las_stats = Counter()
las_ext = Counter()
for dp, _dn, fns in os.walk(ROOT):
    for n in fns:
        if not n.lower().endswith('.las'):
            continue
        path = os.path.join(dp, n)
        try:
            with laspy.open(path) as f:
                h = f.header
                las_stats[(int(h.point_format.id), h.version)] += 1
                las_ext[n.lower().endswith('.las') and path] = 0
        except Exception as e:
            las_stats[f'ОШИБКА {type(e).__name__}'] += 1
for params, c in las_stats.most_common():
    print(f'  {c:3} шт: формат точек {params}')

# --- зоны UTM ---
print('\n=== Зоны UTM (по UTM.json, 192 места) ===')
zones = Counter()
for dp, _dn, fns in os.walk(ROOT):
    if 'UTM.json' in fns:
        try:
            j = json.load(open(os.path.join(dp, 'UTM.json'), encoding='utf-8'))
            zones[j.get('utm', j.get('crs', '?'))] += 1
        except Exception:
            zones['?'] += 1
for z, c in sorted(zones.items()):
    print(f'  {z}: {c} мест')
