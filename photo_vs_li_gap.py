# -*- coding: utf-8 -*-
"""Есть ли объекты, размеченные на фото (Ae/SpOr/Sp), но не размеченные на Li,
в местах, где Li-данные есть? С поправкой на покрытие Li-растра."""
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict

import rasterio
from rasterio.warp import transform_bounds

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
ROOT = r'G:\projects\train-expds-3'
SRC_RE = re.compile(r'_(La|Li|Ae|SpOr|Sp|Or)_')
CLS_RE = re.compile(r'(курганы_целые|курганы_поврежденные|городищ[аа]|фортификации|архитектуры)')
HOMO = str.maketrans({'А': 'A', 'а': 'a', 'Е': 'E', 'е': 'e', 'О': 'O', 'о': 'o',
                      'С': 'C', 'с': 'c', 'Р': 'P', 'р': 'p'})
MATCH_M = 50.0   # допуск совпадения в метрах EPSG:3857 (~25-30 м на земле)


def centroid(rings):
    ring = rings[0] if isinstance(rings[0][0][0], (int, float)) else rings[0][0]
    xs = [c[0] for c in ring]
    ys = [c[1] for c in ring]
    return sum(xs) / len(xs), sum(ys) / len(ys)


stats = Counter()
examples = []
per_site = {}

for top in sorted(os.listdir(ROOT)):
    site_path = os.path.join(ROOT, top)
    if not os.path.isdir(site_path):
        continue
    li_polys, photo_polys = [], []
    li_bounds = None
    for dp, _dn, fns in os.walk(site_path):
        for fn in fns:
            full = os.path.join(dp, fn)
            if fn.endswith('.geojson'):
                src = SRC_RE.search(fn.translate(HOMO))
                if not src:
                    continue
                src = src.group(1)
                cls_m = CLS_RE.search(fn)
                cls = cls_m.group(1) if cls_m else '?'
                g = json.load(open(full, encoding='utf-8'))
                for f in g.get('features', []):
                    if not f.get('geometry'):
                        continue
                    try:
                        c = centroid(f['geometry']['coordinates'])
                    except Exception:
                        continue
                    (li_polys if src == 'Li' else
                     photo_polys if src in ('Ae', 'SpOr', 'Sp') else []).append((c[0], c[1], cls, src))
            elif fn.endswith(('.tif', '.tiff')) and '_Li_' in os.path.relpath(full, site_path):
                try:
                    with rasterio.open(full) as r:
                        b = transform_bounds(r.crs, 'EPSG:3857', *r.bounds, densify_pts=21)
                        li_bounds = b if li_bounds is None else (
                            min(li_bounds[0], b[0]), min(li_bounds[1], b[1]),
                            max(li_bounds[2], b[2]), max(li_bounds[3], b[3]))
                except Exception:
                    pass

    if not (li_polys and photo_polys):
        continue  # интересуют только места с Li-разметкой И фото-разметкой

    inside = unmatched = matched = outside = 0
    for px, py, cls, src in photo_polys:
        if li_bounds and not (li_bounds[0] - 50 <= px <= li_bounds[2] + 50 and
                              li_bounds[1] - 50 <= py <= li_bounds[3] + 50):
            outside += 1
            continue
        best = min((math.hypot(px - lx, py - ly) for lx, ly, _c, _s in li_polys),
                   default=1e9)
        if best <= MATCH_M:
            matched += 1
        else:
            unmatched += 1
            if len(examples) < 200:
                examples.append((top, src, cls, round(px), round(py), round(best)))
    stats['places'] += 1
    stats['photo_objs'] += len(photo_polys)
    stats['matched'] += matched
    stats['photo_only_inside'] += unmatched
    stats['photo_only_outside'] += outside
    per_site[top] = (matched, unmatched, outside)

print('Мест с Li-разметкой и фото-разметкой одновременно:', stats['places'])
print('Фото-объектов всего там:', stats['photo_objs'])
print('  совпадают с Li-разметкой (<=50 м в 3857):', stats['matched'])
print('  НЕ размечены на Li, но ВНУТРИ покрытия Li-растра:', stats['photo_only_inside'])
print('  вне покрытия Li-растра (объяснимо):', stats['photo_only_outside'])

print('\nХудшие места (фото-объектов без Li-разметки внутри покрытия):')
for t, (m, u, o) in sorted(per_site.items(), key=lambda kv: -kv[1][1])[:12]:
    print(f'  {t}: совпало {m}, без Li-разметки {u}, вне покрытия {o}')

print('\nПримеры (место, источник фото, класс, координаты 3857, расстояние до ближайшего Li-объекта):')
for e in examples[:15]:
    print('  ', e)
