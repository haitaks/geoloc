# -*- coding: utf-8 -*-
"""Доразметка переносом для модели 3 (городища/фортификации/архитектуры):
объект, размеченный на одном источнике места, добавляется на все другие
растры этого места, где он в покрытии, но не размечен."""
import json
import math
import os
import re
import sys
from collections import Counter

import rasterio
from rasterio.warp import transform_bounds

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
ROOT = r'G:\projects\train-expds-3'
SRC_RE = re.compile(r'_(La|Li|Ae|SpOr|Sp|Or)_')
CLS_RE = re.compile(r'(курганы_целые|курганы_поврежденные|городищ[аа]|фортификации|архитектуры)')
HOMO = str.maketrans({'А': 'A', 'а': 'a', 'Е': 'E', 'е': 'e', 'О': 'O', 'о': 'o',
                      'С': 'C', 'с': 'c', 'Р': 'P', 'р': 'p'})
MATCH_M = 50.0
CLASSES = ('городища', 'фортификации', 'архитектуры')


def centroid(rings):
    ring = rings[0] if isinstance(rings[0][0][0], (int, float)) else rings[0][0]
    xs = [c[0] for c in ring]
    ys = [c[1] for c in ring]
    return sum(xs) / len(xs), sum(ys) / len(ys)


def site_utm_epsg(top):
    try:
        j = json.load(open(os.path.join(ROOT, top, 'UTM.json'), encoding='utf-8'))
        z = re.match(r'(\d+)', str(j.get('utm', '')))
        return f'EPSG:326{int(z.group(1)):02d}' if z else None
    except Exception:
        return None


adds = Counter()        # (целевой источник, класс) -> добавок
unique_objs = Counter() # уникальные объекты классов по месту (дедуп 50 м)
for top in sorted(os.listdir(ROOT)):
    site_path = os.path.join(ROOT, top)
    if not os.path.isdir(site_path):
        continue
    polys = {'Li': [], 'Ae': [], 'SpOr': [], 'Sp': [], 'Or': []}
    bounds = {}
    for dp, _dn, fns in os.walk(site_path):
        for fn in fns:
            full = os.path.join(dp, fn)
            rel = os.path.relpath(full, site_path)
            if fn.endswith('.geojson'):
                ms = SRC_RE.search(fn.translate(HOMO))
                cm = CLS_RE.search(fn)
                if not (ms and cm):
                    continue
                g = json.load(open(full, encoding='utf-8'))
                for f in g.get('features', []):
                    if f.get('geometry'):
                        try:
                            polys[ms.group(1)].append(centroid(f['geometry']['coordinates']) + (cm.group(1),))
                        except Exception:
                            pass
            elif fn.endswith(('.tif', '.tiff')):
                m = SRC_RE.search(rel.translate(HOMO))
                if not m:
                    continue
                try:
                    with rasterio.open(full) as r:
                        crs = r.crs
                        if crs is None or crs.to_string().startswith('LOCAL_CS'):
                            crs = site_utm_epsg(top)
                        if crs is None:
                            continue
                        b = transform_bounds(crs, 'EPSG:3857', *r.bounds, densify_pts=21)
                        key = m.group(1)
                        bounds[key] = (min(bounds[key][0], b[0]), min(bounds[key][1], b[1]),
                                       max(bounds[key][2], b[2]), max(bounds[key][3], b[3])) if key in bounds else b
                except Exception:
                    pass

    # уникальные объекты места (любой источник, дедуп 50 м)
    uniq = []
    for src, pts in polys.items():
        for p in pts:
            if not any(math.hypot(p[0] - u[0], p[1] - u[1]) <= MATCH_M for u in uniq):
                uniq.append(p)
    for _x, _y, cls in uniq:
        if cls in CLASSES:
            unique_objs[cls] += 1

    # перенос: объект -> на каждый источник с растром, где он в покрытии, но не размечен
    for x, y, cls in uniq:
        if cls not in CLASSES:
            continue
        for tgt in ('Li', 'Ae', 'SpOr', 'Sp', 'Or'):
            if tgt not in bounds:
                continue
            b = bounds[tgt]
            if not (b[0] - 50 <= x <= b[2] + 50 and b[1] - 50 <= y <= b[3] + 50):
                continue
            has = any(math.hypot(x - px, y - py) <= MATCH_M for px, py, _c in polys[tgt])
            if not has:
                adds[(tgt, cls)] += 1

print('Уникальных объектов (после дедупа 50 м):', dict(unique_objs))
print('\nДобавок переносом (растр, класс):')
for (tgt, cls), n in sorted(adds.items(), key=lambda kv: -kv[1]):
    print(f'  {tgt:5} + {cls}: {n}')
print('Сумма добавок:', sum(adds.values()))
