# -*- coding: utf-8 -*-
"""Оценка стратегии: перенос на Li всех фото-классов кроме поврежденных курганов
+ шум для SpOr-модели поврежденных курганов от чужой разметки."""
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


def centroid(rings):
    ring = rings[0] if isinstance(rings[0][0][0], (int, float)) else rings[0][0]
    xs = [c[0] for c in ring]
    ys = [c[1] for c in ring]
    return sum(xs) / len(xs), sum(ys) / len(ys)


def site_utm_epsg(top):
    """Зона UTM места из UTM.json — запасная СК для LOCAL_CS/None растров."""
    try:
        j = json.load(open(os.path.join(ROOT, top, 'UTM.json'), encoding='utf-8'))
        z = re.match(r'(\d+)', str(j.get('utm', '')))
        return f'EPSG:326{int(z.group(1)):02d}' if z else None
    except Exception:
        return None


def load_site(top):
    """-> полигоны по источникам, границы растров по источникам (в 3857)."""
    polys = {'Li': [], 'Ae': [], 'SpOr': [], 'Sp': [], 'Or': []}
    bounds = {}
    for dp, _dn, fns in os.walk(os.path.join(ROOT, top)):
        for fn in fns:
            full = os.path.join(dp, fn)
            rel = os.path.relpath(full, os.path.join(ROOT, top))
            if fn.endswith('.geojson'):
                ms = SRC_RE.search(fn.translate(HOMO))
                if not ms:
                    continue
                src = ms.group(1)
                cls = (CLS_RE.search(fn) or [None])
                cls = cls.group(1) if hasattr(cls, 'group') else '?'
                g = json.load(open(full, encoding='utf-8'))
                for f in g.get('features', []):
                    if f.get('geometry'):
                        try:
                            cx, cy = centroid(f['geometry']['coordinates'])
                            polys[src].append((cx, cy, cls))
                        except Exception:
                            pass
            elif fn.endswith(('.tif', '.tiff')):
                m = SRC_RE.search(rel.translate(HOMO))
                if not m:
                    continue
                src = m.group(1)
                try:
                    with rasterio.open(full) as r:
                        crs = r.crs
                        if crs is None or crs.to_string().startswith('LOCAL_CS'):
                            crs = site_utm_epsg(top)   # локальная СК Li = метры UTM зоны места
                        if crs is None:
                            continue
                        b = transform_bounds(crs, 'EPSG:3857', *r.bounds, densify_pts=21)
                        if src in bounds:
                            bounds[src] = (min(bounds[src][0], b[0]), min(bounds[src][1], b[1]),
                                           max(bounds[src][2], b[2]), max(bounds[src][3], b[3]))
                        else:
                            bounds[src] = b
                except Exception:
                    pass
    return polys, bounds


def inside(b, x, y, pad=50):
    return b is not None and b[0] - pad <= x <= b[2] + pad and b[1] - pad <= y <= b[3] + pad


def nearest_dist(pts, x, y):
    return min((math.hypot(x - px, y - py) for px, py, _c in pts), default=1e9)


# ===== 1. Перенос фото->Li (кроме поврежденных) на 25 местах с двойной разметкой =====
transfer = Counter()          # класс -> добавится к Li
excluded_damaged = 0          # поврежденные с фото внутри Li-покрытия без Li-пары
outside_li = 0                # фото-объекты вне рамки Li-растра
li_base = Counter()           # текущая Li-разметка по классам
for top in sorted(os.listdir(ROOT)):
    if not os.path.isdir(os.path.join(ROOT, top)):
        continue
    polys, bounds = load_site(top)
    if not (polys['Li'] and (polys['Ae'] or polys['SpOr'] or polys['Sp'])):
        continue
    for cls in (c for _x, _y, c in polys['Li']):
        li_base[cls] += 1
    for src in ('Ae', 'SpOr', 'Sp'):
        for px, py, cls in polys[src]:
            if not inside(bounds.get('Li'), px, py):
                outside_li += 1
                continue
            if nearest_dist(polys['Li'], px, py) <= MATCH_M:
                continue
            if cls == 'курганы_поврежденные':
                excluded_damaged += 1
            else:
                transfer[cls] += 1

print('=== Li-модель после переноса (25 мест с двойной разметкой) ===')
print('Было в Li-разметке:', dict(li_base))
print('Добавится переносом с фото (внутри Li-покрытия, без пары в Li):', dict(transfer))
print('Поврежденных с фото внутри Li-покрытия (в перенос не идут):', excluded_damaged)
print('Фото-объектов вне рамки Li-растра (не переносимы):', outside_li)

# ===== 2. Шум для SpOr-модели поврежденных: чужие объекты внутри SpOr-рамок =====
noise = Counter()
spor_places = 0
spor_dmg = 0
for top in sorted(os.listdir(ROOT)):
    if not os.path.isdir(os.path.join(ROOT, top)):
        continue
    polys, bounds = load_site(top)
    spor_objs = polys['SpOr']
    if not spor_objs:
        continue
    spor_places += 1
    spor_dmg += sum(1 for _x, _y, c in spor_objs if c == 'курганы_поврежденные')
    for src in ('Li', 'Ae'):
        for px, py, cls in polys[src]:
            if cls != 'курганы_поврежденные':
                continue
            if not inside(bounds.get('SpOr'), px, py):
                continue
            if nearest_dist(spor_objs, px, py) <= MATCH_M:
                continue
            noise[f'{src}->SpOr'] += 1

print('\n=== SpOr-модель поврежденных курганов ===')
print('Мест с SpOr-разметкой:', spor_places,
      '| поврежденных в SpOr-разметке:', spor_dmg)
print('Чужие поврежденные внутри SpOr-рамок без SpOr-пары (потенциальный ложный фон):', dict(noise))
