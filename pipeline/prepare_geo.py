# -*- coding: utf-8 -*-
"""Препроцессинг геофизики (папки 200-217) под модель 4.

Выходы в work/geo/:
  rasters.json   — реестр: путь, модальность, глубина среза (EtGr), разметка в CRS растра,
                   зоны шума/границы из extra/
  tiles.csv      — манифест окон 512@0.1 м/пикс со счётчиками классов
  split.json     — сплит по местам (val ~25%, стратификация по модальности)
"""
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import rasterio
from pyproj import CRS, Transformer
from rasterio.warp import transform_bounds

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
ROOT = Path(r'G:\projects\train-expds-3')
WORK = Path(r'G:\projects\geoloc\work\geo')
TILE, RES, OVERLAP = 512, 0.1, 0.25


def kind_of(stem):
    if '_MagGr_серый' in stem:
        return 'maggr_gray', None
    if '_MagGr_цвет' in stem:
        return 'maggr_color', None
    m = re.search(r'_EtGr_карта_\d+_(\d+(?:\.\d+)?)$', stem)
    if m:
        return 'etgr_map', float(m.group(1))       # глубина среза, м
    if '_EtGr_профиль' in stem:
        return 'etgr_prof', None
    if '_TerFl' in stem:
        return 'terfl', None
    return None, None


def transform_feats(geoms, src_crs, dst_crs):
    tr = Transformer.from_crs(CRS.from_string(src_crs), CRS.from_string(dst_crs), always_xy=True)

    def ring(r):
        return [[float(x), float(y)] for x, y in (tr.transform(px, py) for px, py in r)]

    out = []
    for geom in geoms:
        if geom['type'] == 'Point':
            x, y = tr.transform(*geom['coordinates'][:2])
            out.append({'type': 'Point', 'coordinates': [float(x), float(y)]})
        elif geom['type'] == 'Polygon':
            out.append({'type': 'Polygon', 'coordinates': [ring(r) for r in geom['coordinates']]})
        elif geom['type'] == 'MultiPolygon':
            out.append({'type': 'MultiPolygon',
                        'coordinates': [[ring(r) for r in poly] for poly in geom['coordinates']]})
    return out


def main():
    WORK.mkdir(parents=True, exist_ok=True)
    rasters = {}
    for site_dir in sorted(p for p in ROOT.iterdir() if p.is_dir() and re.match(r'2\d\d_', p.name)):
        site = site_dir.name
        subs = {s.name: s for s in site_dir.iterdir() if s.is_dir()}
        marks_dir = next((d for n, d in subs.items() if 'разметка' in n), None)
        extra_dir = subs.get('extra')
        all_marks = {}
        if marks_dir:
            for g in sorted(marks_dir.glob('*.geojson')):
                fc = json.load(open(g, encoding='utf-8'))
                crs = fc.get('crs', {}).get('properties', {}).get('name', 'EPSG:32636')
                stem = g.stem
                if stem.endswith('_FindsPoints'):
                    cls, base = 'points', stem[:-len('_FindsPoints')]
                elif stem.endswith('_ObjectPoly'):
                    cls, base = 'polys', stem[:-len('_ObjectPoly')]
                else:
                    continue
                all_marks.setdefault(base, {})[cls] = (fc['features'], crs)
        noise, bounds = [], []
        if extra_dir:
            for g in sorted(extra_dir.glob('*.geojson')):
                fc = json.load(open(g, encoding='utf-8'))
                crs = fc.get('crs', {}).get('properties', {}).get('name', 'EPSG:32636')
                if 'шум' in g.stem:
                    noise.append((fc['features'], crs))
                elif 'границы' in g.stem:
                    bounds.append((fc['features'], crs))

        for sub in subs.values():
            for f in sorted(sub.glob('*.tif')):
                kind, depth = kind_of(f.stem)
                if not kind:
                    continue
                with rasterio.open(f) as src:
                    ras_crs = src.crs.to_string() if src.crs else 'EPSG:32636'
                    rec = dict(path=str(f), site=site, kind=kind, depth=depth, crs=ras_crs,
                               w=src.width, h=src.height,
                               res=float(max(abs(src.transform.a), abs(src.transform.e))),
                               transform=[src.transform.a, src.transform.b, src.transform.c,
                                          src.transform.d, src.transform.e, src.transform.f],
                               bounds=list(src.bounds), stem=f.stem)
                # разметка: EtGr-карта -> своя (по базовому имени файла);
                # MagGr/TerFl -> разметка места своей модальности
                base = f.stem
                marks = all_marks.get(base)
                if not marks and kind in ('maggr_gray', 'maggr_color'):
                    cand = [b for b in all_marks if '_MagGr' in b]
                    marks = all_marks[cand[0]] if cand else None
                if not marks and kind == 'terfl':
                    cand = [b for b in all_marks if '_TerFl' in b]
                    marks = all_marks[cand[0]] if cand else None
                rec['marks'] = {}
                if marks:
                    for cls, (feats, crs) in marks.items():
                        geoms = [ft['geometry'] for ft in feats if ft.get('geometry')]
                        rec['marks'][cls] = transform_feats(geoms, crs, ras_crs)
                rec['noise'] = [g for feats, crs in noise for g in transform_feats(
                    [f2['geometry'] for f2 in feats if f2.get('geometry')], crs, ras_crs)]
                rec['bounds_zone'] = [g for feats, crs in bounds for g in transform_feats(
                    [f2['geometry'] for f2 in feats if f2.get('geometry')], crs, ras_crs)]
                rasters[f.stem] = rec

    json.dump(rasters, open(WORK / 'rasters.json', 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    n_pts = sum(len(r['marks'].get('points', [])) for r in rasters.values())
    n_pol = sum(len(r['marks'].get('polys', [])) for r in rasters.values())
    print(f"Растров: {len(rasters)} | точек: {n_pts} | полигонов: {n_pol} | "
          f"с разметкой: {sum(1 for r in rasters.values() if r['marks'])}")

    # манифест окон
    rows = []
    for stem, r in sorted(rasters.items()):
        if r['kind'] == 'etgr_prof':
            continue                                   # профили — второй эшелон
        win = max(8, min(r['w'], round(TILE * RES / r['res'])))
        win = min(win, r['w'])
        step = max(1, round(win * (1 - OVERLAP)))
        ny = 1 if r['h'] <= win else (r['h'] - win) // step + 1
        inv = ~rasterio.transform.Affine(*r['transform'])
        pts = []
        for g in r['marks'].get('points', []):
            px, py = inv * (g['coordinates'][0], g['coordinates'][1])
            pts.append((px, py))
        pgs = []
        for g in r['marks'].get('polys', []):
            ring = g['coordinates'][0] if isinstance(g['coordinates'][0][0][0], (int, float)) \
                else g['coordinates'][0][0]
            cx = sum(c[0] for c in ring) / len(ring)
            cy = sum(c[1] for c in ring) / len(ring)
            pgs.append((*inv * (cx, cy),))
        for iy in range(ny):
            for ix in range((1 if r['w'] <= win else (r['w'] - win) // step + 1)):
                col = min(ix * step, r['w'] - win)
                row = min(iy * step, r['h'] - win)
                n1 = sum(1 for px, py in pts if col <= px < col + win and row <= py < row + win)
                n2 = sum(1 for (px, py) in pgs if col <= px < col + win and row <= py < row + win)
                if n1 + n2 == 0 and (ix + iy) % 3:      # экономим пустые окна (каждое 3-е)
                    continue
                rows.append((stem, r['site'], r['kind'], col, row, win, win, n1, n2))
    with open(WORK / 'tiles.csv', 'w', encoding='utf-8') as fh:
        fh.write('raster;site;kind;col;row;win_x;win_y;n_points;n_polys\n')
        for t in rows:
            fh.write(';'.join(map(str, t)) + '\n')
    pos = sum(1 for t in rows if t[7] + t[8] > 0)
    print(f'Окон: {len(rows)} (с объектами: {pos})')

    # сплит по местам: val 25%, стратификация по модальности
    import random
    rng = random.Random(42)
    sites = sorted({r['site'] for r in rasters.values() if r['kind'] != 'etgr_prof'})
    mag = [s for s in sites if any(r['kind'].startswith('maggr') and r['site'] == s
                                   for r in rasters.values())]
    et = [s for s in sites if s not in mag]
    rng.shuffle(mag)
    rng.shuffle(et)
    val = mag[:max(1, len(mag) // 4)] + et[:max(1, len(et) // 4)]
    split = {'train': sorted(set(sites) - set(val)), 'val': sorted(val)}
    json.dump(split, open(WORK / 'split.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print('Сплит: train', len(split['train']), '/ val', len(split['val']), '| val:', split['val'])


if __name__ == '__main__':
    main()
