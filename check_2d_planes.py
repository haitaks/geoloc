# -*- coding: utf-8 -*-
"""Можно ли все источники визуализировать на одной 2D-плоскости?
Проверяем: размеры/привязку растров, атрибуты и охват облаков точек,
совпадение координатных рамок LAS и Li-растра одного места."""
import os
import sys
from pathlib import Path

import laspy
import rasterio

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
ROOT = Path('G:/projects/train-expds-3')


def raster_info(path):
    with rasterio.open(path) as src:
        crs = src.crs.to_string() if src.crs else 'НЕТ CRS'
        return {
            'crs': crs, 'w': src.width, 'h': src.height,
            'res': src.res[0], 'bounds': tuple(src.bounds), 'count': src.count,
            'dt': src.dtypes[0],
        }


# 1) Все ли растры имеют валидную геопривязку?
bad = []
n_raster = 0
per_src_crs = {}
for p in ROOT.rglob('*.tif'):
    n_raster += 1
    with rasterio.open(p) as src:
        crs = src.crs.to_string() if src.crs else 'NO CRS'
        res = src.res[0]
        if res is None or res <= 0 or (crs == 'NO CRS'):
            bad.append((str(p.relative_to(ROOT)), crs, res))
        rel = str(p.relative_to(ROOT))
        src_tag = 'Li' if '_Li_' in rel else ('Sp' if ('_Sp' in rel or '_SpOr' in rel)
                                              else ('Ae' if '_Ae' in rel or '_Ае' in rel else 'Or'))
        per_src_crs.setdefault(src_tag, set()).add(crs.split('[')[0][:20])
print(f'Растров: {n_raster}, без привязки/нулевого разрешения: {len(bad)}')
for b in bad:
    print('  ПРОБЛЕМА:', b)
print('CRS по источникам:', {k: sorted(v) for k, v in per_src_crs.items()})

# 2) Or в EPSG:4326 — какое разрешение в метрах?
print('\n--- Or-растры (детали) ---')
for p in sorted(ROOT.rglob('*.tif')):
    rel = str(p.relative_to(ROOT))
    if '_Or_' not in rel and '_ортофото' not in rel:
        continue
    if any(x in rel for x in ('_Li_', '_Ae_', '_Ае_', '_Sp')):
        continue
    ri = raster_info(p)
    res_m = ri['res'] * (111320 if '4326' in ri['crs'] else 1)
    print(f'  {rel.split("/")[0]}: {ri["crs"]}, {ri["w"]}x{ri["h"]}, '
          f'{ri["res"]:.3g} ед/пикс (~{res_m:.3f} м/пикс)')

# 3) LAS: атрибуты и совпадение рамки с Li-растром того же места
print('\n--- LAS: атрибуты и рамки vs Li-растр ---')
for site in ['015_Демидовка', '035_Шумгора', '002_Тешилово', '009_Хальвасборген']:
    site_dir = ROOT / site
    las_files = list(site_dir.rglob('*.las'))
    li_files = [p for p in site_dir.rglob('*.tif') if '_Li_' in str(p)]
    if not las_files:
        continue
    with laspy.open(las_files[0]) as f:
        h = f.header
        xs = (h.x_min, h.x_max)
        ys = (h.y_min, h.y_max)
        zs = (h.z_min, h.z_max)
        n = h.point_count
        fmt = int(h.point_format.id)
        dims = [d.name for d in h.point_format.dimensions]
    line = (f'  {site}: {n/1e6:.1f} млн тчк, формат {fmt}, '
            f'X {xs[0]:.0f}..{xs[1]:.0f}, Y {ys[0]:.0f}..{ys[1]:.0f} '
            f'({(xs[1]-xs[0])/1000:.2f} x {(ys[1]-ys[0])/1000:.2f} км), '
            f'Z {zs[0]:.0f}..{zs[1]:.0f}')
    print(line)
    print(f'    измерения точки: {", ".join(dims)}')
    if li_files:
        ri = raster_info(li_files[0])
        b = ri['bounds']
        print(f'    Li-растр: X {b[0]:.0f}..{b[2]:.0f}, Y {b[1]:.0f}..{b[3]:.0f} '
              f'({ri["crs"][:30]}) — пересечение с LAS: '
              f'{"ДА" if (min(xs[1], b[2]) - max(xs[0], b[0])) > 0 and (min(ys[1], b[3]) - max(ys[0], b[1])) > 0 else "НЕТ"}')
