# -*- coding: utf-8 -*-
"""Разбор геофизических данных (папки 200-217): растры, разметка, extra-зоны."""
import json
import re
import sys
from collections import Counter
from pathlib import Path

import rasterio

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
ROOT = Path(r'G:\projects\train-expds-3')

raster_meta = Counter()
mark_counts = Counter()
sites_by_kind = Counter()
errors = []

for site_dir in sorted(p for p in ROOT.iterdir() if p.is_dir()
                       and re.match(r'2\d\d_', p.name)):
    kinds = set()
    for sub in sorted(site_dir.iterdir()):
        if not sub.is_dir():
            continue
        for f in sorted(sub.glob('*.tif')):
            key = ('MagGr-серый' if '_MagGr_серый' in f.stem else
                   'MagGr-цвет' if '_MagGr_цвет' in f.stem else
                   'EtGr-карта' if '_EtGr_карта' in f.stem else
                   'EtGr-профиль' if '_EtGr_профиль' in f.stem else
                   'TerFl' if '_TerFl' in f.stem else '?')
            try:
                with rasterio.open(f) as src:
                    raster_meta[key] += 1
                    if raster_meta[key] <= 2:
                        print(f'{key}: {f.name} | {src.width}x{src.height} | '
                              f'{src.count}кн {src.dtypes[0]} | '
                              f'{abs(src.transform.a):.3f} м/пикс | '
                              f'{src.crs.to_string()[:18] if src.crs else "нет CRS"}')
            except Exception as e:
                errors.append(f'{f.name}: {e}')
            m = re.search(r'_(MagGr|EtGr|TerFl)_', f.stem)
            if m:
                kinds.add(m.group(1))
    sites_by_kind[frozenset(kinds) or {'?'}] += 1
    md = site_dir / [x.name for x in site_dir.iterdir() if 'разметка' in x.name][0] \
        if any('разметка' in x.name for x in site_dir.iterdir()) else None
    if md:
        for g in sorted(md.glob('*.geojson')):
            fc = json.load(open(g, encoding='utf-8'))
            m = re.search(r'_(FindsPoints|ObjectPoly|StructPoly)$', g.stem)
            cls = m.group(1) if m else g.stem.split('_')[-1]
            geoms = Counter(f['geometry']['type'] for f in fc['features'] if f.get('geometry'))
            mark_counts[(cls, tuple(sorted(geoms)))] += sum(1 for f in fc['features']
                                                             if f.get('geometry'))

print('\nРастры по типам:', dict(raster_meta))
print('Места по набору модальностей:',
      {', '.join(sorted(k)): v for k, v in sites_by_kind.items()})
print('\nРазметка (класс, геометрии) -> объектов:')
for (cls, geoms), n in sorted(mark_counts.items()):
    print(f'  {cls:12} {geoms}: {n}')
extra = list((ROOT / '200_Судьбищи_2').glob('extra/*.geojson'))
for e in extra:
    fc = json.load(open(e, encoding='utf-8'))
    print(f'extra {e.name}: {len(fc["features"])} фич, '
          f'{fc["features"][0]["geometry"]["type"] if fc["features"] else "-"}')
if errors:
    print('Ошибки:', errors[:3])
