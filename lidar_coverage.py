# -*- coding: utf-8 -*-
"""Есть ли у всех размеченных мест лидарные данные (La/Li)?
Сколько мест и объектов разметки остаётся, если брать только лидарные места."""
import json
import os
import re
import sys
from collections import Counter, defaultdict

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
ROOT = r'G:\projects\train-expds-3'
SRC_RE = re.compile(r'_(La|Li|Ae|SpOr|Sp|Or)_')
HOMO = str.maketrans({'А': 'A', 'а': 'a', 'Е': 'E', 'е': 'e', 'О': 'O', 'о': 'o',
                      'С': 'C', 'с': 'c', 'Р': 'P', 'р': 'p'})

sites = {}
for top in sorted(os.listdir(ROOT)):
    top_path = os.path.join(ROOT, top)
    if not os.path.isdir(top_path):
        continue
    sources, markup = set(), Counter()   # markup: источник -> объектов
    for layer in os.listdir(top_path):
        layer_path = os.path.join(top_path, layer)
        if not os.path.isdir(layer_path):
            continue
        if 'разметка' in layer:
            for fn in os.listdir(layer_path):
                if not fn.endswith('.geojson'):
                    continue
                ms = SRC_RE.search(fn.translate(HOMO))
                src = ms.group(1) if ms else '?'
                g = json.load(open(os.path.join(layer_path, fn), encoding='utf-8'))
                n = sum(1 for f in g.get('features', []) if f.get('geometry'))
                markup[src] += n
        else:
            m = SRC_RE.search((layer + '_').translate(HOMO))
            if m:
                sources.add(m.group(1))
    sites[top] = {'sources': sources, 'markup': markup}

lidar = {t: ('La' in i['sources'] or 'Li' in i['sources']) for t, i in sites.items()}
marked = {t: i for t, i in sites.items() if sum(i['markup'].values()) > 0}

print(f'Мест с разметкой: {len(marked)} из {len(sites)}')
print(f'Мест с лидаром (La/Li): {sum(lidar.values())}')
print(f'Мест с разметкой БЕЗ лидара: {sum(1 for t in marked if not lidar[t])}')

no_lidar_sites = [t for t in marked if not lidar[t]]
obj_no_lidar = sum(sum(sites[t]['markup'].values()) for t in no_lidar_sites)
obj_lidar = sum(sum(sites[t]['markup'].values()) for t in marked if lidar[t])
print(f'Объектов на лидарных местах: {obj_lidar}, на безлидарных: {obj_no_lidar}')

# какие источники размечены на безлидарных местах
src_counter = Counter()
for t in no_lidar_sites:
    src_counter.update(sites[t]['markup'])
print('Разметка на безлидарных местах по источникам:', dict(src_counter))

# а на лидарных местах — сколько объектов размечено именно на Li
li_obj = sum(sites[t]['markup'].get('Li', 0) for t in marked if lidar[t])
other_on_lidar = sum(v for t in marked if lidar[t] for s, v in sites[t]['markup'].items() if s != 'Li')
print(f'На лидарных местах: на Li размечено {li_obj} объектов, на других источниках {other_on_lidar}')

print('\nПримеры безлидарных мест (первые 15):')
for t in no_lidar_sites[:15]:
    print(f'  {t}: источники {sorted(sites[t]["sources"])}, '
          f'разметка {dict(sites[t]["markup"])}')
