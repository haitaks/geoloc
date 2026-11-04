# -*- coding: utf-8 -*-
"""Оценка трёх моделей: по классам и источникам — сколько объектов
в местах с обеими модальностями (Li+фото) vs только с одной."""
import json
import os
import re
import sys
from collections import Counter

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
ROOT = r'G:\projects\train-expds-3'
SRC_RE = re.compile(r'_(La|Li|Ae|SpOr|Sp|Or)_')
CLS_RE = re.compile(r'(курганы_целые|курганы_поврежденные|городищ[аа]|фортификации|архитектуры)')
HOMO = str.maketrans({'А': 'A', 'а': 'a', 'Е': 'E', 'е': 'e', 'О': 'O', 'о': 'o',
                      'С': 'C', 'с': 'c', 'Р': 'P', 'р': 'p'})

# класс -> modality_combo -> объектов;  класс -> источник -> объектов
by_combo = {}
by_src = {}
places_combo = Counter()

for top in sorted(os.listdir(ROOT)):
    site_path = os.path.join(ROOT, top)
    if not os.path.isdir(site_path):
        continue
    sources, markup = set(), Counter()   # markup: (src, cls) -> объектов
    for layer in os.listdir(site_path):
        layer_path = os.path.join(site_path, layer)
        if not os.path.isdir(layer_path):
            continue
        m = SRC_RE.search((layer + '_').translate(HOMO))
        if m:
            sources.add(m.group(1))
        if 'разметка' in layer:
            for fn in os.listdir(layer_path):
                if not fn.endswith('.geojson'):
                    continue
                ms = SRC_RE.search(fn.translate(HOMO))
                cm = CLS_RE.search(fn)
                if not (ms and cm):
                    continue
                g = json.load(open(os.path.join(layer_path, fn), encoding='utf-8'))
                markup[(ms.group(1), cm.group(1))] += sum(
                    1 for f in g.get('features', []) if f.get('geometry'))
    has_li = bool(sources & {'La', 'Li'})
    has_photo = bool(sources & {'Ae', 'SpOr', 'Sp', 'Or'})
    combo = 'Li+фото' if (has_li and has_photo) else ('только Li' if has_li else
                                                      ('только фото' if has_photo else 'нет данных'))
    places_combo[combo] += 1
    for (src, cls), n in markup.items():
        by_src.setdefault(cls, Counter())[src] += n
        by_combo.setdefault(cls, Counter())[combo] += n

print('Мест по модальностям:', dict(places_combo))
for cls in ('курганы_поврежденные', 'курганы_целые', 'городища', 'фортификации', 'архитектуры'):
    print(f'\n=== {cls} ===')
    print('  по источникам разметки:', dict(by_src.get(cls, {})))
    print('  по модальности места:  ', dict(by_combo.get(cls, {})))
