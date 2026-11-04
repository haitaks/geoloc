# -*- coding: utf-8 -*-
"""Анализ пересечения локальных данных (expds-2) с новым датасетом expds-3.

Папки верхнего уровня перенумерованы между версиями, поэтому сопоставляем
по нормализованному имени места и сравниваем внутреннюю структуру файлов.
"""
import json
import re
import sys
from collections import defaultdict
from pathlib import PurePosixPath

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

data = json.load(open('listings.json', encoding='utf-8'))
b3 = data['expds3']
local = data['local']


def norm_site(name: str) -> str:
    """Нормализованное имя места: без номера, нижний регистр, без разделителей."""
    s = re.sub(r'^\d+[_ ]*', '', name)
    s = s.lower().replace('ё', 'е')
    s = re.sub(r'[_\s\-.]+', '', s)
    return s


def group_by_site(files):
    by_site = defaultdict(dict)
    for key, size in files.items():
        top = key.split('/', 1)[0]
        rest = key.split('/', 1)[1] if '/' in key else ''
        by_site[norm_site(top)][rest.lower()] = (key, size)
    return by_site


sites3 = group_by_site(b3)
sitesL = group_by_site(local)

# Несколько локальных папок могут нормализоваться одинаково — проверим
dupl = {s: len([t for t in local if norm_site(t.split('/')[0]) == s])
        for s in sitesL}
dupl = {s: n for s, n in dupl.items() if n > 1}
if dupl:
    print('ЛОКАЛЬНЫЕ ПАППЫ С ОДИНАКОВЫМ НОРМ-ИМЕНЕМ:', dupl)

new_sites = set(sites3) - set(sitesL)
common = set(sites3) & set(sitesL)
print(f'\nМест в expds-3: {len(sites3)}; из них локально есть: {len(common)}; только новых: {len(new_sites)}')

tot_match_bytes = tot_miss_bytes = 0
full, partial = [], []
for site in sorted(common):
    inner3 = sites3[site]
    innerL = sitesL[site]
    matched = sum(size for rest, (key, size) in inner3.items()
                  if rest in innerL and innerL[rest][1] == size)
    missing = sum(size for rest, (key, size) in inner3.items()
                  if rest not in innerL or innerL[rest][1] != size)
    tot_match_bytes += matched
    tot_miss_bytes += missing
    (full if missing == 0 else partial).append((site, matched, missing, len(inner3), len(innerL)))

print(f'\nПересечение по общим местам: {tot_match_bytes/2**30:.2f} GiB совпадает, '
      f'{tot_miss_bytes/2**30:.2f} GiB отсутствует/отличается')
print(f'Полностью совпавших мест: {len(full)}, частично: {len(partial)}')

new_only_bytes = sum(size for site in new_sites for _k, size in
                     [(r[0], r[1][1]) for r in sites3[site].items()])
print(f'Объём полностью новых мест: {new_only_bytes/2**30:.2f} GiB')
print(f'ИТОГО реально качать: {(tot_miss_bytes + new_only_bytes)/2**30:.2f} GiB из 199.43 GiB')

print('\n--- Полностью совпавшие места (внутренние пути и размеры идентичны) ---')
for site, m, _x, n3, nL in full:
    print(f'  {site}: {n3} файлов, {m/2**30:.2f} GiB')

print('\n--- Частично совпавшие (первый десяток) ---')
for site, m, x, n3, nL in sorted(partial, key=lambda t: -t[2])[:10]:
    print(f'  {site}: совпало {m/2**30:.2f} GiB, не хватает {x/2**30:.2f} GiB (файлов: новых {n3}, локально {nL})')

print('\n--- Совсем новые места ---')
for site in sorted(new_sites):
    sz = sum(v[1] for v in sites3[site].values())
    print(f'  {site}: {len(sites3[site])} файлов, {sz/2**30:.2f} GiB')
