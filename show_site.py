# -*- coding: utf-8 -*-
"""Показать файлы выбранного места локально и в expds-3 (по нормализованному имени)."""
import json
import re
import sys

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

data = json.load(open('listings.json', encoding='utf-8'))
b3 = data['expds3']
local = data['local']


def norm_site(name):
    s = re.sub(r'^\d+[_ ]*', '', name)
    s = s.lower().replace('ё', 'е')
    return re.sub(r'[_\s\-.]+', '', s)


for site in sys.argv[1:]:
    print(f'===== {site} =====')
    print('--- ЛОКАЛЬНО (expds-2) ---')
    for k, sz in sorted(local.items()):
        if norm_site(k.split('/')[0]) == site:
            print(f'  {sz/2**20:10.2f} MiB  {k}')
    print('--- В EXPDS-3 ---')
    for k, sz in sorted(b3.items()):
        if norm_site(k.split('/')[0]) == site:
            print(f'  {sz/2**20:10.2f} MiB  {k}')
    print()
