# -*- coding: utf-8 -*-
"""Сравнение локальной папки ../data с бакетами train-expds-2 и train-expds-3."""
import boto3
import json
import os
import sys
from collections import Counter
from pathlib import Path

import urllib3
from botocore.config import Config

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

ENDPOINT = 'https://s3.ru-7.storage.selcloud.ru:443'
EXCLUDES = ('Старая', 'Архив')
ROOT = Path(r'G:\projects\data')


def make_client(key, secret):
    return boto3.client(
        's3', endpoint_url=ENDPOINT,
        aws_access_key_id=key, aws_secret_access_key=secret,
        verify=False,
        config=Config(signature_version='s3v4', max_pool_connections=10),
    )


def list_bucket(client, bucket):
    objs = {}
    for page in client.get_paginator('list_objects_v2').paginate(Bucket=bucket):
        for o in page.get('Contents', []):
            k = o['Key']
            if k.endswith('/') or any(p in EXCLUDES for p in k.split('/')):
                continue
            objs[k] = int(o['Size'])
    return objs


c2 = make_client('a064df53e320474396c1de1c82dd858e', 'b1f66191dfe34927992afe3cc62a66ce')
try:
    b2 = list_bucket(c2, 'train-expds-2')
    print(f"EXPDS-2: доступен, файлов={len(b2)}, объём={sum(b2.values())/2**30:.2f} GiB")
except Exception as e:
    print('EXPDS-2: НЕДОСТУПЕН:', str(e)[:200])
    b2 = {}

c3 = make_client('2c2e18dd7dd54cb89228405480ae6651', '557bddef38bb4b079f52ba017eb94613')
b3 = list_bucket(c3, 'train-expds-3')
print(f"EXPDS-3: файлов={len(b3)}, объём={sum(b3.values())/2**30:.2f} GiB")

local = {}
for dp, _dn, fns in os.walk(ROOT):
    for n in fns:
        p = Path(dp) / n
        local[p.relative_to(ROOT).as_posix()] = p.stat().st_size
print(f"LOCAL:   файлов={len(local)}, объём={sum(local.values())/2**30:.2f} GiB")


def coverage(remote, name):
    ok = sum(1 for k, s in remote.items() if local.get(k) == s)
    miss = [k for k in remote if k not in local]
    mism = [k for k in remote if k in local and local[k] != remote[k]]
    miss_b = sum(remote[k] for k in miss)
    print(f"{name}: совпало={ok}, отсутствует={len(miss)} (~{miss_b/2**30:.2f} GiB), размер-не-тот={len(mism)}")
    return miss, mism


if b2:
    coverage(b2, 'vs EXPDS-2')
coverage(b3, 'vs EXPDS-3')

json.dump({'expds2': b2, 'expds3': b3, 'local': local},
          open('listings.json', 'w', encoding='utf-8'), ensure_ascii=False)
print('listings.json сохранён')
