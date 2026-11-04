# -*- coding: utf-8 -*-
"""ESA WorldCover 2021 (10 м/пикс) как маска леса для фото-источников.

1) По рамкам спутниковых растров (SpOr/Sp) определяет нужные тайлы WorldCover
   (сетка 3x3 градуса, именование N54E031) и скачивает их из открытого S3.
2) Для каждого места собирает маску леса (класс 10 = Tree cover) в EPSG:4326
   в work/wcmask/<место>.tif: 1 — лес, 0 — прочее.
Маска применяется в dataset.py как ignore-зона для model1 (SpOr/Sp).
"""
import json
import math
import sys
import urllib.request
from pathlib import Path

import numpy as np
import rasterio
from pyproj import CRS
from rasterio import mask as riomask
from rasterio.transform import from_bounds as tf_from_bounds
from rasterio.warp import transform_bounds

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
WORK = Path(r'G:\projects\geoloc\work')
WC_DIR = WORK / 'worldcover'
OUT_DIR = WORK / 'wcmask'
S3 = 'https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/ESA_WorldCover_10m_2021_v200_{tile}_Map.tif'
# маски нужны всем местам: и фото (ignore), и лидарным (вырезание лесных ЛП на инференсе)
SAT_KINDS = ('SpOr', 'Sp', 'Li', 'La')


def site_bbox_4326(ras):
    crs = CRS.from_string(ras['crs'])
    tf = rasterio.transform.Affine(*ras['transform'])
    b = rasterio.transform.array_bounds(ras['h'], ras['w'], tf)  # l,b,r,t
    return transform_bounds(crs, CRS.from_epsg(4326), b[0], b[1], b[2], b[3], densify_pts=21)


def tile_names(bbox):
    tiles = set()
    for lon in range(int(math.floor(bbox[0])), int(math.floor(bbox[2])) + 1, 3):
        for lat in range(int(math.floor(bbox[1])), int(math.floor(bbox[3])) + 1, 3):
            lat0 = lat - ((lat % 3) if lat >= 0 else 0)
            lon0 = lon - ((lon % 3) if lon >= 0 else 0)
            tiles.add(f"{'N' if lat0 >= 0 else 'S'}{abs(lat0):02d}{'E' if lon0 >= 0 else 'W'}{abs(lon0):03d}")
    return tiles


def scan_test_regions(test_dir):
    """Тест-каталог (region_XXXX): фото-растры по регионам -> псевдо-rasters.json
    записи (crs/transform/w/h), чтобы маски строились как для train-мест."""
    import re
    homo = str.maketrans({'А': 'A', 'а': 'a', 'Е': 'E', 'е': 'e', 'О': 'O', 'о': 'o',
                          'С': 'C', 'с': 'c', 'Р': 'P', 'р': 'p'})
    sat_sites = {}
    for rd in sorted(p for p in Path(test_dir).iterdir() if p.is_dir()):
        for f in sorted(rd.rglob('*.tif')):
            if not re.search(r'_(SpOr|Sp|Ae|Or)_', f.name.translate(homo)):
                continue
            try:
                with rasterio.open(f) as src:
                    if src.crs is None:
                        continue
                    sat_sites.setdefault(rd.name, []).append(dict(
                        crs=src.crs.to_string(), w=src.width, h=src.height,
                        transform=[src.transform.a, src.transform.b, src.transform.c,
                                   src.transform.d, src.transform.e, src.transform.f]))
            except Exception as e:
                print(f'  ! {f.name}: {e}')
    return sat_sites


def main():
    WC_DIR.mkdir(exist_ok=True, parents=True)
    OUT_DIR.mkdir(exist_ok=True, parents=True)
    if '--test-dir' in sys.argv:
        # маски для тест-регионов (готовятся заранее: в контейнере сабмита сети нет;
        # infer_all читает их как work/wcmask/<регион>.tif — фильтр леса для m1)
        sat_sites = scan_test_regions(sys.argv[sys.argv.index('--test-dir') + 1])
    else:
        rasters = json.load(open(WORK / 'rasters.json', encoding='utf-8'))
        sat_sites = {}
        for ras in rasters.values():
            if ras['kind'] in SAT_KINDS:
                sat_sites.setdefault(ras['site'], []).append(ras)

    need = set()
    for site, rs in sat_sites.items():
        for ras in rs:
            need |= tile_names(site_bbox_4326(ras))
    print(f'Спутниковых мест: {len(sat_sites)}; тайлов WorldCover нужно: {len(need)}')

    for t in sorted(need):
        dst = WC_DIR / f'{t}.tif'
        if dst.exists() and dst.stat().st_size > 1_000_000:
            continue
        url = S3.format(tile=t)
        print(f'  качаю {t}...', flush=True)
        try:
            urllib.request.urlretrieve(url, dst)
        except Exception as e:
            print(f'    ОШИБКА {t}: {e}')

    # маска леса на место
    for site, rs in sorted(sat_sites.items()):
        bb = None
        for ras in rs:
            b = site_bbox_4326(ras)
            bb = b if bb is None else (min(bb[0], b[0]), min(bb[1], b[1]),
                                       max(bb[2], b[2]), max(bb[3], b[3]))
        pad = 0.005
        bb = (bb[0] - pad, bb[1] - pad, bb[2] + pad, bb[3] + pad)
        tiles = [WC_DIR / f'{t}.tif' for t in sorted(tile_names(bb))]
        tiles = [t for t in tiles if t.exists()]
        if not tiles:
            print(f'  ! {site}: нет тайлов WorldCover')
            continue
        # читаем окно из каждого тайла и собираем бинарную маску на общей сетке ~10 м
        w_px = max(8, int((bb[2] - bb[0]) / 0.000125))
        h_px = max(8, int((bb[3] - bb[1]) / 0.000125))
        grid_tf = tf_from_bounds(bb[0], bb[1], bb[2], bb[3], w_px, h_px)
        out = np.zeros((h_px, w_px), 'uint8')      # ignore: лес/кустарник/болота/застройка/вода
        out_pos = np.zeros((h_px, w_px), 'uint8')  # рабочие территории: степь/пашня/пустошь
        for t in tiles:
            with rasterio.open(t) as src:
                try:
                    win = rasterio.windows.from_bounds(bb[0], bb[1], bb[2], bb[3],
                                                       src.transform)
                    arr = src.read(1, window=win, out_shape=(h_px, w_px),
                                   boundless=True, fill_value=0,
                                   resampling=rasterio.enums.Resampling.nearest)
                    out |= np.isin(arr, (10, 20, 90)).astype('uint8')   # лес, кустарник, болота
                    out_pos |= np.isin(arr, (30, 40, 60)).astype('uint8')
                except Exception as e:
                    print(f'    ! {t.name}: {e}')
        profile = dict(driver='GTiff', width=w_px, height=h_px, count=1, dtype='uint8',
                       crs='EPSG:4326', transform=grid_tf, compress='deflate')
        with rasterio.open(OUT_DIR / f'{site}.tif', 'w', **profile) as dst:
            dst.write(out, 1)
        pos_dir = WORK / 'wcmask_pos'
        pos_dir.mkdir(exist_ok=True)
        with rasterio.open(pos_dir / f'{site}.tif', 'w', **profile) as dst:
            dst.write(out_pos, 1)
        frac = out.mean()
        if frac > 0.05:
            note = ' <-- много ignore'
        else:
            note = ''
        print(f'  {site}: ignore {frac*100:.1f}%, рабочая территория {out_pos.mean()*100:.1f}%{note}')


if __name__ == '__main__':
    main()
