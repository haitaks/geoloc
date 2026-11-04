# -*- coding: utf-8 -*-
"""Датасет геофизики (модель 4): точки -> диски r=2.5 м, полигоны -> класс 2,
ignore: зоны шума, вне границ съёмки, nodata. Каналы: img 3 + one-hot{maggr,etgr,terfl} 3
+ глубина среза 1 = 7."""
import csv
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio import features as rfeatures
from rasterio.windows import Window

sys.path.insert(0, str(Path(__file__).parent))
WORK = Path(r'G:\projects\geoloc\work\geo')
CLASSES = ('finds_points', 'object_poly')
KIND_ONEHOT = {'maggr_gray': 0, 'maggr_color': 0, 'etgr_map': 1, 'terfl': 2}
DISK_R = 2.5   # метры; в метрике допуск точек 2 м


def disk(cx_m, cy_m, r_m, tf, tile):
    """Круг-полигон в CRS растра, затем растеризация в сетке окна."""
    n = 12
    ring = [[cx_m + r_m * math.cos(2 * math.pi * i / n),
             cy_m + r_m * math.sin(2 * math.pi * i / n)] for i in range(n + 1)]
    return [{'type': 'Polygon', 'coordinates': [ring]}]


class GeoTileDataset:
    def __init__(self, split, neg_frac=0.3, seed=7, augment=True):
        self.augment = augment and split == 'train'
        self.rasters = json.load(open(WORK / 'rasters.json', encoding='utf-8'))
        split_sites = set(json.load(open(WORK / 'split.json', encoding='utf-8'))[split])
        pos, neg = [], []
        with open(WORK / 'tiles.csv', encoding='utf-8') as fh:
            for row in csv.DictReader(fh, delimiter=';'):
                if row['site'] not in split_sites:
                    continue
                (pos if int(row['n_points']) + int(row['n_polys']) > 0 else neg).append(row)
        rng = random.Random(seed)
        if split == 'train' and neg:
            neg = rng.sample(neg, min(len(neg), round(neg_frac * len(pos))))
        self.rows = pos + neg
        rng.shuffle(self.rows)
        self.n_pos = len(pos)
        self._src = {}

    def __len__(self):
        return len(self.rows)

    def _open(self, path):
        if path not in self._src:
            self._src[path] = rasterio.open(path)
        return self._src[path]

    def __getitem__(self, idx):
        row = self.rows[idx]
        ras = self.rasters[row['raster']]
        src = self._open(ras['path'])
        tile = 512
        win = Window(int(row['col']), int(row['row']), int(row['win_x']), int(row['win_y']))
        arr = src.read(window=win, out_shape=(src.count, tile, tile),
                       resampling=rasterio.enums.Resampling.average,
                       boundless=True, fill_value=0)
        img = arr[:3].astype(np.float32) / 255.0
        valid = arr.sum(axis=0) > 0

        tf = rasterio.windows.transform(win, src.transform) * rasterio.transform.Affine.scale(
            win.width / tile, win.height / tile)

        shapes = []
        for g in ras['marks'].get('points', []):          # точки -> диски, класс 1
            shapes.extend((d, 1) for d in disk(g['coordinates'][0], g['coordinates'][1],
                                               DISK_R, tf, tile))
        for g in ras['marks'].get('polys', []):           # полигоны -> класс 2
            shapes.append((g, 2))
        mask = rfeatures.rasterize(shapes, out_shape=(tile, tile), transform=tf,
                                   fill=0, dtype='int32') if shapes else \
            np.zeros((tile, tile), 'int32')
        ign = list(ras['noise'])                          # шум и вне границ съёмки -> ignore
        outside = []
        if ras['bounds_zone']:
            cov = rfeatures.rasterize([(g, 1) for g in ras['bounds_zone']],
                                      out_shape=(tile, tile), transform=tf, fill=0, dtype='uint8')
            mask[cov == 0] = 255
        if ign:
            noise = rfeatures.rasterize([(g, 1) for g in ign],
                                        out_shape=(tile, tile), transform=tf, fill=0, dtype='uint8')
            mask[noise > 0] = 255
        mask[~valid] = 255

        onehot = np.zeros((3, tile, tile), np.float32)
        onehot[KIND_ONEHOT.get(ras['kind'], 0)] = 1.0
        depth = np.full((1, tile, tile), (ras['depth'] or 0.0) / 2.0, np.float32)
        x = np.concatenate([img, onehot, depth], 0)

        if self.augment:
            k = random.randint(0, 7)
            if k & 1:
                x, mask = x[:, :, ::-1], mask[:, ::-1]
            if k & 2:
                x, mask = x[:, ::-1, :], mask[::-1, :]
            if k & 4:
                x, mask = np.rot90(x, 1, (1, 2)), np.rot90(mask, 1, (0, 1))
        return np.ascontiguousarray(x), np.ascontiguousarray(mask)
