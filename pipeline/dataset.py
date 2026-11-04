# -*- coding: utf-8 -*-
"""Датасеты тайлов для трёх моделей: чтение окна из GeoTIFF + растеризация маски на лету.

Маска: 0..K-1 — позитивные классы модели, 255 — ignore (чужие классы, nodata).
Каналы входа:
  model1 (повреждённые курганы, фото): img 3кн (SpOr RGB / Ae gray×3) + one-hot {SpOr+Sp, Ae} = 5
  model2 (целые курганы, Li):          c-RGB 3кн + g-уклоны 1кн = 4
  model3 (прочие классы, multi-source): img 3кн + valid 1кн + one-hot {Li, SpOr+Sp, Ae, Or} = 8
"""
import csv
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import rasterio
from pyproj import CRS as pCRS
from pyproj import Transformer
from rasterio import features
from rasterio.windows import Window, from_bounds, bounds as win_bounds

WORK = Path(r'G:\projects\geoloc\work')
MODELS = {
    'model1': dict(classes=('курганы_поврежденные',), tile=512),
    'model2': dict(classes=('курганы_целые',), tile=512),
    'model3': dict(classes=('городища', 'фортификации', 'архитектуры'), tile=512),
}
KNOWN = frozenset(('курганы_целые', 'курганы_поврежденные',
                   'городища', 'фортификации', 'архитектуры'))

# Версия кэша тайлов: поднимается при любом изменении разметки (marks/), ignore-масок
# WorldCover или сетки тайлов в манифестах — иначе кэш тихо подсунет старые маски.
# v3: ignore возвращён к 10+20+90 (v2 содержал расширенный 10+20+50+80+90 прогона r50wc).
CACHE_VER = 3


class GeoTileDataset:
    """tile rows подгружаются из манифеста; растры и разметка открываются лениво."""

    def __init__(self, model: str, split: str, neg_per_pos: float = 2.0,
                 max_tiles: int = 0, seed: int = 7, augment: bool = True,
                 wc_channel: bool = False, oversample: dict = None):
        """oversample: {класс: множитель} — дублировать позитивные тайлы с редкими
        классами (городищ ~100 уникальных — план §4.2); кэш тайлов не затрагивается."""
        self.cfg = MODELS[model]
        self.model = model
        self.augment = augment and split == 'train'
        self.wc_channel = wc_channel
        rng = random.Random(seed)
        self.rasters = json.load(open(WORK / 'rasters.json', encoding='utf-8'))
        split_sites = set(json.load(open(WORK / 'splits' / f'{model}_split.json',
                                         encoding='utf-8'))[split])
        pos, neg = [], []
        with open(WORK / 'manifests' / f'{model}_tiles.csv', encoding='utf-8') as fh:
            for row in csv.DictReader(fh, delimiter=';'):
                if row['site'] not in split_sites:
                    continue
                (pos if int(row['pos']) > 0 else neg).append(row)
        if split == 'train' and oversample:
            dup = []
            for row in pos:
                f = 1
                for cls, k in oversample.items():
                    if int(row.get(f'n_{cls}', 0) or 0) > 0:
                        f = max(f, k)
                dup.extend([row] * (f - 1))
            pos.extend(dup)
        if split == 'train' and neg:
            keep = min(len(neg), round(neg_per_pos * len(pos)))
            neg = rng.sample(neg, keep)
        elif split == 'val' and neg:
            neg = rng.sample(neg, min(len(neg), max(50, len(pos))))   # FP-контроль
        self.rows = pos + neg
        rng.shuffle(self.rows)
        if max_tiles:
            self.rows = self.rows[:max_tiles]
        self.n_pos = len(pos)
        self._src_cache = {}
        self._marks_cache = {}
        self._wc_cache = {}

    def __len__(self):
        return len(self.rows)

    # --- ленивые кэши ---
    def _src(self, path):
        if path not in self._src_cache:
            self._src_cache[path] = rasterio.open(path)
        return self._src_cache[path]

    def _marks(self, site, stem):
        key = f'{site}/{stem}'
        if key not in self._marks_cache:
            g = json.load(open(WORK / 'marks' / site / f'{stem}.geojson', encoding='utf-8'))
            self._marks_cache[key] = g['features']
        return self._marks_cache[key]

    # --- чтение ---
    def _read(self, src, row, bands=3):
        tile = self.cfg['tile']
        win = Window(int(row['col']), int(row['row']), int(row['win_x']), int(row['win_y']))
        out = src.read(window=win, out_shape=(src.count, tile, tile),
                       resampling=rasterio.enums.Resampling.average,
                       boundless=True, fill_value=0)
        return out, win

    def _image3(self, row, ras):
        """3-канальное изображение float [0..1] + признак int16."""
        src = self._src(ras['path'])
        arr, win = self._read(src, row)
        kind = ras['kind']
        if kind == 'SpOr' or kind == 'Sp' or kind == 'Or':
            if arr.shape[0] >= 3:
                img = arr[:3]
            else:
                img = np.repeat(arr[:1], 3, 0)
            return img.astype(np.float32) / 255.0, False
        if kind == 'Ae':
            a = arr[0].astype(np.float32)
            if arr.dtype != np.uint8:      # редкие int16-сканы
                lo, hi = np.percentile(a[a > 0], [1, 99]) if (a > 0).any() else (0, 1)
                a = np.clip((a - lo) / max(1e-6, hi - lo), 0, 1)
                return np.repeat(a[None], 3, 0), False
            return np.repeat(a[None], 3, 0) / 255.0, False
        # Li: 3-канальный RGB-рендер
        img = arr[:3] if arr.shape[0] >= 3 else np.repeat(arr[:1], 3, 0)
        return img.astype(np.float32) / 255.0, False

    def _mask(self, row, ras, win):
        """Маска (tile,tile) int32: 0..K-1 классы, 255 ignore."""
        classes = self.cfg['classes']
        tile = self.cfg['tile']
        cls_code = {c: i for i, c in enumerate(classes)}
        src = self._src(ras['path'])
        tf = rasterio.windows.transform(win, src.transform) * rasterio.transform.Affine.scale(
            win.width / tile, win.height / tile)   # сетка выхода tile×tile, а не источника
        shapes = []
        for f in self._marks(row['site'], row['raster']):
            cls = f['properties']['cls']
            code = cls_code.get(cls, 255 if cls in KNOWN else None)
            if code is None:
                continue
            shapes.append((f['geometry'], code + 1 if code != 255 else 255))
        m = features.rasterize(shapes, out_shape=(tile, tile), transform=tf,
                               fill=0, dtype='int32') if shapes else \
            np.zeros((tile, tile), 'int32')
        return m

    def _read_wc(self, path, ras, win, src, tile):
        """Окно из маски WorldCover (4326) в сетке тайла; None, если файла нет."""
        if not path.exists():
            return None
        if str(path) not in self._wc_cache:
            ds = rasterio.open(path)
            tr = Transformer.from_crs(pCRS.from_string(ras['crs']), pCRS.from_epsg(4326),
                                      always_xy=True)
            self._wc_cache[str(path)] = (ds, tr)
        ds, tr = self._wc_cache[str(path)]
        b = tr.transform_bounds(*win_bounds(win, src.transform), densify_pts=21)
        w = from_bounds(*b, transform=ds.transform)
        return ds.read(1, window=w, out_shape=(tile, tile), boundless=True, fill_value=0,
                       resampling=rasterio.enums.Resampling.nearest)

    def _cache_path(self, row):
        d = WORK / 'tilecache' / self.model / f'v{CACHE_VER}'
        d.mkdir(parents=True, exist_ok=True)
        key = f"{row['raster']}|{row['col']}|{row['row']}|{row['win_x']}|{row['win_y']}|{self.cfg['tile']}"
        h = hashlib.md5(key.encode()).hexdigest()[:10]
        return d / f"{row['raster']}_{row['col']}_{row['row']}_{h}.npz"

    def __getitem__(self, idx):
        for _try in range(6):
            row = self.rows[idx]
            ras = self.rasters[row['raster']]
            src = self._src(ras['path'])
            tile = self.cfg['tile']
            win = Window(int(row['col']), int(row['row']), int(row['win_x']), int(row['win_y']))
            cp = self._cache_path(row)
            if cp.exists():
                z = np.load(cp)
                arr, mask, wc = z['img'], z['mask'].astype('int32'), z['wc']
            else:
                arr = src.read(window=win, out_shape=(src.count, tile, tile),
                               resampling=rasterio.enums.Resampling.average,
                               boundless=True, fill_value=0)
                mask = self._mask(row, ras, win).astype('uint8')
                wc = self._read_wc(WORK / 'wcmask' / f"{row['site']}.tif", ras, win, src, tile)
                if wc is None:
                    wc = np.zeros(1, 'uint8')
                wc = wc.astype('uint8')
                try:
                    np.savez(cp, img=arr, mask=mask, wc=wc)
                except OSError:
                    pass
            valid = arr.sum(axis=0) > 0
            if valid.mean() >= 0.05 or _try == 5:   # почти пустой тайл (край растра) — берём другой
                break
            idx = (idx + 7919) % len(self.rows)     # детерминированный перескок
        kind = ras['kind']

        # изображение: до 4 каналов (RGB + NIR, если есть)
        if arr.dtype == np.uint8:
            base = arr[:4].astype(np.float32) / 255.0
        else:                                   # int16 (Ae) и прочие нестандарты
            a = arr[0].astype(np.float32)
            pos = a[a > 0]
            lo, hi = (np.percentile(pos, [1, 99]) if pos.size else (0.0, 1.0))
            base = np.repeat(np.clip((a - lo) / max(1e-6, hi - lo), 0, 1)[None], 4, 0)
        base = base.astype(np.float32, copy=False)   # percentile продвигает тип до float64
        if base.shape[0] < 4:                       # недостающие каналы — нулями
            base = np.concatenate(
                [base, np.zeros((4 - base.shape[0], *base.shape[1:]), np.float32)], 0)
        valid = valid.astype(np.float32)

        mask[valid == 0] = 255                      # вне данных — ignore
        if self.model == 'model1' and kind in ('SpOr', 'Sp') and wc.size > 1:
            mask[wc > 0] = 255                      # WorldCover ignore: лес/кустарник/болота/застройка/вода

        # каналы входа по моделям
        if self.model == 'model2':
            g = np.zeros((tile, tile), np.float32)
            if ras.get('aux_g'):
                try:
                    gsrc = self._src(ras['aux_g'])
                    b = win_bounds(win, src.transform)
                    gw = from_bounds(*b, transform=gsrc.transform,
                                     width=tile, height=tile)
                    ga = gsrc.read(window=gw, out_shape=(1, tile, tile),
                                   resampling=rasterio.enums.Resampling.average,
                                   boundless=True, fill_value=0)
                    g = (ga[0].astype(np.float32) - ga.min()) / max(1e-6, ga.max() - ga.min())
                except Exception:
                    pass
            x = np.concatenate([base[:3], g[None]], 0)
        elif self.model == 'model1':
            onehot = np.zeros((2, tile, tile), np.float32)
            onehot[0 if kind in ('SpOr', 'Sp') else 1] = 1.0
            x = [base, onehot]                      # 4 канала (RGB+NIR) + источник
            if self.wc_channel:                     # позитивный приор: степь/пашня/пустошь
                posm = self._read_wc(WORK / 'wcmask_pos' / f"{row['site']}.tif",
                                     ras, win, src, tile)
                x.append((posm > 0).astype(np.float32)[None] if posm is not None
                         else np.zeros((1, tile, tile), np.float32))
            x = np.concatenate(x, 0)
        else:  # model3
            onehot = np.zeros((4, tile, tile), np.float32)
            kmap = {'Li': 0, 'SpOr': 1, 'Sp': 1, 'Ae': 2, 'Or': 3}
            onehot[kmap.get(kind, 0)] = 1.0
            x = np.concatenate([base[:3], valid[None], onehot], 0)

        if self.augment:                        # флипы + повороты на 90
            k = random.randint(0, 7)
            if k & 1:
                x, mask = x[:, :, ::-1], mask[:, ::-1]
            if k & 2:
                x, mask = x[:, ::-1, :], mask[::-1, :]
            if k & 4:
                x, mask = np.rot90(x, 1, (1, 2)), np.rot90(mask, 1, (0, 1))
        return np.ascontiguousarray(x), np.ascontiguousarray(mask)


if __name__ == '__main__':                      # превью для визуальной сверки
    import sys
    from PIL import Image
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    out = WORK / 'preview'
    out.mkdir(exist_ok=True)
    palette = {1: (255, 0, 0), 2: (0, 255, 0), 3: (0, 220, 255), 255: (160, 0, 160)}
    for model in MODELS:
        ds = GeoTileDataset(model, 'train', neg_per_pos=0, augment=False)
        for i in range(min(4, len(ds))):
            x, m = ds[i]
            img = (np.moveaxis(x[:3], 0, -1) * 255).astype(np.uint8)
            ov = img.copy()
            for code, color in palette.items():
                ov[m == code] = color
            Image.fromarray(ov).save(out / f'{model}_{i}_site_{ds.rows[i]["site"]}.png')
            print(f'{model}[{i}]: {ds.rows[i]["site"]} {ds.rows[i]["raster"]} '
                  f'x{x.shape} mask_uniq={np.unique(m)[:6]}')
    print('Превью:', out)
