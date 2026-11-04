# -*- coding: utf-8 -*-
"""Инференс: скользящее окно по растру, пост-обработка, экспорт GeoJSON.

Пример:
  python pipeline/infer.py --model model2 --raster 01_Хога_Li_c
  python pipeline/infer.py --model model1 --raster all_val
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import rasterio
from pyproj import CRS as pCRS
from pyproj import Transformer
from rasterio import features
from rasterio.enums import Resampling
from rasterio.windows import Window, from_bounds, bounds as win_bounds

sys.path.insert(0, str(Path(__file__).parent))
from dataset import GeoTileDataset, WORK, MODELS  # noqa: E402

import segmentation_models_pytorch as smp  # noqa: E402
import torch  # noqa: E402
from scipy import ndimage  # noqa: E402

try:
    import torch_directml as dml
    DEVICE = dml.device() if dml.device_count() else 'cpu'
except ImportError:
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

IN_CH = {'model1': 6, 'model2': 4, 'model3': 8}
# Минимальная площадь компоненты по классам, м² (официальные пороги КОЗ №3: курганы 10, городища 25)
MIN_AREA_M2 = {
    'model1': {'курганы_поврежденные': 10},
    'model2': {'курганы_целые': 10},
    'model3': {'городища': 25, 'фортификации': 10, 'архитектуры': 10},
}


def build_input(arr, ras, valid, model, aux_arr=None):
    """Тот же формат входа, что в dataset.py."""
    if arr.dtype == np.uint8:
        base = arr[:4].astype(np.float32) / 255.0
    else:
        a = arr[0].astype(np.float32)
        pos = a[a > 0]
        lo, hi = (np.percentile(pos, [1, 99]) if pos.size else (0.0, 1.0))
        base = np.repeat(np.clip((a - lo) / max(1e-6, hi - lo), 0, 1)[None], 4, 0)
    base = base.astype(np.float32, copy=False)
    if base.shape[0] < 4:
        base = np.concatenate([base, np.zeros((4 - base.shape[0], *base.shape[1:]), np.float32)], 0)
    kind = ras['kind']
    if model == 'model2':
        g = np.zeros(base.shape[1:], np.float32)
        if aux_arr is not None:
            g = (aux_arr - aux_arr.min()) / max(1e-6, float(aux_arr.max() - aux_arr.min()))
        return np.concatenate([base[:3], g[None]], 0)
    if model == 'model1':
        onehot = np.zeros((2, *base.shape[1:]), np.float32)
        onehot[0 if kind in ('SpOr', 'Sp') else 1] = 1.0
        return np.concatenate([base, onehot], 0)
    onehot = np.zeros((4, *base.shape[1:]), np.float32)
    onehot[{'Li': 0, 'SpOr': 1, 'Sp': 1, 'Ae': 2, 'Or': 3}.get(kind, 0)] = 1.0
    return np.concatenate([base[:3], valid[None], onehot], 0)


def read_wc(path, ras, b, tile):
    """Окно маски WorldCover (4326) в сетке тайла; None, если файла нет."""
    if not path.exists():
        return None
    with rasterio.open(path) as ds:
        tr = Transformer.from_crs(pCRS.from_string(ras['crs']), pCRS.from_epsg(4326),
                                  always_xy=True)
        b4326 = tr.transform_bounds(*b, densify_pts=21)
        w = from_bounds(*b4326, transform=ds.transform)
        return ds.read(1, window=w, out_shape=(tile, tile), boundless=True, fill_value=0,
                       resampling=Resampling.nearest)


def read_forest(site, ras, b, tile):
    """Маска леса (WorldCover 10/20/90) на окно, либо None."""
    path = WORK / 'wcmask' / f'{site}.tif'
    if not path.exists() or ras['kind'] not in ('SpOr', 'Sp'):
        return None
    return read_wc(path, ras, b, tile)


def run_raster(model, ckpt, stem, ras, out_dir, tile=512, overlap=0.5,
               args_threshold=0.0):
    classes = MODELS[model]['classes']
    src = rasterio.open(ras['path'])
    res_target = {'model1': 0.25, 'model2': 0.2, 'model3': 0.5}[model]  # m1-чемпион обучен на 0.25
    win_x = min(ras['w'], max(8, round(tile * res_target / ras['res'])))
    win_y = min(ras['h'], max(8, round(tile * res_target / ras['res'])))
    step_x = max(1, round(win_x * (1 - overlap)))
    step_y = max(1, round(win_y * (1 - overlap)))

    net = smp.Unet(encoder_name='resnet18', encoder_weights=None,
                   in_channels=IN_CH[model], classes=len(classes) + 1)
    net.load_state_dict(torch.load(ckpt, map_location='cpu'))
    net.to(DEVICE).eval()

    probs = np.zeros((len(classes) + 1, ras['h'], ras['w']), 'float32')  # аккумулятор max-вероятности
    skipped_win = 0
    with torch.no_grad():
        for row in range(0, ras['h'], step_y):
            for col in range(0, ras['w'], step_x):
                col = min(col, ras['w'] - win_x)
                row = min(row, ras['h'] - win_y)
                win = Window(col, row, win_x, win_y)
                b = win_bounds(win, src.transform)
                # жёсткий фильтр «не-рабочих» территорий: без степи/пашни/пустоши — пропуск окна
                if model == 'model1' and ras['kind'] in ('SpOr', 'Sp'):
                    pm = read_wc(WORK / 'wcmask_pos' / f"{ras['site']}.tif", ras, b, tile)
                    if pm is not None and (pm > 0).mean() < 0.05:
                        skipped_win += 1
                        continue
                arr = src.read(window=win, out_shape=(src.count, tile, tile),
                               resampling=Resampling.average, boundless=True, fill_value=0)
                valid = (arr.sum(axis=0) > 0).astype(np.float32)
                aux = None
                if model == 'model2' and ras.get('aux_g'):
                    with rasterio.open(ras['aux_g']) as gsrc:
                        b = win_bounds(win, src.transform)
                        gw = from_bounds(*b, transform=gsrc.transform, width=tile, height=tile)
                        aux = gsrc.read(1, window=gw, out_shape=(tile, tile), boundless=True,
                                        fill_value=0, resampling=Resampling.average).astype(np.float32)
                x = torch.from_numpy(build_input(arr, ras, valid, model, aux)[None]).to(DEVICE)
                p = torch.softmax(net(x), 1)[0].cpu().numpy()
                # растягиваем вероятности тайла на окно растра (replicated + краевой пад)
                ry, rx = max(1, win_y // tile), max(1, win_x // tile)
                pr = np.kron(p, np.ones((1, ry, rx)))
                ph, pw = pr.shape[1:]
                if ph < win_y:                                    # окно крупнее тайла — достраиваем
                    pr = np.pad(pr, ((0, 0), (0, win_y - ph), (0, win_x - pw)), mode='edge')
                region = probs[:, row:row + win_y, col:col + win_x]
                np.maximum(region, pr[:, :win_y, :win_x], out=region)
    conf_map = probs[1:] if len(classes) > 0 else None
    if args_threshold <= 0:
        pred = probs.argmax(0)
    else:
        pred = np.zeros(probs.shape[1:], 'int32')
        for ci in range(1, len(classes) + 1):
            pred[(probs[ci] >= args_threshold) & (pred == 0)] = ci

    # лес-постфильтр (модель 1): вырезать WorldCover 10/20/90 по всей рамке растра
    if model == 'model1':
        b = (src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top)
        fw = from_bounds(*b, transform=src.transform, width=ras['w'], height=ras['h'])
        fm_path = WORK / 'wcmask' / f"{ras['site']}.tif"
        if fm_path.exists():
            with rasterio.open(fm_path) as ds:
                tr = Transformer.from_crs(pCRS.from_string(ras['crs']), pCRS.from_epsg(4326),
                                          always_xy=True)
                b4326 = tr.transform_bounds(*b, densify_pts=21)
                w = from_bounds(*b4326, transform=ds.transform)
                fm = ds.read(1, window=w, out_shape=(ras['h'], ras['w']), boundless=True,
                             fill_value=0, resampling=Resampling.nearest)
                pred[fm > 0] = 0

    # компоненты связности -> полигоны
    fc = []
    for ci in range(1, len(classes) + 1):
        min_px = MIN_AREA_M2[model].get(classes[ci - 1], 10) / (ras['res'] ** 2)  # растровые пиксели
        m = pred == ci
        lab, n = ndimage.label(m)
        if not n:
            continue
        sizes = ndimage.sum(m, lab, range(1, n + 1))
        keep = [i + 1 for i, s in enumerate(sizes) if s >= min_px]
        if not keep:
            continue
        mm = np.isin(lab, keep)
        # уверенность компоненты = средняя вероятность класса внутри неё
        conf_of = {i + 1: float(probs[ci][lab == i + 1].mean()) for i in range(n)}
        for geom, val in features.shapes(mm.astype('uint8'), mask=mm, transform=src.transform):
            comp_ids = set(lab[lab > 0].tolist()) & set(keep)
            conf = max(conf_of.get(k, 0) for k in comp_ids) if comp_ids else 0.0
            fc.append({'type': 'Feature',
                       'properties': {'class': classes[ci - 1], 'area_px': int(val),
                                      'confidence': round(conf, 3)},
                       'geometry': geom})
    out = {'type': 'FeatureCollection', 'crs': {
        'type': 'name', 'properties': {'name': ras['crs']}}, 'features': fc}
    dst = out_dir / f'{stem}.geojson'
    json.dump(out, open(dst, 'w', encoding='utf-8'), ensure_ascii=False)
    return len(fc), dst, skipped_win


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True, choices=list(MODELS))
    ap.add_argument('--checkpoint', default='')
    ap.add_argument('--raster', default='all_val', help='stem растра, "all" или "all_val"')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--threshold', type=float, default=0.0,
                    help='0 = argmax; иначе P(класс) >= порог')
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    ckpt = args.checkpoint or str(WORK / 'runs' / args.model / 'best.pth')
    out_dir = WORK / 'infer' / args.model
    out_dir.mkdir(parents=True, exist_ok=True)
    rasters = json.load(open(WORK / 'rasters.json', encoding='utf-8'))
    if args.raster in ('all', 'all_val'):
        split = json.load(open(WORK / 'splits' / f"{args.model}_split.json", encoding='utf-8'))
        sites = set(split['val'] if args.raster == 'all_val' else
                    split['train'] + split['val'])
        if args.model in ('model2', 'model3'):   # только первичные растры Li
            stems = [s for s, r in rasters.items()
                     if r['site'] in sites and (r['kind'] != 'Li' or s.endswith('_Li_c'))]
        else:
            stems = [s for s, r in rasters.items() if r['site'] in sites]
    else:
        stems = [args.raster]
    if args.limit:
        stems = stems[:args.limit]
    print(f'Устройство: {DEVICE} | чекпойнт: {ckpt} | растров: {len(stems)}')
    for stem in stems:
        n, dst, skipped = run_raster(args.model, ckpt, stem, rasters[stem], out_dir,
                                     args_threshold=args.threshold)
        print(f'  {stem}: {n} объектов -> {dst.name} (пропущено окон вне рабочих территорий: {skipped})',
              flush=True)


if __name__ == '__main__':
    main()
