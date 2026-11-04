# -*- coding: utf-8 -*-
"""Общий движок инференса: растр → карты вероятностей классов → GeoJSON-фичи.

Отличия от pipeline/infer.py (который остаётся для val-прогонов по train-индексу):
  * усреднение вероятностей в перекрытиях окон (вместо max) — план §6, шаг 5;
  * TTA дигональными флипами (id/h/v/hv) с усреднением;
  * батчи окон, AMP на CUDA (контейнер RTX 4090);
  * scale-фактор для гигантских растров (Or 2.7 ГБ): канва и чтение прорежаются,
    чтобы уложиться в бюджет памяти;
  * постобработка под платформенную метрику: порог по классу, мин-площадь,
    буфер-расширение (cov = area(P∩G)/min(area) ≥ 0.5 → oversize бесплатен),
    упрощение контура (вершин ≤ 3000, контур не оценивается);
  * finds_points → точка по центроиду компоненты (сценарий Б метрики: круг r=2 м
    вокруг центроида, допуск 2 м).

Формат входов 1:1 как в dataset.py (m1/m2/m3) и dataset_geo.py (m4).
"""
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from scipy import ndimage

import rasterio
from rasterio import features as rfeatures
from rasterio.enums import Resampling
from rasterio.windows import Window, from_bounds, bounds as win_bounds

import segmentation_models_pytorch as smp

import geoloc_paths

try:
    from shapely.geometry import Polygon, Point, mapping
    from shapely.ops import unary_union
    from shapely import make_valid
    SHAPELY = True
except ImportError:  # pragma: no cover
    SHAPELY = False

# ---------------------------------------------------------------------------
# Спецификации моделей (классы как в обучении; веса target-разрешения — чемпионы)
MODEL_SPECS = {
    'model1': dict(classes=('курганы_поврежденные',), in_ch=6, res=0.25),
    'model2': dict(classes=('курганы_целые',), in_ch=4, res=0.2),
    'model3': dict(classes=('городища', 'фортификации', 'архитектуры'), in_ch=8, res=0.5),
    'model4': dict(classes=('finds_points', 'object_poly'), in_ch=7, res=0.1),
}
# Какая модель на каких источниках работает (predict kind -> модели)
KIND2MODELS = {
    'Li': ('model2', 'model3'),
    'La': ('model2', 'model3'),      # La идёт через деривацию la_derive (kind=Li у производного растра)
    'SpOr': ('model1', 'model3'),
    'Sp': ('model1', 'model3'),
    'Ae': ('model1', 'model3'),
    'Or': ('model3',),
    'MagGr': ('model4',),
    'EtGr': ('model4',),
    'TerFl': ('model4',),
}


@dataclass
class ModelSpec:
    name: str
    ckpt: str
    encoder: str = 'resnet18'
    classes: Tuple[str, ...] = ()
    in_ch: int = 0
    res: float = 0.5                     # целевое разрешение обучения, м/пикс
    tile: int = 512
    net: Optional[torch.nn.Module] = field(default=None, repr=False)


def load_spec(model: str, ckpt: str = '', run: str = '') -> ModelSpec:
    """Собирает спецификацию: энкодер берётся из report.json прогона, если есть."""
    base = MODEL_SPECS[model]
    runs = geoloc_paths.runs_dir()
    if not ckpt:
        cand = runs / (model + run) / 'best.pth'
        ckpt = str(cand)
    encoder, res = 'resnet18', base['res']
    rp = Path(ckpt).parent / 'report.json'
    if rp.exists():
        try:
            a = json.load(open(rp, encoding='utf-8'))['args']
            encoder = a.get('encoder', encoder)
        except Exception:
            pass
    return ModelSpec(name=model, ckpt=ckpt, encoder=encoder,
                     classes=tuple(base['classes']), in_ch=base['in_ch'], res=res)


def build_net(spec: ModelSpec, device) -> torch.nn.Module:
    net = smp.Unet(encoder_name=spec.encoder, encoder_weights=None,
                   in_channels=spec.in_ch, classes=len(spec.classes) + 1)
    sd = torch.load(spec.ckpt, map_location='cpu')
    net.load_state_dict(sd)
    net.to(device).eval()
    return net


def pick_device() -> torch.device:
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ---------------------------------------------------------------------------
# Входные тензоры — строго как в dataset.py (m1/m2/m3) и dataset_geo.py (m4)
def build_input(arr: np.ndarray, kind: str, model: str,
                aux_arr: Optional[np.ndarray] = None,
                depth: Optional[float] = None,
                geo_kind: str = '') -> np.ndarray:
    if arr.dtype == np.uint8:
        base = arr[:4].astype(np.float32) / 255.0
    else:                                   # int16 (Ae) и прочие нестандарты
        a = arr[0].astype(np.float32)
        pos = a[a > 0]
        lo, hi = (np.percentile(pos, [1, 99]) if pos.size else (0.0, 1.0))
        base = np.repeat(np.clip((a - lo) / max(1e-6, hi - lo), 0, 1)[None], 4, 0)
    base = base.astype(np.float32, copy=False)
    if base.shape[0] < 4:
        base = np.concatenate(
            [base, np.zeros((4 - base.shape[0], *base.shape[1:]), np.float32)], 0)
    valid = (arr.sum(axis=0) > 0).astype(np.float32)

    if model == 'model2':                  # RGB-рендер + нормированные уклоны
        g = np.zeros(base.shape[1:], np.float32)
        if aux_arr is not None:
            g = (aux_arr - aux_arr.min()) / max(1e-6, float(aux_arr.max() - aux_arr.min()))
        return np.concatenate([base[:3], g[None]], 0)
    if model == 'model1':                  # RGB+NIR + one-hot {SpOr/Sp, Ae}
        onehot = np.zeros((2, *base.shape[1:]), np.float32)
        onehot[0 if kind in ('SpOr', 'Sp') else 1] = 1.0
        return np.concatenate([base, onehot], 0)
    if model == 'model3':                  # RGB + valid + one-hot источника
        onehot = np.zeros((4, *base.shape[1:]), np.float32)
        onehot[{'Li': 0, 'SpOr': 1, 'Sp': 1, 'Ae': 2, 'Or': 3}.get(kind, 0)] = 1.0
        return np.concatenate([base[:3], valid[None], onehot], 0)
    if model == 'model4':                  # геофизика: img3 + one-hot{maggr,etgr,terfl} + глубина/2
        onehot = np.zeros((3, *base.shape[1:]), np.float32)
        onehot[{'maggr_gray': 0, 'maggr_color': 0, 'etgr_map': 1, 'etgr_prof': 1,
                'terfl': 2}.get(geo_kind, 0)] = 1.0
        dep = np.full((1, *base.shape[1:]), (depth or 0.0) / 2.0, np.float32)
        return np.concatenate([base[:3], onehot, dep], 0)
    raise ValueError(model)


def _tta(x: 'torch.Tensor'):
    """Диагональные флипы батча-тензора: (id, h, v, hv). torch не умеет
    отрицательный шаг в слайсах — только torch.flip."""
    return (x, torch.flip(x, (2,)), torch.flip(x, (1,)), torch.flip(x, (1, 2)))


def _untta(p: np.ndarray, k: int) -> np.ndarray:
    if k == 0:
        return p
    if k == 1:
        return p[:, :, ::-1]
    if k == 2:
        return p[:, ::-1, :]
    return p[:, ::-1, ::-1]


@torch.no_grad()
def predict_raster(spec: ModelSpec, net: torch.nn.Module, device,
                   raster_path: str, kind: str, aux_path: str = '',
                   geo_kind: str = '', depth: float = None,
                   tile: int = 512, overlap: float = 0.5, tta: bool = True,
                   batch: int = 4, amp: bool = True,
                   max_canvas_bytes: float = 6.0e9, max_windows: int = 20000,
                   keep_mask: Optional[np.ndarray] = None,
                   progress=None) -> Tuple[np.ndarray, dict]:
    """Карты вероятностей (C+1, H', W') float16 на сетке растра (возможно прореженной).

    keep_mask — bool (H', W'): где False, вероятности зануляются (границы съёмки m4).
    Гварды: память канвы (гиганты типа Or 2.7 ГБ) и число окон (грубые Ae/SpOr
    при целевом 0.25 м дают десятки тысяч окон) — сначала снимается перекрытие,
    затем канва/окна прореживаются (scale ×2).
    Возвращает (probs, info: scale/res_eff/win/n_windows).
    """
    with rasterio.open(raster_path) as src:
        H, W, ras_res = src.height, src.width, float(max(abs(src.transform.a),
                                                          abs(src.transform.e)))
        nc = len(spec.classes) + 1
        scale, ov = 1, overlap
        while True:
            win_x = max(8, min(W, round(tile * spec.res / ras_res * scale)))
            win_y = max(8, min(H, round(tile * spec.res / ras_res * scale)))
            step_x = max(1, round(win_x * (1 - ov)))
            step_y = max(1, round(win_y * (1 - ov)))
            nx = 1 if W <= win_x else (W - win_x) // step_x + 1
            ny = 1 if H <= win_y else (H - win_y) // step_y + 1
            canvas_bytes = (np.ceil(H / scale) * np.ceil(W / scale) * (nc * 2 + 1))
            if canvas_bytes <= max_canvas_bytes and nx * ny <= max_windows:
                break
            if ov > 0:                       # сперва жертвуем перекрытием
                ov = 0.0
                continue
            scale *= 2                       # затем прореживаем канву/окна
        ch, cw = int(np.ceil(H / scale)), int(np.ceil(W / scale))

        # канва усреднения: сумма fp16 + счётчик
        csum = np.zeros((nc, ch, cw), np.float16)
        cnt = np.zeros((ch, cw), np.uint8)

        aux_src = rasterio.open(aux_path) if aux_path else None
        # растяжение вероятностей тайла на канву окна (кратно scale)
        ry = max(1, (win_y // tile) // scale)
        rx = max(1, (win_x // tile) // scale)

        wins = []
        for row in range(0, H, step_y):
            for col in range(0, W, step_x):
                wins.append((min(col, W - win_x), min(row, H - win_y)))
        # дедуп краёв (min может повторять последние окна)
        wins = sorted(set(wins))

        amp_ctx = (torch.autocast('cuda', dtype=torch.float16)
                   if (amp and device.type == 'cuda') else torch.enable_grad(False))

        with amp_ctx:
            for i0 in range(0, len(wins), batch):
                xs, metas = [], []
                for col, row in wins[i0:i0 + batch]:
                    win = Window(col, row, win_x, win_y)
                    arr = src.read(window=win, out_shape=(src.count, tile, tile),
                                   resampling=Resampling.average, boundless=True,
                                   fill_value=0)
                    aux = None
                    if aux_src is not None:
                        b = win_bounds(win, src.transform)
                        gw = from_bounds(*b, transform=aux_src.transform,
                                         width=tile, height=tile)
                        aux = aux_src.read(window=gw, out_shape=(1, tile, tile),
                                           boundless=True, fill_value=0,
                                           resampling=Resampling.average)[0].astype(np.float32)
                    x = build_input(arr, kind, spec.name, aux, depth, geo_kind)
                    xs.append(x)
                    metas.append((col, row))
                xb = torch.from_numpy(np.stack(xs)).to(device)
                variants = _tta(xb) if tta else (xb,)
                acc = None
                for k, xv in enumerate(variants):
                    p = torch.softmax(net(xv).float(), 1).cpu().numpy()
                    p = _untta(p, k)
                    acc = p if acc is None else acc + p
                p = acc / len(variants)                    # (B, nc, tile, tile)

                for bi, (col, row) in enumerate(metas):
                    pr = np.kron(p[bi], np.ones((1, ry, rx), np.float16))
                    ph, pw = pr.shape[1:]
                    th, tw = max(1, win_y // scale), max(1, win_x // scale)
                    if ph < th or pw < tw:                 # краевое достраивание
                        pr = np.pad(pr, ((0, 0), (0, max(0, th - ph)), (0, max(0, tw - pw))),
                                    mode='edge')
                    r0, c0 = row // scale, col // scale
                    r1, c1 = min(ch, r0 + th), min(cw, c0 + tw)
                    csum[:, r0:r1, c0:c1] += pr[:, :r1 - r0, :c1 - c0].astype(np.float16)
                    cnt[r0:r1, c0:c1] += 1
                if progress:
                    progress(i0 + len(metas), len(wins))
        if aux_src is not None:
            aux_src.close()

    valid = cnt > 0
    probs = np.zeros_like(csum)
    np.divide(csum, cnt[None].astype(np.float16), out=probs, where=valid[None])
    if keep_mask is not None:
        probs[:, ~keep_mask] = 0.0
    info = dict(scale=scale, res_eff=ras_res * scale, tile=tile,
                win=(win_x, win_y), n_windows=len(wins))
    return probs, info


# ---------------------------------------------------------------------------
# Постобработка: вероятности → фичи (шейпли-геометрия в CRS растра)
def probmaps_to_features(probs: np.ndarray, classes: Sequence[str],
                         canvas_transform, res_eff: float,
                         thresholds: Dict[str, float],
                         min_area_m2: Dict[str, float],
                         buffer_m: Dict[str, float] = None,
                         simplify_m: float = 1.0,
                         as_points: Sequence[str] = (),
                         max_vertices: int = 2900) -> List[dict]:
    """Компоненты связности по классам → полигоны (или точки) + уверенность.

    canvas_transform — аффин пикселей КАНАВЫ (уже умноженный на scale) в CRS растра;
    res_eff — размер пикселя канвы, м. Порог 0/отсутствует → argmax. Буфер
    расширяет полигон (метрика: cov≥0.5 по меньшей фигуре, контур не оценивается).
    Выход — геометрия в CRS растра (обычно UTM-метры).
    """
    assert SHAPELY, 'нужен shapely'
    tf = canvas_transform
    buffer_m = buffer_m or {}
    out = []
    argmax = probs.argmax(0)
    for ci, cls in enumerate(classes, start=1):
        thr = float(thresholds.get(cls, 0.0) or 0.0)
        if thr <= 0:
            m = argmax == ci
        else:
            m = (probs[ci] >= thr) & ((argmax == ci) | (argmax == 0))
        if not m.any():
            continue
        lab, n = ndimage.label(m)
        if not n:
            continue
        min_px = max(4.0, float(min_area_m2.get(cls, 1.0)) / (res_eff ** 2))
        sizes = ndimage.sum(m, lab, range(1, n + 1))
        keep = [i + 1 for i, s in enumerate(sizes) if s >= min_px]
        if not keep:
            continue
        mm = np.isin(lab, keep)
        # уверенность компоненты — средняя вероятность класса внутри неё
        conf_of = {i + 1: float(probs[ci][lab == i + 1].mean()) for i in range(n)}
        for geom, val in rfeatures.shapes(mm.astype('uint8'), mask=mm, transform=tf):
            poly = Polygon(geom['coordinates'][0], geom['coordinates'][1:])
            if not poly.is_valid:
                poly = make_valid(poly)
            if poly.is_empty:
                continue
            # к какой компоненте относится (для уверенности/точек)
            cx, cy = poly.representative_point().x, poly.representative_point().y
            col, row = ~tf * (cx, cy)
            lid = lab[min(int(round(row)), lab.shape[0] - 1),
                      min(int(round(col)), lab.shape[1] - 1)]
            conf = conf_of.get(int(lid), 0.0)
            if cls in as_points:
                pt = poly.centroid
                out.append(dict(cls=cls, conf=conf, geometry=Point(pt.x, pt.y)))
                continue
            buf = float(buffer_m.get(cls, 0.0))
            if buf > 0:
                poly = poly.buffer(buf)
                if not poly.is_valid:
                    poly = make_valid(poly)
            if poly.geom_type not in ('Polygon', 'MultiPolygon'):
                continue
            # лимит 3000 вершин: упрощение с нарастающим допуском
            simp = simplify_m
            while _n_vertices(poly) > max_vertices and simp < 200:
                simp *= 2
                poly = poly.simplify(simp, preserve_topology=True)
            out.append(dict(cls=cls, conf=conf, geometry=poly))
    return out


def _n_vertices(geom) -> int:
    if geom.geom_type == 'Polygon':
        return len(geom.exterior.coords) + sum(len(r.coords) for r in geom.interiors)
    return sum(_n_vertices(g) for g in geom.geoms)


def features_intersect_mask(feats: List[dict], mask_polys: List,
                            mode: str = 'drop') -> List[dict]:
    """m4: выбросить фичи, пересекающие зоны шума; клип по границам съёмки."""
    if not mask_polys:
        return feats
    zone = unary_union(mask_polys) if len(mask_polys) > 1 else mask_polys[0]
    out = []
    for f in feats:
        g = f['geometry']
        try:
            if mode == 'drop' and g.intersects(zone):
                continue
            if mode == 'clip':
                g = g.intersection(zone)
                if g.is_empty:
                    continue
                f = dict(f, geometry=g)
        except Exception:
            continue
        out.append(f)
    return out
