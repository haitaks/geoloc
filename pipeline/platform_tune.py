# -*- coding: utf-8 -*-
"""Этап Б плана §8: тюнинг постобработки ПОД платформенную метрику.

Проблема: threshold_sweep.py оптимизирует object-F1 с матчингом IoU≥0.25 —
это не та целевая функция. Платформа (compute_metrics.py): cov≥0.5 по меньшей
фигуре, внутренняя точка в GT, FP×0.5, взвешенный F1 по классам. Здесь свипаем
(порог × буфер × мин-площадь) прямо по F1Calculator на val-инференсе и пишем
лучший набор в work/infer_config.json, который читает infer_all.py.

Классы моделей не пересекаются → каждую модель тюним независимо.

  python pipeline/platform_tune.py --model model1 [--limit 12] [--grid coarse]
  python pipeline/platform_tune.py --model model3 --grid fine
"""
import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # корень репо
import geoloc_paths  # noqa: E402
import predict  # noqa: E402
from infer_all import RU2EN, CLASS_WEIGHT, DEFAULT_CONFIG, merge_config  # noqa: E402

import compute_metrics  # noqa: E402  (метрика платформы, корень репозитория)

from pyproj import CRS as pCRS
from pyproj import Transformer
from rasterio.transform import Affine
from shapely.geometry import mapping
from shapely.ops import transform as shp_transform

# Сетки свипа: порог × буфер (м) × множитель мин-площади
GRIDS = {
    'coarse': dict(thr=(0.35, 0.5, 0.65), buf=(0.0, 3.0, 8.0), area=(0.5, 1.0)),
    'fine': dict(thr=(0.25, 0.35, 0.45, 0.55, 0.65, 0.75),
                 buf=(0.0, 2.0, 4.0, 8.0, 15.0), area=(0.25, 0.5, 1.0)),
}
# Для m4 (точки) буфер не важен — свипаем только порог/площадь.


def gt_for_model(model: str, work: Path) -> dict:
    """GT val-сплита модели из work/marks -> FeatureCollection в 3857 (формат платформы)."""
    split = json.load(open(work / 'splits' / f'{model}_split.json',
                           encoding='utf-8'))['val']
    feats = []
    for site in sorted(set(split)):
        md = work / 'marks' / site
        if not md.is_dir():
            continue
        for f in sorted(md.glob('*.geojson')):
            g = json.load(open(f, encoding='utf-8'))
            crs = g.get('crs', {}).get('properties', {}).get('name', 'EPSG:3857')
            tr = Transformer.from_crs(pCRS.from_string(crs), pCRS.from_epsg(3857),
                                      always_xy=True)
            for ft in g['features']:
                en = RU2EN.get(ft['properties'].get('cls', ''))
                if not en:
                    continue
                feats.append({'type': 'Feature', 'properties': {
                    'class_name': en, 'confidence': 1.0}, 'geometry':
                    geojson_3857(ft['geometry'], tr)})
    return {'type': 'FeatureCollection', 'features': feats}


def geojson_3857(geom, tr):
    def ring(r):
        return [[float(x), float(y)] for x, y in (tr.transform(px, py) for px, py in r)]
    if geom['type'] == 'Polygon':
        return {'type': 'Polygon', 'coordinates': [ring(r) for r in geom['coordinates']]}
    if geom['type'] == 'MultiPolygon':
        return {'type': 'MultiPolygon',
                'coordinates': [[ring(r) for r in poly] for poly in geom['coordinates']]}
    return geom


def val_rasters(model: str, work: Path, limit: int):
    """Val-растры модели (как infer.py all_val): m2/m3 — только первичные Li."""
    rasters = json.load(open(work / 'rasters.json', encoding='utf-8'))
    split = set(json.load(open(work / 'splits' / f'{model}_split.json',
                               encoding='utf-8'))['val'])
    src_kinds = {'model1': ('SpOr', 'Sp', 'Ae'), 'model2': ('Li',),
                 'model3': ('Li', 'SpOr', 'Sp', 'Ae', 'Or')}[model]
    out = []
    for stem, r in rasters.items():
        if r['site'] not in split or r['kind'] not in src_kinds:
            continue
        if model in ('model2', 'model3') and r['kind'] == 'Li' \
                and not stem.endswith('_Li_c'):
            continue
        out.append((stem, r))
    out.sort(key=lambda kv: -max(kv[1]['w'], kv[1]['h']))      # крупные первыми
    return out[:limit] if limit else out


def pred_fc(feats, model_classes):
    """Фичи predict.probmaps_to_features (в CRS растра) -> FC в 3857."""
    out = []
    for f in feats:
        tr = f['tf']
        g = mapping(shp_transform(
            lambda x, y, z=None: tr.transform(x, y), f['geometry']))
        out.append({'type': 'Feature', 'properties': {
            'class_name': RU2EN[f['cls']], 'confidence': round(float(f['conf']), 3)},
            'geometry': g})
    return {'type': 'FeatureCollection', 'features': out}


def model_score(calc, pred: dict, gt: dict, model_classes) -> float:
    """Взвешенный F1 только по классам модели (final_score платформы включал бы
    чужие классы с GT и нулём предсказаний — для пер-модельного тюнинга не нужно)."""
    res = calc.calculate_f1(pred, gt)['per_class']
    num = den = 0.0
    for cls_ru, cls_en in zip(model_classes, [RU2EN[c] for c in model_classes]):
        r = res.get(cls_en, {})
        if r.get('num_ground_truth', 0) == 0:
            continue                       # класса нет в этом val — веса не влияет
        num += r['f1'] * CLASS_WEIGHT[cls_en]
        den += CLASS_WEIGHT[cls_en]
    return num / den if den else 0.0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model', required=True,
                    choices=['model1', 'model2', 'model3', 'model4'])
    ap.add_argument('--limit', type=int, default=12,
                    help='сколько val-растров брать (0 = все)')
    ap.add_argument('--grid', default='coarse', choices=list(GRIDS))
    ap.add_argument('--run', default='', help='суффикс прогона чекпойнта')
    ap.add_argument('--checkpoint', default='')
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

    work = geoloc_paths.work_dir()
    cfg = merge_config(geoloc_paths.infer_config_path())
    spec = predict.load_spec(args.model, args.checkpoint, args.run)
    print(f'{args.model}: {spec.ckpt} ({spec.encoder})')
    device = predict.pick_device()
    net = predict.build_net(spec, device)

    if args.model == 'model4':
        return tune_model4(spec, net, device, work, args)

    # ---- val-растры -> кэш карт вероятностей -------------------------------
    stems = val_rasters(args.model, work, args.limit)
    print(f'val-растров: {len(stems)}; инференс...', flush=True)
    cache = []                                   # (probs, canvas_tf, res_eff, tf3857)
    import rasterio
    for stem, r in stems:
        t0 = time.time()
        with rasterio.open(r['path']) as src:
            probs, info = predict.predict_raster(
                spec, net, device, r['path'], kind=r['kind'],
                aux_path=r.get('aux_g', ''), overlap=cfg['overlap'],
                tta=cfg['tta'], batch=cfg['batch'])
            tf = src.transform * Affine.scale(info['scale'], info['scale'])
            crs_str = r['crs']
        to3857 = Transformer.from_crs(pCRS.from_string(crs_str),
                                      pCRS.from_epsg(3857), always_xy=True)
        cache.append((probs, tf, info['res_eff'], to3857))
        print(f'  {stem}: scale={info["scale"]} {info["n_windows"]} окон, '
              f'{time.time() - t0:.0f} с', flush=True)

    gt = gt_for_model(args.model, work)
    n_gt = sum(1 for f in gt['features']
               if f['properties']['class_name'] in
               [RU2EN[c] for c in spec.classes])
    print(f'GT-объектов классов модели: {n_gt}')

    calc = compute_metrics.F1Calculator()
    g = GRIDS[args.grid]
    base_area = cfg['min_area_m2']
    best = (-1, None)
    rows = []
    for thr, buf, amul in itertools.product(g['thr'], g['buf'], g['area']):
        t0 = time.time()
        feats_all = []
        for probs, tf, res_eff, to3857 in cache:
            min_area = {c: base_area.get(c, 10.0) * amul for c in spec.classes}
            buffer = {c: buf for c in spec.classes}
            thresholds = {c: thr for c in spec.classes}
            fs = predict.probmaps_to_features(
                probs, spec.classes, tf, res_eff, thresholds, min_area, buffer,
                as_points=('finds_points',))
            feats_all += [dict(f, tf=to3857) for f in fs]
        s = model_score(calc, pred_fc(feats_all, spec.classes), gt, spec.classes)
        rows.append((thr, buf, amul, s))
        print(f'  порог {thr:.2f} буфер {buf:4.1f} м площадь ×{amul:.2f}: '
              f'{s:.4f}  ({time.time() - t0:.0f} с, {len(feats_all)} пред.)',
              flush=True)
        if s > best[0]:
            best = (s, (thr, buf, amul))

    print(f'\nЛучший: score {best[0]:.4f} при порог {best[1][0]}, буфер '
          f'{best[1][1]} м, площадь ×{best[1][2]}')
    write_config(args.model, spec, best[1], base_area)


def write_config(model, spec, best, base_area):
    """Лучшие параметры -> work/infer_config.json (merge)."""
    thr, buf, amul = best
    path = geoloc_paths.infer_config_path()
    cfg = json.load(open(path, encoding='utf-8')) if path.exists() else {}
    cfg.setdefault('checkpoints', {})[model] = str(spec.ckpt)
    cfg.setdefault('thresholds', {})[model] = {c: thr for c in spec.classes}
    for c in spec.classes:
        cfg.setdefault('buffer_m', {})[c] = buf
        cfg.setdefault('min_area_m2', {})[c] = round(base_area.get(c, 10.0) * amul, 2)
    path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(cfg, open(path, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print(f'-> {path}')


def tune_model4(spec, net, device, work, args):
    """m4: точки finds_points матчатся по центроиду (допуск 2 м) — буфер не нужен."""
    raise SystemExit('m4: сначала обучите (TRAINING_NEXT.md), затем добавьте '
                     'GT из work/geo и запустите этот же свип с порогами/площадью')


if __name__ == '__main__':
    main()
