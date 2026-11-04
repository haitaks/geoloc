# -*- coding: utf-8 -*-
"""Локальный стенд платформенной метрики (compute_metrics.py).

1. GT: разметка val-мест (work/marks) -> solution-формат (EPSG:3857, транслит классов).
2. Предсказания: результат infer.py (GeoJSON в CRS растра) -> solution-формат.
3. Вызов compute_metrics CLI -> final_score (взвешенный F1, FP x0.5, cov>=0.5).

  python pipeline/eval_platform.py --gt-model model2                # GT val-мест модели
  python pipeline/eval_platform.py --pred work/infer/model2/xx.geojson --gt-model model2
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

from pyproj import CRS as pCRS
from pyproj import Transformer

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
GEO = Path(r'G:\projects\geoloc')
WORK = GEO / 'work'
METRICS = GEO / 'compute_metrics (1).py'

RU2EN = {
    'курганы_целые': 'kurgany_tselye',
    'курганы_поврежденные': 'kurgany_povrezhdennye',
    'городища': 'gorodishcha',
    'фортификации': 'fortifikatsii',
    'архитектуры': 'arkhitektury',
    'FindsPoints': 'finds_points',
    'findspoints': 'finds_points',
    'ObjectPoly': 'object_poly',
    'objectpoly': 'object_poly',
}
KIND2MT = {'Li': 'li', 'La': 'la', 'Ae': 'ae', 'SpOr': 'sp_or', 'Sp': 'sp_or',
           'Or': 'or', 'MagGr': 'maggr', 'EtGr': 'etgr', 'TerFl': 'terfl'}
M3857 = pCRS.from_epsg(3857)


def to_3857(geom, crs):
    tr = Transformer.from_crs(pCRS.from_string(crs), M3857, always_xy=True)

    def ring(r):
        return [[float(x), float(y)] for x, y in (tr.transform(px, py) for px, py in r)]

    if geom['type'] == 'Polygon':
        return {'type': 'Polygon', 'coordinates': [ring(r) for r in geom['coordinates']]}
    if geom['type'] == 'MultiPolygon':
        return {'type': 'MultiPolygon',
                'coordinates': [[ring(r) for r in poly] for poly in geom['coordinates']]}
    if geom['type'] == 'Point':
        x, y = tr.transform(geom['coordinates'][0], geom['coordinates'][1])
        return {'type': 'Point', 'coordinates': [float(x), float(y)]}
    return geom


def fc_from_marks(model, out, only_kind=None):
    """GT из work/marks val-мест модели (все источники; опционально фильтр по kind растра)."""
    split = json.load(open(WORK / 'splits' / f'{model}_split.json', encoding='utf-8'))['val']
    sites = set(split)
    feats = []
    for site in sorted(sites):
        md = WORK / 'marks' / site
        if not md.is_dir():
            continue
        for f in sorted(md.glob('*.geojson')):
            g = json.load(open(f, encoding='utf-8'))
            crs = g.get('crs', {}).get('properties', {}).get('name', 'EPSG:3857')
            kind = next((v for k, v in KIND2MT.items() if f'_{k}_' in f.stem), None)
            kind = kind or ('etgr' if '_EtGr_' in f.stem else 'maggr' if '_MagGr_' in f.stem
                            else 'terfl' if '_TerFl_' in f.stem else 'li')
            if only_kind and kind != only_kind:
                continue
            for ft in g['features']:
                cls = RU2EN.get(ft['properties'].get('cls', ''), None)
                if not cls:
                    continue
                feats.append({'type': 'Feature', 'properties': {
                    'class_name': cls, 'region_name': site, 'markup_type': kind,
                    'confidence': 1.0, 'crs': 'EPSG:3857'},
                    'geometry': to_3857(ft['geometry'], crs)})
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({'type': 'FeatureCollection', 'features': feats},
              open(out, 'w', encoding='utf-8'), ensure_ascii=False)
    print(f'GT: {len(feats)} объектов -> {out}')
    return out


def fc_from_infer(pred_path, out):
    """Предсказания infer.py (файл или каталог) -> solution-формат.

    Регион и markup_type берутся из work/rasters.json по имени растра."""
    rasters = json.load(open(WORK / 'rasters.json', encoding='utf-8'))
    paths = ([Path(pred_path)] if Path(pred_path).is_file()
             else sorted(Path(pred_path).glob('*.geojson')))
    feats = []
    for p in paths:
        g = json.load(open(p, encoding='utf-8'))
        crs = g.get('crs', {}).get('properties', {}).get('name', 'EPSG:3857')
        ras = rasters.get(p.stem, {})
        site = ras.get('site', p.stem.split('_', 1)[-1].rsplit('_', 2)[0])
        kind = KIND2MT.get(ras.get('kind', 'Li'), 'li')
        for ft in g['features']:
            cls = RU2EN.get(ft['properties'].get('class', ft['properties'].get('cls', '')))
            if not cls:
                continue
            feats.append({'type': 'Feature', 'properties': {
                'class_name': cls, 'region_name': site, 'markup_type': kind,
                'confidence': ft['properties'].get('confidence', 1.0),
                'crs': 'EPSG:3857'},
                'geometry': to_3857(ft['geometry'], crs)})
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({'type': 'FeatureCollection', 'features': feats},
              open(out, 'w', encoding='utf-8'), ensure_ascii=False)
    print(f'Pred: {len(feats)} объектов из {len(paths)} растров -> {out}')
    return out


def run_metrics(pred, gt):
    r = subprocess.run([sys.executable, str(METRICS), '-p', str(pred), '-g', str(gt)],
                       capture_output=True, text=True, encoding='utf-8', errors='replace')
    out = r.stdout.strip().splitlines()
    print('\n'.join(out[-40:]))
    if r.returncode != 0:
        print('STDERR:', r.stderr[-1500:])


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--gt-model', default='model2', help='модель, чей val-сплит брать для GT')
    ap.add_argument('--pred', default='', help='GeoJSON инференса (или пусто — только GT)')
    ap.add_argument('--only-kind', default='', help='фильтр GT по markup_type (li/sp_or/...)')
    ap.add_argument('--min-confidence', type=float, default=0.0,
                    help='фильтр предсказаний по confidence')
    args = ap.parse_args()
    suffix = f'_{args.only_kind}' if args.only_kind else ''
    gt = fc_from_marks(args.gt_model, WORK / 'platform' / f'gt_{args.gt_model}{suffix}.geojson',
                       args.only_kind or None)
    if args.pred:
        pred = fc_from_infer(args.pred, WORK / 'platform' / 'pred.geojson')
        if args.min_confidence > 0:
            g = json.load(open(pred, encoding='utf-8'))
            g['features'] = [f for f in g['features']
                             if f['properties'].get('confidence', 1.0) >= args.min_confidence]
            pred_filtered = WORK / 'platform' / f'pred_c{args.min_confidence:.2f}.geojson'
            json.dump(g, open(pred_filtered, 'w', encoding='utf-8'), ensure_ascii=False)
            print(f"фильтр confidence>={args.min_confidence}: осталось {len(g['features'])}")
            pred = pred_filtered
        run_metrics(pred, gt)
