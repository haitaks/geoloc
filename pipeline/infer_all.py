#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Этап A плана §8: универсальный инференс тест-каталога → result.geojson.

Обходит регионы (region_XXXX или *_FINAL), сам определяет модальность слоёв по
именам файлов и запускает нужные модели на каждом растре:

  SpOr/Sp/Ae -> model1 (повреждённые курганы) + model3
  Or         -> model3
  Li_c(+g)   -> model2 (целые курганы) + model3            [в тесте не встречается]
  La (.las)  -> деривация la_derive (хиллшейд+уклоны) -> model2 + model3 (markup 'la')
  MagGr/EtGr/TerFl -> model4 (finds_points -> точки, object_poly),
                клип по extra/*_границы.geojson, выброс extra/*_шум*

Постобработка под платформенную метрику: порог/мин-площадь/буфер по классам из
work/infer_config.json (пишется platform_tune.py; без него — дефолты ниже).
Выход: EPSG:3857, 7 канонических классов, лимиты 100 МБ / 20 000 объектов /
3 000 вершин, бюджет времени (--time-budget, по умолчанию 25 из 30 минут).

  python pipeline/infer_all.py --input <каталог регионов> --output <каталог>
  python pipeline/infer_all.py --input ... --dry-run        # только план заданий
"""
import argparse
import json
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import geoloc_paths  # noqa: E402
import la_derive  # noqa: E402
import predict  # noqa: E402

from pyproj import CRS as pCRS
from pyproj import Transformer
from rasterio import features as rfeatures

HOMO = str.maketrans({'А': 'A', 'а': 'a', 'Е': 'E', 'е': 'e', 'О': 'O', 'о': 'o',
                      'С': 'C', 'с': 'c', 'Р': 'P', 'р': 'p'})

RU2EN = {
    'курганы_поврежденные': 'kurgany_povrezhdennye',
    'курганы_целые': 'kurgany_tselye',
    'городища': 'gorodishcha',
    'фортификации': 'fortifikatsii',
    'архитектуры': 'arkhitektury',
    'finds_points': 'finds_points',
    'object_poly': 'object_poly',
}
CLASS_WEIGHT = {
    'kurgany_povrezhdennye': 30.0, 'kurgany_tselye': 20.0, 'gorodishcha': 15.0,
    'arkhitektury': 10.0, 'object_poly': 10.0, 'fortifikatsii': 5.0,
    'finds_points': 10.0,
}
MODEL_WEIGHT = {'model1': 30.0, 'model2': 20.0, 'model3': 30.0, 'model4': 20.0}
MARKUP_TYPE = {'Li': 'li', 'La': 'la', 'SpOr': 'sp_or', 'Sp': 'sp_or',
               'Ae': 'ae', 'Or': 'or', 'MagGr': 'maggr', 'EtGr': 'etgr',
               'TerFl': 'terfl'}

# Дефолты постобработки (до первого запуска platform_tune.py). Пороги ниже
# F1-оптимальных: FP стоит 0.5, а cov-матчинг мягкий — ставка на полноту (§8Б).
DEFAULT_CONFIG = {
    'checkpoints': {},                     # model -> путь к best.pth
    'thresholds': {
        'model1': {'курганы_поврежденные': 0.5},
        'model2': {'курганы_целые': 0.5},
        'model3': {'городища': 0.4, 'фортификации': 0.4, 'архитектуры': 0.4},
        'model4': {'finds_points': 0.5, 'object_poly': 0.5},
    },
    'min_area_m2': {
        'курганы_поврежденные': 10, 'курганы_целые': 10, 'городища': 25,
        'фортификации': 10, 'архитектуры': 10,
        'finds_points': 1, 'object_poly': 2,
    },
    'buffer_m': {
        'курганы_поврежденные': 3.0, 'курганы_целые': 3.0, 'городища': 10.0,
        'фортификации': 3.0, 'архитектуры': 3.0, 'object_poly': 2.0,
    },
    'overlap': 0.5, 'tta': True, 'batch': 4, 'time_budget_s': 1500.0,
}

LIMIT_OBJECTS = 20000
LIMIT_BYTES = 100 * 1024 * 1024
MAX_VERTICES = 2900


def merge_config(path: Path) -> dict:
    cfg = json.loads(json.dumps(DEFAULT_CONFIG))           # глубокая копия
    if path.exists():
        user = json.load(open(path, encoding='utf-8'))
        for k, v in user.items():
            if isinstance(v, dict) and isinstance(cfg.get(k), dict):
                cfg[k].update(v)
            else:
                cfg[k] = v
    # относительные пути чекпойнтов — от каталога конфига (пакет сабмита)
    for m, ck in list(cfg.get('checkpoints', {}).items()):
        p = Path(ck)
        if not p.is_absolute() and not p.exists():
            cand = path.parent / p
            if cand.exists():
                cfg['checkpoints'][m] = str(cand)
    return cfg


# ---------------------------------------------------------------------------
# Скан тест-каталога
@dataclass
class Layer:
    path: Path
    kind: str                       # Li/La/SpOr/Sp/Ae/Or/MagGr/EtGr/TerFl
    stem: str
    primary: bool = True            # Li: только *_Li_c (остальные — производные)
    aux_g: str = ''
    geo_kind: str = ''              # m4: maggr_gray/maggr_color/etgr_map/terfl
    depth: Optional[float] = None   # EtGr: глубина среза
    derived: bool = False


@dataclass
class Region:
    name: str
    root: Path
    layers: List[Layer] = field(default_factory=list)
    bounds_geo: List[dict] = field(default_factory=list)   # extra/*_границы
    noise_geo: List[dict] = field(default_factory=list)    # extra/*_шум


def classify_layer(rel: str, f: Path) -> Optional[Layer]:
    """Модальность слоя по имени. Латинские токены — по транслитерированному
    имени (HOMO), русские ключевые слова (профиль/цвет) — по исходному: HOMO
    переводит их буквы в латинские и сравнение молча ломается."""
    name = rel.translate(HOMO)
    m = re.search(r'_(SpOr|Sp|Ae|Or|Li|La)_', name)
    geo = re.search(r'_(MagGr|EtGr|TerFl)', name)
    if m:
        k = m.group(1)
        if k == 'Li':
            stem = f.stem
            if stem.endswith('_Li_c'):
                aux = list(f.parent.glob('*_Li_g.tif'))
                return Layer(f, 'Li', stem, True, str(aux[0]) if aux else '')
            return None                                 # Li_g/Li_ch — не первичные
        return Layer(f, k, f.stem)
    if geo:
        g = geo.group(1)
        if g == 'MagGr':
            gk = 'maggr_color' if '_MagGr_цвет' in rel else 'maggr_gray'
            return Layer(f, 'MagGr', f.stem, geo_kind=gk)
        if g == 'EtGr':
            if '_EtGr_профиль' in rel:
                return None                             # профили — второй эшелон
            dm = re.search(r'_(\d+(?:\.\d+)?)$', f.stem)
            return Layer(f, 'EtGr', f.stem, geo_kind='etgr_map',
                        depth=float(dm.group(1)) if dm else None)
        return Layer(f, 'TerFl', f.stem, geo_kind='terfl')
    return None


def scan_regions(input_dir: Path) -> List[Region]:
    regions = []
    for rd in sorted(p for p in input_dir.iterdir() if p.is_dir()):
        reg = Region(rd.name, rd)
        for f in sorted(rd.rglob('*')):
            if not f.is_file():
                continue
            rel = str(f.relative_to(rd))
            if f.suffix.lower() in ('.tif', '.tiff'):
                lay = classify_layer(rel, f)
                if lay:
                    reg.layers.append(lay)
            elif f.suffix.lower() in ('.las', '.laz'):
                if re.search(r'_La_', rel.translate(HOMO)):
                    reg.layers.append(Layer(f, 'La', f.stem))
            elif f.suffix.lower() == '.geojson' and 'extra' in f.parent.name:
                try:
                    g = json.load(open(f, encoding='utf-8'))
                except Exception:
                    continue
                geoms = [ft.get('geometry') for ft in g.get('features', [])
                         if ft.get('geometry')]
                if not geoms:
                    continue
                crs = g.get('crs', {}).get('properties', {}).get('name', 'EPSG:3857')
                item = dict(geoms=geoms, crs=crs)
                if 'шум' in f.stem:
                    reg.noise_geo.append(item)
                elif 'границы' in f.stem:
                    reg.bounds_geo.append(item)
        if reg.layers:
            regions.append(reg)
    return regions


def raster_crs_str(src, region: Region) -> str:
    """CRS растра с фолбэком на зону UTM.json (LOCAL_CS/None у части Li)."""
    crs = src.crs
    if crs is None or crs.to_string().startswith('LOCAL_CS'):
        utm = la_derive.find_utm_zone(region.root)
        if utm:
            return f'EPSG:{32600 + utm:02d}'
    return crs.to_string() if crs else 'EPSG:3857'


# ---------------------------------------------------------------------------
# Задания
@dataclass
class Job:
    region: Region
    layer: Layer
    model: str
    mpx: float                          # мегапиксели растра — оценка стоимости
    priority: float


def derive_la(region: Region, work: Path, res: float = 0.2,
              allow_derive: bool = False) -> List[Layer]:
    """La без Li в регионе -> производные растры (кэш в work/derived).

    allow_derive=False — только кэш (фаза деривации уже прошла с учётом бюджета).
    """
    has_li = any(l.kind == 'Li' for l in region.layers)
    if has_li:
        return []
    out = []
    for l in region.layers:
        if l.kind != 'La':
            continue
        out_dir = work / 'derived' / region.name
        c = out_dir / f'{l.stem}_der_c.tif'
        g = out_dir / f'{l.stem}_der_g.tif'
        if not c.exists() or not g.exists():
            if not allow_derive:
                continue
            print(f'  [deriver] {region.name}/{l.path.name}: DTM {res} м ...',
                  flush=True)
            rec = la_derive.derive(l.path, out_dir, res)
            c, g = Path(rec['path']), Path(rec['aux_g'])
        out.append(Layer(c, 'Li', c.stem, True, str(g)))
    return out


def derive_phase(regions: List[Region], work: Path, deadline: float,
                 res: float = 0.2):
    """Деривация La под бюджетом времени: большие облака могут не влезть."""
    for reg in regions:
        if time.time() > deadline - 120:          # инференс важнее деривации
            print(f'  [бюджет] деривация пропущена: {reg.name}')
            continue
        derive_la(reg, work, res, allow_derive=True)


def build_jobs(regions: List[Region], models_avail: set, work: Path) -> List[Job]:
    """Задания по растрам. La-слои заменены производными (derive_phase выше)."""
    import rasterio
    jobs = []
    for reg in regions:
        layers = list(reg.layers) + derive_la(reg, work)
        for l in layers:
            if l.kind == 'La':                    # работает производный растр
                continue
            for model in predict.KIND2MODELS.get(l.kind, ()):
                if model not in models_avail:
                    continue
                try:
                    with rasterio.open(l.path) as src:
                        mpx = src.width * src.height / 1e6
                except Exception as e:
                    print(f'  ! не открылся {l.path.name}: {e}')
                    continue
                prio = MODEL_WEIGHT[model] / max(0.05, mpx)   # вес на мегапиксель
                jobs.append(Job(reg, l, model, mpx, prio))
    jobs.sort(key=lambda j: -j.priority)
    return jobs


# ---------------------------------------------------------------------------
# Инференс по заданию
def read_wc_mask(site: str, src, crs_str: str, shape_hw) -> Optional[np.ndarray]:
    """Маска леса WorldCover (work/wcmask/<место>.tif, EPSG:4326, 1=лес) в сетке канвы.

    Для тест-регионов маски готовятся заранее (submit_package/worldcover.py) и
    кладутся в пакет сабмита — в контейнере сети нет. None, если маски нет.
    """
    import rasterio
    from rasterio.windows import from_bounds
    path = geoloc_paths.work_dir() / 'wcmask' / f'{site}.tif'
    if not path.exists():
        return None
    try:
        with rasterio.open(path) as ds:
            tr = Transformer.from_crs(pCRS.from_string(crs_str), pCRS.from_epsg(4326),
                                      always_xy=True)
            b = tr.transform_bounds(*src.bounds, densify_pts=21)
            w = from_bounds(*b, transform=ds.transform)
            return ds.read(1, window=w, out_shape=shape_hw, boundless=True,
                           fill_value=0, resampling=rasterio.enums.Resampling.nearest)
    except Exception:
        return None


def run_job(job: Job, spec, net, device, cfg, work: Path) -> List[dict]:
    import rasterio
    from rasterio.transform import Affine
    l = job.layer
    t0 = time.time()
    with rasterio.open(l.path) as src:
        crs_str = raster_crs_str(src, job.region)
        keep_mask = None
        if job.model == 'model4' and job.region.bounds_geo:
            keep_mask = rasterize_in_crs(job.region.bounds_geo, crs_str,
                                         src, scale=1)      # канва scale уточнится ниже
        probs, info = predict.predict_raster(
            spec, net, device, str(l.path), kind=l.kind,
            aux_path=l.aux_g, geo_kind=l.geo_kind, depth=l.depth,
            overlap=cfg['overlap'], tta=cfg['tta'], batch=cfg['batch'])
        if keep_mask is not None:
            # границы растеризованы в нативной сетке — приводим к сетке канвы
            s = info['scale']
            if s > 1:
                kh, kw = probs.shape[1], probs.shape[2]
                keep = keep_mask[:kh * s, :kw * s]
                keep_mask = keep.reshape(kh, s, kw, s).mean(axis=(1, 3)) > 0.5
            probs[:, ~keep_mask] = 0.0
        # лес-постфильтр m1: ЛП на кронах — главный источник FP (план §6, факт 2)
        if job.model == 'model1' and l.kind in ('SpOr', 'Sp', 'Ae'):
            wc = read_wc_mask(job.region.name, src, crs_str, probs.shape[1:])
            if wc is not None:
                probs[:, wc > 0] = 0.0
        canvas_tf = src.transform * Affine.scale(info['scale'], info['scale'])
        feats = predict.probmaps_to_features(
            probs, spec.classes, canvas_tf, info['res_eff'],
            thresholds=cfg['thresholds'].get(job.model, {}),
            min_area_m2=cfg['min_area_m2'],
            buffer_m=cfg['buffer_m'],
            as_points=('finds_points',),
            max_vertices=MAX_VERTICES)
        if job.model == 'model4' and job.region.noise_geo:
            noise = polys_in_crs(job.region.noise_geo, crs_str)
            feats = drop_in_zones(feats, noise)
    src_crs = pCRS.from_string(crs_str)
    to3857 = Transformer.from_crs(src_crs, pCRS.from_epsg(3857), always_xy=True)
    markup = MARKUP_TYPE['La' if is_derived(job.layer) else job.layer.kind]
    out = []
    for f in feats:
        out.append(dict(cls=RU2EN[f['cls']], conf=f['conf'], geom=f['geometry'],
                        markup=markup, region=job.region.name,
                        sub=l.path.stem, tf=to3857))
    out and print(f'    {job.model}/{l.path.name}: {len(out)} объектов, '
                  f'{time.time() - t0:.1f} c', flush=True)
    return out


def is_derived(l: Layer) -> bool:
    return l.stem.endswith('_der_c')


def polys_in_crs(items: List[dict], dst_crs: str) -> list:
    """Полилинии из extra-geojson (обычно 3857) в CRS растра (для shapely)."""
    from shapely.geometry import shape
    from shapely.ops import transform as shp_transform
    out = []
    for it in items:
        try:
            tr = Transformer.from_crs(pCRS.from_string(it['crs']),
                                      pCRS.from_string(dst_crs), always_xy=True)
            for g in it['geoms']:
                if g.get('type') not in ('Polygon', 'MultiPolygon'):
                    continue
                out.append(shp_transform(lambda x, y, z=None: tr.transform(x, y),
                                         shape(g)))
        except Exception:
            continue
    return out


def rasterize_in_crs(items: List[dict], dst_crs: str, src, scale: int) -> np.ndarray:
    """Булева маска «внутри границ съёмки» в сетке растра (нативной)."""
    polys = polys_in_crs(items, dst_crs)
    if not polys:
        return None
    h = int(np.ceil(src.height / scale))
    w = int(np.ceil(src.width / scale))
    from rasterio.transform import Affine
    tf = src.transform * Affine.scale(scale, scale)
    m = rfeatures.rasterize(((g, 1) for g in polys), out_shape=(h, w),
                            transform=tf, fill=0, dtype='uint8')
    return m > 0


def drop_in_zones(feats: List[dict], zones: list) -> List[dict]:
    return [f for f in feats
            if not any(z.intersects(f['geometry']) for z in zones)]


# ---------------------------------------------------------------------------
# Экспорт result.geojson
def geom_to_fc(geom, tf):
    """Геометрия shapely в CRS растра -> GeoJSON-словарь в EPSG:3857 (округление 1 см)."""
    from shapely.geometry import mapping
    from shapely.ops import transform as shp_transform

    def pt(x, y, z=None):
        a, b = tf.transform(x, y)
        return (round(a, 2), round(b, 2))

    return mapping(shp_transform(pt, geom))


def export(features: List[dict], out_path: Path):
    """Лимиты платформы: ≤20 000 объектов (по conf×вес), ≤100 МБ."""
    features = sorted(features, key=lambda f: -f['conf'] * CLASS_WEIGHT.get(f['cls'], 1))
    features = features[:LIMIT_OBJECTS]
    fc = {'type': 'FeatureCollection',
          'crs': {'type': 'name', 'properties': {'name': 'urn:ogc:def:crs:EPSG::3857'}},
          'features': []}
    for i, f in enumerate(features):
        fc['features'].append({
            'type': 'Feature',
            'properties': {
                'class_name': f['cls'], 'region_name': f['region'],
                'sub_region_name': f.get('sub', ''), 'markup_type': f['markup'],
                'crs': 'urn:ogc:def:crs:EPSG::3857', 'fid': i,
                'confidence': round(float(f['conf']), 3)},
            'geometry': geom_to_fc(f['geom'], f['tf'])})
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as fh:
        json.dump(fc, fh, ensure_ascii=False)
    # контроль размера: режем хвост по conf×вес
    while out_path.stat().st_size > LIMIT_BYTES and len(fc['features']) > 100:
        fc['features'] = fc['features'][:int(len(fc['features']) * 0.8)]
        with open(out_path, 'w', encoding='utf-8') as fh:
            json.dump(fc, fh, ensure_ascii=False)
    return len(fc['features']), out_path.stat().st_size


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input', required=True, help='каталог с регионами теста')
    ap.add_argument('--output', required=True, help='куда писать result.geojson')
    ap.add_argument('--config', default='', help='путь к infer_config.json (иначе work/)')
    ap.add_argument('--models', default='', help='подмножество моделей, напр. model1,model3')
    ap.add_argument('--time-budget', type=float, default=0,
                    help='сек; 0 = взять из конфига (по умолчанию 1500)')
    ap.add_argument('--dry-run', action='store_true', help='только план заданий')
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

    work = geoloc_paths.work_dir()
    cfg_path = Path(args.config) if args.config else geoloc_paths.infer_config_path()
    cfg = merge_config(cfg_path)
    if args.time_budget > 0:
        cfg['time_budget_s'] = args.time_budget

    regions = scan_regions(Path(args.input))
    print(f'Регионов: {len(regions)}; слоёв: {sum(len(r.layers) for r in regions)}')
    for r in regions:
        kinds = defaultdict(int)
        for l in r.layers:
            kinds[l.kind] += 1
        print(f'  {r.name}: {dict(kinds)}'
              + (f' + границы{len(r.bounds_geo)}/шум{len(r.noise_geo)}'
                 if r.bounds_geo or r.noise_geo else ''))

    # какие модели доступны (чекпойнты существуют)
    avail = set()
    for m in predict.MODEL_SPECS:
        ck = cfg['checkpoints'].get(m) or str(geoloc_paths.runs_dir() / m / 'best.pth')
        if Path(ck).exists():
            avail.add(m)
        else:
            print(f'  ! {m}: чекпойнт не найден ({ck}) — модель пропущена')
    if args.models:
        avail &= set(args.models.split(','))

    # Деривация La (до инференса, внутри общего бюджета времени)
    deadline = time.time() + cfg['time_budget_s']
    if not args.dry_run:
        derive_phase(regions, work, deadline)

    jobs = build_jobs(regions, avail, work)
    total_mpx = sum(j.mpx for j in jobs)
    print(f'Заданий: {len(jobs)} ({total_mpx:.0f} Мпикс), модели: {sorted(avail)}')
    if args.dry_run:
        for j in jobs:
            print(f'  {j.priority:7.1f}  {j.model}  {j.region.name}/{j.layer.path.name}'
                  f'  {j.mpx:.1f} Мпикс')
        return

    device = predict.pick_device()
    print('Устройство:', device)
    specs, nets = {}, {}
    for m in sorted(avail):
        specs[m] = predict.load_spec(m, cfg['checkpoints'].get(m, ''))
        nets[m] = predict.build_net(specs[m], device)
        print(f'  {m}: {specs[m].ckpt} ({specs[m].encoder}, {specs[m].in_ch} кан.)')

    rate = defaultdict(float)               # сек на Мпикс по моделям (скользящая)
    feats: List[dict] = []
    done = skipped = 0
    for job in jobs:
        remain = deadline - time.time()
        est = job.mpx * rate[job.model] if rate[job.model] else 60.0
        if remain < est + 15:
            print(f'  [бюджет] пропуск {job.model} {job.layer.path.name} '
                  f'(осталось {remain:.0f} с, оценка {est:.0f} с)', flush=True)
            skipped += 1
            continue
        t0 = time.time()
        try:
            out = run_job(job, specs[job.model], nets[job.model], device, cfg, work)
            feats.extend(out)
            dt = time.time() - t0
            r = dt / max(0.1, job.mpx)
            rate[job.model] = 0.5 * rate[job.model] + 0.5 * r if rate[job.model] else r
            done += 1
        except Exception as e:
            print(f'  ! ошибка {job.model} {job.layer.path.name}: {e}', flush=True)
    print(f'Выполнено заданий: {done}, пропущено по бюджету: {skipped}; '
          f'объектов: {len(feats)}')

    out_path = Path(args.output)
    n, size = export(feats, out_path / 'result.geojson' if out_path.suffix == '' \
                     else out_path)
    print(f'result.geojson: {n} объектов, {size / 1e6:.1f} МБ')


if __name__ == '__main__':
    main()
