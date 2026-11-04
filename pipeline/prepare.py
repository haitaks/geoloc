# -*- coding: utf-8 -*-
"""Препроцессинг train-expds-3 под три модели (см. План_трёх_моделей.md).

Правила разметки на растр:
  * нативная разметка источника попадает в маску полностью (без дедупликации);
  * кластеры (объект, размеченный на нескольких источниках, ~50 м в 3857) используются
    только для переноса: если у кластера нет представителя источника растра,
    переносится лучший полигон кластера.

Выходы в work/:
  rasters.json               — индекс растров (id -> путь, CRS, источник, размер, aux_g)
  marks/<место>/<стем>.geojson — объекты в CRS растра: {cls, native, cid}
  manifests/model{1,2,3}_tiles.csv — тайлы: растр, окно в пикселях, счётчики классов
  splits/model{1,2,3}_split.json  — сплит по местам (train/val), детерминированный
  prepare_report.md          — сводка
"""
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import rasterio
from pyproj import CRS, Transformer
from rasterio.warp import transform_bounds

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
ROOT = Path(r'G:\projects\train-expds-3')
WORK = Path(r'G:\projects\geoloc\work')
SRC_RE = re.compile(r'_(La|Li|Ae|SpOr|Sp|Or)_')
CLS_RE = re.compile(r'(курганы_целые|курганы_поврежденные|городищ[аа]|фортификации|архитектуры)')
HOMO = str.maketrans({'А': 'A', 'а': 'a', 'Е': 'E', 'е': 'e', 'О': 'O', 'о': 'o',
                      'С': 'C', 'с': 'c', 'Р': 'P', 'р': 'p'})
M3857 = CRS.from_epsg(3857)

# Порог кластеризации «тот же объект» в метрах 3857 (~x1.7-2 к наземным метрам).
# Курганы: официально «ближе 50 м — отдельные объекты» (КОЗ №2), поэтому узкий порог ~13 м
# на местности; крупные классы (городища с валами ≤100 м) — шире.
CLUSTER_M_DEFAULT = 50.0
CLUSTER_M_BY_CLASS = {'курганы_целые': 25.0, 'курганы_поврежденные': 25.0}

MODELS = {
    'model1': dict(sources=('SpOr', 'Sp', 'Ae'), classes=('курганы_поврежденные',),
                   tile=512, res=0.5, overlap=0.25),
    'model2': dict(sources=('Li',), classes=('курганы_целые',),
                   tile=512, res=0.2, overlap=0.25, li_primary_only=True),
    'model3': dict(sources=('Li', 'SpOr', 'Sp', 'Ae', 'Or'),
                   classes=('городища', 'фортификации', 'архитектуры'),
                   tile=512, res=0.5, overlap=0.25, li_primary_only=True),
}
MAX_WIN_PX = 16384   # сверхдетальные растры (Or без пирамид) при таком окне не читаются


def poly_area_centroid(rings):
    """Площадь и центроид внешнего кольца; None для битых полигонов."""
    try:
        ring = rings[0] if isinstance(rings[0][0][0], (int, float)) else rings[0][0]
    except (IndexError, TypeError):
        return None
    if len(ring) < 3:
        return None
    s = 0.0
    for i in range(len(ring) - 1):
        s += ring[i][0] * ring[i + 1][1] - ring[i + 1][0] * ring[i][1]
    cx = sum(p[0] for p in ring[:-1]) / (len(ring) - 1)
    cy = sum(p[1] for p in ring[:-1]) / (len(ring) - 1)
    return abs(s) / 2.0, (cx, cy)


def site_utm_epsg(site_dir):
    try:
        j = json.load(open(site_dir / 'UTM.json', encoding='utf-8'))
        z = re.match(r'(\d+)', str(j.get('utm', '')))
        return int(z.group(1)) if z else None
    except Exception:
        return None


# ---------------------------------------------------------------------------
def scan_sites():
    sites = {}
    for site_dir in sorted(p for p in ROOT.iterdir() if p.is_dir()):
        feats, rasters = [], []
        utm_zone = site_utm_epsg(site_dir)
        for dp, _dn, fns in os.walk(site_dir):
            for fn in fns:
                full = Path(dp) / fn
                if fn.endswith('.geojson'):
                    ms = SRC_RE.search(fn.translate(HOMO))
                    cm = CLS_RE.search(fn)
                    if not (ms and cm):
                        continue
                    try:
                        g = json.load(open(full, encoding='utf-8'))
                    except Exception:
                        continue
                    for f in g.get('features', []):
                        geom = f.get('geometry')
                        if not geom or geom.get('type') not in ('Polygon', 'MultiPolygon'):
                            continue
                        ac = poly_area_centroid(geom['coordinates'])
                        if ac:
                            feats.append((ms.group(1), cm.group(1), geom, ac[0], ac[1]))
                elif fn.lower().endswith(('.tif', '.tiff')):
                    rel = str(full.relative_to(site_dir))
                    m = SRC_RE.search(rel.translate(HOMO))
                    if not m:
                        continue
                    try:
                        with rasterio.open(full) as r:
                            crs = r.crs
                            local = crs is None or crs.to_string().startswith('LOCAL_CS')
                            if local and utm_zone:
                                crs = CRS.from_epsg(32600 + utm_zone)
                            if crs is None:
                                print(f'  ! без CRS, пропущен: {rel}')
                                continue
                            aux_g = None
                            if m.group(1) == 'Li' and full.stem.endswith('_Li_c'):
                                cand = list(full.parent.glob('*_Li_g.tif'))
                                aux_g = str(cand[0]) if cand else None
                            rasters.append(dict(
                                path=str(full), kind=m.group(1), crs=crs.to_string(),
                                local_crs=local, w=r.width, h=r.height,
                                res=float(max(abs(r.transform.a), abs(r.transform.e))),
                                bounds3857=transform_bounds(crs, M3857, *r.bounds, densify_pts=21),
                                transform=[r.transform.a, r.transform.b, r.transform.c,
                                           r.transform.d, r.transform.e, r.transform.f],
                                site=site_dir.name, stem=full.stem, aux_g=aux_g))
                    except Exception as e:
                        print(f'  ! не открылся {rel}: {e}')
        if feats or rasters:
            sites[site_dir.name] = dict(feats=feats, rasters=rasters)
    return sites


def cluster_objects(feats):
    """Кластеры «тот же объект на разных источниках» (union-find по центроидам)."""
    n = len(feats)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        for j in range(i + 1, n):
            dx = feats[i][4][0] - feats[j][4][0]
            dy = feats[i][4][1] - feats[j][4][1]
            thr = CLUSTER_M_BY_CLASS.get(feats[i][1], CLUSTER_M_DEFAULT)
            if dx * dx + dy * dy <= thr * thr:
                parent[find(i)] = find(j)
    groups = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(feats[i])
    clusters = []
    for members in groups.values():
        cx = sum(m[4][0] for m in members) / len(members)
        cy = sum(m[4][1] for m in members) / len(members)
        clusters.append(dict(members=members, kinds={m[0] for m in members}, cen=(cx, cy)))
    return clusters


def raster_feats(site, clusters):
    """Объекты на каждый растр: вся нативная разметка + переносы недостающих кластеров."""
    out = {}
    for ras in site['rasters']:
        crs = CRS.from_string(ras['crs'])
        fwd = Transformer.from_crs(M3857, crs, always_xy=True)
        inv_t = rasterio.transform.Affine(*ras['transform'])  # пиксели -> коорд. СК растра

        def to_px(cen3857):
            x, y = fwd.transform(*cen3857)
            col, row = ~inv_t * (x, y)
            return col, row

        feats = []
        for cid, cl in enumerate(clusters):
            b = ras['bounds3857']
            cx, cy = cl['cen']
            if not (b[0] - 100 <= cx <= b[2] + 100 and b[1] - 100 <= cy <= b[3] + 100):
                continue
            native = [m for m in cl['members'] if m[0] == ras['kind']]
            if native:   # вся нативная разметка источника — как есть
                for src, cls, geom, _a, cen in native:
                    feats.append(dict(cls=cls, native=True, cid=cid, geometry=geom,
                                      px=to_px(cen)))
            else:        # переноса недостающего: лучший (максимальный) полигон кластера
                best = max(cl['members'], key=lambda m: m[3])
                feats.append(dict(cls=best[1], native=False, cid=cid,
                                  geometry=best[2], px=to_px(cl['cen'])))
        out[ras['stem']] = dict(raster=ras, feats=feats, fwd=fwd)
    return out


def transform_geom(geom, tr):
    def ring(r):
        return [[float(x), float(y)] for x, y in (tr.transform(x, y) for x, y in r)]
    if geom['type'] == 'Polygon':
        return {'type': 'Polygon', 'coordinates': [ring(r) for r in geom['coordinates']]}
    return {'type': 'MultiPolygon',
            'coordinates': [[ring(r) for r in poly] for poly in geom['coordinates']]}


def build_manifest(cfg, marks_by_site):
    rows = []
    tile, res, ov = cfg['tile'], cfg['res'], cfg['overlap']
    for site, rmarks in marks_by_site.items():
        for stem, mk in rmarks.items():
            ras = mk['raster']
            if ras['kind'] not in cfg['sources']:
                continue
            if cfg.get('li_primary_only') and ras['kind'] == 'Li' and not ras.get('aux_g') \
                    and not ras['stem'].endswith('_Li_c'):
                continue
            if cfg.get('li_primary_only') and ras['kind'] == 'Li' and not ras['stem'].endswith('_Li_c'):
                continue
            if round(tile * res / ras['res']) > MAX_WIN_PX:
                print(f'  ! пропущен сверхдетальный растр (окно > {MAX_WIN_PX} px): '
                      f'{site}/{stem} (res {ras["res"]:.4f})')
                continue
            win_x = min(ras['w'], max(8, round(tile * res / ras['res'])))
            win_y = min(ras['h'], max(8, round(tile * res / ras['res'])))
            step_x = max(1, round(win_x * (1 - ov)))
            step_y = max(1, round(win_y * (1 - ov)))
            nx = 1 if ras['w'] <= win_x else (ras['w'] - win_x) // step_x + 1
            ny = 1 if ras['h'] <= win_y else (ras['h'] - win_y) // step_y + 1
            cent = [(f['px'][0], f['px'][1], f['cls']) for f in mk['feats']]
            for iy in range(ny):
                for ix in range(nx):
                    col = min(ix * step_x, ras['w'] - win_x)
                    row = min(iy * step_y, ras['h'] - win_y)
                    counts = Counter(cls for px, py, cls in cent
                                     if col <= px < col + win_x and row <= py < row + win_y)
                    rows.append(dict(
                        raster=stem, site=site, kind=ras['kind'],
                        col=col, row=row, win_x=win_x, win_y=win_y,
                        pos=sum(v for k, v in counts.items() if k in cfg['classes']),
                        **{f'n_{c}': counts.get(c, 0) for c in cfg['classes']}))
    return rows


def make_split(cfg, rows):
    import random
    rng = random.Random(42)
    by_site = defaultdict(Counter)
    for r in rows:
        by_site[r['site']][r['kind']] += r['pos']
    sites = sorted(by_site)
    rng.shuffle(sites)
    val, train = [], []
    quota = Counter()
    total = Counter()
    for s, c in by_site.items():
        total.update(c)
    for s in sites:
        if len(val) < max(1, round(0.2 * len(sites))) and \
                all(quota[k] + by_site[s][k] <= 0.35 * total[k] for k in total):
            val.append(s)
            quota.update(by_site[s])
        else:
            train.append(s)
    return {'train': sorted(train), 'val': sorted(val)}


def main():
    WORK.mkdir(exist_ok=True)
    # --model1-res 0.25 / --model3-res 0.25: пересобрать манифест на другом разрешении
    # (рецепт чемпиона m1: курганам и архитектурам нужно 0.25 м; план §8Д)
    for name in ('model1', 'model3'):
        flag = f'--{name}-res'
        if flag in sys.argv:
            res = float(sys.argv[sys.argv.index(flag) + 1])
            MODELS[name]['res'] = res
            print(f'Разрешение {name}: {res} м/пикс')
    print('Сканирование мест...')
    sites = scan_sites()
    print(f'Мест: {len(sites)}; растров: {sum(len(s["rasters"]) for s in sites.values())}')

    marks_by_site, raster_index = {}, {}
    n_native = n_trans = 0
    uniq = set()
    for site, s in sites.items():
        clusters = cluster_objects(s['feats'])
        for cl in clusters:
            uniq.add((site, round(cl['cen'][0] / 10), round(cl['cen'][1] / 10)))
        rmarks = raster_feats(s, clusters)
        out_dir = WORK / 'marks' / site
        out_dir.mkdir(parents=True, exist_ok=True)
        for stem, mk in rmarks.items():
            fc = {'type': 'FeatureCollection', 'crs': {
                'type': 'name', 'properties': {'name': mk['raster']['crs']}}, 'features': []}
            for f in mk['feats']:
                fc['features'].append({
                    'type': 'Feature',
                    'properties': {'cls': f['cls'], 'native': f['native'], 'cid': f['cid']},
                    'geometry': transform_geom(f['geometry'], mk['fwd'])})
                n_native += f['native']
                n_trans += not f['native']
            json.dump(fc, open(out_dir / f'{stem}.geojson', 'w', encoding='utf-8'),
                      ensure_ascii=False)
            raster_index[stem] = mk['raster']
        marks_by_site[site] = rmarks
    json.dump(raster_index, open(WORK / 'rasters.json', 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    print(f'Кластеров-объектов: {len(uniq)}; полигонов native={n_native}, переносов={n_trans}')

    (WORK / 'manifests').mkdir(exist_ok=True)
    (WORK / 'splits').mkdir(exist_ok=True)
    report = ['# Сводка prepare', '',
              f'Мест: {len(sites)}, растров: {len(raster_index)}, '
              f'объектов: {len(uniq)}, native-полигонов: {n_native}, переносов: {n_trans}', '']
    for name, cfg in MODELS.items():
        rows = build_manifest(cfg, marks_by_site)
        cols = list(rows[0].keys()) if rows else []
        with open(WORK / 'manifests' / f'{name}_tiles.csv', 'w', encoding='utf-8') as fh:
            fh.write(';'.join(cols) + '\n')
            for r in rows:
                fh.write(';'.join(str(r[c]) for c in cols) + '\n')
        split = make_split(cfg, rows)
        json.dump(split, open(WORK / 'splits' / f'{name}_split.json', 'w', encoding='utf-8'),
                  ensure_ascii=False, indent=1)
        pos_tiles = sum(1 for r in rows if r['pos'] > 0)
        pos_train = sum(r['pos'] for r in rows if r['site'] in set(split['train']))
        pos_val = sum(r['pos'] for r in rows if r['site'] in set(split['val']))
        report.append(
            f'## {name}\n- тайлов: {len(rows)} (с объектами: {pos_tiles}) '
            f'из {len({r["raster"] for r in rows})} растров\n'
            f'- мест train/val: {len(split["train"])}/{len(split["val"])}; '
            f'объектов в тайлах train/val: {pos_train}/{pos_val}\n')
    open(WORK / 'prepare_report.md', 'w', encoding='utf-8').write('\n'.join(report))
    print('\n'.join(report))
    print('Готово:', WORK)


if __name__ == '__main__':
    main()
