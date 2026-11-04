# -*- coding: utf-8 -*-
"""Этап В плана: LAS-облако → производные рельефа в формате Li-карт.

Без этого модели 2/3 не работают в тест-регионах, где есть только La (Li-растров
в публичной выборке нет вовсе): region_4AR4, 5JID, ESME, JTKK, MYV9 и т.д.

Что делает:
  1. читает .las/.laz потоково (чанки laspy), CRS — из заголовка (WKT) или UTM.json места;
  2. DTM: минимум z по ячейкам сетки res (по умолчанию 0.2 м = разрешение чемпиона m2)
     по точкам класса «земля» (classification 2); если классификации нет — по всем точкам;
     пустоты заливаются ближайшим значением (distance transform);
  3. растры-производные как у Li-карт:
       <stem>_der_c.tif — хиллшейд, 3 канала uint8 (серый ×3, как Ae в dataset.py);
       <stem>_der_g.tif — уклоны, 1 канал uint8 (градусы × 255/90);
     LZW, тайловые — пригодны для скользящего окна;
  4. печатает JSON-запись, совместимую с rasters.json (kind='Li', aux_g=...),
     и (по --register) дописывает её в work/rasters.json.

  python pipeline/la_derive.py --region <путь к региону> [--res 0.2]
  python pipeline/la_derive.py --las <файл.las> --res 0.2 [--register]
"""
import argparse
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import geoloc_paths  # noqa: E402

CHUNK = 5_000_000


def find_utm_zone(las_path: Path):
    """UTM-зона из UTM.json (ищем по родителям файла) → EPSG:326NN."""
    for p in las_path.parents[:3]:
        f = p / 'UTM.json'
        if f.exists():
            try:
                j = json.load(open(f, encoding='utf-8'))
                z = re.match(r'(\d+)', str(j.get('utm', '')))
                if z:
                    return int(z.group(1))
            except Exception:
                pass
    return None


def grid_min_z(las_path: Path, res: float, use_ground: bool):
    """DTM: минимум z по ячейкам. Возвращает (zmin с nan, xmin, ymax, ch, cw, crs)."""
    import laspy

    with laspy.open(las_path) as f:
        header = f.header
        crs = None
        try:
            crs = header.parse_crs()
        except Exception:
            crs = None
        crs_str = None
        if crs is not None:
            crs_str = crs.to_wkt()
        scale, offset = header.scales, header.offsets
        mins, maxs = header.mins, header.maxs

        first = True
        zmin = cnt = ch = cw = None
        xmin = ymax = None
        n_ground = n_total = 0
        for chunk in f.chunk_iterator(CHUNK):
            x = np.asarray(chunk.x, np.float64)
            y = np.asarray(chunk.y, np.float64)
            z = np.asarray(chunk.z, np.float64)
            n_total += len(x)
            if use_ground:
                cls = np.asarray(chunk.classification)
                m = (cls == 2) | (cls == 3) | (cls == 4)
                if not m.any():
                    continue                    # в чанке нет земли — не мусорим в DTM
                n_ground += int(m.sum())
                x, y, z = x[m], y[m], z[m]
            if first:
                xmin = float(mins[0])
                ymax = float(maxs[1])
                cw = max(1, int(math.ceil((maxs[0] - mins[0]) / res)))
                ch = max(1, int(math.ceil((maxs[1] - mins[1]) / res)))
                zmin = np.full(ch * cw, np.nan, np.float32)
                cnt = np.zeros(ch * cw, np.uint32)
                first = False
            col = ((x - xmin) / res).astype(np.int64)
            row = ((ymax - y) / res).astype(np.int64)
            ok = (col >= 0) & (col < cw) & (row >= 0) & (row < ch)
            if not ok.any():
                continue
            flat = (row[ok] * cw + col[ok])
            zf = z[ok].astype(np.float32)
            # минимум по ячейкам чанка: сортировка по ячейке + reduceat
            order = np.argsort(flat, kind='stable')
            fs, zs = flat[order], zf[order]
            uniq, start = np.unique(fs, return_index=True)
            zc = np.minimum.reduceat(zs, start)
            prev = zmin[uniq]
            zmin[uniq] = np.where(np.isnan(prev), zc, np.minimum(prev, zc))
            cnt[uniq] += 1
        if use_ground and n_total and n_ground / max(1, n_total) < 0.005:
            # классификации фактически нет — пересобрать по всем точкам
            print(f'  класс «земля» {n_ground}/{n_total} — переключаюсь на все точки')
            return grid_min_z(las_path, res, use_ground=False)
    return zmin, xmin, ymax, ch, cw, crs_str


def fill_nan_nearest(a: np.ndarray) -> np.ndarray:
    from scipy import ndimage
    if not np.isnan(a).any():
        return a
    idx = ndimage.distance_transform_edt(np.isnan(a), return_indices=True)[1]
    return a[tuple(idx)]


def hillshade_dtm(dtm: np.ndarray, res: float, azimuth=315.0, altitude=45.0):
    gy, gx = np.gradient(dtm, res)
    slope = np.arctan(np.hypot(gx, gy))
    aspect = np.arctan2(-gx, gy)          # рад, «откуда направлен скат»
    zen = math.radians(90 - altitude)
    az = math.radians(azimuth)
    hs = (np.cos(zen) * np.cos(slope)
          + np.sin(zen) * np.sin(slope) * np.cos(az - aspect - math.pi / 2))
    return np.clip(hs * 255, 0, 255).astype(np.uint8)


def slope_deg(dtm: np.ndarray, res: float) -> np.ndarray:
    gy, gx = np.gradient(dtm, res)
    return np.clip(np.degrees(np.arctan(np.hypot(gx, gy))) * 255 / 90,
                   0, 255).astype(np.uint8)


def derive(las_path: Path, out_dir: Path, res: float) -> dict:
    import rasterio
    from rasterio.crs import CRS as RioCRS
    from rasterio.transform import Affine

    use_ground = True
    zmin, xmin, ymax, ch, cw, crs_wkt = grid_min_z(las_path, res, use_ground)
    if zmin is None or np.isnan(zmin).all():
        raise RuntimeError(f'нет точек в {las_path}')
    filled = fill_nan_nearest(zmin.reshape(ch, cw))

    zone = None
    crs_obj = None
    if crs_wkt:
        crs_obj = RioCRS.from_wkt(crs_wkt)
    if (crs_obj is None or crs_obj.is_geographic) :
        zone = find_utm_zone(las_path)
        if zone is None:
            raise RuntimeError(f'не удалось определить CRS (нет WKT и UTM.json рядом с {las_path})')
        crs_obj = RioCRS.from_epsg(32600 + int(zone))
    transform = Affine(res, 0.0, xmin, 0.0, -res, ymax)

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = las_path.stem
    c_path = out_dir / f'{stem}_der_c.tif'
    g_path = out_dir / f'{stem}_der_g.tif'
    hs = hillshade_dtm(filled, res)
    sl = slope_deg(filled, res)
    profile = dict(driver='GTiff', height=ch, width=cw, dtype='uint8',
                   crs=crs_obj, transform=transform, tiled=True,
                   blockxsize=512, blockysize=512, compress='lzw')
    with rasterio.open(c_path, 'w', count=3, **profile) as dst:
        for b in range(3):
            dst.write(hs, b + 1)
    with rasterio.open(g_path, 'w', count=1, **profile) as dst:
        dst.write(sl, 1)

    rec = dict(path=str(c_path), kind='Li', crs=crs_obj.to_string(),
               w=cw, h=ch, res=res, aux_g=str(g_path), stem=f'{stem}_der_c',
               derived_from=str(las_path), site=las_path.parent.parent.name)
    return rec


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--las', default='', help='конкретный .las/.laz')
    ap.add_argument('--region', default='', help='каталог региона: найти все .las/.laz')
    ap.add_argument('--res', type=float, default=0.2, help='шаг сетки DTM, м')
    ap.add_argument('--outdir', default='', help='куда класть растры (по умолчанию work/derived/<место>)')
    ap.add_argument('--register', action='store_true',
                    help='дописать записи в work/rasters.json (для infer/val-прогонов)')
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

    files = []
    if args.las:
        files = [Path(args.las)]
    elif args.region:
        files = sorted(Path(args.region).rglob('*.la[sz]'))
    if not files:
        ap.error('укажите --las или --region с файлами .las/.laz')

    work = geoloc_paths.work_dir()
    records = []
    for f in files:
        out = Path(args.outdir) if args.outdir else work / 'derived' / f.parent.parent.name
        print(f'{f.name}: сетка {args.res} м ...', flush=True)
        rec = derive(f, out, args.res)
        records.append(rec)
        print(f"  -> {rec['path']} ({rec['w']}x{rec['h']}, CRS {rec['crs']})")

    if args.register:
        rp = work / 'rasters.json'
        idx = json.load(open(rp, encoding='utf-8')) if rp.exists() else {}
        for rec in records:
            idx[rec['stem']] = rec
        json.dump(idx, open(rp, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
        print(f'rasters.json: +{len(records)} записей')
    print(json.dumps(records, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
