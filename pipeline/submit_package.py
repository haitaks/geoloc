# -*- coding: utf-8 -*-
"""Этап A3 плана §8: сборка пакета сабмита (ZIP ≤ 2 ГБ + Docker-образ).

Что кладёт в <out>/:
  solution.py          — обёртка: python solution.py <input_dir> <output_dir>
  src/                 — infer_all/predict/la_derive/geoloc_paths
  weights/<model>.pth  — чекпойнты из work/infer_config.json (или work/runs)
  infer_config.json    — тюнинг постобработки (platform_tune.py) + чекпойнты
  wcmask/              — маски леса WorldCover для тест-регионов (готовятся заранее,
                         в контейнере сети нет)
  Dockerfile, metadata.json, и zip-архив рядом с <out>

  python pipeline/submit_package.py --out submit_v1 [--models model1,model2,model3]
                                    [--test-dir <публичная выборка>]
"""
import argparse
import json
import shutil
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import geoloc_paths  # noqa: E402
from infer_all import merge_config  # noqa: E402

SRC_FILES = ('infer_all.py', 'predict.py', 'la_derive.py', 'geoloc_paths.py')

SOLUTION_PY = '''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Точка входа сабмита: python solution.py <input_dir> <output_dir>.

См. pipeline/infer_all.py — универсальный обход регионов, все модальности,
бюджет времени, лимиты результата. Пути внутри пакета, никакой сети.
"""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
os.environ.setdefault('GEOLOC_WORK', str(HERE))          # веса/конфиг/кэш — в пакете
os.environ.setdefault('GEOLOC_RUNS', str(HERE / 'weights'))
sys.path.insert(0, str(HERE / 'src'))

from infer_all import main                               # noqa: E402

if __name__ == '__main__':
    inp = sys.argv[1] if len(sys.argv) > 1 else '/app/input'
    out = sys.argv[2] if len(sys.argv) > 2 else '/app/output'
    sys.argv = ['infer_all', '--input', inp, '--output', out,
                '--config', str(HERE / 'infer_config.json')]
    main()
'''

DOCKERFILE = '''# Образ сабмита КОЗ-3 (платформа: RTX 4090, 30 мин, лимит образа 50 ГБ)
FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn8-runtime
RUN pip install --no-cache-dir rasterio shapely pyproj scipy \\
    "laspy[lazrs]" segmentation-models-pytorch
WORKDIR /app
COPY . /app
ENTRYPOINT ["python", "solution.py"]
'''


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', default='submit_v1', help='каталог пакета')
    ap.add_argument('--models', default='model1,model2,model3',
                    help='какие модели включать (m4 — после обучения)')
    ap.add_argument('--image', default='geoloc-koz3:latest',
                    help='имя Docker-образа для metadata.json')
    ap.add_argument('--test-dir', default='',
                    help='каталог публичной выборки: для её регионов взять маски WC')
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

    work = geoloc_paths.work_dir()
    cfg = merge_config(geoloc_paths.infer_config_path())
    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    (out / 'src').mkdir(parents=True)
    (out / 'weights').mkdir()

    # 1. код
    here = Path(__file__).parent
    for f in SRC_FILES:
        shutil.copy2(here / f, out / 'src' / f)
    (out / 'solution.py').write_text(SOLUTION_PY, encoding='utf-8')
    (out / 'Dockerfile').write_text(DOCKERFILE, encoding='utf-8')

    # 2. веса + конфиг (чекпойнты — пути из конфига, имена в пакете короткие)
    models = [m for m in args.models.split(',') if m]
    cfg_pkg = json.loads(json.dumps(cfg))
    cfg_pkg.setdefault('checkpoints', {})
    total = 0
    for m in models:
        ck = cfg['checkpoints'].get(m) or str(geoloc_paths.runs_dir() / m / 'best.pth')
        if not Path(ck).exists():
            print(f'  ! {m}: нет {ck} — модель НЕ включена')
            cfg_pkg['checkpoints'].pop(m, None)
            continue
        dst = out / 'weights' / f'{m}.pth'
        shutil.copy2(ck, dst)
        total += dst.stat().st_size
        # путь ОТНОСИТЕЛЬНО пакета: абсолютный путь машины сборки в контейнере невалиден
        cfg_pkg['checkpoints'][m] = str(dst.relative_to(out))
        print(f'  {m}: {Path(ck).name} ({dst.stat().st_size / 1e6:.0f} МБ)')
    json.dump(cfg_pkg, open(out / 'infer_config.json', 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)

    # 3. маски леса WorldCover: для известных тест-регионов (если подготовлены)
    wc_src = work / 'wcmask'
    if wc_src.is_dir():
        names = []
        if args.test_dir:
            names = [p.name for p in Path(args.test_dir).iterdir() if p.is_dir()]
        (out / 'wcmask').mkdir(exist_ok=True)
        for f in sorted(wc_src.glob('*.tif')):
            if not names or f.stem in names:
                shutil.copy2(f, out / 'wcmask' / f.name)
        print(f'  wcmask: {len(list((out / "wcmask").glob("*.tif")))} масок')

    # 4. metadata.json
    json.dump({'image': args.image, 'entry_point': 'python solution.py'},
              open(out / 'metadata.json', 'w', encoding='utf-8'), indent=2)

    # 5. zip ≤ 2 ГБ
    zp = out.with_suffix('.zip')
    with zipfile.ZipFile(zp, 'w', zipfile.ZIP_DEFLATED) as z:
        for f in sorted(out.rglob('*')):
            if f.is_file():
                z.write(f, f.relative_to(out.parent))
    zsize = zp.stat().st_size
    print(f'\nПакет: {out}/ ({total / 1e6:.0f} МБ весов); архив {zp.name} '
          f'{zsize / 1e6:.0f} МБ')
    if zsize > 2 * 1024 ** 3:
        print('! архив больше 2 ГБ — уменьшите набор весов (fp16) или моделей')
    print('Дальше: docker build -t ' + args.image + ' ' + str(out)
          + ' && загрузить образ в регистри; на платформу — сам zip')


if __name__ == '__main__':
    main()
