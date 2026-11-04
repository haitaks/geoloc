# -*- coding: utf-8 -*-
"""Одноразовая доработка: пирамиды (overviews) во все растры датасета.

ВНИМАНИЕ: пирамиды ВНУТРЕННИЕ (in-place) — файлы растут на ~30%, и следующая
синхронизация с S3 посчитает их «изменёнными» и перекачает (~100 ГБ).
Правило: сначала s3_sync (update), потом этот скрипт.
(Внешние .ovr через rasterio невозможны: build_overviews требует 'r+', а GTiff
пишет .ovr только для датасетов, открытых read-only.)

Повторный запуск безопасен: готовые пирамиды пропускаются.
"""
import json
import sys
import time
from pathlib import Path

import rasterio

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
WORK = Path(r'G:\projects\geoloc\work')
LEVELS = [2, 4, 8, 16]


def main():
    rasters = json.load(open(WORK / 'rasters.json', encoding='utf-8'))
    todo = [r for stem, r in sorted(rasters.items())
            if r['kind'] != 'Or' and not Path(r['path']).name.startswith('._')]
    t0 = time.time()
    done = skipped = failed = 0
    for i, ras in enumerate(todo, 1):
        if max(ras['w'], ras['h']) < 2048:   # мелким растрам пирамиды не нужны
            skipped += 1
            continue
        try:
            with rasterio.open(ras['path'], 'r+') as src:
                if src.overviews(1):
                    skipped += 1
                    continue
                src.build_overviews(LEVELS, resampling=rasterio.enums.Resampling.average)
                done += 1
        except Exception as e:
            failed += 1
            print(f'  ! {ras["path"]}: {e}', flush=True)
        if i % 40 == 0:
            print(f'  {i}/{len(todo)} ({time.time()-t0:.0f} с)', flush=True)
    print(f'Пирамиды: построено {done}, пропущено {skipped}, ошибок {failed}, '
          f'время {time.time()-t0:.0f} с')


if __name__ == '__main__':
    main()
