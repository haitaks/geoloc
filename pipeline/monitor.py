# -*- coding: utf-8 -*-
"""Живой мониторинг обучения.

  python pipeline/monitor.py                 — сам найдёт самый свежий идущий прогон
  python pipeline/monitor.py model1_r50      — следить за конкретным прогоном
  python pipeline/monitor.py model1_r50 150  — указать всего эпох (иначе возьмёт из отчёта/150)
"""
import json
import sys
import time
from pathlib import Path

RUNS = Path(r'G:\projects\geoloc\work\runs')
REFRESH = 15


def pick_run():
    logs = sorted(RUNS.glob('*/log.csv'), key=lambda p: p.stat().st_mtime, reverse=True)
    if not logs:
        sys.exit(f'Не найдено ни одного log.csv в {RUNS}')
    return logs[0].parent.name, logs[0]


def read_csv(path):
    try:
        lines = [l.strip() for l in open(path, encoding='utf-8') if l.strip()]
        if len(lines) < 2:
            return None
        header, row = lines[0].split(';'), lines[-1].split(';')
        if len(row) < 2 or not row[0].isdigit():
            return None
        return header, row
    except OSError:
        return None


def total_epochs(run_dir, arg_total):
    if arg_total:
        return arg_total
    try:
        return int(json.load(open(run_dir / 'report.json', encoding='utf-8'))['args']['epochs'])
    except Exception:
        return 150


def main():
    arg_run = sys.argv[1] if len(sys.argv) > 1 else ''
    arg_total = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    if arg_run:
        run_dir = RUNS / arg_run
        path = run_dir / 'log.csv'
        if not path.exists():
            sys.exit(f'Нет {path}. Доступны: {[p.name for p in RUNS.iterdir() if p.is_dir()]}')
    else:
        arg_run, path = pick_run()
        run_dir = RUNS / arg_run
    total = total_epochs(run_dir, arg_total)
    print(f'Прогон: {arg_run} | всего эпох: {total} | источник: {path}', flush=True)
    print(f'Обновление каждые {REFRESH} с, выход — Ctrl+C\n', flush=True)

    ep0 = t0 = None
    last_printed = None
    finished = False
    while not finished:
        data = read_csv(path)
        if data:
            header, row = data
            ep, loss = int(row[0]), row[1]
            extra = ' '.join(f'{k}={v}' for k, v in zip(header, row)
                             if k not in ('epoch', 'loss', 'sec') and v)
            now = time.time()
            if ep0 is None:
                ep0, t0 = ep, now
                msg = f'[старт наблюдения] эпоха {ep}/{total} | loss={loss}'
            elif ep == ep0:
                msg = f'эпоха {ep}/{total} | loss={loss} | скорость: измеряю...'
            else:
                speed = (ep - ep0) / (now - t0) * 3600            # эпох/час
                eta_h = (total - ep) / speed if speed > 0 else 0
                eta = f'{eta_h:.1f} ч' if eta_h >= 1 else f'{eta_h * 60:.0f} мин'
                msg = (f'эпоха {ep}/{total} | loss={loss} | {speed:.0f} эпох/ч '
                       f'(~{3600 / max(speed, 1e-9) / 60:.1f} мин/эпоху) | осталось ~{eta}')
            if extra:
                msg += f' | {extra}'
            if msg != last_printed:
                stamp = time.strftime('%H:%M:%S')
                print(f'[{stamp}] {msg}', flush=True)
                last_printed = msg
            finished = (run_dir / 'report.json').exists() and ep >= total
            if finished:
                print('\nОбучение завершено — итог в report.json этого прогона.')
        time.sleep(REFRESH)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nвыход')
