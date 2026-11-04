# -*- coding: utf-8 -*-
"""Быстрая оценка чекпойнта на валидации (CPU, не мешает фоновому обучению).

Пример: python pipeline/eval_ckpt.py --model model1 --run _r50 --encoder resnet50
"""
import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))
from dataset import GeoTileDataset, MODELS  # noqa: E402
from train import collate, validate  # noqa: E402

import segmentation_models_pytorch as smp  # noqa: E402

RUNS = Path(r'G:\projects\geoloc\work\runs')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True, choices=list(MODELS))
    ap.add_argument('--run', default='', help='суффикс запуска, напр. _long; можно без суффикса')
    ap.add_argument('--encoder', default='')
    ap.add_argument('--wc-channel', action='store_true')
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    torch.set_num_threads(6)
    run_dir = RUNS / (args.model + args.run)
    ckpt = run_dir / 'best.pth'
    encoder = args.encoder
    wc = args.wc_channel
    if not encoder:   # берём архитектуру из отчёта прогона, если он есть
        try:
            r = json.load(open(run_dir / 'report.json', encoding='utf-8'))['args']
            encoder = r['encoder']
            wc = bool(r.get('wc_channel'))
        except Exception:
            encoder = 'resnet18'
    n_pos = len(MODELS[args.model]['classes'])
    in_ch = {'model1': 7 if wc else 6, 'model2': 4, 'model3': 8}[args.model]
    net = smp.Unet(encoder_name=encoder, encoder_weights=None,
                   in_channels=in_ch, classes=n_pos + 1)
    net.load_state_dict(torch.load(ckpt, map_location='cpu'))
    ds = GeoTileDataset(args.model, 'val', seed=7, augment=False, wc_channel=wc)
    ld = DataLoader(ds, batch_size=4, collate_fn=collate)
    print(f'{args.model}{args.run}: {len(ds)} тайлов val | {encoder}'
          f'{" +WC" if wc else ""} | чекпойнт {ckpt.name}')
    m = validate(net, ld, n_pos, 'cpu')
    print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()})


if __name__ == '__main__':
    main()
