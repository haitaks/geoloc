# -*- coding: utf-8 -*-
"""Подбор порога инференса: вместо argmax — P(класс) >= t. Один проход по val,
метрики для всех порогов сразу. Работает на CPU, обучению не мешает.

  python pipeline/threshold_sweep.py --model model1 --run _long
  python pipeline/threshold_sweep.py --model model1 --run _r50 --encoder resnet50
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))
from dataset import GeoTileDataset, MODELS  # noqa: E402
from train import collate  # noqa: E402

import segmentation_models_pytorch as smp  # noqa: E402

RUNS = Path(r'G:\projects\geoloc\work\runs')
THRESHOLDS = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8)


def object_metrics(pred, gt):
    """TP/FP/FN по компонентам связности, матчинг IoU>=0.25 (как в train.validate)."""
    tp = fp = fn = 0
    for b in range(pred.shape[0]):
        pc, _ = ndimage.label(pred[b])
        gc, _ = ndimage.label(gt[b])
        pids = list(range(1, pc.max() + 1))
        used = set()
        for g in range(1, gc.max() + 1):
            gm = gc == g
            best, bi = 0.25, None
            for p in pids:
                if p in used:
                    continue
                pm = pc == p
                iou = (gm & pm).sum() / max(1, (gm | pm).sum())
                if iou > best:
                    best, bi = iou, p
            if bi is not None:
                used.add(bi)
                tp += 1
            else:
                fn += 1
        fp += len(pids) - len(used)
    return tp, fp, fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True, choices=list(MODELS))
    ap.add_argument('--run', default='')
    ap.add_argument('--encoder', default='')
    ap.add_argument('--wc-channel', action='store_true')
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    torch.set_num_threads(6)
    run_dir = RUNS / (args.model + args.run)
    ckpt = run_dir / 'best.pth'
    encoder, wc = args.encoder, args.wc_channel
    if not encoder:
        try:
            r = json.load(open(run_dir / 'report.json', encoding='utf-8'))['args']
            encoder, wc = r['encoder'], bool(r.get('wc_channel'))
        except Exception:
            encoder = 'resnet18'

    n_pos = len(MODELS[args.model]['classes'])
    in_ch = {'model1': 7 if wc else 6, 'model2': 4, 'model3': 8}[args.model]
    net = smp.Unet(encoder_name=encoder, encoder_weights=None,
                   in_channels=in_ch, classes=n_pos + 1)
    net.load_state_dict(torch.load(ckpt, map_location='cpu'))
    net.eval()
    ds = GeoTileDataset(args.model, 'val', seed=7, augment=False, wc_channel=wc)
    ld = DataLoader(ds, batch_size=4, collate_fn=collate)
    print(f'{args.model}{args.run} | {encoder}{" +WC" if wc else ""} | {len(ds)} тайлов val')

    probs_all, ys = [], []
    with torch.no_grad():
        for x, y in ld:
            p = torch.softmax(net(x), 1)
            probs_all.append(p[:, 1].numpy())   # вероятность позитивного класса
            ys.append(y.numpy())
    probs = np.concatenate(probs_all)
    ys = np.concatenate(ys)

    print(f'{"порог":>6} {"mIoU":>7} {"точн.":>7} {"полн.":>7} {"F1":>7} {"FP-фон%":>8}')
    best_f1, best_t = -1, None
    for t in THRESHOLDS:
        pred = probs >= t
        valid = ys != 255
        inter = (pred & (ys == 1) & valid).sum()
        union = ((pred | (ys == 1)) & valid).sum()
        iou = inter / union if union else float('nan')
        tp, fp, fn = object_metrics(pred, (ys == 1) & valid)
        p = tp / max(1, tp + fp)
        r = tp / max(1, tp + fn)
        f1 = 2 * p * r / max(1e-9, p + r)
        fp_rate = (pred & (ys == 0) & valid).sum() / max(1, valid.sum())
        print(f'{t:>6.2f} {iou:>7.3f} {p*100:>6.1f}% {r*100:>6.1f}% {f1:>7.3f} {fp_rate*100:>7.2f}%')
        if f1 > best_f1:
            best_f1, best_t = f1, t
    print(f'\nЛучший F1 {best_f1:.3f} при пороге {best_t}')


if __name__ == '__main__':
    main()
