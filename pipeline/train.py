# -*- coding: utf-8 -*-
"""Обучение сегментационной модели (SMP U-Net) на тайлах train-expds-3.

Пример:
  python pipeline/train.py --model model2 --epochs 8 --encoder resnet18
"""
import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))
from dataset import GeoTileDataset, MODELS  # noqa: E402

import segmentation_models_pytorch as smp  # noqa: E402

WORK = Path(r'G:\projects\geoloc\work')

try:
    import torch_directml as dml
    DML = dml.device_count() > 0
except ImportError:
    DML = False


def pick_device():
    if DML:
        return dml.device()
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def collate(batch):
    xs = torch.from_numpy(np.stack([b[0] for b in batch]))
    ys = torch.from_numpy(np.stack([b[1] for b in batch]).astype(np.int64))
    return xs, ys


class LossCE_Dice(nn.Module):
    def __init__(self, n_classes, pos_weight=1.0):
        """pos_weight: скаляр или список весов по классам «1,3,2»
        (городища ~100 уникальных против 551 фортификации — план §4.2)."""
        super().__init__()
        if isinstance(pos_weight, str) and ',' in pos_weight:
            per_class = [float(v) for v in pos_weight.split(',')]
            assert len(per_class) == n_classes, \
                f'pos_weight: {len(per_class)} весов, классов {n_classes}'
        else:
            per_class = [float(pos_weight)] * n_classes
        w = torch.ones(n_classes + 1)
        w[1:] = torch.tensor(per_class)   # код 0 — фон, 1..n — классы
        self.register_buffer('w', w)
        self.n = n_classes           # позитивные классы (коды 1..n)

    def forward(self, logits, target):
        valid = target != 255
        n_valid = valid.sum()
        if n_valid > 0:
            per = F.cross_entropy(logits, target, reduction='none', ignore_index=255,
                                  weight=self.w)
            ce = per.sum() / n_valid
        else:
            ce = logits.sum() * 0.0
        probs = torch.softmax(logits, 1)
        validf = valid.to(probs.dtype)      # bool-тензор в произведении роняет автоград DirectML
        dice = 0.0
        for i in range(1, self.n + 1):
            p = probs[:, i] * validf
            t = (target == i).float()
            denom = p.sum(dim=(1, 2)) + t.sum(dim=(1, 2))
            num = (p * t).sum(dim=(1, 2))
            dice += (1 - ((2 * num + 1) / (denom + 1))).mean()
        return ce + dice / self.n


@torch.no_grad()
def validate(model, loader, n_classes, device):
    model.eval()
    inter = np.zeros(n_classes + 1)
    union = np.zeros(n_classes + 1)
    tp = fp = fn = 0
    neg_px = neg_tot = 0
    from scipy import ndimage
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(1)
        valid = y != 255
        for i in range(1, n_classes + 1):
            pi = (pred == i) & valid
            gi = (y == i) & valid
            inter[i] += (pi & gi).sum().item()
            union[i] += (pi | gi).sum().item()
            # object-level: компоненты связности, матчинг по IoU>=0.25
            for b in range(x.shape[0]):
                pc, _ = ndimage.label((pred[b] == i).cpu().numpy())
                gc, _ = ndimage.label(gi[b].cpu().numpy())
                pids = [p for p in range(1, pc.max() + 1)]
                gids = [g for g in range(1, gc.max() + 1)]
                used = set()
                for g in gids:
                    gmask = gc == g
                    best, bi = 0.25, None
                    for p in pids:
                        if p in used:
                            continue
                        pmask = pc == p
                        iou = (gmask & pmask).sum() / max(1, (gmask | pmask).sum())
                        if iou > best:
                            best, bi = iou, p
                    if bi is not None:
                        used.add(bi)
                        tp += 1
                    else:
                        fn += 1
                fp += len(pids) - len(used)
        neg_px += ((pred > 0) & (y == 0) & valid).sum().item()
        neg_tot += valid.sum().item()
    ious = {f'IoU_{i}': (inter[i] / union[i] if union[i] else float('nan'))
            for i in range(1, n_classes + 1)}
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    return dict(ious, objP=prec, objR=rec,
                objF1=2 * prec * rec / max(1e-9, prec + rec),
                fp_rate=neg_px / max(1, neg_tot))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True, choices=list(MODELS))
    ap.add_argument('--epochs', type=int, default=8)
    ap.add_argument('--batch', type=int, default=4)
    ap.add_argument('--encoder', default='resnet18')
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--neg-per-pos', type=float, default=2.0)
    ap.add_argument('--pos-weight', default='1.0',
                    help='скаляр или веса по классам через запятую: «1,3,2»')
    ap.add_argument('--oversample', default='',
                    help='дублировать позитивные тайлы редких классов: «городища:3,архитектуры:2»')
    ap.add_argument('--max-tiles', type=int, default=0)
    ap.add_argument('--out-suffix', default='')
    ap.add_argument('--val-every', type=int, default=1,
                    help='Валидировать каждые N эпох (0 — только в конце)')
    ap.add_argument('--init-encoder-from', default='',
                    help='Путь к чекпойнту другой модели: взять её encoder (без первой свёртки)')
    ap.add_argument('--wc-channel', action='store_true',
                    help='(model1) добавить канал позитивного приора WorldCover (вход 7 кан.)')
    ap.add_argument('--workers', type=int, default=0)
    ap.add_argument('--threads', type=int, default=10)
    ap.add_argument('--seed', type=int, default=7)
    out_dir = WORK / 'runs' / (ap.parse_known_args()[0].model + ap.parse_known_args()[0].out_suffix)
    ap.parse_args()
    args = ap.parse_args()
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)

    n_pos = len(MODELS[args.model]['classes'])
    in_ch = {'model1': 7 if args.wc_channel else 6, 'model2': 4, 'model3': 8}[args.model]
    oversample = {}
    for part in filter(None, args.oversample.split(',')):
        cls, _, k = part.partition(':')
        oversample[cls.strip()] = max(1, int(k or 2))
    train_ds = GeoTileDataset(args.model, 'train', neg_per_pos=args.neg_per_pos,
                              seed=args.seed, augment=True, max_tiles=args.max_tiles,
                              wc_channel=args.wc_channel, oversample=oversample)
    val_ds = GeoTileDataset(args.model, 'val', seed=args.seed, augment=False,
                            wc_channel=args.wc_channel)
    print(f'train: {len(train_ds)} тайлов ({train_ds.n_pos} с объектами); '
          f'val: {len(val_ds)} тайлов ({val_ds.n_pos} с объектами)')
    train_ld = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                          num_workers=args.workers, collate_fn=collate,
                          persistent_workers=args.workers > 0)
    val_ld = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                        num_workers=max(0, args.workers), collate_fn=collate)

    model = smp.Unet(encoder_name=args.encoder, encoder_weights='imagenet',
                     in_channels=in_ch, classes=n_pos + 1)
    if args.init_encoder_from:
        sd = torch.load(args.init_encoder_from, map_location='cpu')
        enc = {k[len('encoder.'):]: v for k, v in sd.items() if k.startswith('encoder.')}
        enc = {k: v for k, v in enc.items() if not k.startswith('conv1.')}  # входы разной ширины
        model.encoder.load_state_dict(enc, strict=False)
        print(f'encoder инициализирован из {args.init_encoder_from} ({len(enc)} тензоров)')
    device = pick_device()
    print('Устройство:', dml.device_name(0) if DML else str(device))
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    crit = LossCE_Dice(n_pos, args.pos_weight).to(device)

    log_rows = []
    keys = None

    def flush_log():
        nonlocal keys
        if not log_rows:
            return
        if keys is None:
            keys = list(log_rows[-1].keys())
        for r in log_rows:
            for k in keys:
                r.setdefault(k, '')
        with open(out_dir / 'log.csv', 'w', newline='', encoding='utf-8') as fh:
            fh.write(';'.join(keys) + '\n')
            for r in log_rows:
                fh.write(';'.join(str(r[k]) for k in keys) + '\n')

    best = -1
    for ep in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        tot = nb = 0
        for x, y in train_ld:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward()
            opt.step()
            tot += loss.item()
            nb += 1
        sched.step()
        tr_loss = tot / max(1, nb)
        do_val = (ep == args.epochs) or (args.val_every > 0 and ep % args.val_every == 0)
        if not do_val:
            row = dict(epoch=ep, loss=round(tr_loss, 4))
            log_rows.append(row)
            print(row, flush=True)
            flush_log()
            continue
        m = validate(model, val_ld, n_pos, device)
        miou = np.nanmean(list(m.values())[:n_pos])
        row = dict(epoch=ep, loss=round(tr_loss, 4), mIoU=round(float(miou), 4),
                   **{k: (round(v, 4) if v == v else 'nan') for k, v in m.items()},
                   sec=round(time.time() - t0))
        log_rows.append(row)
        print(row, flush=True)
        flush_log()
        if miou > best:
            best = miou
            torch.save(model.state_dict(), out_dir / 'best.pth')
    json.dump(dict(args=vars(args), best_mIoU=float(best), log=log_rows),
              open(out_dir / 'report.json', 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1, default=str)
    print(f'Лучший mIoU: {best:.4f}; артефакты в {out_dir}')


if __name__ == '__main__':
    main()
