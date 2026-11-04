# -*- coding: utf-8 -*-
"""Обучение модели 4 (геофизика): finds_points + object_poly.

  python pipeline/train_geo.py --epochs 120 --init-encoder-from work/runs/model1_hr025_100/best.pth
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))
from dataset_geo import GeoTileDataset, CLASSES  # noqa: E402
from train import LossCE_Dice, collate, pick_device, validate  # noqa: E402

try:
    import torch_directml as dml
    DML = dml.device_count() > 0
except ImportError:
    DML = False

import segmentation_models_pytorch as smp  # noqa: E402

WORK = Path(r'G:\projects\geoloc\work')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=120)
    ap.add_argument('--batch', type=int, default=4)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--pos-weight', type=float, default=3.0)
    ap.add_argument('--encoder', default='resnet18')
    ap.add_argument('--init-encoder-from', default='')
    ap.add_argument('--val-every', type=int, default=5)
    ap.add_argument('--threads', type=int, default=4)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    torch.set_num_threads(args.threads)
    torch.manual_seed(7)
    out_dir = WORK / 'runs' / 'model4'
    out_dir.mkdir(parents=True, exist_ok=True)

    tr = GeoTileDataset('train')
    vl = GeoTileDataset('val', augment=False)
    print(f'train: {len(tr)} ({tr.n_pos} с объектами) | val: {len(vl)} ({vl.n_pos})')
    tr_ld = DataLoader(tr, batch_size=args.batch, shuffle=True, collate_fn=collate)
    vl_ld = DataLoader(vl, batch_size=args.batch, shuffle=False, collate_fn=collate)

    net = smp.Unet(encoder_name=args.encoder, encoder_weights='imagenet',
                   in_channels=7, classes=len(CLASSES) + 1)
    if args.init_encoder_from:
        sd = torch.load(args.init_encoder_from, map_location='cpu')
        enc = {k[len('encoder.'):]: v for k, v in sd.items() if k.startswith('encoder.')}
        enc = {k: v for k, v in enc.items() if not k.startswith('conv1.')}
        net.encoder.load_state_dict(enc, strict=False)
        print(f'encoder из {args.init_encoder_from}: {len(enc)} тензоров')
    device = pick_device()
    print('Устройство:', dml.device_name(0) if DML else str(device))
    net.to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    crit = LossCE_Dice(len(CLASSES), args.pos_weight).to(device)

    log = []
    best = -1
    for ep in range(1, args.epochs + 1):
        net.train()
        t0 = time.time()
        tot = nb = 0
        for x, y in tr_ld:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = crit(net(x), y)
            loss.backward()
            opt.step()
            tot += loss.item()
            nb += 1
        sched.step()
        tr_loss = tot / max(1, nb)
        row = {'epoch': ep, 'loss': round(tr_loss, 4)}
        if ep % args.val_every == 0 or ep == args.epochs:
            m = validate(net, vl_ld, len(CLASSES), device)
            miou = float(np.nanmean([m[f'IoU_{i}'] for i in range(1, len(CLASSES) + 1)]))
            row.update({k: round(v, 4) for k, v in m.items()})
            print(row, flush=True)
            if miou > best:
                best = miou
                torch.save(net.state_dict(), out_dir / 'best.pth')
        else:
            print(row, flush=True)
        log.append(row)
        with open(out_dir / 'log.csv', 'w', newline='', encoding='utf-8') as fh:
            w = csv.DictWriter(fh, fieldnames=list(log[-1].keys()), restval='')
            w.writeheader()
            w.writerows(log)
    json.dump(dict(args=vars(args), best_mIoU=best, log=log),
              open(out_dir / 'report.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print(f'Лучший mIoU: {best:.4f} -> {out_dir}')


if __name__ == '__main__':
    main()
