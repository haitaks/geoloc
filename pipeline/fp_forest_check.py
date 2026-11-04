# -*- coding: utf-8 -*-
"""Где концентрируются ложные срабатывания модели 1: в лесу или в поле?

Берём валидационные НЕГАТИВНЫЕ тайлы (без объектов), считаем долю пикселей,
предсказанных позитивом, и сохраняем превью худших/лучших для визуальной оценки.
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from dataset import GeoTileDataset, WORK, MODELS  # noqa: E402

import segmentation_models_pytorch as smp  # noqa: E402

try:
    import torch_directml as dml
    device = dml.device() if dml.device_count() else 'cpu'
except ImportError:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
OUT = WORK / 'fp_check'
OUT.mkdir(exist_ok=True)

model = smp.Unet(encoder_name='resnet18', encoder_weights=None, in_channels=6, classes=2)
model.load_state_dict(torch.load(WORK / 'runs/model1/best.pth', map_location='cpu'))
model.to(device).eval()

ds = GeoTileDataset('model1', 'val', augment=False)
neg_rows = [i for i, r in enumerate(ds.rows) if int(r['pos']) == 0]
print(f'негативных тайлов в val: {len(neg_rows)}')
scores = []
with torch.no_grad():
    for i in neg_rows:
        x, m = ds[i]
        xt = torch.from_numpy(x[None]).to(device)
        pred = model(xt).argmax(1)[0].cpu().numpy()
        frac = float((pred == 1).mean())
        scores.append((frac, i))
scores.sort(reverse=True)

for rank, (frac, i) in enumerate(scores[:10] + scores[-3:]):
    x, m = ds[i]
    img = (np.moveaxis(x[:3], 0, -1) * 255).astype(np.uint8)
    ov = img.copy()
    ov[x[3] > 0.5] = (ov[x[3] > 0.5] * 0.5 + np.array([120, 255, 120]) * 0.5).astype(np.uint8)
    red = pred_cache = None
    xt = torch.from_numpy(x[None]).to(device)
    pred = model(xt).argmax(1)[0].cpu().numpy()
    ov[pred == 1] = (255, 60, 60)
    tag = 'top' if rank < 10 else 'low'
    Image.fromarray(ov).save(OUT / f'{tag}_{rank}_fp{frac:.3f}_{ds.rows[i]["site"]}.png')
    print(f'{tag} {rank}: fp={frac:.3f} {ds.rows[i]["site"]} {ds.rows[i]["kind"]}')
print('Превью:', OUT)
