# Дообучение: готовые команды (НЕ запущены — запускать вручную)

Обоснование каждого прогона — в `ANALIZ_I_RESHENIE.md` (история попыток §2,
причины провалов §5). Ключевой вывод истории: данные выучиваемы (оверфит-тест
32×400, loss 0.066); провал m3 — режим обучения (30 эпох, 0.5 м, без oversampling
и transfer), а не данные. Ниже — рецепт чемпиона m1, применённый к остальным.

Все команды — на виндовой тренировочной машине (DirectML), из корня `geoloc`.
Порядок = приоритет вклада в метрику.

## 1. model3 @ 0.25 м + oversampling + transfer (вес 30: 15+10+5)

Сначала пересобрать манифест m3 на 0.25 м (сплиты по местам сохраняются —
детерминизм seed 42; val-состав не меняется):

```bash
python pipeline/prepare.py --model3-res 0.25
```

Прогон с переносом энкодера из чемпиона m1 (без conv1 — входы разной ширины),
весами классов по обратной частоте (городища ~100, архитектуры ~149,
фортификации ~551 ≈ 1 : 0.7 : 0.2 → «5,1,3») и дублированием тайлов с городищами:

```bash
python pipeline/train.py --model model3 --epochs 150 --encoder resnet18 ^
  --init-encoder-from work/runs/model1_hr025_100/best.pth ^
  --pos-weight 5,1,3 --oversample "городища:4,архитектуры:2" --out-suffix _hr025
```

Суффиксы прогонов как в истории (`_hr025` = high-res 0.25). Сравнение с базой —
`python pipeline/eval_ckpt.py --model model3 --run _hr025` (строгий IoU-монитор)
и `python pipeline/platform_tune.py --model model3` (платформенная метрика —
главная, см. ANALIZ §3: локальный IoU-F1 систематически занижает).

## 2. model1 на обновлённых данных (вес 30)

Датасет досинхронизирован 30.10 (Гнёздово, чиненые LAS, разметка Сакских) —
перенатренировать чемпиона на свежих масках (CACHE_VER поднять, если prepare
переписал marks/):

```bash
python pipeline/train.py --model model1 --epochs 100 --encoder resnet18 ^
  --out-suffix _hr025_100b
```

(эквивалент чемпиона `model1_hr025_100`: r18, 0.25 м — манифест уже собран
с этим разрешением; порог теперь подбирает platform_tune, не threshold_sweep).

## 3. model4 геофизика (вес 20) — прогонов ещё не было

```bash
python pipeline/prepare_geo.py
python pipeline/train_geo.py --epochs 120 --pos-weight 3.0 ^
  --init-encoder-from work/runs/model1_hr025_100/best.pth
```

После обучения — включить в platform_tune (классы finds_points/object_poly,
GT из work/geo) и в пакет сабмита.

## 4. model2 на обновлённых данных (вес 20)

```bash
python pipeline/train.py --model model2 --epochs 16 --encoder resnet18 ^
  --out-suffix _v2
```

(базовая m2 = 16 эпох была рабочей; дольше — по времени после пунктов 1–3.)

## 5. После каждого обучения — тюнинг под платформу и сборка

```bash
python pipeline/platform_tune.py --model model1 --grid fine   # пишет work/infer_config.json
python pipeline/infer_all.py --input <публичная выборка> --output out_smoke --time-budget 1500
python pipeline/submit_package.py --out submit_v1 --test-dir <публичная выборка>
```

Смоук на публичной выборке локально = проверка тайминга (<25 мин на 4090) и
лимитов результата до траты попыток (10/сутки, 3 успешных).

## Контрольные точки перед сабмитом

- [ ] `work/infer_config.json` заполнен platform_tune для всех обученных моделей
- [ ] маски WorldCover для всех тест-регионов скачаны (`pipeline/worldcover.py`
      по рамкам регионов публичной выборки) — в контейнере сети нет
- [ ] `infer_all --dry-run` на публичной выборке: все 44 региона покрыты заданиями
- [ ] zip ≤ 2 ГБ, время смоука ≤ 25 мин
