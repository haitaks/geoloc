# geoloc

Экспедиция Data Science, КОЗ №3 — геолокация по спутниковым данным.

## Структура

- `pipeline/` — основной пайплайн: подготовка данных, обучение (`train*.py`), инференс, оценка
  - `infer_all.py` — **универсальный инференс тест-каталога → result.geojson** (все модальности,
    лимиты платформы, бюджет 30 мин; пути через env `GEOLOC_WORK`/`GEOLOC_DATA`)
  - `predict.py` — общий движок инференса (TTA, усреднение окон, AMP CUDA)
  - `la_derive.py` — La→DTM/хиллшейд/уклоны (модели 2/3 на La-регионах теста)
  - `platform_tune.py` — свип порогов/буферов под реальную метрику платформы → `work/infer_config.json`
  - `submit_package.py` — сборка пакета сабмита (solution.py + веса + Dockerfile + zip)
- `work/splits/` — сплиты по тайлам для трёх моделей
- `baseline_solution/`, `simple_solution/` — решения для платформы соревнования
- `Скрипт для датасета S3/` — скрипт синхронизации датасета из S3
- Корневые `*.py` — исследовательские скрипты и анализ данных
- `compute_metrics.py` — метрики платформы
- Документация: `ANALIZ_I_RESHENIE.md` (анализ и решение), `TRAINING_NEXT.md` (команды
  дообучения), `План_трёх_моделей.md`, `baseline_readme.md`, `data_structure.md`, PDF

Датасеты, кэши растров и чекпойнты (`work/runs/` и т.п.) в репозиторий не входят — см. `.gitignore`.

## Как запушить на GitHub

```bash
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/haitaks/geoloc.git
git push -u origin main
```
