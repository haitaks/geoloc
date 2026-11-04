# -*- coding: utf-8 -*-
"""Переносимые пути проекта: Windows-машина обучения и Linux-контейнер сабмита.

Приоритет: переменные окружения GEOLOC_WORK / GEOLOC_DATA → автопоиск.
Используется новыми скриптами (predict/infer_all/la_derive/platform_tune);
старые скрипты со своими G:-путями не трогаем.

  GEOLOC_WORK — рабочий каталог (work/): rasters.json, splits/, runs/, infer_config.json
  GEOLOC_DATA — корень датасета (train-expds-3) или каталог тест-регионов
"""
import os
from pathlib import Path

# G:\projects\geoloc на виндовой машине; здесь — родитель каталога pipeline/
_REPO = Path(__file__).resolve().parent.parent
_WIN_WORK = Path(r'G:\projects\geoloc\work')
_WIN_DATA = Path(r'G:\projects\train-expds-3')


def work_dir() -> Path:
    v = os.environ.get('GEOLOC_WORK')
    if v:
        return Path(v)
    if _WIN_WORK.is_dir():          # виндовая тренировочная машина
        return _WIN_WORK
    return _REPO / 'work'


def data_root() -> Path:
    v = os.environ.get('GEOLOC_DATA')
    if v:
        return Path(v)
    if _WIN_DATA.is_dir():
        return _WIN_DATA
    return _REPO / 'train-expds-3'


def runs_dir() -> Path:
    v = os.environ.get('GEOLOC_RUNS')
    return Path(v) if v else work_dir() / 'runs'


def infer_config_path() -> Path:
    """Пути чекпойнтов + тюнинг постобработки (пишется platform_tune.py)."""
    v = os.environ.get('GEOLOC_INFER_CONFIG')
    return Path(v) if v else work_dir() / 'infer_config.json'
