#!/usr/bin/env python3
"""
Решение для конкурса по археологической разметке.

Этот скрипт создает объединенную разметку археологических объектов
на основе структуры датасета.

Режимы работы:
1. ГЕНЕРАЦИЯ (по умолчанию):
   - Сканирует структуру датасета (папки с изображениями, данными LiDAR и т.д.)
   - Извлекает метаданные из путей к файлам (region, markup_type)
   - Генерирует случайные полигоны с правильными метаданными
   - Основной сценарий для папки valid (без файлов .geojson разметки)

2. ИЗ ФАЙЛОВ РАЗМЕТКИ (--use-gt):
   - Читает реальные GeoJSON файлы из датасета
   - Извлекает реальные археологические объекты
   - Разбивает MultiPolygon на отдельные Polygon
   - Удаляет объекты с пустыми координатами
   - Используется для тестирования на датасетах с разметкой
"""

import os
import sys
import json
import math
import random
import re
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def _get_pixel_resolution(region_dir: Path, markup_type: str) -> Optional[float]:
    """Читает pixel_resolution из GeoTIFF для геомагнитной разведки.

    Возвращает значение только для markup_type='maggr'. Для остальных — None.
    """
    if markup_type != "maggr":
        return None
    if not region_dir.is_dir():
        return None
    from fnmatch import fnmatch
    for subdir in sorted(region_dir.iterdir()):
        if subdir.is_dir() and fnmatch(subdir.name, "*MagGr*"):
            for tif in sorted(subdir.glob("*.tif")):
                try:
                    import rasterio
                    with rasterio.open(tif) as src:
                        return abs(src.transform.a)
                except Exception:
                    return None
    return None


# Фиксируем сид для воспроизводимости (для режима --random)
SEED = 42
random.seed(SEED)

# Классы и веса метрики — из общего конфига платформы (config.py), чтобы скрипт
# у команд считал ровно те же классы, что сервер. Fallback — встроенная копия
# (если скрипт запускают отдельно от репозитория).
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from config import (ALL_VALID_CLASSES, CLASS_NAME_MAPPING, CLASS_WEIGHTS,
                        normalize_class_name)
    PLATFORM_CONFIG_LOADED = True
except ImportError:  # pragma: no cover — одиночная сборка скрипта
    PLATFORM_CONFIG_LOADED = False
    CLASS_WEIGHTS = {
        "kurgany_povrezhdennye": 30.0, "kurgany_tselye": 20.0,
        "gorodishcha": 15.0, "arkhitektury": 10.0, "object_poly": 10.0,
        "fortifikatsii": 5.0, "finds_points": 10.0,
    }
    ALL_VALID_CLASSES = list(CLASS_WEIGHTS.keys())
    CLASS_NAME_MAPPING = {
        "курганы_целые": "kurgany_tselye",
        "курганы_поврежденные": "kurgany_povrezhdennye",
        "курганы_разрушенные": "kurgany_povrezhdennye",
        "городища": "gorodishcha",
        "архитектуры": "arkhitektury",
        "фортификации": "fortifikatsii",
        "FindsPoints": "finds_points",
        "findspoints": "finds_points",
        "finds_points": "finds_points",
        "ObjectPoly": "object_poly",
        "ObjectsPoly": "object_poly",
        "objectpoly": "object_poly",
        "object_poly": "object_poly",
    }

    def normalize_class_name(raw: str):
        if raw in CLASS_NAME_MAPPING:
            return CLASS_NAME_MAPPING[raw]
        low = str(raw).lower()
        if low in CLASS_NAME_MAPPING:
            return CLASS_NAME_MAPPING[low]
        return raw

# Стандартизованные названия классов метрики (для режима генерации)
CLASS_NAMES = list(ALL_VALID_CLASSES)
VALID_CLASS_NAMES = set(ALL_VALID_CLASSES)


# Система координат выходного файла решения (требование формата: координаты в EPSG:3857).
CRS_OUTPUT = "urn:ogc:def:crs:EPSG::3857"

# Допустимые классы объектов для каждого типа данных — по эталонным наборам КОЗ-3
# (finds_points встречается только у геофизики maggr, etgr/terfl — только object_poly,
# у ae/li/sp_or/or — курганы/городища/фортификации/архитектуры). Неизвестный тип —
# без ограничений (все классы метрики).
CLASSES_BY_MARKUP = {
    "li": ["kurgany_tselye", "kurgany_povrezhdennye", "gorodishcha", "fortifikatsii", "arkhitektury"],
    "ae": ["kurgany_tselye", "kurgany_povrezhdennye", "gorodishcha", "fortifikatsii", "arkhitektury"],
    "sp_or": ["kurgany_tselye", "kurgany_povrezhdennye", "gorodishcha", "fortifikatsii", "arkhitektury"],
    "or": ["arkhitektury", "gorodishcha", "fortifikatsii"],
    "maggr": ["finds_points", "object_poly"],
    "etgr": ["object_poly"],
    "terfl": ["object_poly"],
    "geogr": ["object_poly"],
}

# Токены типов данных в именах папок/файлов (сравнение целыми токенами,
# регистронезависимо; побеждает самый правый распознанный токен). «La» (облака точек)
# и «Li» (производные растры) — один тип данных «лидар» → li.
_MARKUP_TOKEN_TYPES = {
    "la": "li",
    "li": "li",
    "lidar": "li",
    "ae": "ae",
    "ае": "ae",
    "spor": "sp_or",
    "sp": "sp_or",
    "or": "or",
    "maggr": "maggr",
    "mggr": "maggr",
    "mg": "maggr",
    "etgr": "etgr",
    "geogr": "geogr",
    "terfl": "terfl",
    "terfi": "terfl",
    "terf1": "terfl",
    "trfl": "terfl",
    "trfi": "terfl",
}


def determine_markup_type(file_path: Path) -> str:
    """
    Определение типа данных (markup_type) по пути к файлу.

    Разбор по токенам (участки из букв/цифр): совпадение «внутри слова» типом не
    считается; побеждает самый правый распознанный токен (тип данных стоит в конце
    имени папки/файла). Результат — как в разметке датасета: li, ae, sp_or, or,
    maggr, etgr, geogr, terfl; по умолчанию — ae.

    Args:
        file_path: Путь к файлу

    Returns:
        str: Тип данных (markup_type)
    """
    text = str(file_path).lower()
    # «sp_or»/«sp or»/«sp-or» — один токен типа (иначе распался бы на sp + or)
    text = text.replace("sp_or", "spor").replace("sp-or", "spor").replace("sp or", "spor")
    result = None
    for token in re.split(r"[^0-9a-zа-яё]+", text):
        found = _MARKUP_TOKEN_TYPES.get(token)
        if found is not None:
            result = found
    return result or "ae"


def extract_region_name(file_path: Path, root_dataset: Path) -> str:
    """Извлечение названия региона из пути к файлу."""
    try:
        relative_path = file_path.relative_to(root_dataset)
    except ValueError:
        return root_dataset.name
    parts = list(relative_path.parts)
    if len(parts) == 0:
        return root_dataset.name
    return parts[0]


def read_utm_json(region_path: Path) -> Optional[str]:
    """
    Читает файл UTM.json из папки региона и возвращает CRS
    
    Args:
        region_path: Путь к папке региона
        
    Returns:
        str или None: CRS из файла UTM.json или None если файл не найден
    """
    utm_file = region_path / "UTM.json"
    
    if not utm_file.exists():
        return None
    
    try:
        with open(utm_file, 'r', encoding='utf-8') as f:
            utm_data = json.load(f)
        
        crs = utm_data.get("crs")
        if crs:
            logger.info(f"Найден файл UTM.json в {region_path.name}, CRS: {crs}")
            return crs
        else:
            logger.warning(f"Файл UTM.json в {region_path.name} не содержит поле 'crs'")
            return None
            
    except Exception as e:
        logger.error(f"Ошибка при чтении файла UTM.json из {region_path}: {e}")
        return None


def extract_class_name_from_filename(filename: str) -> Optional[str]:
    """
    Извлечение названия класса из имени файла
    
    Args:
        filename: Имя файла
        
    Returns:
        Название класса или None
    """
    name_without_ext = Path(filename).stem
    parts = name_without_ext.replace(' ', '_').split('_')
    
    if len(parts) < 2:
        return None
    
    technical_terms = {'ae', 'li', 'gae', 'lidar', 'tif', 'geojson', 'spor', 'or'}
    
    for i in range(len(parts) - 1, -1, -1):
        part = parts[i].lower()
        if part not in technical_terms:
            return normalize_class_name(part)

    return None


def has_empty_coordinates(geom_type: str, coordinates: Any) -> bool:
    """
    Проверка на пустые координаты
    
    Args:
        geom_type: Тип геометрии
        coordinates: Координаты
        
    Returns:
        True если координаты пустые
    """
    if not isinstance(coordinates, list):
        return True
    
    if geom_type == "Polygon":
        if len(coordinates) == 0:
            return True
        if len(coordinates) > 0:
            first_ring = coordinates[0]
            if not isinstance(first_ring, list) or len(first_ring) == 0:
                return True
    
    elif geom_type == "MultiPolygon":
        if len(coordinates) == 0:
            return True
        if len(coordinates) > 0:
            first_polygon = coordinates[0]
            if not isinstance(first_polygon, list) or len(first_polygon) == 0:
                return True
    
    else:
        if len(coordinates) == 0:
            return True
    
    return False


def split_multipolygon_to_polygons(geometry: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Разбивка MultiPolygon на отдельные Polygon
    
    Args:
        geometry: Геометрия
        
    Returns:
        Список геометрий (Polygon)
    """
    if geometry.get("type") != "MultiPolygon":
        return [geometry]
    
    coordinates = geometry.get("coordinates", [])
    
    if len(coordinates) == 1:
        # Если только один полигон, конвертируем в Polygon
        return [{
            "type": "Polygon",
            "coordinates": coordinates[0]
        }]
    else:
        # Разбиваем MultiPolygon на отдельные Polygon'ы
        polygon_geometries = []
        for polygon_coords in coordinates:
            polygon_geometries.append({
                "type": "Polygon",
                "coordinates": polygon_coords
            })
        return polygon_geometries


def point_to_circle_polygon(x: float, y: float, radius: float = 2.0, num_points: int = 32) -> List[List[float]]:
    """Строит круговой полигон вокруг точки (аппроксимация окружности).

    Args:
        x: координата X (долгота в EPSG:3857)
        y: координата Y (широта в EPSG:3857)
        radius: радиус окружности в единицах координат, метры для EPSG:3857
            (2,0 м — допуск метрики КОЗ №3 для точечной разметки)
        num_points: количество точек аппроксимации

    Returns:
        Координаты замкнутого полигона в формате GeoJSON
    """
    coords = []
    for i in range(num_points):
        angle = 2 * math.pi * i / num_points
        px = x + radius * math.cos(angle)
        py = y + radius * math.sin(angle)
        coords.append([px, py])
    coords.append(coords[0])  # замыкаем
    return coords


def scan_dataset_folders(input_path: Path) -> List[Path]:
    """
    Сканирует датасет и возвращает список папок с данными.

    Находит только папки, которые:
    1. Содержат файлы данных (любые файлы кроме .geojson)
    
    Args:
        input_path: Путь к корневому датасету
        
    Returns:
        List[Path]: Список путей к папкам с данными
    """
    folders = []
    seen_paths = set()
    
    # Ищем все файлы данных (любые файлы кроме .geojson и служебных)
    data_extensions = {'.tif', '.tiff', '.gml', '.jpg', '.jpeg', '.png', '.asc', '.las', '.laz'}

    for file_path in input_path.rglob("*"):
        if file_path.is_file():
            # Проверяем расширение файла
            if file_path.suffix.lower() in data_extensions:
                # Берем parent директорию файла
                folder = file_path.parent
                
                # Добавляем только если папка не скрытая и не была обработана
                if (not folder.name.startswith('.') and 
                    not folder.name.startswith('__') and
                    str(folder) not in seen_paths):
                    folders.append(folder)
                    seen_paths.add(str(folder))
    
    # Если не нашли файлов данных, берем папки на уровне регионов
    # Рекурсивно обходим структуру
    if not folders:
        def collect_valid_folders(path: Path) -> None:
            """Рекурсивно собирает папки"""
            for item in path.iterdir():
                if item.is_dir():
                    # Пропускаем скрытые папки
                    if (not item.name.startswith('.') and
                        not item.name.startswith('__')):
                        folders.append(item)
                        # Рекурсивно обходим подпапки
                        collect_valid_folders(item)
        
        collect_valid_folders(input_path)
    
    logger.info(f"Найдено {len(folders)} папок с данными в датасете")
    return folders


def generate_features_for_folder(folder_path: Path, root_dataset: Path, 
                                current_fid: int, region_crs: Optional[str] = None) -> Tuple[List[Dict[str, Any]], int]:
    """
    Генерирует features для папки (как в solution_old.py)
    
    Args:
        folder_path: Путь к папке
        root_dataset: Корневой датасет
        current_fid: Текущий fid для нумерации
        region_crs: CRS региона из UTM.json (если есть)
        
    Returns:
        Tuple[List[Dict], int]: (список features, следующий fid)
    """
    # Извлекаем метаданные из пути
    region_name = extract_region_name(folder_path, root_dataset)
    markup_type = determine_markup_type(folder_path)

    # pixel_resolution из GeoTIFF
    region_dir = root_dataset / region_name if region_name else None
    feat_pixel_resolution = None
    if region_dir and region_dir.is_dir():
        feat_pixel_resolution = _get_pixel_resolution(region_dir, markup_type)
    
    features = []

    # Классы — только допустимые для этого типа данных (CLASSES_BY_MARKUP)
    allowed_classes = CLASSES_BY_MARKUP.get(markup_type, CLASS_NAMES)

    # Генерируем случайное количество классов (от 0 до 4)
    num_classes = random.randint(0, 4)
    if num_classes > 0:
        selected_classes = random.sample(allowed_classes, min(num_classes, len(allowed_classes)))
        
        for class_name in selected_classes:
            # Случайное количество объектов этого класса (от 1 до 7)
            num_objects = random.randint(1, 7)
            
            for _ in range(num_objects):
                # Генерируем случайный полигон
                polygon_coords = generate_random_polygon_epsg3857()
                
                # Система координат: crs — СК выходного файла (координаты примера
                # генерируются в EPSG:3857); original_crs — СК исходных данных
                # региона (UTM.json), если известна, иначе совпадает с crs.
                crs = CRS_OUTPUT
                original_crs = region_crs or crs
                
                props_out = {
                    "class_name": class_name,
                    "region_name": region_name,
                    "markup_type": markup_type,
                    "original_crs": original_crs,
                    "crs": crs,
                    "fid": current_fid,
                    "confidence": round(random.uniform(0.3, 1.0), 3),
                }
                if markup_type == "maggr":
                    props_out["pixel_resolution"] = feat_pixel_resolution
                feature = {
                    "type": "Feature",
                    "properties": props_out,
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [polygon_coords]
                    }
                }

                features.append(feature)
                current_fid += 1

    return features, current_fid


# ============================================================================
# ФУНКЦИИ ДЛЯ РЕЖИМА --random (ГЕНЕРАЦИЯ СЛУЧАЙНЫХ ДАННЫХ)
# ============================================================================

def generate_random_polygon_epsg3857() -> List[List[float]]:
    """
    Генерирует случайный полигон в координатах EPSG:3857 (Web Mercator).
    
    Координаты генерируются в разумных пределах для территории России.
    Полигон гарантированно замкнут и без самопересечений.
    
    Returns:
        List[List[float]]: Координаты полигона в формате GeoJSON [x, y]
    """
    # Границы для территории России в EPSG:3857
    min_x = random.uniform(2500000, 9000000)
    min_y = random.uniform(5600000, 11000000)
    
    # Размеры полигона в метрах (от 5 до 200 метров)
    width = random.uniform(5, 200)
    height = random.uniform(5, 200)
    
    # Создаем полигон с небольшими случайными отклонениями
    half_width = width / 2
    half_height = height / 2
    
    # Уменьшаем шум до 5% чтобы избежать самопересечений
    noise_factor = 0.05
    noise_w = half_width * noise_factor
    noise_h = half_height * noise_factor
    
    # Генерируем четыре угловые точки
    center_x = min_x
    center_y = min_y
    
    p1 = [center_x - half_width + random.uniform(-noise_w, noise_w), 
          center_y - half_height + random.uniform(-noise_h, noise_h)]
    p2 = [center_x + half_width + random.uniform(-noise_w, noise_w), 
          center_y - half_height + random.uniform(-noise_h, noise_h)]
    p3 = [center_x + half_width + random.uniform(-noise_w, noise_w), 
          center_y + half_height + random.uniform(-noise_h, noise_h)]
    p4 = [center_x - half_width + random.uniform(-noise_w, noise_w), 
          center_y + half_height + random.uniform(-noise_h, noise_h)]
    
    # ВАЖНО: Замыкаем полигон - последняя точка должна быть ИДЕНТИЧНА первой
    polygon = [p1, p2, p3, p4, p1]
    
    return polygon


def generate_random_features_for_file(file_path: Path, root_dataset: Path, 
                                     current_fid: int, region_crs: Optional[str] = None) -> Tuple[List[Dict[str, Any]], int]:
    """
    Генерирует случайные features с метаданными из реального файла
    
    Args:
        file_path: Путь к файлу (используется для извлечения метаданных)
        root_dataset: Корневой датасет
        current_fid: Текущий fid для нумерации
        region_crs: CRS региона из UTM.json (если есть)
        
    Returns:
        Tuple[List[Dict], int]: (список features, следующий fid)
    """
    # Извлекаем метаданные из пути к файлу
    region_name = extract_region_name(file_path, root_dataset)
    markup_type = determine_markup_type(file_path)
    
    # Извлекаем класс из имени файла; при отсутствии — случайный из допустимых
    # для этого типа данных
    class_name = extract_class_name_from_filename(file_path.name)
    if not class_name:
        class_name = random.choice(CLASSES_BY_MARKUP.get(markup_type, CLASS_NAMES))
    
    features = []
    
    # Генерируем случайное количество объектов (от 1 до 5)
    num_objects = random.randint(1, 5)
    
    for _ in range(num_objects):
        # Генерируем случайный полигон
        polygon_coords = generate_random_polygon_epsg3857()
        
        # Система координат: crs — СК выходного файла (EPSG:3857); original_crs —
        # СК исходных данных региона (UTM.json), если известна.
        crs = CRS_OUTPUT
        original_crs = region_crs or crs
        
        feature = {
            "type": "Feature",
            "properties": {
                "class_name": class_name,
                "region_name": region_name,
                "markup_type": markup_type,
                "original_crs": original_crs,
                "crs": crs,
                "fid": current_fid,
                "confidence": round(random.uniform(0.3, 1.0), 3)
            },
            "geometry": {
                "type": "Polygon",
                "coordinates": [polygon_coords]
            }
        }
        
        features.append(feature)
        current_fid += 1
    
    return features, current_fid


def process_geojson_file(file_path: Path, root_dataset: Path, current_fid: int,
                         region_crs: Optional[str] = None) -> Tuple[List[Dict[str, Any]], int]:
    """
    Обработка одного GeoJSON файла
    
    Args:
        file_path: Путь к файлу
        root_dataset: Корневой датасет
        current_fid: Текущий fid для нумерации
        region_crs: CRS региона из UTM.json (если есть)
        
    Returns:
        Tuple[List[Dict], int]: (список features, следующий fid)
    """
    try:
        # Загружаем файл
        with open(file_path, 'r', encoding='utf-8') as f:
            geojson_data = json.load(f)
        
        # Проверка структуры
        if not isinstance(geojson_data, dict) or geojson_data.get("type") != "FeatureCollection":
            logger.warning(f"Файл {file_path.name} не является FeatureCollection")
            return [], current_fid
        
        features_in = geojson_data.get("features", [])
        if not isinstance(features_in, list):
            return [], current_fid
        
        # Извлекаем метаданные из пути
        region_name = extract_region_name(file_path, root_dataset)
        markup_type = determine_markup_type(file_path)

        # pixel_resolution из GeoTIFF
        region_dir = root_dataset / region_name if region_name else None
        feat_pixel_resolution = None
        if region_dir and region_dir.is_dir():
            feat_pixel_resolution = _get_pixel_resolution(region_dir, markup_type)

        # Извлекаем класс из имени файла
        class_name = extract_class_name_from_filename(file_path.name)
        if not class_name:
            class_name = "unknown"
        
        # Обрабатываем каждую feature
        features_out = []
        
        for feature in features_in:
            if not isinstance(feature, dict):
                continue
            
            geometry = feature.get("geometry")
            if not geometry:
                continue
            
            # Проверяем тип геометрии
            geom_type = geometry.get("type")
            coordinates = geometry.get("coordinates", [])

            if has_empty_coordinates(geom_type, coordinates):
                logger.debug(f"Пропущена Feature с пустыми координатами из файла {file_path.name}")
                continue

            # Система координат: crs — СК выходного файла; original_crs — исходная
            # СК (из свойств разметки, при отсутствии — СК данных региона из UTM.json).
            props = feature.get("properties", {})
            crs = props.get("crs") or CRS_OUTPUT
            original_crs = props.get("original_crs") or region_crs or crs

            # Разбиваем MultiPolygon на отдельные Polygon
            geometries = split_multipolygon_to_polygons(geometry)

            # Создаем features для каждой геометрии
            for geom in geometries:
                props_out = {
                    "class_name": class_name,
                    "region_name": region_name,
                    "markup_type": markup_type,
                    "original_crs": original_crs,
                    "crs": crs,
                    "fid": current_fid,
                }
                if markup_type == "maggr":
                    props_out["pixel_resolution"] = feat_pixel_resolution
                new_feature = {
                    "type": "Feature",
                    "properties": props_out,
                    "geometry": geom
                }
                features_out.append(new_feature)
                current_fid += 1
        
        return features_out, current_fid
        
    except Exception as e:
        logger.error(f"Ошибка при обработке файла {file_path}: {e}")
        return [], current_fid


def predict(input_dir: str, output_dir: str, use_gt: bool = False) -> Dict[str, Any]:
    """
    Основная функция для создания объединенной разметки.
    
    Два режима работы:
    1. use_gt=False (по умолчанию): генерирует случайные полигоны с метаданными из структуры
    2. use_gt=True: читает реальные GeoJSON файлы из датасета
    
    Args:
        input_dir: Путь к входному датасету
        output_dir: Путь для сохранения результатов (путь к папке, куда сохранить файл result.geojson)
        use_gt: Использовать файлы разметки (Ground Truth) вместо генерации
        
    Returns:
        Dict: Статистика обработки
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    if not input_path.exists():
        raise ValueError(f"Входная директория не существует: {input_dir}")
    
    mode_str = "ИЗ ФАЙЛОВ РАЗМЕТКИ (GT)" if use_gt else "ГЕНЕРАЦИЯ"
    logger.info(f"Начинаем создание объединенной разметки (режим: {mode_str})")
    logger.info(f"Входная директория: {input_dir}")
    logger.info(f"Выходная директория: {output_dir}")
    if not use_gt:
        logger.info(f"Сид генератора: {SEED}")
    
    # Создаем выходную директорию
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Сканируем регионы и создаем кэш CRS из файлов UTM.json
    logger.info("Сканирование регионов и загрузка UTM.json...")
    region_crs_cache = {}
    
    # Сначала проверяем UTM.json в корне входной директории
    # (для случая когда весь input_dir - это один регион)
    root_crs = read_utm_json(input_path)
    if root_crs:
        # Используем имя корневой папки как ключ
        region_crs_cache[input_path.name] = root_crs
    
    # Затем проверяем UTM.json в подпапках
    # (для случая когда input_dir содержит несколько регионов)
    for item in input_path.iterdir():
        if item.is_dir():
            crs = read_utm_json(item)
            if crs:
                region_crs_cache[item.name] = crs
    
    if region_crs_cache:
        logger.info(f"Загружено CRS для {len(region_crs_cache)} регионов:")
        for region, crs in region_crs_cache.items():
            logger.info(f"  {region}: {crs}")
    else:
        logger.info("Файлы UTM.json не найдены, будет использоваться CRS из разметки или рандом")
    
    # Создаем структуру для объединенного файла
    merged_data = {
        "type": "FeatureCollection",
        "features": []
    }
    
    current_fid = 0
    total_items_processed = 0
    
    if use_gt:
        # Режим чтения реальных файлов разметки - сканируем GeoJSON файлы
        geojson_files = list(input_path.rglob("*.geojson"))
        
        if not geojson_files:
            logger.warning(f"Не найдено GeoJSON файлов в датасете: {input_dir}")
        
        logger.info(f"Найдено {len(geojson_files)} GeoJSON файлов")
        
        # Обрабатываем каждый GeoJSON файл
        for file_path in geojson_files:
            try:
                logger.info(f"Обработка файла: {file_path.name}")
                
                # Определяем регион для получения CRS
                region_name = extract_region_name(file_path, input_path)
                region_crs = region_crs_cache.get(region_name)
                
                # Если CRS для региона не найден, используем корневой CRS
                if not region_crs:
                    region_crs = region_crs_cache.get(input_path.name)
                
                features, current_fid = process_geojson_file(
                    file_path, input_path, current_fid, region_crs,
                )
                
                if features:
                    merged_data["features"].extend(features)
                    total_items_processed += 1
                    logger.info(f"  Обработано {len(features)} features")
                
            except Exception as e:
                logger.error(f"Ошибка при обработке файла {file_path}: {e}")
                continue
    else:
        # Режим генерации (по умолчанию) - сканируем папки как в solution_old.py
        all_folders = scan_dataset_folders(input_path)
        
        if not all_folders:
            logger.warning(f"Не найдено папок в датасете: {input_dir}")
            # Создаем пустой файл
            all_folders = [input_path]
        
        # Генерируем features для каждой папки
        for folder in all_folders:
            try:
                logger.info(f"Обработка папки: {folder.relative_to(input_path)}")
                
                # Определяем регион для получения CRS
                region_name = extract_region_name(folder, input_path)
                region_crs = region_crs_cache.get(region_name)
                
                # Если CRS для региона не найден, используем корневой CRS
                if not region_crs:
                    region_crs = region_crs_cache.get(input_path.name)
                
                features, current_fid = generate_features_for_folder(
                    folder, input_path, current_fid, region_crs
                )
                
                if features:
                    merged_data["features"].extend(features)
                    total_items_processed += 1
                    logger.info(f"  Сгенерировано {len(features)} features")
                
            except Exception as e:
                logger.error(f"Ошибка при обработке папки {folder}: {e}")
                continue
    
    # Сохраняем объединенный файл
    output_file = output_path / "result.geojson"
    
    logger.info(f"Сохранение объединенного файла: {output_file}")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(merged_data, f, ensure_ascii=False, indent=2)
    
    # Статистика
    if use_gt:
        stats = {
            "mode": "gt",
            "total_files": len(geojson_files) if 'geojson_files' in locals() else 0,
            "files_processed": total_items_processed,
            "total_features": len(merged_data["features"]),
            "output_file": str(output_file)
        }
        logger.info(f"Обработка завершена!")
        logger.info(f"Обработано файлов: {total_items_processed}/{len(geojson_files) if 'geojson_files' in locals() else 0}")
        logger.info(f"Создано features: {len(merged_data['features'])}")
        logger.info(f"Результат сохранен в: {output_file}")
    else:
        stats = {
            "mode": "generated",
            "total_folders": len(all_folders) if 'all_folders' in locals() else 0,
            "folders_processed": total_items_processed,
            "total_features": len(merged_data["features"]),
            "output_file": str(output_file)
        }
        logger.info(f"Обработка завершена!")
        logger.info(f"Обработано папок: {total_items_processed}/{len(all_folders) if 'all_folders' in locals() else 0}")
        logger.info(f"Сгенерировано features: {len(merged_data['features'])}")
        logger.info(f"Результат сохранен в: {output_file}")
    
    return stats


def main():
    """
    Основная функция для запуска из командной строки.
    
    Использование:
        # Генерация (по умолчанию) - для папки valid без разметки
        python solution.py <input_dir> <output_dir>
        
        # Из файлов разметки - для тестирования с ground truth
        python solution.py <input_dir> <output_dir> --use-gt
        
    Примеры:
        python solution.py DATA/valid output/
        python solution.py DATA/ТЕСТ output/ --use-gt
    """
    parser = argparse.ArgumentParser(
        description='Создание объединенной разметки археологических объектов',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Режимы работы:
  1. ГЕНЕРАЦИЯ (по умолчанию):
     - Сканирует структуру датасета (папки с изображениями, LiDAR и т.д.)
     - Извлекает метаданные из путей (region, markup_type)
     - Генерирует случайные полигоны с правильными метаданными
     - Использует фиксированный сид (SEED = 42) для воспроизводимости
     - Основной сценарий для папки valid (без файлов .geojson разметки)
     
  2. ИЗ ФАЙЛОВ РАЗМЕТКИ (--use-gt):
     - Читает реальные GeoJSON файлы из датасета
     - Извлекает реальные археологические объекты
     - Разбивает MultiPolygon на отдельные Polygon
     - Удаляет объекты с пустыми координатами
     - Используется для тестирования на датасетах с разметкой

Примеры:
  # Генерация для папки valid (основной сценарий)
  python solution.py DATA/valid output/
  
  # Использование файлов разметки для тестирования
  python solution.py DATA/ТЕСТ output/ --use-gt
        """
    )
    
    parser.add_argument(
        'input_dir',
        help='Путь к входному датасету'
    )
    
    parser.add_argument(
        'output_dir',
        help='Путь для сохранения результатов'
    )
    
    parser.add_argument(
        '--use-gt',
        action='store_true',
        help='Использовать файлы разметки (Ground Truth) вместо генерации'
    )

    args = parser.parse_args()

    try:
        results = predict(
            args.input_dir,
            args.output_dir,
            use_gt=args.use_gt,
        )
        
        print("\n" + "="*70)
        print("ОБРАБОТКА ЗАВЕРШЕНА УСПЕШНО!")
        print("="*70)
        print(f"Режим: {results['mode'].upper()}")
        if results['mode'] == 'gt':
            print(f"Обработано файлов: {results['files_processed']}/{results['total_files']}")
        else:
            print(f"Обработано папок: {results['folders_processed']}/{results['total_folders']}")
        print(f"Создано features: {results['total_features']}")
        print(f"Результат сохранен в: {results['output_file']}")
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"\n{'='*70}", file=sys.stderr)
        print(f"ОШИБКА ПРИ ОБРАБОТКЕ", file=sys.stderr)
        print(f"{'='*70}", file=sys.stderr)
        print(f"{e}", file=sys.stderr)
        print(f"{'='*70}\n", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()