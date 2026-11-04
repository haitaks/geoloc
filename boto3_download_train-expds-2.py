#!/usr/bin/env python3
"""
Скрипт для скачивания и синхронизации обучающей выборки (train-expds-2) из хранилища S3.
Скрипт оптимизирован для работы с большими файлами: он поддерживает многопоточность,
автоматические повторные попытки при ошибках сети (с экспоненциальной задержкой) и
проверку целостности (сопоставление по размеру файлов). Также скрипт полностью поддерживает
сценарий "дозаливки" или "синхронизации": он пропускает уже скачанные файлы корректного размера,
что позволяет использовать его для докачивания данных (без необходимости качать все заново).
"""
import boto3
import os
import argparse
import urllib3
import time
import threading
import logging
import random
import re
from pathlib import Path
from botocore.config import Config
from concurrent.futures import ThreadPoolExecutor, as_completed
from boto3.s3.transfer import TransferConfig

# Отключение предупреждений SSL
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Конфигурация подключения S3
ENDPOINT_URL = 'https://s3.ru-7.storage.selcloud.ru:443'
BUCKET_NAME = 'train-expds-2'
LOCAL_DIR = '../data'
LOG_FILE = 'download_train-expds-2.log'

# Настройки многопоточной передачи (Multipart) для больших файлов.
# use_threads=False крайне важно: если установить True, внутри каждого 
# потока воркера создается дочерний вложенный пул потоков, что приведет к падению
# с ошибкой "cannot schedule new futures after interpreter shutdown",
# когда внешний ThreadPoolExecutor будет завершать работу.
TRANSFER_CONFIG = TransferConfig(
    multipart_threshold=64 * 1024 * 1024,
    multipart_chunksize=16 * 1024 * 1024,
    max_concurrency=1,   # игнорируется при use_threads=False, оставлено для наглядности
    use_threads=False
)

MAX_FILE_RETRIES = 6      # количество попыток повторной загрузки файла при сбоях
RETRY_BASE_DELAY = 3.0    # секунды; базовая задержка (ограничена капом во избежание слишком долгого сна)
RETRY_MAX_DELAY  = 30.0   # максимальная задержка (хард кап) перед повторной попыткой

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

# Инициализация клиента S3 с отключенной проверкой SSL и оптимизированными настройками
s3_client = boto3.client(
    's3',
    endpoint_url=ENDPOINT_URL,
    aws_access_key_id='a064df53e320474396c1de1c82dd858e', 
    aws_secret_access_key='b1f66191dfe34927992afe3cc62a66ce', 
    verify=False,  # Отключение проверки SSL сертификатов
    config=Config(
        signature_version='s3v4',
        max_pool_connections=20,
        connect_timeout=30,
        read_timeout=120,        # для больших файлов: позволяем до 2 мин паузы между чанками
        # режим 'adaptive' автоматически увеличивает задержки при 429/503 ответах от сервера
        retries={
            'max_attempts': 5,
            'mode': 'adaptive',
        }
    )
)

# Глобальные счетчики для визуализации и отслеживания прогресса
download_stats = {
    'completed': 0,
    'total': 0,
    'downloaded_bytes': 0,
    'total_bytes': 0,
    'start_time': None,
    'last_render_time': 0.0,
    'render_interval': 0.2,
    'lock': threading.Lock()
}


def format_bytes(bytes_size):
    """Форматирование байтов в читабельный вид (B, KB, MB, GB, TB)"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.2f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.2f} TB"


def render_progress(force=False):
    """Форматированный вывод текущего прогресса скачивания в консоль"""
    with download_stats['lock']:
        now = time.time()
        if not force and (now - download_stats['last_render_time']) < download_stats['render_interval']:
            return

        download_stats['last_render_time'] = now
        
        elapsed = time.time() - download_stats['start_time']
        speed = download_stats['downloaded_bytes'] / elapsed if elapsed > 0 else 0
        progress = (download_stats['completed'] / download_stats['total']) * 100 if download_stats['total'] > 0 else 0
        
        print(f"\r  Прогресс: [{download_stats['completed']}/{download_stats['total']}] "
              f"{progress:.1f}% | {format_bytes(download_stats['downloaded_bytes'])}/{format_bytes(download_stats['total_bytes'])} | "
              f"Скорость: {format_bytes(speed)}/s", end='', flush=True)


def add_downloaded_bytes(num_bytes):
    """Добавить количество байт к статистике текущей сессии"""
    with download_stats['lock']:
        download_stats['downloaded_bytes'] += num_bytes
    render_progress()


def complete_file(progress_bytes=0):
    """Отметить один файл как завершенный и обновить счетчик байт"""
    with download_stats['lock']:
        if progress_bytes > 0:
            download_stats['downloaded_bytes'] += progress_bytes
        download_stats['completed'] += 1
    render_progress(force=True)


class ProgressCallback:
    """Обертка для обратного вызова (callback) библиотеки boto3 (считает скачанные байты чанка)"""

    def __init__(self):
        self.transferred = 0

    def __call__(self, bytes_amount):
        self.transferred += bytes_amount
        add_downloaded_bytes(bytes_amount)


def _rollback_bytes(n):
    """Отнимаем байты, которые были учтены во время неудачной попытки частичного скачивания."""
    if n <= 0:
        return
    with download_stats['lock']:
        download_stats['downloaded_bytes'] = max(0, download_stats['downloaded_bytes'] - n)


# Регулярное выражение для временных файлов boto3 TransferManager: исходное имя + '.' + 8 hex-символов
_PARTIAL_SUFFIX_RE = re.compile(r'\.[0-9A-Fa-f]{8}$')


def _cleanup_partial_file(local_file, expected_size=None):
    """Удаление неполных/поврежденных локальных файлов и временных файлов boto3 (с hex-суффиксом)."""
    parent = os.path.dirname(local_file)
    basename = os.path.basename(local_file)

    # Удаление самого файла назначения, если он неполный
    try:
        if os.path.exists(local_file):
            if expected_size is None or os.path.getsize(local_file) != expected_size:
                os.remove(local_file)
                logger.info(f"Удален поврежденный/частичный файл: {local_file}")
    except Exception as e:
        logger.warning(f"Не удалось удалить частичный файл {local_file}: {e}")

    # Удаление временных файлов (мусора) boto3: <basename>.<8 hex chars>
    try:
        if os.path.isdir(parent):
            siblings = sorted(
                (f for f in os.listdir(parent)
                 if f.startswith(basename + '.') and _PARTIAL_SUFFIX_RE.search(f)),
                key=lambda f: f.rsplit('.', 1)[-1]
            )
            for name in siblings:
                sibling_path = os.path.join(parent, name)
                try:
                    os.remove(sibling_path)
                    logger.info(f"Удален временный файл boto3: {sibling_path}")
                except Exception as e:
                    logger.warning(f"Не удалось удалить временный файл boto3 {sibling_path}: {e}")
    except Exception as e:
        logger.warning(f"Не удалось сканировать временные файлы boto3 в {parent}: {e}")


def cleanup_existing_partial_files(local_dir):
    """Рекурсивное сканирование *local_dir* с удалением всех временных (hex-суффикс) файлов boto3.

    Файлы обрабатываются с сортировкой для обеспечения детерминированности логов.
    Возвращает количество удаленных файлов.
    """
    removed = 0
    found = []
    for dirpath, _, filenames in os.walk(local_dir):
        for name in filenames:
            if _PARTIAL_SUFFIX_RE.search(name):
                found.append(os.path.join(dirpath, name))

    # Сортировка по hex-суффиксу
    found.sort(key=lambda p: p.rsplit('.', 1)[-1])

    if found:
        print(f"  Найдено {len(found)} остаточных временных файлов boto3 — производим очистку...")
        for path in found:
            try:
                os.remove(path)
                logger.info(f"Удален существующий временный файл: {path}")
                print(f"    Удалено: {os.path.relpath(path, local_dir)}")
                removed += 1
            except Exception as e:
                logger.warning(f"Ошибка при удалении существующего временного файла {path}: {e}")
                print(f"    Ошибка удаления: {path}: {e}")
    return removed


def download_file(bucket, key, local_file, file_size):
    """Загрузка одиночного файла с обработкой повторных попыток (retries) и прогрессивной задержкой."""
    # Создание директории при необходимости
    local_file_dir = os.path.dirname(local_file)
    Path(local_file_dir).mkdir(parents=True, exist_ok=True)

    # Пропуск, если файл существует и имеет корректный размер (сценарий синхронизации/докачивания)
    if os.path.exists(local_file):
        if os.path.getsize(local_file) == file_size:
            logger.info(f"Пропуск (уже существует): {key}")
            complete_file(progress_bytes=file_size)
            return True
        else:
            logger.info(f"Найден поврежденный/неполный локальный файл (размер не совпадает), очистка перед загрузкой: {local_file}")
            _cleanup_partial_file(local_file)

    for attempt in range(1, MAX_FILE_RETRIES + 1):
        progress_callback = ProgressCallback()
        try:
            logger.info(f"Скачивание (попытка {attempt}): {key} -> {local_file}")
            s3_client.download_file(
                bucket,
                key,
                local_file,
                Config=TRANSFER_CONFIG,
                Callback=progress_callback
            )

            # Проверка размера загруженного файла
            if os.path.getsize(local_file) != file_size:
                logger.warning(f"Несовпадение размера после загрузки: {key}")
                _rollback_bytes(progress_callback.transferred)
                _cleanup_partial_file(local_file, expected_size=file_size)
                # Вызов ошибки для инициации повторной попытки скачивания
                raise ValueError(f"Несовпадение размера (получено {os.path.getsize(local_file)}, ожидалось {file_size})")

            missing_bytes = max(0, file_size - progress_callback.transferred)
            logger.info(f"Скачано успешно: {key}")
            complete_file(progress_bytes=missing_bytes)
            return True

        except Exception as e:
            # Откат байт, которые уже были учтены в ходе этой неудачной попытки
            _rollback_bytes(progress_callback.transferred)
            _cleanup_partial_file(local_file, expected_size=file_size)

            if attempt == MAX_FILE_RETRIES:
                print(f"\nОшибка после {MAX_FILE_RETRIES} попыток: {key}: {e}")
                logger.error(f"Остановка скачивания для файла {key} после {MAX_FILE_RETRIES} попыток: {e}")
                return False

            delay = min(RETRY_BASE_DELAY * (2 ** (attempt - 1)), RETRY_MAX_DELAY) + random.uniform(0, 2)
            print(f"\nПовторная попытка скачивания {attempt}/{MAX_FILE_RETRIES - 1} для {os.path.basename(key)} через {delay:.1f}с: {e}")
            logger.warning(f"Повторная загрузка файла {key} (попытка {attempt}) через {delay:.1f}с: {e}")
            time.sleep(delay)


def download_folder_parallel(bucket, prefix, local_dir, max_workers=20):
    """Скачивание всех файлов из папки S3 с использованием многопоточности"""
    # Удаление любых остаточных hex-суффикс файлов перед запуском
    cleanup_existing_partial_files(local_dir)

    print(f"Сканирование файлов в папке/пути: {prefix}")
    logger.info(f"Процесс сканирования для пути (префикса): {prefix}")
    
    # Получение списка файлов с поддержкой пагинации AWS
    paginator = s3_client.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=bucket, Prefix=prefix)
    
    files_to_download = []
    total_size = 0
    
    for page in pages:
        if 'Contents' not in page:
            print(f"Файлы не найдены по префиксу {prefix}")
            return
            
        for obj in page['Contents']:
            key = obj['Key']
            if not key.endswith('/'):
                # Пропуск файлов внутри исключенных служебных подпапок
                parts = key.split('/')
                if any(p in ('Старая', 'Архив') for p in parts):
                    logger.info(f"Пропуск (запрещенная/скрытая директория): {key}")
                    continue
                local_file = os.path.join(local_dir, key)
                files_to_download.append((key, local_file, obj['Size']))
                total_size += obj['Size']
    
    if not files_to_download:
        print(f"Нет файлов для скачивания в префиксе: {prefix}")
        logger.info(f"Не найдены файлы в префиксе: {prefix}")
        return
    
    # Инициализация отслеживания состояния
    download_stats['total'] = len(files_to_download)
    download_stats['completed'] = 0
    download_stats['downloaded_bytes'] = 0
    download_stats['total_bytes'] = total_size
    download_stats['start_time'] = time.time()
    download_stats['last_render_time'] = 0.0
    
    print(f"Ожидается загрузка {len(files_to_download)} файлов ({format_bytes(total_size)}) с использованием {max_workers} потоков...")
    logger.info(
        f"Старт скачивания: файлов={len(files_to_download)}, суммарный размер={total_size} байт, потоков={max_workers}"
    )
    
    # Скачивание в многопоточном режиме
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = {
            executor.submit(download_file, bucket, key, local_file, file_size): key
            for key, local_file, file_size in files_to_download
        }
        
        failed_files = []
        for future in as_completed(future_to_file):
            key = future_to_file[future]
            try:
                success = future.result()
                if not success:
                    failed_files.append(key)
            except Exception as e:
                print(f"\nНепредвиденная ошибка при скачивании файла {key}: {e}")
                logger.exception(f"Непредвиденная внутренняя ошибка для {key}")
                failed_files.append(key)
    
    render_progress(force=True)
    print()  # Перенос строки после визуализации финального прогресса
    
    if failed_files:
        print(f"Не удалось скачать ряд файлов (ошибок: {len(failed_files)}):")
        logger.error(f"Не удалось скачать файлов: {len(failed_files)}")
        for key in failed_files[:10]:  # Показываем только первые 10
            print(f"  - {key}")
            logger.error(f"Сбойный файл: {key}")
        if len(failed_files) > 10:
            print(f"  ... и еще {len(failed_files) - 10} файлов")
            logger.error(f"... и еще {len(failed_files) - 10} сбойных файлов")
    else:
        print("✓ Все файлы загружены успешно!")
        logger.info("Все файлы загружены успешно")


def verify_downloads(bucket, prefix, local_dir):
    """Верификация скачанных данных путем сличения локальных размеров с размерами в S3 бакете"""
    print(f"\nПроверка целостности файлов по пути: {prefix}")
    
    paginator = s3_client.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=bucket, Prefix=prefix)
    
    total_files = 0
    missing_files = 0
    size_mismatches = 0
    correct_files = 0
    
    for page in pages:
        if 'Contents' not in page:
            continue
            
        for obj in page['Contents']:
            key = obj['Key']
            if key.endswith('/'):
                continue
                
            total_files += 1
            remote_size = obj['Size']
            local_file = os.path.join(local_dir, key)
            
            if not os.path.exists(local_file):
                missing_files += 1
                if missing_files < 10:
                    print(f"Отсутствует локально: {key}")
            else:
                local_size = os.path.getsize(local_file)
                if local_size != remote_size:
                    size_mismatches += 1
                    if size_mismatches < 10:
                        print(f"Размер файла не совпадает: {key} (локально: {format_bytes(local_size)}, на сервере: {format_bytes(remote_size)})")
                else:
                    correct_files += 1
            
            # Визуализация прогресса верификации
            print(f"\r  Проверено: {total_files} файлов | ✓ {correct_files} | ✗ {missing_files + size_mismatches}", end='', flush=True)
    
    print()  # Перенос строки
    print(f"\nСводка проверки файлов:")
    print(f"  Всего файлов: {total_files}")
    print(f"  Корректных: {correct_files}")
    print(f"  Отсутствует: {missing_files}")
    print(f"  Несовпадение размера (поврежденных): {size_mismatches}")
    
    if missing_files == 0 and size_mismatches == 0:
        print("✓ Все файлы успешно проверены и являются консистентными!")
    else:
        print("✗ Некоторые файлы имеют проблемы! (Ошибки загрузки)")
    
    return missing_files + size_mismatches == 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Скрипт оптимизированного скачивания файлов из S3 (обучающая выборка) с валидацией')
    parser.add_argument('--folder', type=str, default='', 
                       help='Путь/префикс к целевой папке (по умолчанию: скачать все файлы из бакета)')
    parser.add_argument('--local-dir', type=str, default=LOCAL_DIR, 
                       help=f'Локальная папка (директория) для сохранения (по умолчанию: {LOCAL_DIR})')
    parser.add_argument('--threads', type=int, default=4,
                       help='Количество параллельных потоков скачивания (по умолчанию: 4)')
    parser.add_argument('--verify-only', action='store_true', 
                       help='Только проверить существующие локальные файлы (сравнить с S3), без скачивания')
    parser.add_argument('--no-verify', action='store_true', 
                       help='Пропустить проверку файлов после их загрузки')
    args = parser.parse_args()

    local_directory = args.local_dir
    Path(local_directory).mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print(f"Загрузчик из S3 - папка: {'ВСЕ ДАННЫЕ' if not args.folder else args.folder.upper()}")
    print(f"Локальная директория сохранения: {local_directory}")
    print(f"Количество потоков: {args.threads}")
    print(f"Файл логов: {LOG_FILE}")
    print("=" * 80)
    logger.info(
        f"Запуск: папка/путь='{args.folder}', local_dir='{local_directory}', потоков={args.threads}, "
        f"verify_only={args.verify_only}, no_verify={args.no_verify}"
    )
    
    if args.verify_only:
        print("РЕЖИМ ПРОВЕРКИ/СВЕРКИ ДАННЫХ")
        verify_downloads(BUCKET_NAME, args.folder, local_directory)
    else:
        print("РЕЖИМ СКАЧИВАНИЯ")
        download_folder_parallel(BUCKET_NAME, args.folder, local_directory, args.threads)
        
        if not args.no_verify:
            verify_downloads(BUCKET_NAME, args.folder, local_directory)
    
    print("=" * 80)
    print("Работа программы завершена!")
    print("=" * 80)
    logger.info("Работа скрипта была успешно завершена")
