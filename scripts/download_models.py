"""Скрипт для загрузки моделей из GitHub репозитория.

Этот скрипт автоматически загружает обученные модели из GitHub Releases
или из самого репозитория, если модели хранятся там.
"""

import os
import sys
import requests
import zipfile
from pathlib import Path
from typing import Optional
import json

# Конфигурация
REPO_OWNER = "MichailLepin"  # Замените на ваш GitHub username
REPO_NAME = "Fake-News-Classifier"  # Замените на название репозитория
GITHUB_API_BASE = "https://api.github.com"
GITHUB_RELEASES_URL = f"{GITHUB_API_BASE}/repos/{REPO_OWNER}/{REPO_NAME}/releases/latest"
GITHUB_CONTENT_URL = f"https://raw.githubusercontent.com/{REPO_OWNER}/{REPO_NAME}/main"

# Файлы для загрузки
MODELS_TO_DOWNLOAD = {
    "best_cnn_model.pth": "models/best_cnn_model.pth",
    "best_lstm_model.pth": "models/best_lstm_model.pth",
    "best_bert_model.zip": "models/best_bert_model.zip",
    "best_distilbert_model.zip": "models/best_distilbert_model.zip",
    "vocab.json": "vocab/vocab.json"
}

# Папки для распаковки
FOLDERS_TO_EXTRACT = {
    "best_bert_model.zip": "models/best_bert_model",
    "best_distilbert_model.zip": "models/best_distilbert_model"
}


def download_file(url: str, destination: Path, chunk_size: int = 8192) -> bool:
    """
    Загружает файл по URL.
    
    Args:
        url: URL файла для загрузки
        destination: Путь для сохранения файла
        chunk_size: Размер чанка для загрузки
        
    Returns:
        True если загрузка успешна, False иначе
    """
    try:
        print(f"Загрузка {url}...")
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        # Создаем директорию если нужно
        destination.parent.mkdir(parents=True, exist_ok=True)
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(destination, 'wb') as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        print(f"\r  Прогресс: {percent:.1f}%", end='', flush=True)
        
        print(f"\n  ✓ Файл сохранен: {destination}")
        return True
        
    except requests.exceptions.RequestException as e:
        print(f"\n  ✗ Ошибка при загрузке: {e}")
        return False


def download_from_releases() -> bool:
    """
    Загружает модели из последнего GitHub Release.
    
    Returns:
        True если хотя бы один файл загружен успешно
    """
    try:
        print("=" * 70)
        print("ЗАГРУЗКА МОДЕЛЕЙ ИЗ GITHUB RELEASES")
        print("=" * 70)
        print()
        
        # Получаем информацию о последнем релизе
        print(f"Получение информации о релизах из {REPO_OWNER}/{REPO_NAME}...")
        response = requests.get(GITHUB_RELEASES_URL, timeout=10)
        
        if response.status_code == 404:
            print("⚠ Релизы не найдены. Попробуйте загрузить из репозитория.")
            return False
        
        response.raise_for_status()
        release_data = response.json()
        
        print(f"✓ Найден релиз: {release_data.get('tag_name', 'latest')}")
        print(f"  Название: {release_data.get('name', 'N/A')}")
        print()
        
        # Получаем список ассетов
        assets = release_data.get('assets', [])
        if not assets:
            print("⚠ В релизе нет файлов для загрузки.")
            return False
        
        # Создаем словарь ассетов по имени файла
        assets_dict = {asset['name']: asset['browser_download_url'] for asset in assets}
        
        success_count = 0
        
        # Загружаем модели
        for filename, local_path in MODELS_TO_DOWNLOAD.items():
            if filename in assets_dict:
                destination = Path(local_path)
                
                # Пропускаем если файл уже существует
                if destination.exists():
                    print(f"⏭ {filename} уже существует, пропускаем")
                    continue
                
                url = assets_dict[filename]
                if download_file(url, destination):
                    success_count += 1
                    
                    # Распаковываем zip файлы
                    if filename.endswith('.zip') and filename in FOLDERS_TO_EXTRACT:
                        extract_path = Path(FOLDERS_TO_EXTRACT[filename])
                        print(f"  Распаковка {filename} в {extract_path}...")
                        try:
                            with zipfile.ZipFile(destination, 'r') as zip_ref:
                                zip_ref.extractall(extract_path.parent)
                            print(f"  ✓ Распаковано в {extract_path}")
                            # Удаляем zip файл после распаковки
                            destination.unlink()
                            print(f"  ✓ Временный файл {filename} удален")
                        except Exception as e:
                            print(f"  ✗ Ошибка при распаковке: {e}")
            else:
                print(f"⚠ {filename} не найден в релизе")
        
        print()
        print("=" * 70)
        if success_count > 0:
            print(f"✓ Успешно загружено файлов: {success_count}")
        else:
            print("⚠ Файлы не были загружены")
        print("=" * 70)
        
        return success_count > 0
        
    except requests.exceptions.RequestException as e:
        print(f"✗ Ошибка при обращении к GitHub API: {e}")
        return False


def download_from_repo() -> bool:
    """
    Загружает модели напрямую из репозитория (если они там хранятся).
    
    Returns:
        True если хотя бы один файл загружен успешно
    """
    try:
        print("=" * 70)
        print("ЗАГРУЗКА МОДЕЛЕЙ ИЗ РЕПОЗИТОРИЯ")
        print("=" * 70)
        print()
        
        success_count = 0
        
        for filename, local_path in MODELS_TO_DOWNLOAD.items():
            destination = Path(local_path)
            
            # Пропускаем если файл уже существует
            if destination.exists():
                print(f"⏭ {filename} уже существует, пропускаем")
                continue
            
            # Формируем URL для загрузки из репозитория
            url = f"{GITHUB_CONTENT_URL}/{local_path}"
            
            if download_file(url, destination):
                success_count += 1
                
                # Распаковываем zip файлы
                if filename.endswith('.zip') and filename in FOLDERS_TO_EXTRACT:
                    extract_path = Path(FOLDERS_TO_EXTRACT[filename])
                    print(f"  Распаковка {filename} в {extract_path}...")
                    try:
                        with zipfile.ZipFile(destination, 'r') as zip_ref:
                            zip_ref.extractall(extract_path.parent)
                        print(f"  ✓ Распаковано в {extract_path}")
                        # Удаляем zip файл после распаковки
                        destination.unlink()
                        print(f"  ✓ Временный файл {filename} удален")
                    except Exception as e:
                        print(f"  ✗ Ошибка при распаковке: {e}")
        
        print()
        print("=" * 70)
        if success_count > 0:
            print(f"✓ Успешно загружено файлов: {success_count}")
        else:
            print("⚠ Файлы не были загружены")
        print("=" * 70)
        
        return success_count > 0
        
    except Exception as e:
        print(f"✗ Ошибка при загрузке из репозитория: {e}")
        return False


def main():
    """Основная функция."""
    print()
    print("=" * 70)
    print("СКРИПТ ЗАГРУЗКИ МОДЕЛЕЙ ИЗ GITHUB")
    print("=" * 70)
    print()
    print(f"Репозиторий: {REPO_OWNER}/{REPO_NAME}")
    print()
    
    # Сначала пробуем загрузить из Releases
    if download_from_releases():
        print("\n✓ Загрузка из Releases завершена")
    else:
        print("\n⚠ Не удалось загрузить из Releases, пробуем из репозитория...")
        print()
        download_from_repo()
    
    print()
    print("Проверка наличия файлов:")
    print("-" * 70)
    
    all_exist = True
    for filename, local_path in MODELS_TO_DOWNLOAD.items():
        destination = Path(local_path)
        
        # Для zip файлов проверяем распакованную папку
        if filename.endswith('.zip') and filename in FOLDERS_TO_EXTRACT:
            check_path = Path(FOLDERS_TO_EXTRACT[filename])
            if check_path.exists() and check_path.is_dir():
                print(f"✓ {filename} → {check_path}/")
            else:
                print(f"✗ {filename} → {check_path}/ (не найдено)")
                all_exist = False
        else:
            if destination.exists():
                size_mb = destination.stat().st_size / (1024 * 1024)
                print(f"✓ {filename} → {destination} ({size_mb:.2f} MB)")
            else:
                print(f"✗ {filename} → {destination} (не найдено)")
                all_exist = False
    
    print("-" * 70)
    
    if all_exist:
        print("\n✓ Все файлы успешно загружены!")
    else:
        print("\n⚠ Некоторые файлы отсутствуют.")
        print("\nРекомендации:")
        print("1. Убедитесь, что модели загружены в GitHub Releases или репозиторий")
        print("2. Проверьте правильность REPO_OWNER и REPO_NAME в скрипте")
        print("3. Для больших файлов (>100MB) используйте GitHub Releases или Git LFS")
    
    return 0 if all_exist else 1


if __name__ == "__main__":
    sys.exit(main())

