"""
Скрипт для подготовки файлов к деплою на Railway.

Проверяет наличие всех необходимых файлов (модели и словарь)
и создает инструкцию по их размещению.
"""

import os
import json
from pathlib import Path


def check_file_exists(filepath, description):
    """Проверка существования файла."""
    exists = os.path.exists(filepath)
    status = "✓" if exists else "✗"
    size_info = ""
    
    if exists:
        size_mb = os.path.getsize(filepath) / (1024 * 1024)
        size_info = f" ({size_mb:.2f} MB)"
    
    print(f"{status} {description}: {filepath}{size_info}")
    return exists


def check_vocab_structure(vocab_path):
    """Проверка структуры словаря."""
    try:
        with open(vocab_path, 'r', encoding='utf-8') as f:
            vocab = json.load(f)
        
        required_keys = ['<PAD>', '<UNK>']
        missing_keys = [key for key in required_keys if key not in vocab]
        
        if missing_keys:
            print(f"  ⚠ Отсутствуют обязательные ключи: {missing_keys}")
            return False
        
        print(f"  ✓ Размер словаря: {len(vocab)} слов")
        print(f"  ✓ <PAD>: {vocab.get('<PAD>')}")
        print(f"  ✓ <UNK>: {vocab.get('<UNK>')}")
        return True
    except Exception as e:
        print(f"  ✗ Ошибка при чтении словаря: {e}")
        return False


def main():
    """Основная функция проверки."""
    print("=" * 70)
    print("ПРОВЕРКА ФАЙЛОВ ДЛЯ ДЕПЛОЯ НА RAILWAY")
    print("=" * 70)
    print()
    
    # Пути к файлам
    files_to_check = [
        ("models/best_cnn_model.pth", "CNN модель"),
        ("models/best_lstm_model.pth", "LSTM модель"),
        ("vocab/vocab.json", "Словарь (vocab)"),
    ]
    
    all_exist = True
    results = {}
    
    # Проверка файлов
    for filepath, description in files_to_check:
        exists = check_file_exists(filepath, description)
        results[filepath] = exists
        if not exists:
            all_exist = False
    
    print()
    
    # Проверка структуры словаря
    if results.get("vocab/vocab.json"):
        print("Проверка структуры словаря:")
        vocab_valid = check_vocab_structure("vocab/vocab.json")
        if not vocab_valid:
            all_exist = False
    
    print()
    print("=" * 70)
    
    if all_exist:
        print("✓ ВСЕ ФАЙЛЫ НАЙДЕНЫ И ГОТОВЫ К ДЕПЛОЮ!")
        print()
        print("Следующие шаги:")
        print("1. Убедитесь, что все файлы находятся в правильных папках:")
        print("   - models/best_cnn_model.pth")
        print("   - models/best_lstm_model.pth")
        print("   - vocab/vocab.json")
        print()
        print("2. Следуйте инструкциям в RAILWAY_DEPLOYMENT.md для деплоя")
        print()
        print("3. После деплоя на Railway загрузите файлы через:")
        print("   - Railway Dashboard → Files")
        print("   - Или Railway CLI: railway up <file_path>")
    else:
        print("✗ НЕКОТОРЫЕ ФАЙЛЫ ОТСУТСТВУЮТ!")
        print()
        print("Инструкция по получению файлов:")
        print()
        print("1. ЗАПУСТИТЕ НОУТБУКИ В GOOGLE COLAB:")
        print("   - notebooks/cnn_training.ipynb")
        print("   - notebooks/lstm_training.ipynb")
        print()
        print("2. ВЫПОЛНИТЕ ВСЕ ЯЧЕЙКИ В НОУТБУКАХ:")
        print("   - Обучение моделей (может занять время)")
        print("   - Сохранение словаря (vocab.json)")
        print("   - Скачивание моделей (best_*_model.pth)")
        print()
        print("3. СКАЧАЙТЕ ФАЙЛЫ ИЗ COLAB:")
        print("   - Файлы автоматически скачаются в папку Downloads")
        print("   - Или используйте ячейки 'Download Model for Railway Deployment'")
        print()
        print("4. СКОПИРУЙТЕ ФАЙЛЫ В ПРОЕКТ:")
        print("   - best_cnn_model.pth → models/best_cnn_model.pth")
        print("   - best_lstm_model.pth → models/best_lstm_model.pth")
        print("   - vocab.json → vocab/vocab.json")
        print()
        print("5. ЗАПУСТИТЕ ЭТОТ СКРИПТ СНОВА ДЛЯ ПРОВЕРКИ")
    
    print("=" * 70)
    
    return 0 if all_exist else 1


if __name__ == "__main__":
    exit(main())

