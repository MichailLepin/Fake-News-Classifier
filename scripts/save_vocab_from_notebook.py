"""Скрипт для сохранения vocab из ноутбуков обучения.

Этот скрипт помогает извлечь vocab из ноутбуков и сохранить его в JSON формате
для использования в бэкенде.
"""

import json
import sys
from pathlib import Path
from collections import Counter


def build_vocab_from_texts(texts, min_freq=2):
    """
    Построение словаря из текстов.
    
    Args:
        texts: Список текстов
        min_freq: Минимальная частота слова для включения в словарь
        
    Returns:
        Словарь слово -> индекс
    """
    word_counts = Counter()
    for text in texts:
        words = str(text).lower().split()
        word_counts.update(words)
    
    vocab = {'<PAD>': 0, '<UNK>': 1}
    idx = 2
    
    for word, count in word_counts.items():
        if count >= min_freq:
            vocab[word] = idx
            idx += 1
    
    return vocab


def main():
    """Основная функция."""
    print("=" * 60)
    print("Vocab Saver from Notebooks")
    print("=" * 60)
    print("\nЭтот скрипт создает vocab.json на основе структуры из ноутбуков.")
    print("Для полного vocab нужно запустить ноутбук и сохранить vocab.")
    print("\nСоздаю базовый vocab...")
    
    # Базовый vocab с минимальными токенами
    # В реальности vocab должен быть сохранен из ноутбука после обучения
    vocab = {
        '<PAD>': 0,
        '<UNK>': 1
    }
    
    # Создание директории vocab если не существует
    vocab_dir = Path("vocab")
    vocab_dir.mkdir(exist_ok=True)
    
    vocab_path = vocab_dir / "vocab.json"
    
    # Сохранение vocab
    with open(vocab_path, 'w', encoding='utf-8') as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)
    
    print(f"✓ Базовый vocab сохранен в {vocab_path}")
    print("\n⚠ ВАЖНО: Для работы моделей нужно:")
    print("  1. Запустить ноутбук обучения (lstm_training.ipynb или cnn_training.ipynb)")
    print("  2. Сохранить vocab после построения:")
    print("     import json")
    print("     with open('vocab/vocab.json', 'w') as f:")
    print("         json.dump(vocab, f)")
    print("  3. Скопировать vocab.json в корень проекта")
    print("\nИли использовать скрипт из ноутбука для автоматического сохранения.")


if __name__ == "__main__":
    main()

