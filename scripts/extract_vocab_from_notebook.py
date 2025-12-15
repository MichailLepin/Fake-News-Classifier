"""Скрипт для извлечения и сохранения vocab из ноутбуков.

Этот скрипт можно запустить в ноутбуке после обучения для сохранения vocab.
"""

import json
from collections import Counter


def save_vocab_from_notebook(vocab, output_path="vocab/vocab.json"):
    """
    Сохранение vocab в JSON файл.
    
    Использование в ноутбуке:
        from scripts.extract_vocab_from_notebook import save_vocab_from_notebook
        save_vocab_from_notebook(vocab)
    
    Args:
        vocab: Словарь слово -> индекс
        output_path: Путь для сохранения
    """
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)
    
    print(f"✓ Vocab сохранен в {output_path}")
    print(f"  Размер словаря: {len(vocab)}")


# Пример использования в ноутбуке:
"""
# После построения vocab в ноутбуке:
import sys
sys.path.append('..')
from scripts.extract_vocab_from_notebook import save_vocab_from_notebook

# Сохранение vocab
save_vocab_from_notebook(vocab, output_path="../vocab/vocab.json")
"""

