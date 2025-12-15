"""Загрузка словаря (vocab) для моделей."""

import json
import os
from typing import Dict, Optional


class VocabLoader:
    """Класс для загрузки словаря из файла."""
    
    @staticmethod
    def load_vocab(vocab_path: str) -> Dict[str, int]:
        """
        Загрузка словаря из JSON файла.
        
        Args:
            vocab_path: Путь к файлу словаря
            
        Returns:
            Словарь слово -> индекс
            
        Raises:
            FileNotFoundError: Если файл не найден
            json.JSONDecodeError: Если файл невалидный JSON
        """
        if not os.path.exists(vocab_path):
            raise FileNotFoundError(f"Vocab file not found: {vocab_path}")
        
        with open(vocab_path, 'r', encoding='utf-8') as f:
            vocab = json.load(f)
        
        return vocab
    
    @staticmethod
    def save_vocab(vocab: Dict[str, int], vocab_path: str):
        """
        Сохранение словаря в JSON файл.
        
        Args:
            vocab: Словарь для сохранения
            vocab_path: Путь для сохранения
        """
        os.makedirs(os.path.dirname(vocab_path), exist_ok=True)
        
        with open(vocab_path, 'w', encoding='utf-8') as f:
            json.dump(vocab, f, ensure_ascii=False, indent=2)

