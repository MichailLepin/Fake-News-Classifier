"""Предобработка текста для моделей."""

import re
from typing import List, Dict


class TextProcessor:
    """Класс для предобработки текста перед подачей в модель."""
    
    def __init__(self, vocab: Dict[str, int], max_len: int = 256):
        """
        Инициализация процессора текста.
        
        Args:
            vocab: Словарь слово -> индекс
            max_len: Максимальная длина последовательности
        """
        self.vocab = vocab
        self.max_len = max_len
        self.pad_token = vocab.get('<PAD>', 0)
        self.unk_token = vocab.get('<UNK>', 1)
    
    def clean_text(self, text: str) -> str:
        """
        Очистка текста: lowercase, удаление URL, нормализация пробелов.
        
        Args:
            text: Исходный текст
            
        Returns:
            Очищенный текст
        """
        if not text:
            return ""
        
        text = str(text)
        # Приведение к нижнему регистру
        text = text.lower()
        # Удаление URL
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        # Нормализация пробелов
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        return text
    
    def text_to_sequence(self, text: str) -> List[int]:
        """
        Преобразование текста в последовательность индексов.
        
        Args:
            text: Текст для обработки
            
        Returns:
            Список индексов токенов
        """
        # Очистка текста
        cleaned = self.clean_text(text)
        
        # Разбиение на слова
        words = cleaned.split()
        
        # Преобразование в индексы
        sequence = [
            self.vocab.get(word, self.unk_token) 
            for word in words[:self.max_len]
        ]
        
        # Padding до max_len
        if len(sequence) < self.max_len:
            sequence.extend([self.pad_token] * (self.max_len - len(sequence)))
        
        return sequence[:self.max_len]

