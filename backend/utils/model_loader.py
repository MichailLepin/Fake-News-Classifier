"""Загрузка и управление моделями."""

import os
import torch
import numpy as np
from typing import Optional, Dict, Tuple
from pathlib import Path

from ..models.lstm_model import LSTMModel
from ..models.cnn_model import CNNModel
from ..preprocessing.vocab_loader import VocabLoader


class ModelLoader:
    """Класс для загрузки обученных моделей."""
    
    def __init__(
        self,
        models_dir: str = "models",
        vocab_path: str = "vocab/vocab.json",
        glove_path: Optional[str] = None
    ):
        """
        Инициализация загрузчика моделей.
        
        Args:
            models_dir: Директория с сохраненными моделями
            vocab_path: Путь к файлу словаря
            glove_path: Путь к файлу GloVe embeddings (опционально)
        """
        self.models_dir = Path(models_dir)
        self.vocab_path = vocab_path
        self.glove_path = glove_path
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.vocab: Optional[Dict[str, int]] = None
        self.embedding_matrix: Optional[np.ndarray] = None
        
        self.lstm_model: Optional[LSTMModel] = None
        self.cnn_model: Optional[CNNModel] = None
    
    def load_vocab(self) -> Dict[str, int]:
        """Загрузка словаря."""
        if self.vocab is None:
            self.vocab = VocabLoader.load_vocab(self.vocab_path)
        return self.vocab
    
    def load_glove_embeddings(self, vocab: Dict[str, int], embedding_dim: int = 100) -> np.ndarray:
        """
        Загрузка GloVe embeddings.
        
        Args:
            vocab: Словарь слово -> индекс
            embedding_dim: Размерность эмбеддингов
            
        Returns:
            Матрица эмбеддингов
        """
        if self.embedding_matrix is not None:
            return self.embedding_matrix
        
        vocab_size = len(vocab)
        embedding_matrix = np.zeros((vocab_size, embedding_dim))
        
        if self.glove_path and os.path.exists(self.glove_path):
            embeddings_index = {}
            with open(self.glove_path, 'r', encoding='utf-8') as f:
                for line in f:
                    values = line.split()
                    word = values[0]
                    coefs = np.asarray(values[1:], dtype='float32')
                    embeddings_index[word] = coefs
            
            found = 0
            for word, idx in vocab.items():
                if word in embeddings_index:
                    embedding_matrix[idx] = embeddings_index[word]
                    found += 1
                else:
                    # Случайная инициализация для неизвестных слов
                    embedding_matrix[idx] = np.random.normal(scale=0.6, size=(embedding_dim,))
            
            print(f"Found embeddings for {found}/{vocab_size} words ({found/vocab_size*100:.2f}%)")
        else:
            # Случайная инициализация, если GloVe не найден
            print("GloVe embeddings not found, using random initialization")
            for idx in range(vocab_size):
                embedding_matrix[idx] = np.random.normal(scale=0.6, size=(embedding_dim,))
        
        self.embedding_matrix = embedding_matrix
        return embedding_matrix
    
    def load_lstm_model(
        self,
        model_path: Optional[str] = None,
        vocab_size: Optional[int] = None,
        embedding_dim: int = 100,
        hidden_dim: int = 128
    ) -> LSTMModel:
        """
        Загрузка LSTM модели.
        
        Args:
            model_path: Путь к файлу модели (по умолчанию models/best_lstm_model.pth)
            vocab_size: Размер словаря
            embedding_dim: Размерность эмбеддингов
            hidden_dim: Размерность скрытого слоя
            
        Returns:
            Загруженная модель
        """
        if self.lstm_model is not None:
            return self.lstm_model
        
        vocab = self.load_vocab()
        if vocab_size is None:
            vocab_size = len(vocab)
        
        # Загрузка эмбеддингов
        embedding_matrix = self.load_glove_embeddings(vocab, embedding_dim)
        
        # Создание модели
        model = LSTMModel(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            embedding_matrix=embedding_matrix
        ).to(self.device)
        
        # Загрузка весов
        if model_path is None:
            model_path = self.models_dir / "best_lstm_model.pth"
        
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=self.device))
            model.eval()
            print(f"✓ LSTM model loaded from {model_path}")
        else:
            print(f"⚠ Warning: Model file not found at {model_path}, using untrained model")
        
        self.lstm_model = model
        return model
    
    def load_cnn_model(
        self,
        model_path: Optional[str] = None,
        vocab_size: Optional[int] = None,
        embedding_dim: int = 100,
        num_filters: int = 100,
        filter_sizes: list = [3, 4, 5]
    ) -> CNNModel:
        """
        Загрузка CNN модели.
        
        Args:
            model_path: Путь к файлу модели (по умолчанию models/best_cnn_model.pth)
            vocab_size: Размер словаря
            embedding_dim: Размерность эмбеддингов
            num_filters: Количество фильтров
            filter_sizes: Размеры фильтров
            
        Returns:
            Загруженная модель
        """
        if self.cnn_model is not None:
            return self.cnn_model
        
        vocab = self.load_vocab()
        if vocab_size is None:
            vocab_size = len(vocab)
        
        # Загрузка эмбеддингов
        embedding_matrix = self.load_glove_embeddings(vocab, embedding_dim)
        
        # Создание модели
        model = CNNModel(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            num_filters=num_filters,
            filter_sizes=filter_sizes,
            embedding_matrix=embedding_matrix
        ).to(self.device)
        
        # Загрузка весов
        if model_path is None:
            model_path = self.models_dir / "best_cnn_model.pth"
        
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=self.device))
            model.eval()
            print(f"✓ CNN model loaded from {model_path}")
        else:
            print(f"⚠ Warning: Model file not found at {model_path}, using untrained model")
        
        self.cnn_model = model
        return model

