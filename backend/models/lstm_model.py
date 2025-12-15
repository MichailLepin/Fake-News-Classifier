"""LSTM модель для классификации фейковых новостей."""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional


class LSTMModel(nn.Module):
    """Bidirectional LSTM модель для классификации текста."""
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 100,
        hidden_dim: int = 128,
        num_layers: int = 1,
        dropout: float = 0.3,
        num_classes: int = 2,
        embedding_matrix: Optional[np.ndarray] = None
    ):
        """
        Инициализация LSTM модели.
        
        Args:
            vocab_size: Размер словаря
            embedding_dim: Размерность эмбеддингов
            hidden_dim: Размерность скрытого слоя LSTM
            num_layers: Количество слоев LSTM
            dropout: Dropout вероятность
            num_classes: Количество классов (2: fake/real)
            embedding_matrix: Матрица предобученных эмбеддингов (GloVe)
        """
        super(LSTMModel, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Инициализация предобученными эмбеддингами
        if embedding_matrix is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(embedding_matrix))
            self.embedding.weight.requires_grad = True
        
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Прямой проход через модель.
        
        Args:
            x: Входной тензор формы (batch_size, seq_len)
            
        Returns:
            Выходной тензор формы (batch_size, num_classes)
        """
        embedded = self.embedding(x)
        lstm_out, (hidden, cell) = self.lstm(embedded)
        
        # Объединение прямого и обратного скрытых состояний
        output = torch.cat((hidden[-2], hidden[-1]), dim=1)
        output = self.dropout(output)
        output = self.fc(output)
        
        return output

