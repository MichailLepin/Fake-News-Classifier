"""CNN модель для классификации фейковых новостей."""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, List


class CNNModel(nn.Module):
    """CNN модель с несколькими размерами фильтров для классификации текста."""
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 100,
        num_filters: int = 100,
        filter_sizes: List[int] = [3, 4, 5],
        num_classes: int = 2,
        dropout: float = 0.3,
        embedding_matrix: Optional[np.ndarray] = None
    ):
        """
        Инициализация CNN модели.
        
        Args:
            vocab_size: Размер словаря
            embedding_dim: Размерность эмбеддингов
            num_filters: Количество фильтров для каждого размера
            filter_sizes: Список размеров фильтров
            num_classes: Количество классов (2: fake/real)
            dropout: Dropout вероятность
            embedding_matrix: Матрица предобученных эмбеддингов (GloVe)
        """
        super(CNNModel, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Инициализация предобученными эмбеддингами
        if embedding_matrix is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(embedding_matrix))
            self.embedding.weight.requires_grad = True
        
        # Сверточные слои с разными размерами фильтров
        self.convs = nn.ModuleList([
            nn.Conv1d(embedding_dim, num_filters, kernel_size=fs)
            for fs in filter_sizes
        ])
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(len(filter_sizes) * num_filters, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Прямой проход через модель.
        
        Args:
            x: Входной тензор формы (batch_size, seq_len)
            
        Returns:
            Выходной тензор формы (batch_size, num_classes)
        """
        # x shape: (batch_size, seq_len)
        embedded = self.embedding(x)  # (batch_size, seq_len, embedding_dim)
        
        # Conv1d ожидает (batch_size, embedding_dim, seq_len)
        embedded = embedded.permute(0, 2, 1)  # (batch_size, embedding_dim, seq_len)
        
        # Применение сверток и max pooling
        conv_outputs = []
        for conv in self.convs:
            conv_out = conv(embedded)  # (batch_size, num_filters, seq_len - filter_size + 1)
            conv_out = torch.relu(conv_out)
            pooled = torch.max_pool1d(conv_out, kernel_size=conv_out.size(2))  # (batch_size, num_filters, 1)
            pooled = pooled.squeeze(2)  # (batch_size, num_filters)
            conv_outputs.append(pooled)
        
        # Объединение выходов всех сверток
        concatenated = torch.cat(conv_outputs, dim=1)  # (batch_size, len(filter_sizes) * num_filters)
        
        concatenated = self.dropout(concatenated)
        output = self.fc(concatenated)  # (batch_size, num_classes)
        
        return output

