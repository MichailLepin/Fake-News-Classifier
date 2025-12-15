"""Модели для классификации фейковых новостей."""

from .lstm_model import LSTMModel
from .cnn_model import CNNModel

__all__ = ['LSTMModel', 'CNNModel']

