# Инструкция по использованию скриптов обучения в Google Colab

## Подготовка

1. **Откройте Google Colab**: https://colab.research.google.com/

2. **Создайте новый ноутбук** или используйте существующий

3. **Загрузите данные** одним из способов:
   - **Вариант A**: Загрузите файлы вручную через `Files -> Upload`:
     - `data/processed/isot_processed.csv`
     - `data/processed/liar_train_processed.csv`
     - `data/processed/liar_test_processed.csv` (опционально)
     - `data/processed/liar_valid_processed.csv` (опционально)
   
   - **Вариант B**: Клонируйте репозиторий:
     ```python
     !git clone https://github.com/YOUR_USERNAME/Fake-News-Classifier.git
     %cd Fake-News-Classifier
     ```

## Обучение LSTM модели

1. **Скопируйте содержимое** файла `notebooks/lstm_training.py` в ячейки Colab

2. **Или загрузите файл напрямую**:
   ```python
   # В первой ячейке Colab
   !wget https://raw.githubusercontent.com/YOUR_USERNAME/Fake-News-Classifier/main/notebooks/lstm_training.py
   %run lstm_training.py
   ```

3. **Измените URL репозитория** в скрипте:
   ```python
   REPO_URL = "https://raw.githubusercontent.com/YOUR_USERNAME/Fake-News-Classifier/main"
   ```
   Замените `YOUR_USERNAME` на ваш GitHub username

4. **Запустите все ячейки** последовательно

## Обучение CNN модели

1. **Скопируйте содержимое** файла `notebooks/cnn_training.py` в ячейки Colab

2. **Или загрузите файл напрямую**:
   ```python
   !wget https://raw.githubusercontent.com/YOUR_USERNAME/Fake-News-Classifier/main/notebooks/cnn_training.py
   %run cnn_training.py
   ```

3. **Измените URL репозитория** аналогично LSTM

4. **Запустите все ячейки** последовательно

## Структура ноутбука

Рекомендуемая структура ячеек в Colab:

### Ячейка 1: Установка зависимостей
```python
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
!pip install transformers scikit-learn pandas numpy matplotlib seaborn tqdm
!pip install gensim
```

### Ячейка 2: Импорты и проверка GPU
```python
# Импорты из скрипта
# Проверка device
```

### Ячейка 3: Загрузка данных
```python
# Код загрузки данных
```

### Ячейка 4: Подготовка данных
```python
# Создание словаря, токенизация
# Загрузка GloVe
```

### Ячейка 5: Определение модели
```python
# Класс модели (LSTM или CNN)
# Создание экземпляра модели
```

### Ячейка 6: Обучение
```python
# Функции train_epoch и evaluate
# Цикл обучения
```

### Ячейка 7: Оценка
```python
# Оценка на тестовом наборе
# Визуализация результатов
```

## Параметры обучения

Обе модели используют одинаковые параметры:

- **Batch size**: 16
- **Max sequence length**: 256
- **Learning rate**: 2e-5
- **Embedding dimension**: 100 (GloVe)
- **Early stopping**: patience=3 на основе F1-score валидации
- **Optimizer**: Adam
- **Loss**: CrossEntropyLoss

### LSTM специфичные параметры:
- **Hidden dimension**: 128
- **Layers**: 1 (bidirectional)
- **Dropout**: 0.3

### CNN специфичные параметры:
- **Number of filters**: 100
- **Filter sizes**: [3, 4, 5]
- **Dropout**: 0.3

## Сохранение моделей

Модели автоматически сохраняются в файлы:
- `best_lstm_model.pth` - лучшая LSTM модель
- `best_cnn_model.pth` - лучшая CNN модель

Для скачивания моделей из Colab:
```python
from google.colab import files
files.download('best_lstm_model.pth')
files.download('best_cnn_model.pth')
```

## Сравнение моделей

После обучения обеих моделей можно сравнить результаты:

```python
import pandas as pd

comparison_df = pd.DataFrame({
    'Model': ['LSTM', 'CNN'],
    'Accuracy': [lstm_results['test_accuracy'], cnn_results['test_accuracy']],
    'F1-Score': [lstm_results['test_f1'], cnn_results['test_f1']],
    'Precision': [lstm_results['test_precision'], cnn_results['test_precision']],
    'Recall': [lstm_results['test_recall'], cnn_results['test_recall']]
})

print(comparison_df)
```

## Устранение проблем

### Проблема: Не хватает памяти GPU
**Решение**: Уменьшите batch_size или max_len

### Проблема: Медленное обучение
**Решение**: 
- Убедитесь, что используете GPU (Runtime -> Change runtime type -> GPU)
- Уменьшите размер датасета для тестирования

### Проблема: GloVe не загружается
**Решение**: Скрипт автоматически скачает GloVe, но если есть проблемы:
```python
# Загрузите вручную
!wget http://nlp.stanford.edu/data/glove.6B.zip
!unzip glove.6B.zip
```

### Проблема: Данные не загружаются из GitHub
**Решение**: Загрузите файлы вручную через `Files -> Upload` в Colab

## Дополнительные ресурсы

- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [GloVe Embeddings](https://nlp.stanford.edu/projects/glove/)
- [Google Colab Guide](https://colab.research.google.com/notebooks/intro.ipynb)

