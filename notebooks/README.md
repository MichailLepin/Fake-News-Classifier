# Notebooks Directory

This directory contains Jupyter notebooks and interactive dashboards for data exploration and analysis.

## Files

### `eda_dashboard.html`
Interactive web-based dashboard for Exploratory Data Analysis (EDA) visualization.

**Usage:**
1. First, run the EDA analysis script to generate data:
   ```bash
   python scripts/eda_analysis.py
   ```

2. Open `eda_dashboard.html` in a web browser:
   - Double-click the file, or
   - Right-click → Open with → Web browser
   - Or use a local web server:
     ```bash
     # Python 3
     python -m http.server 8000
     # Then open: http://localhost:8000/notebooks/eda_dashboard.html
     ```

**Features:**
- Label distribution charts
- Text length analysis
- Top words frequency visualization
- Interactive dataset switching (ISOT/Kaggle vs LIAR)
- Responsive design

**Note:** The dashboard requires `reports/data/eda_data.json` to be generated first by running `scripts/eda_analysis.py`.

---

## Baseline Models Training

### `lstm_training.ipynb` и `lstm_training.py`
Полный код для обучения LSTM (Bidirectional LSTM) модели в Google Colab.

**Рекомендуется**: Использовать `.ipynb` файл для прямого открытия в Google Colab.

**Альтернатива**: `.py` файл можно скопировать в ячейки Colab вручную или использовать скрипт конвертации.

**Архитектура:**
- GloVe 100d embeddings
- Bidirectional LSTM (128 units)
- Dropout (0.3)
- Полносвязный слой для классификации

**Использование (.ipynb):**
1. Откройте Google Colab: https://colab.research.google.com/
2. File → Upload notebook → выберите `lstm_training.ipynb`
3. Загрузите данные (см. `COLAB_SETUP.md`)
4. Запустите все ячейки последовательно

**Использование (.py):**
1. Откройте Google Colab: https://colab.research.google.com/
2. Загрузите данные (см. `COLAB_SETUP.md`)
3. Скопируйте содержимое `lstm_training.py` в ячейки Colab
4. Запустите все ячейки последовательно

### `cnn_training.ipynb` и `cnn_training.py`
Полный код для обучения CNN (1D Convolutional) модели в Google Colab.

**Рекомендуется**: Использовать `.ipynb` файл для прямого открытия в Google Colab.

**Альтернатива**: `.py` файл можно скопировать в ячейки Colab вручную или использовать скрипт конвертации.

**Архитектура:**
- GloVe 100d embeddings
- 1D Convolutional layers (filter sizes: 3, 4, 5)
- Max pooling
- Dropout (0.3)
- Полносвязный слой для классификации

**Использование (.ipynb):**
1. Откройте Google Colab: https://colab.research.google.com/
2. File → Upload notebook → выберите `cnn_training.ipynb`
3. Загрузите данные (см. `COLAB_SETUP.md`)
4. Запустите все ячейки последовательно

**Использование (.py):**
1. Откройте Google Colab: https://colab.research.google.com/
2. Загрузите данные (см. `COLAB_SETUP.md`)
3. Скопируйте содержимое `cnn_training.py` в ячейки Colab
4. Запустите все ячейки последовательно

### `COLAB_SETUP.md`
Подробная инструкция по настройке и использованию скриптов обучения в Google Colab.

**Содержит:**
- Инструкции по загрузке данных
- Параметры обучения
- Устранение проблем
- Сохранение моделей

**Параметры обучения (общие для обеих моделей):**
- Batch size: 16
- Max sequence length: 256
- Learning rate: 2e-5
- Early stopping: patience=3 (на основе F1-score)
- Optimizer: Adam
- Loss: CrossEntropyLoss

**Результаты:**
После обучения модели сохраняются в:
- `best_lstm_model.pth`
- `best_cnn_model.pth`

Обе модели оцениваются на тестовом наборе с выводом:
- Accuracy
- F1-Score
- Precision
- Recall
- Confusion Matrix
- Classification Report


