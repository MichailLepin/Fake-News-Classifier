# Capstone 3: Интеграция реальных моделей

## Быстрый старт

### 1. Подготовка моделей и vocab

Перед запуском бэкенда необходимо:

1. **Обучить модели** (если еще не обучены):
   - Запустите `notebooks/lstm_training.ipynb`
   - Запустите `notebooks/cnn_training.ipynb`
   - Модели сохранятся как `best_lstm_model.pth` и `best_cnn_model.pth`

2. **Сохранить vocab** из ноутбука:
   ```python
   # В ноутбуке после построения vocab:
   import json
   import os
   
   os.makedirs('vocab', exist_ok=True)
   with open('vocab/vocab.json', 'w', encoding='utf-8') as f:
       json.dump(vocab, f, ensure_ascii=False, indent=2)
   ```

3. **Скопировать файлы**:
   - Скопируйте `best_lstm_model.pth` и `best_cnn_model.pth` в папку `models/`
   - Скопируйте `vocab.json` в папку `vocab/`
   - (Опционально) Скопируйте `glove.6B.100d.txt` в папку `vocab/` для предобученных эмбеддингов

### 2. Установка зависимостей

```bash
pip install -r requirements.txt
```

### 3. Запуск бэкенда

```bash
# Вариант 1: Через uvicorn напрямую
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000

# Вариант 2: Через Python модуль
python -m backend.main
```

Бэкенд будет доступен по адресу: `http://localhost:8000`

### 4. Открытие фронтенда

Откройте `docs/index.html` в браузере или запустите локальный сервер:

```bash
# Python
python -m http.server 8080 --directory docs

# Или просто откройте файл в браузере
```

Фронтенд автоматически определит доступность API и переключится между реальными моделями и симуляцией.

## Структура проекта

```
.
├── backend/                    # FastAPI бэкенд
│   ├── main.py                # Главное приложение
│   ├── models/                # Определения моделей
│   │   ├── lstm_model.py
│   │   └── cnn_model.py
│   ├── preprocessing/         # Предобработка текста
│   │   ├── text_processor.py
│   │   └── vocab_loader.py
│   └── utils/                 # Утилиты
│       └── model_loader.py
├── models/                    # Сохраненные модели
│   ├── best_lstm_model.pth
│   └── best_cnn_model.pth
├── vocab/                     # Vocab и embeddings
│   ├── vocab.json
│   └── glove.6B.100d.txt (опционально)
└── docs/
    └── index.html             # Фронтенд с интеграцией API
```

## API Endpoints

### `GET /api/health`
Проверка статуса API и загруженных моделей.

**Ответ:**
```json
{
  "status": "healthy",
  "models_loaded": {
    "lstm": true,
    "cnn": true
  }
}
```

### `POST /api/predict/lstm`
Предсказание с использованием LSTM модели.

**Запрос:**
```json
{
  "text": "Your news text here..."
}
```

**Ответ:**
```json
{
  "label": "fake",
  "confidence": 0.95,
  "fake_score": 0.95,
  "real_score": 0.05,
  "inference_time_ms": 45.2
}
```

### `POST /api/predict/cnn`
Предсказание с использованием CNN модели.

**Запрос и ответ:** аналогично LSTM.

### `POST /api/predict/all`
Сравнение предсказаний всех доступных моделей.

**Ответ:**
```json
{
  "predictions": [
    {
      "model": "LSTM Baseline",
      "model_id": "lstm",
      "label": "fake",
      "confidence": 0.92,
      "fake_score": 0.92,
      "real_score": 0.08,
      "inference_time_ms": 42.1
    },
    {
      "model": "CNN-Text Baseline",
      "model_id": "cnn",
      "label": "fake",
      "confidence": 0.94,
      "fake_score": 0.94,
      "real_score": 0.06,
      "inference_time_ms": 38.5
    }
  ],
  "average_confidence": 0.93,
  "consensus_label": "fake"
}
```

## Переменные окружения

Можно настроить через переменные окружения:

```bash
export MODELS_DIR=models          # Директория с моделями
export VOCAB_PATH=vocab/vocab.json # Путь к vocab
export GLOVE_PATH=vocab/glove.6B.100d.txt  # Путь к GloVe (опционально)
```

## Устранение проблем

### Модели не загружаются

1. Проверьте, что файлы моделей существуют в `models/`
2. Проверьте, что vocab.json существует в `vocab/`
3. Проверьте логи бэкенда на наличие ошибок

### API недоступен

1. Убедитесь, что бэкенд запущен на порту 8000
2. Проверьте CORS настройки в `backend/main.py`
3. Проверьте консоль браузера на ошибки CORS

### Предсказания не работают

1. Проверьте формат входного текста (не должен быть пустым)
2. Проверьте логи бэкенда на ошибки предобработки
3. Убедитесь, что vocab соответствует обученным моделям

## Следующие шаги

- [ ] Интеграция BERT и DistilBERT моделей
- [ ] Оптимизация через ONNX для браузера
- [ ] Добавление батч-обработки
- [ ] Кэширование предсказаний
- [ ] Мониторинг и логирование

