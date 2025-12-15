"""FastAPI приложение для классификации фейковых новостей."""

import os
import time
from typing import Dict, List, Optional, Any
import torch
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .utils.model_loader import ModelLoader
from .preprocessing.text_processor import TextProcessor


# Инициализация FastAPI приложения
app = FastAPI(
    title="Fake News Classifier API",
    description="API для классификации фейковых новостей с использованием LSTM и CNN моделей",
    version="1.0.0"
)

# CORS настройки для фронтенда
# Получаем разрешенные домены из переменных окружения или используем по умолчанию
allowed_origins = os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:8080,http://localhost:3000,http://127.0.0.1:8080"
).split(",")

# Добавляем GitHub Pages домен если указан
github_pages_domain = os.getenv("GITHUB_PAGES_DOMAIN")
if github_pages_domain:
    allowed_origins.append(github_pages_domain)

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Глобальные переменные для моделей и процессора
model_loader: Optional[ModelLoader] = None
text_processor: Optional[TextProcessor] = None


# Модели запросов/ответов
class PredictionRequest(BaseModel):
    """Запрос на предсказание."""
    text: str


class PredictionResponse(BaseModel):
    """Ответ с предсказанием."""
    label: str  # "fake" или "real"
    confidence: float  # Уверенность в предсказании (0-1)
    fake_score: float  # Вероятность фейковой новости (0-1)
    real_score: float  # Вероятность реальной новости (0-1)
    inference_time_ms: float  # Время инференса в миллисекундах


class ModelComparisonResponse(BaseModel):
    """Ответ со сравнением моделей."""
    predictions: List[Dict[str, Any]]
    average_confidence: float
    consensus_label: str


@app.on_event("startup")
async def startup_event():
    """Инициализация моделей при запуске приложения."""
    global model_loader, text_processor
    
    print("=" * 60)
    print("Initializing Fake News Classifier Backend...")
    print("=" * 60)
    
    # Определение путей (Railway использует переменные окружения)
    models_dir = os.getenv("MODELS_DIR", "models")
    vocab_path = os.getenv("VOCAB_PATH", "vocab/vocab.json")
    glove_path = os.getenv("GLOVE_PATH", None)
    
    print(f"Models directory: {models_dir}")
    print(f"Vocab path: {vocab_path}")
    print(f"GLOVE path: {glove_path if glove_path else 'Not specified (using random init)'}")
    
    # Инициализация загрузчика моделей
    model_loader = ModelLoader(
        models_dir=models_dir,
        vocab_path=vocab_path,
        glove_path=glove_path
    )
    
    # Загрузка словаря
    vocab = model_loader.load_vocab()
    
    # Инициализация процессора текста
    text_processor = TextProcessor(vocab=vocab, max_len=256)
    
    # Предзагрузка моделей (опционально, можно загружать по требованию)
    try:
        model_loader.load_lstm_model()
        print("✓ LSTM model ready")
    except Exception as e:
        print(f"⚠ Could not load LSTM model: {e}")
    
    try:
        model_loader.load_cnn_model()
        print("✓ CNN model ready")
    except Exception as e:
        print(f"⚠ Could not load CNN model: {e}")
    
    print("✓ Models initialized")


@app.get("/api/health")
async def health_check():
    """Проверка статуса API."""
    return {
        "status": "healthy",
        "models_loaded": {
            "lstm": model_loader.lstm_model is not None if model_loader else False,
            "cnn": model_loader.cnn_model is not None if model_loader else False
        }
    }


def predict_with_model(
    model: torch.nn.Module,
    text: str,
    device: torch.device
) -> Dict[str, Any]:
    """
    Выполнение предсказания с моделью.
    
    Args:
        model: PyTorch модель
        text: Текст для классификации
        device: Устройство (CPU/GPU)
        
    Returns:
        Словарь с результатами предсказания
    """
    start_time = time.time()
    
    # Предобработка текста
    sequence = text_processor.text_to_sequence(text)
    
    # Преобразование в тензор
    input_tensor = torch.LongTensor([sequence]).to(device)
    
    # Предсказание
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
    
    # Извлечение вероятностей
    fake_score = probabilities[0][1].item()  # Индекс 1 = fake
    real_score = probabilities[0][0].item()   # Индекс 0 = real
    
    label = "fake" if predicted_class == 1 else "real"
    confidence = max(fake_score, real_score)
    
    inference_time = (time.time() - start_time) * 1000  # в миллисекундах
    
    return {
        "label": label,
        "confidence": confidence,
        "fake_score": fake_score,
        "real_score": real_score,
        "inference_time_ms": inference_time
    }


@app.post("/api/predict/lstm", response_model=PredictionResponse)
async def predict_lstm(request: PredictionRequest):
    """Предсказание с использованием LSTM модели."""
    if model_loader is None or model_loader.lstm_model is None:
        raise HTTPException(status_code=503, detail="LSTM model not loaded")
    
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    try:
        result = predict_with_model(
            model_loader.lstm_model,
            request.text,
            model_loader.device
        )
        return PredictionResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/api/predict/cnn", response_model=PredictionResponse)
async def predict_cnn(request: PredictionRequest):
    """Предсказание с использованием CNN модели."""
    if model_loader is None or model_loader.cnn_model is None:
        raise HTTPException(status_code=503, detail="CNN model not loaded")
    
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    try:
        result = predict_with_model(
            model_loader.cnn_model,
            request.text,
            model_loader.device
        )
        return PredictionResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/api/predict/all", response_model=ModelComparisonResponse)
async def predict_all(request: PredictionRequest):
    """Сравнение предсказаний всех доступных моделей."""
    if model_loader is None:
        raise HTTPException(status_code=503, detail="Models not initialized")
    
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    predictions = []
    
    # LSTM предсказание
    if model_loader.lstm_model is not None:
        try:
            lstm_result = predict_with_model(
                model_loader.lstm_model,
                request.text,
                model_loader.device
            )
            predictions.append({
                "model": "LSTM Baseline",
                "model_id": "lstm",
                **lstm_result
            })
        except Exception as e:
            print(f"Error predicting with LSTM: {e}")
    
    # CNN предсказание
    if model_loader.cnn_model is not None:
        try:
            cnn_result = predict_with_model(
                model_loader.cnn_model,
                request.text,
                model_loader.device
            )
            predictions.append({
                "model": "CNN-Text Baseline",
                "model_id": "cnn",
                **cnn_result
            })
        except Exception as e:
            print(f"Error predicting with CNN: {e}")
    
    if not predictions:
        raise HTTPException(status_code=503, detail="No models available")
    
    # Вычисление средних значений
    avg_confidence = sum(p["confidence"] for p in predictions) / len(predictions)
    avg_fake_score = sum(p["fake_score"] for p in predictions) / len(predictions)
    consensus_label = "fake" if avg_fake_score > 0.5 else "real"
    
    return ModelComparisonResponse(
        predictions=predictions,
        average_confidence=avg_confidence,
        consensus_label=consensus_label
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

