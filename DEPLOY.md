# 🚀 Quick Deployment Guide

## Railway.app Deployment

### Шаг 1: Подготовка
1. Обучите модели в ноутбуках (`notebooks/lstm_training.ipynb`, `notebooks/cnn_training.ipynb`)
2. Сохраните vocab используя Cell 10-11 в ноутбуках
3. Скачайте файлы: `best_lstm_model.pth`, `best_cnn_model.pth`, `vocab.json`

### Шаг 2: Деплой на Railway
1. Зайдите на https://railway.app → New Project → Deploy from GitHub
2. Выберите ваш репозиторий
3. Railway автоматически определит и развернет приложение
4. Добавьте переменные окружения:
   ```
   MODELS_DIR=models
   VOCAB_PATH=vocab/vocab.json
   ALLOWED_ORIGINS=http://localhost:8080,https://YOUR_USERNAME.github.io
   GITHUB_PAGES_DOMAIN=https://YOUR_USERNAME.github.io
   ```
5. Загрузите файлы моделей через Railway Dashboard или CLI
6. Скопируйте URL вашего приложения (например: `https://your-app.up.railway.app`)

### Шаг 3: Обновление фронтенда
1. Откройте `docs/index.html`
2. Найдите `getAPIBaseURL()` (строка ~803)
3. Замените `YOUR_BACKEND_URL.com` на ваш Railway URL
4. Commit и push

**Подробности:** См. `RAILWAY_DEPLOYMENT.md`

