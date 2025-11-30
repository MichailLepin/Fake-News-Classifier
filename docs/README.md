# EDA Dashboard - GitHub Pages

Этот дашборд с результатами EDA (Exploratory Data Analysis) настроен для публикации на GitHub Pages.

## Настройка GitHub Pages

Чтобы опубликовать этот дашборд на GitHub Pages:

1. **Убедитесь, что файлы находятся в папке `docs/`**:
   - `docs/index.html` - главный HTML файл дашборда
   - `docs/data/eda_data.json` - данные для визуализации

2. **Настройте GitHub Pages в настройках репозитория**:
   - Перейдите в Settings → Pages
   - В разделе "Source" выберите:
     - Branch: `main` (или ваша основная ветка)
     - Folder: `/docs`
   - Нажмите "Save"

3. **Дождитесь публикации**:
   - GitHub автоматически опубликует сайт через несколько минут
   - Ссылка будет доступна в формате: `https://[ваш-username].github.io/[название-репозитория]/`

## Обновление данных

Если вы обновили данные EDA:

1. Запустите скрипт анализа:
   ```bash
   python scripts/eda_analysis.py
   ```

2. Скопируйте обновленный файл данных:
   ```bash
   copy reports\data\eda_data.json docs\data\eda_data.json
   ```

3. Закоммитьте и запушьте изменения:
   ```bash
   git add docs/data/eda_data.json
   git commit -m "Update EDA data"
   git push
   ```

GitHub Pages автоматически обновится после пуша.

## Локальный просмотр

Для локального просмотра дашборда:

1. Откройте `docs/index.html` в браузере
2. Или используйте локальный сервер:
   ```bash
   cd docs
   python -m http.server 8000
   ```
   Затем откройте `http://localhost:8000` в браузере

