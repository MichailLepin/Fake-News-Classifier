# Fake News Classifier

## Project Description

This project is an automated news classification system that distinguishes between fake and real news using Natural Language Processing (NLP) methods. The system is designed to assist journalists, fact-checkers, content moderators, and researchers in combating the spread of misinformation.

## Problem Definition

Online misinformation spreads rapidly, influencing public perception in politics, health, and global affairs. We aim to develop an NLP classifier that automatically distinguishes fake vs. real news using headline/article text and meta-information.

### Target Users

- **Journalists and fact-checkers** — prioritize suspicious content
- **Content moderators** — integrate into workflows
- **Researchers** — analyze misinformation trends
- **End-users** — benefit from cleaner information

### Project Value

- Reduces reputational risk for media platforms
- Improves fact-checking throughput
- Provides benchmark datasets for research

## Data & EDA Plan

### Datasets

1. **LIAR Dataset** — 12,800 political statements with 6-tier labels (pants-fire to true), speaker metadata, and historical credibility
2. **ISOT/Kaggle** — 25,000 news articles (True.csv, Fake.csv) with title, full text, subject, and date

### Data Integration

- Normalize labels to binary (fake/real)
- Merge into unified schema
- Clean text (lowercase, remove URLs/punctuation)
- Tokenize with BERT (256–512 tokens)

### EDA (Exploratory Data Analysis) Plan

- Label distribution
- Text length comparison
- Top n-grams (TF-IDF)
- Word clouds
- Subject/party bias analysis
- Temporal trends
- t-SNE embedding visualization
- Sentiment polarity

## Core Features & UI/UX

### Workflow

1. User inputs text
2. Model predicts "Fake" or "Real" with confidence score
3. UI shows color-coded result + attention highlights

### Interface

- Text input box
- "Classify" button
- Prediction display with confidence bar
- Attention weight visualization (explainability)
- Disclaimer about limitations

## Milestones

### Checkpoint 1: Model & Data Ready

- Cleaned/tokenized dataset
- Baseline models (LSTM, CNN) trained and evaluated
- EDA report completed

### Checkpoint 2: Core Functionality Integrated

- Working web demo (frontend + backend)
- All 4 models compared
- Explainability visualization implemented
- Documentation finalized

## Modeling & Deployment Strategy

### Training Environment

- **Platform**: Google Colab Pro (GPU)
- **Frameworks**: PyTorch + Hugging Face Transformers

### Deployment Target

- FastAPI or TensorFlow.js for browser-based inference

### Models

1. **LSTM Baseline**
   - GloVe 100d + BiLSTM (128 units)
   - Fast training, interpretable

2. **CNN-Text Baseline**
   - 1D convolutions
   - Captures local n-grams

3. **BERT-base-uncased**
   - 12-layer Transformer
   - Handles context/sarcasm

4. **DistilBERT**
   - 40% smaller than BERT
   - Similar accuracy, faster inference (ideal for deployment)

### Training Parameters

- Batch size: 16
- Max length: 256
- Learning rate: 2e-5
- Early stopping on validation F1
- Cross-validation to reduce bias

## Success Metrics & Trade-offs

### Primary Metrics

- **Primary**: F1-score (Fake class)
- **Secondary**: Accuracy, Precision, Recall, ROC-AUC

### Priorities

- **Precision > Recall** (minimize false accusations of legitimate news)

### Evaluation

- Compare all 4 models on test set (table with metrics)
- Record inference latency (Colab GPU vs. browser)
- Generate confusion matrices + analyze misclassified examples

### Trade-offs

- High F1 vs. fast inference
- Smaller model size vs. maximum accuracy

## Installation & Usage

### Requirements

```bash
pip install -r requirements.txt
```

### Running Scripts

```bash
# Data analysis and integration
python scripts/analyze_and_integrate.py

# Verify processed data
python scripts/verify_processed_data.py

# Generate EDA dashboard
python scripts/eda_analysis.py
```

## EDA Dashboard (GitHub Pages)

Интерактивный дашборд с результатами EDA доступен на GitHub Pages:

- **Онлайн**: После настройки GitHub Pages дашборд будет доступен по адресу: `https://[ваш-username].github.io/[название-репозитория]/`
- **Локально**: Откройте `docs/index.html` в браузере

### Настройка GitHub Pages

1. Перейдите в **Settings → Pages** вашего репозитория
2. В разделе "Source" выберите:
   - Branch: `main` (или ваша основная ветка)
   - Folder: `/docs`
3. Нажмите **Save**

Подробные инструкции см. в [docs/README.md](docs/README.md)

### Project Structure

For detailed information about the project structure, see [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md).

## Project Structure

```
.
├── data/                      # Data directory
│   ├── raw/                   # Original, unprocessed datasets
│   │   ├── isot_kaggle/       # ISOT/Kaggle dataset
│   │   └── liar/              # LIAR dataset
│   └── processed/             # Processed and cleaned data
├── src/                       # Source code
│   ├── data/                  # Data processing modules
│   ├── models/                # Model definitions
│   └── utils/                 # Utility functions
├── scripts/                   # Standalone scripts
├── notebooks/                 # Jupyter notebooks for EDA and training
├── models/                    # Saved trained models
├── reports/                    # Analysis reports
├── docs/                      # GitHub Pages documentation and EDA dashboard
│   ├── index.html             # EDA Dashboard (published on GitHub Pages)
│   └── data/                  # Data files for the dashboard
├── requirements.txt           # Python dependencies
└── README.md
```

For detailed structure information, see [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md).

## License

MIT License

## Authors

Fake News Classification Project Team
