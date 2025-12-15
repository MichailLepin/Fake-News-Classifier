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

- **Backend**: FastAPI deployed on Railway.app
- **Frontend**: GitHub Pages (static HTML/JS)

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
├── backend/                   # FastAPI backend application
│   ├── main.py               # Main FastAPI app
│   ├── models/               # Model definitions (LSTM, CNN)
│   ├── preprocessing/        # Text preprocessing
│   └── utils/                # Utilities (model loading)
├── scripts/                   # Standalone scripts
├── notebooks/                 # Jupyter notebooks for EDA and training
├── models/                    # Saved trained models (upload to Railway)
├── vocab/                     # Vocabulary files (upload to Railway)
├── reports/                   # Analysis reports
├── docs/                      # GitHub Pages documentation and EDA dashboard
│   ├── index.html             # EDA Dashboard + Model Demo (published on GitHub Pages)
│   └── data/                  # Data files for the dashboard
├── requirements.txt           # Python dependencies
├── Procfile                  # Railway deployment command
├── railway.json              # Railway configuration
├── runtime.txt                # Python version
└── README.md
```

## Deployment

### Backend Deployment (Railway.app)

1. **Prepare models and vocab:**
   - Train models in `notebooks/lstm_training.ipynb` and `notebooks/cnn_training.ipynb`
   - Save vocab using Cell 10-11 in notebooks
   - Download `best_lstm_model.pth`, `best_cnn_model.pth`, and `vocab.json`

2. **Deploy to Railway:**
   - Sign up at https://railway.app
   - Create new project → Deploy from GitHub repo
   - Select this repository
   - Railway will automatically detect Python and install dependencies
   - Add environment variables:
     ```
     MODELS_DIR=models
     VOCAB_PATH=vocab/vocab.json
     ALLOWED_ORIGINS=https://YOUR_USERNAME.github.io
     GITHUB_PAGES_DOMAIN=https://YOUR_USERNAME.github.io
     ```
   - Upload model files (`best_lstm_model.pth`, `best_cnn_model.pth`) and `vocab.json` via Railway dashboard or CLI
   - Railway will provide a URL like `https://your-app.up.railway.app`

3. **Update frontend:**
   - Edit `docs/index.html`
   - Find `getAPIBaseURL()` function (around line 803)
   - Replace `YOUR_BACKEND_URL.com` with your Railway URL
   - Commit and push changes

**Подробные инструкции:** См. `RAILWAY_DEPLOYMENT.md`

### Frontend Deployment (GitHub Pages)

The frontend is automatically deployed via GitHub Pages from the `docs/` directory.

## Local Development

### Backend

```bash
# Install dependencies
pip install -r requirements.txt

# Run backend
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

### Frontend

Open `docs/index.html` in a browser or use a local server:

```bash
cd docs
python -m http.server 8080
```

## License

MIT License

## Authors

Fake News Classification Project Team

