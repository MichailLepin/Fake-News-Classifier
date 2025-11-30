# Project Structure

This document describes the standard project structure for the Fake News Classification project.

```
.
├── data/                      # Data directory
│   ├── raw/                   # Original, unprocessed data
│   │   ├── isot_kaggle/       # ISOT/Kaggle dataset (original files)
│   │   │   ├── Fake.csv
│   │   │   ├── Fake.csv.zip
│   │   │   ├── True.csv
│   │   │   └── True.csv.zip
│   │   └── liar/              # LIAR dataset (original files)
│   │       ├── train.tsv
│   │       ├── train.tsv.zip
│   │       ├── test.tsv
│   │       ├── valid.tsv
│   │       └── README.txt
│   └── processed/             # Processed and cleaned data
│       ├── isot_processed.csv
│       ├── liar_all_processed.csv
│       ├── liar_train_processed.csv
│       ├── liar_test_processed.csv
│       ├── liar_valid_processed.csv
│       └── analysis_summary.json
│
├── src/                       # Source code
│   ├── __init__.py
│   ├── data/                  # Data processing modules
│   │   └── __init__.py
│   ├── models/                # Model definitions
│   │   └── __init__.py
│   └── utils/                 # Utility functions
│       └── __init__.py
│
├── scripts/                   # Standalone scripts
│   ├── analyze_and_integrate.py
│   ├── eda_analysis.py
│   ├── verify_processed_data.py
│   └── convert_py_to_ipynb.py  # Utility for converting .py to .ipynb
│
├── notebooks/                  # Jupyter notebooks for EDA and experiments
│   ├── lstm_training.ipynb    # LSTM model training notebook (Colab-ready)
│   ├── cnn_training.ipynb      # CNN model training notebook (Colab-ready)
│   ├── lstm_training.py        # LSTM training script (alternative)
│   ├── cnn_training.py         # CNN training script (alternative)
│   ├── eda_dashboard.html      # Interactive EDA dashboard
│   ├── COLAB_SETUP.md          # Colab setup instructions
│   └── README.md               # Notebooks documentation
│
├── docs/                       # GitHub Pages documentation
│   ├── index.html              # EDA Dashboard (GitHub Pages)
│   ├── data/                   # Data for dashboard
│   ├── README.md               # GitHub Pages setup
│   └── GITHUB_PAGES_SETUP.md   # Detailed setup guide
│
├── models/                     # Saved trained models
│
├── reports/                    # Analysis reports and documentation
│   ├── data/                   # Analysis data files
│   ├── figures/                # Generated visualizations
│   ├── DATA_ANALYSIS_REPORT.md
│   ├── EDA_PLAN.md
│   └── EDA_README.md
│
├── requirements.txt           # Python dependencies
├── .gitignore                 # Git ignore rules
├── PROJECT_STRUCTURE.md      # This file
└── README.md                  # Project documentation
```

## Directory Descriptions

### `data/`
- **`raw/`**: Contains original, unmodified datasets. Files here should not be modified.
- **`processed/`**: Contains cleaned, normalized, and processed datasets ready for model training.

### `src/`
- **`data/`**: Modules for data loading, preprocessing, and transformation.
- **`models/`**: Model architectures and definitions (LSTM, CNN, BERT, DistilBERT).
- **`utils/`**: Helper functions, evaluation metrics, and common utilities.

### `scripts/`
Standalone scripts for data processing, analysis, and utility tasks. These can be run independently.

### `notebooks/`
Jupyter notebooks and training scripts for:
- **Model Training**: LSTM and CNN baseline models (`.ipynb` for Colab, `.py` as alternative)
- **EDA Dashboard**: Interactive HTML dashboard for data exploration
- **Documentation**: Setup guides and instructions

**Note**: Both `.ipynb` (Jupyter/Colab) and `.py` (standalone script) versions are provided for flexibility.

### `models/`
Saved model checkpoints, weights, and trained models.

### `reports/`
Documentation, analysis reports, and project documentation:
- Analysis reports and summaries
- EDA visualizations and figures
- Data analysis metadata (JSON files)

### `docs/`
GitHub Pages documentation and static website:
- EDA Dashboard HTML file
- Setup instructions for GitHub Pages
- Data files for dashboard visualization

## Best Practices

1. **Data Organization**: Keep raw data separate from processed data. Never modify files in `data/raw/`.
2. **Code Organization**: Place reusable code in `src/`, standalone scripts in `scripts/`.
3. **Version Control**: Use `.gitignore` to exclude large data files and model checkpoints.
4. **Documentation**: Keep README.md updated and document major changes in reports.

## Usage

### Running Scripts
```bash
# From project root
python scripts/analyze_and_integrate.py
python scripts/verify_processed_data.py
```

### Importing Modules
```python
# From project root or notebooks
from src.data import DataLoader
from src.models import LSTMModel
from src.utils import evaluate_model
```


