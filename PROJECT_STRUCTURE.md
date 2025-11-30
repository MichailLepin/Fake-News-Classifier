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
│   └── verify_processed_data.py
│
├── notebooks/                  # Jupyter notebooks for EDA and experiments
│
├── models/                    # Saved trained models
│
├── reports/                    # Analysis reports and documentation
│   └── DATA_ANALYSIS_REPORT.md
│
├── requirements.txt           # Python dependencies
├── .gitignore                 # Git ignore rules
└── README.md                   # Project documentation
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
Jupyter notebooks for:
- Exploratory Data Analysis (EDA)
- Model experimentation
- Visualization
- Prototyping

### `models/`
Saved model checkpoints, weights, and trained models.

### `reports/`
Documentation, analysis reports, and project documentation.

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


