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


