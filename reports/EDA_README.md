# EDA Analysis - Quick Start Guide

## Overview

This directory contains the Exploratory Data Analysis (EDA) for the Fake News Classification project.

## Files Generated

### Visualizations (`reports/figures/`)
- `isot_label_distribution.png` - Label distribution for ISOT dataset
- `liar_label_distribution.png` - Label distribution for LIAR dataset
- `isot_text_length.png` - Text length analysis for ISOT
- `liar_text_length.png` - Text length analysis for LIAR
- `isot_fake_words.png` - Top words in fake news (ISOT)
- `isot_real_words.png` - Top words in real news (ISOT)
- `liar_fake_words.png` - Top words in fake news (LIAR)
- `liar_real_words.png` - Top words in real news (LIAR)
- `isot_subject_distribution.png` - Subject distribution by label (ISOT)

### Data Files (`reports/data/`)
- `eda_data.json` - Complete EDA data for JavaScript visualization
- `summary_stats.json` - Summary statistics in JSON format

## Running EDA Analysis

### Step 1: Run Python Analysis Script

```bash
python scripts/eda_analysis.py
```

This will:
- Load processed datasets
- Calculate statistics
- Generate visualizations
- Export data for JavaScript dashboard

### Step 2: View Interactive Dashboard

Open `notebooks/eda_dashboard.html` in a web browser:

**Option A: Direct Open**
- Double-click `notebooks/eda_dashboard.html`

**Option B: Local Server (Recommended)**
```bash
# From project root
python -m http.server 8000
# Then open: http://localhost:8000/notebooks/eda_dashboard.html
```

## Key Findings

### ISOT/Kaggle Dataset
- **Total Records:** 44,898
- **Fake News:** 23,481 (52.3%)
- **Real News:** 21,417 (47.7%)
- **Average Text Length (Fake):** 94 characters, 15 words
- **Average Text Length (Real):** 65 characters, 10 words

### LIAR Dataset
- **Total Records:** 12,791
- **Fake News:** 5,657 (44.2%)
- **Real News:** 7,134 (55.8%)
- **Average Text Length (Fake):** 104 characters, 17 words
- **Average Text Length (Real):** 110 characters, 19 words

### Top Words
- **ISOT Fake:** trump, video, obama, hillary, watch
- **ISOT Real:** trump, says, house, russia, north korea
- **LIAR Fake:** the, and, says, obama, has
- **LIAR Real:** the, and, says, percent, than

## Next Steps

1. Review visualizations in `reports/figures/`
2. Explore interactive dashboard
3. Use insights for feature engineering
4. Consider text length differences for tokenization strategy
5. Analyze word frequencies for stop word removal

## Documentation

- **EDA Plan:** `reports/EDA_PLAN.md` - Detailed analysis plan
- **Data Analysis Report:** `reports/DATA_ANALYSIS_REPORT.md` - Previous analysis
- **Project Structure:** `PROJECT_STRUCTURE.md` - Project organization


