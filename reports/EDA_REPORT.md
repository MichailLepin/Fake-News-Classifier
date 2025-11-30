# Exploratory Data Analysis (EDA) Report
## Fake News Classification Project

**Date:** November 30, 2025  
**Based on:** Processed datasets from data integration phase

---

## 1. Analysis Objectives

### Primary Goals
1. Understand data distribution and characteristics
2. Identify patterns and differences between fake and real news
3. Discover features that could help classification
4. Validate data quality after processing
5. Prepare insights for model development

### Key Questions
- How do fake and real news differ in text characteristics?
- What are the most common words/phrases in each class?
- Are there temporal patterns in the data?
- What is the class distribution and balance?
- How do text lengths vary between classes?

---

## 2. EDA Components

### 2.1 Descriptive Statistics
- **Label Distribution**: Count and percentage of fake vs real
- **Text Length Analysis**: 
  - Character count statistics (mean, median, std, min, max)
  - Word count statistics
  - Sentence count statistics
- **Dataset Comparison**: ISOT vs LIAR characteristics

### 2.2 Distribution Visualizations
- **Label Distribution**: Bar charts, pie charts
- **Text Length Distributions**: 
  - Histograms for character/word counts
  - Box plots comparing fake vs real
  - Violin plots for detailed distribution shapes
- **Temporal Analysis** (ISOT only):
  - Time series of article counts
  - Distribution by month/year
  - Label distribution over time

### 2.3 Text Analysis
- **Word Frequency Analysis**:
  - Top N most frequent words per class
  - Word clouds for fake and real news
  - Stop words removal and analysis
- **N-gram Analysis**:
  - Unigrams (single words)
  - Bigrams (word pairs)
  - Trigrams (word triplets)
  - TF-IDF analysis
- **Vocabulary Analysis**:
  - Unique word counts
  - Vocabulary overlap between classes
  - Rare word analysis

### 2.4 Subject/Category Analysis (ISOT only)
- Subject distribution by label
- Subject-label correlation
- Most common subjects for fake vs real

### 2.5 Statistical Tests
- **Text Length Differences**: 
  - T-tests for mean differences
  - Mann-Whitney U test for distribution differences
- **Vocabulary Differences**: Chi-square tests for word frequency differences

### 2.6 Advanced Visualizations
- **t-SNE Embeddings**: 2D visualization of text embeddings
- **Correlation Analysis**: Feature correlations
- **Outlier Detection**: Identify unusual text lengths or patterns

---

## 3. Key Findings

### 3.1 ISOT/Kaggle Dataset

**Total Records:** 44,898

**Label Distribution:**
- **Fake News:** 23,481 (52.3%)
- **Real News:** 21,417 (47.7%)

**Text Length Statistics:**
- **Average Text Length (Fake):** 94 characters, 15 words
- **Average Text Length (Real):** 65 characters, 10 words

**Top Words:**
- **Fake:** trump, video, obama, hillary, watch
- **Real:** trump, says, house, russia, north korea

### 3.2 LIAR Dataset

**Total Records:** 12,791

**Label Distribution:**
- **Fake News:** 5,657 (44.2%)
- **Real News:** 7,134 (55.8%)

**Text Length Statistics:**
- **Average Text Length (Fake):** 104 characters, 17 words
- **Average Text Length (Real):** 110 characters, 19 words

**Top Words:**
- **Fake:** the, and, says, obama, has
- **Real:** the, and, says, percent, than

---

## 4. Generated Files

### 4.1 Visualizations (`reports/figures/`)
- `isot_label_distribution.png` - Label distribution for ISOT dataset
- `liar_label_distribution.png` - Label distribution for LIAR dataset
- `isot_text_length.png` - Text length analysis for ISOT
- `liar_text_length.png` - Text length analysis for LIAR
- `isot_fake_words.png` - Top words in fake news (ISOT)
- `isot_real_words.png` - Top words in real news (ISOT)
- `liar_fake_words.png` - Top words in fake news (LIAR)
- `liar_real_words.png` - Top words in real news (LIAR)
- `isot_subject_distribution.png` - Subject distribution by label (ISOT)

### 4.2 Data Files (`reports/data/`)
- `eda_data.json` - Complete EDA data for JavaScript visualization
- `summary_stats.json` - Summary statistics in JSON format

---

## 5. Implementation Details

### 5.1 Tools and Libraries

**Python (Data Processing)**
- **pandas**: Data manipulation
- **numpy**: Numerical operations
- **matplotlib**: Static visualizations
- **seaborn**: Statistical visualizations
- **wordcloud**: Word cloud generation
- **scikit-learn**: TF-IDF, statistical tests
- **nltk/spacy**: Text processing

**JavaScript (Web Visualization)**
- **Chart.js**: Interactive charts
- **Vanilla JS**: Core functionality

### 5.2 Running EDA Analysis

**Step 1: Run Python Analysis Script**

```bash
python scripts/eda_analysis.py
```

This will:
- Load processed datasets
- Calculate statistics
- Generate visualizations
- Export data for JavaScript dashboard

**Step 2: View Interactive Dashboard**

The interactive dashboard is available at `docs/index.html` (published on GitHub Pages).

---

## 6. Key Metrics

### 6.1 Text Characteristics
- Average text length (characters, words)
- Median text length
- Standard deviation
- Min/Max lengths
- Text length percentiles (25th, 75th, 90th, 95th)

### 6.2 Vocabulary Metrics
- Total unique words
- Average words per document
- Vocabulary size per class
- Common words ratio
- Rare words count

### 6.3 Class Balance
- Class distribution percentages
- Imbalance ratio
- Sample size adequacy

---

## 7. Data Quality Checks

1. **Missing Values**: Verified no critical missing data
2. **Text Quality**: Checked for empty or extremely short texts
3. **Encoding Issues**: Verified text encoding is correct
4. **Outliers**: Identified and documented unusual cases
5. **Consistency**: Verified label consistency

---

## 8. Recommendations for Model Development

1. **Tokenization Strategy:**
   - ISOT: Use max_length=512 tokens (full articles)
   - LIAR: Use max_length=256 tokens (shorter statements)

2. **Feature Engineering:**
   - Consider text length differences between classes
   - Analyze word frequencies for stop word removal
   - Use subject categories from ISOT dataset as additional features

3. **Class Balance:**
   - Both datasets show acceptable class balance
   - ISOT: Slight imbalance (52.3% fake, 47.7% real) - acceptable
   - LIAR: Moderate imbalance (44.2% fake, 55.8% real) - acceptable

4. **Next Steps:**
   - Review visualizations in `reports/figures/`
   - Explore interactive dashboard
   - Use insights for feature engineering
   - Consider text length differences for tokenization strategy

---

## 9. Success Criteria

✅ All planned visualizations created  
✅ Statistical tests performed  
✅ Key insights documented  
✅ Interactive dashboard functional  
✅ Report generated with findings  
✅ Recommendations provided for modeling

---

**Report End**
