# Exploratory Data Analysis (EDA) Plan
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

## 3. Implementation Strategy

### Phase 1: Basic Statistics
1. Load processed datasets
2. Calculate descriptive statistics
3. Generate summary tables
4. Create basic distribution plots

### Phase 2: Text Analysis
1. Tokenize texts
2. Calculate word frequencies
3. Generate n-grams
4. Create word clouds
5. Analyze vocabulary

### Phase 3: Advanced Analysis
1. Generate embeddings (if needed)
2. Create t-SNE visualizations
3. Perform statistical tests
4. Identify patterns and insights

### Phase 4: Reporting
1. Compile all visualizations
2. Document findings
3. Create summary report
4. Generate interactive dashboard (JavaScript)

---

## 4. Tools and Libraries

### Python (Data Processing)
- **pandas**: Data manipulation
- **numpy**: Numerical operations
- **matplotlib**: Static visualizations
- **seaborn**: Statistical visualizations
- **wordcloud**: Word cloud generation
- **scikit-learn**: TF-IDF, statistical tests
- **nltk/spacy**: Text processing

### JavaScript (Web Visualization)
- **Chart.js** or **D3.js**: Interactive charts
- **PapaParse**: CSV parsing
- **Plotly.js**: Advanced visualizations
- **Vanilla JS**: Core functionality

---

## 5. Expected Outputs

### 5.1 Static Visualizations (Python)
- Label distribution charts
- Text length histograms and box plots
- Word frequency bar charts
- Word clouds
- Temporal analysis charts (ISOT)
- Subject distribution charts (ISOT)
- t-SNE scatter plots

### 5.2 Interactive Dashboard (JavaScript)
- Interactive charts
- Filterable visualizations
- Dataset comparison views
- Real-time data exploration

### 5.3 Reports
- EDA summary report (Markdown)
- Statistical findings document
- Recommendations for model development

---

## 6. Key Metrics to Calculate

### Text Characteristics
- Average text length (characters, words)
- Median text length
- Standard deviation
- Min/Max lengths
- Text length percentiles (25th, 75th, 90th, 95th)

### Vocabulary Metrics
- Total unique words
- Average words per document
- Vocabulary size per class
- Common words ratio
- Rare words count

### Class Balance
- Class distribution percentages
- Imbalance ratio
- Sample size adequacy

---

## 7. Data Quality Checks

1. **Missing Values**: Verify no critical missing data
2. **Text Quality**: Check for empty or extremely short texts
3. **Encoding Issues**: Verify text encoding is correct
4. **Outliers**: Identify and document unusual cases
5. **Consistency**: Verify label consistency

---

## 8. Deliverables

1. **Python EDA Script**: `scripts/eda_analysis.py`
2. **JavaScript Visualization**: `notebooks/eda_dashboard.html`
3. **EDA Report**: `reports/EDA_REPORT.md`
4. **Visualization Files**: `reports/figures/` (PNG/SVG)
5. **Data Exports**: JSON files for JavaScript consumption

---

## 9. Timeline

- **Phase 1**: 2-3 hours (Basic statistics and distributions)
- **Phase 2**: 3-4 hours (Text analysis and n-grams)
- **Phase 3**: 2-3 hours (Advanced analysis)
- **Phase 4**: 2-3 hours (Reporting and dashboard)
- **Total**: ~10-13 hours

---

## 10. Success Criteria

✅ All planned visualizations created  
✅ Statistical tests performed  
✅ Key insights documented  
✅ Interactive dashboard functional  
✅ Report generated with findings  
✅ Recommendations provided for modeling


