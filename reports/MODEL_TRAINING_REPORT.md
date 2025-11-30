# Model Training Report
## Fake News Classification Project

**Date:** November 30, 2025  
**Author:** Model Development Team  
**Project:** Fake News Classifier - NLP-based Classification System

---

## Executive Summary

This report documents the model training process for the Fake News Classification project. Two baseline models were implemented and trained: a Convolutional Neural Network (CNN) and a Bidirectional Long Short-Term Memory (LSTM) network. Both models were trained on the ISOT/Kaggle dataset using GloVe embeddings for text representation. The training was conducted in Google Colab with GPU acceleration, following best practices for deep learning model development.

---

## 1. Training Environment

### 1.1 Platform and Infrastructure

- **Platform:** Google Colab
- **GPU:** CUDA-enabled GPU (when available)
- **Framework:** PyTorch
- **Python Version:** 3.8+

### 1.2 Dependencies

Core libraries used:
- `torch`, `torchvision`, `torchaudio` - Deep learning framework
- `pandas`, `numpy` - Data manipulation
- `scikit-learn` - Data splitting and metrics
- `matplotlib`, `seaborn` - Visualization
- `tqdm` - Progress bars
- `kagglehub` - Dataset download (no API keys required)

---

## 2. Dataset and Preprocessing

### 2.1 Dataset

**Source:** ISOT/Kaggle Fake and Real News Dataset  
**Dataset ID:** `clmentbisaillon/fake-and-real-news-dataset`  
**Total Records:** 44,898 articles
- Fake news: 23,481 (52.3%)
- Real news: 21,417 (47.7%)

### 2.2 Data Preprocessing Pipeline

The preprocessing pipeline includes the following steps:

1. **Dataset Download**
   - Automated download via `kagglehub` (no API keys required)
   - Loads `Fake.csv` and `True.csv` files

2. **Text Cleaning**
   - Convert to lowercase
   - Remove URLs (http, https, www)
   - Normalize whitespace (multiple spaces → single space)
   - Strip leading/trailing whitespace

3. **Label Encoding**
   - Binary labels: `fake` → 1, `real` → 0

4. **Data Splitting**
   - Stratified train/validation/test split
   - Train: 64% (28,734 samples)
   - Validation: 16% (7,184 samples)
   - Test: 20% (8,980 samples)
   - Stratification ensures balanced class distribution in each split

### 2.3 Vocabulary and Tokenization

- **Vocabulary Building:**
  - Built from training data only
  - Minimum word frequency: 2
  - Special tokens: `<PAD>` (0), `<UNK>` (1)
  - Vocabulary size: ~50,000-100,000 words (varies by run)

- **Text to Sequence:**
  - Maximum sequence length: 256 tokens
  - Padding: Right-side padding with `<PAD>`
  - Truncation: Left-side truncation if longer than max_len
  - Unknown words: Mapped to `<UNK>` token

---

## 3. Embeddings

### 3.1 GloVe Embeddings

**Source:** GloVe 6B.100d (Stanford NLP)  
**Dimensions:** 100  
**Vocabulary:** 400,000 words  
**Download:** Automatic via `wget` if not present

### 3.2 Embedding Matrix Construction

- Initialize embedding matrix with zeros: `(vocab_size, 100)`
- For each word in vocabulary:
  - If word exists in GloVe: use pre-trained embedding
  - If word not found: random initialization (normal distribution, scale=0.6)
- Coverage: Typically 60-80% of vocabulary words have pre-trained embeddings

---

## 4. Model Architectures

### 4.1 CNN Model

#### Architecture Overview

The CNN model uses multiple convolutional filters of different sizes to capture n-gram patterns in text.

**Components:**
1. **Embedding Layer**
   - Input: Vocabulary indices
   - Output: 100-dimensional embeddings
   - Initialization: GloVe embeddings (if available)

2. **Convolutional Layers**
   - Multiple 1D convolutions with different filter sizes: [3, 4, 5]
   - Number of filters per size: 100
   - Activation: ReLU
   - Captures 3-grams, 4-grams, and 5-grams

3. **Max Pooling**
   - Global max pooling over sequence dimension
   - Produces fixed-size feature vectors

4. **Concatenation**
   - Concatenate outputs from all filter sizes
   - Feature vector size: 300 (100 × 3)

5. **Fully Connected Layer**
   - Input: 300 features
   - Output: 2 classes (fake/real)
   - Dropout: 0.3

**Model Parameters:**
- Total parameters: ~1-2 million (depending on vocabulary size)
- Trainable parameters: All (including embeddings)

#### Key Design Decisions

- **Multiple filter sizes:** Captures different n-gram patterns simultaneously
- **Max pooling:** Reduces sequence to fixed-size representation
- **Dropout:** Prevents overfitting (0.3 rate)

### 4.2 LSTM Model

#### Architecture Overview

The LSTM model uses bidirectional LSTM to capture sequential dependencies in text.

**Components:**
1. **Embedding Layer**
   - Input: Vocabulary indices
   - Output: 100-dimensional embeddings
   - Initialization: GloVe embeddings (if available)

2. **Bidirectional LSTM**
   - Hidden dimension: 128
   - Number of layers: 1
   - Bidirectional: Yes (forward + backward)
   - Effective hidden size: 256 (128 × 2)
   - Dropout: 0.3 (applied between layers if num_layers > 1)

3. **Output Extraction**
   - Concatenate last hidden states from both directions
   - Uses `hidden[-2]` (forward) and `hidden[-1]` (backward)

4. **Fully Connected Layer**
   - Input: 256 features (128 × 2)
   - Output: 2 classes (fake/real)
   - Dropout: 0.3

**Model Parameters:**
- Total parameters: ~500K-1M (depending on vocabulary size)
- Trainable parameters: All (including embeddings)

#### Key Design Decisions

- **Bidirectional:** Captures context from both directions
- **Single layer:** Simpler architecture, faster training
- **Last hidden states:** Concatenates forward and backward final states

---

## 5. Training Configuration

### 5.1 Hyperparameters

**Common Parameters:**
- Batch size: 16
- Learning rate: 2e-5
- Optimizer: Adam
- Loss function: CrossEntropyLoss
- Max epochs: 10
- Early stopping patience: 3 epochs
- Gradient clipping: max_norm=1.0

**Model-Specific Parameters:**

**CNN:**
- Filter sizes: [3, 4, 5]
- Number of filters: 100 per size
- Embedding dimension: 100
- Dropout: 0.3

**LSTM:**
- Hidden dimension: 128
- Number of layers: 1
- Bidirectional: True
- Embedding dimension: 100
- Dropout: 0.3

### 5.2 Training Procedure

1. **Initialization**
   - Load pre-trained GloVe embeddings (if available)
   - Initialize model on GPU (if available)

2. **Training Loop**
   - For each epoch:
     - Train on training set
     - Evaluate on validation set
     - Calculate metrics (loss, accuracy, F1-score)
     - Save best model based on validation F1-score
     - Early stopping if no improvement for 3 epochs

3. **Evaluation Metrics**
   - Loss: Cross-entropy loss
   - Accuracy: Overall classification accuracy
   - F1-score: Weighted F1-score (primary metric)
   - Precision: Weighted precision
   - Recall: Weighted recall

4. **Model Selection**
   - Best model selected based on validation F1-score
   - Model state saved to `best_cnn_model.pth` or `best_lstm_model.pth`

### 5.3 Early Stopping

- **Criterion:** Validation F1-score
- **Patience:** 3 epochs
- **Purpose:** Prevent overfitting and reduce training time
- **Behavior:** Training stops if validation F1-score doesn't improve for 3 consecutive epochs

---

## 6. Evaluation

### 6.1 Test Set Evaluation

After training, the best model (based on validation F1-score) is evaluated on the test set.

**Metrics Reported:**
- Test Loss
- Test Accuracy
- Test F1-Score
- Test Precision
- Test Recall
- Classification Report (per-class metrics)
- Confusion Matrix

### 6.2 Visualization

- **Confusion Matrix:** Heatmap showing true vs predicted labels
- **Color scheme:**
  - CNN: Green colormap
  - LSTM: Blue colormap

---

## 7. Model Comparison

### 7.1 Architecture Comparison

| Feature | CNN | LSTM |
|---------|-----|------|
| **Architecture Type** | Convolutional | Recurrent |
| **Feature Extraction** | N-gram patterns | Sequential dependencies |
| **Parallelization** | High | Moderate |
| **Training Speed** | Faster | Slower |
| **Memory Usage** | Lower | Higher |
| **Context Handling** | Local (n-grams) | Global (full sequence) |

### 7.2 Expected Performance Characteristics

**CNN Advantages:**
- Fast training and inference
- Good at capturing local patterns (phrases, n-grams)
- Less prone to vanishing gradients
- Better parallelization

**LSTM Advantages:**
- Better at capturing long-range dependencies
- Sequential processing maintains order information
- Can handle variable-length sequences more naturally

### 7.3 Use Case Recommendations

**Choose CNN if:**
- Fast inference is critical
- Local patterns (phrases, keywords) are most important
- Computational resources are limited

**Choose LSTM if:**
- Long-range dependencies matter
- Sequential order is crucial
- More computational resources available

---

## 8. Training Best Practices

### 8.1 Data Handling

- ✅ Stratified splitting ensures balanced classes
- ✅ Text cleaning removes noise
- ✅ Vocabulary built only from training data (prevents data leakage)
- ✅ GloVe embeddings provide semantic information

### 8.2 Model Training

- ✅ Early stopping prevents overfitting
- ✅ Gradient clipping stabilizes training
- ✅ Dropout regularization reduces overfitting
- ✅ Learning rate (2e-5) suitable for fine-tuning pre-trained embeddings

### 8.3 Evaluation

- ✅ Separate test set for final evaluation
- ✅ Multiple metrics provide comprehensive assessment
- ✅ Confusion matrix shows class-specific performance

---

## 9. Implementation Details

### 9.1 PyTorch Dataset Class

```python
class NewsDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len=256):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len
    
    def __getitem__(self, idx):
        sequence = text_to_sequence(self.texts[idx], self.vocab, self.max_len)
        return torch.LongTensor(sequence), torch.LongTensor([self.labels[idx]])
```

### 9.2 Training Function

- Sets model to training mode
- Iterates through batches
- Computes loss and gradients
- Applies gradient clipping
- Updates parameters
- Tracks accuracy and loss

### 9.3 Evaluation Function

- Sets model to evaluation mode
- Disables gradient computation
- Collects all predictions and labels
- Computes metrics (loss, accuracy, F1-score)
- Returns comprehensive results

---

## 10. Results and Metrics

### 10.1 Expected Performance Range

Based on similar implementations and dataset characteristics:

**CNN Model:**
- Accuracy: 85-92%
- F1-Score: 0.85-0.92
- Training time: ~10-20 minutes per epoch (GPU)

**LSTM Model:**
- Accuracy: 87-93%
- F1-Score: 0.87-0.93
- Training time: ~15-30 minutes per epoch (GPU)

*Note: Actual results depend on hyperparameters, data quality, and training conditions.*

### 10.2 Model Selection Criteria

- **Primary Metric:** Validation F1-score
- **Secondary Metrics:** Accuracy, Precision, Recall
- **Practical Considerations:** Training time, inference speed, model size

---

## 11. Future Improvements

### 11.1 Model Enhancements

1. **Architecture Improvements:**
   - Attention mechanisms
   - Multi-layer LSTM/CNN
   - Residual connections
   - Transformer-based models (BERT, DistilBERT)

2. **Training Improvements:**
   - Learning rate scheduling
   - Data augmentation
   - Ensemble methods
   - Cross-validation

3. **Feature Engineering:**
   - Character-level embeddings
   - Additional metadata features
   - Domain-specific preprocessing

### 11.2 Evaluation Enhancements

- Cross-validation for more robust evaluation
- Error analysis on misclassified examples
- Per-class detailed metrics
- ROC-AUC curves
- Precision-Recall curves

---

## 12. Files and Artifacts

### 12.1 Training Notebooks

- `notebooks/cnn_training.ipynb` - CNN model training
- `notebooks/lstm_training.ipynb` - LSTM model training

### 12.2 Saved Models

- `best_cnn_model.pth` - Best CNN model weights
- `best_lstm_model.pth` - Best LSTM model weights

### 12.3 Training Logs

Training progress is printed during execution:
- Per-epoch metrics (loss, accuracy, F1-score)
- Best model checkpoints
- Early stopping notifications

---

## 13. Conclusion

This report documented the training process for two baseline models (CNN and LSTM) for fake news classification. Both models were implemented using PyTorch, trained on the ISOT/Kaggle dataset with GloVe embeddings, and evaluated using comprehensive metrics. The models serve as baselines for comparison with more advanced transformer-based models (BERT, DistilBERT) in future work.

The training pipeline is fully automated and optimized for Google Colab, making it easy to reproduce and iterate on model improvements.

---

**End of Report**

