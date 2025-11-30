"""
CNN Model Training Script for Fake News Classification
Этот скрипт можно скопировать в Google Colab для обучения CNN модели
"""

# ============================================================================
# УСТАНОВКА ЗАВИСИМОСТЕЙ (выполнить в Colab)
# ============================================================================
"""
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
!pip install transformers scikit-learn pandas numpy matplotlib seaborn tqdm
!pip install gensim
"""

# ============================================================================
# ИМПОРТЫ
# ============================================================================
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Проверка GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB')

# ============================================================================
# ЗАГРУЗКА ДАННЫХ
# ============================================================================
# Вариант 1: Загрузка из GitHub
REPO_URL = "https://raw.githubusercontent.com/YOUR_USERNAME/Fake-News-Classifier/main"

# Вариант 2: Загрузка файлов вручную в Colab
# Используйте: Files -> Upload и загрузите файлы из data/processed/

try:
    isot_df = pd.read_csv(f'{REPO_URL}/data/processed/isot_processed.csv')
    print(f"✓ ISOT dataset loaded: {isot_df.shape}")
except:
    print("⚠ Загрузите isot_processed.csv вручную в Colab")
    # isot_df = pd.read_csv('/content/isot_processed.csv')

try:
    liar_train = pd.read_csv(f'{REPO_URL}/data/processed/liar_train_processed.csv')
    print(f"✓ LIAR train loaded: {liar_train.shape}")
except:
    print("⚠ Загрузите liar_train_processed.csv вручную в Colab")
    # liar_train = pd.read_csv('/content/liar_train_processed.csv')

# Объединение данных
if 'isot_df' in locals() and 'liar_train' in locals():
    isot_data = isot_df[['text', 'label_binary']].copy()
    isot_data.columns = ['text', 'label']
    liar_data = liar_train[['text', 'label_binary']].copy()
    liar_data.columns = ['text', 'label']
    combined_data = pd.concat([isot_data, liar_data], ignore_index=True)
elif 'isot_df' in locals():
    combined_data = isot_df[['text', 'label_binary']].copy()
    combined_data.columns = ['text', 'label']
elif 'liar_train' in locals():
    combined_data = liar_train[['text', 'label_binary']].copy()
    combined_data.columns = ['text', 'label']
else:
    raise ValueError("Нет данных для обучения!")

# Очистка
combined_data = combined_data[combined_data['text'].notna() & (combined_data['text'].str.len() > 0)]
print(f"\nОбъединенный датасет: {combined_data.shape}")
print(f"Распределение меток: {combined_data['label'].value_counts().to_dict()}")

# Разделение на train/val/test
X = combined_data['text'].values
y = combined_data['label'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
)

print(f"\nTrain: {len(X_train)}, Validation: {len(X_val)}, Test: {len(X_test)}")

# ============================================================================
# СОЗДАНИЕ СЛОВАРЯ И ТОКЕНИЗАЦИЯ
# ============================================================================
def build_vocab(texts, min_freq=2):
    """Создание словаря из текстов"""
    word_counts = Counter()
    for text in texts:
        words = str(text).lower().split()
        word_counts.update(words)
    
    vocab = {'<PAD>': 0, '<UNK>': 1}
    idx = 2
    
    for word, count in word_counts.items():
        if count >= min_freq:
            vocab[word] = idx
            idx += 1
    
    return vocab

def text_to_sequence(text, vocab, max_len=256):
    """Преобразование текста в последовательность индексов"""
    words = str(text).lower().split()
    sequence = [vocab.get(word, vocab['<UNK>']) for word in words[:max_len]]
    
    if len(sequence) < max_len:
        sequence.extend([vocab['<PAD>']] * (max_len - len(sequence)))
    
    return sequence[:max_len]

# Создание словаря
print("\nСоздание словаря...")
vocab = build_vocab(X_train, min_freq=2)
vocab_size = len(vocab)
print(f"Размер словаря: {vocab_size}")

MAX_LEN = 256
EMBEDDING_DIM = 100

# ============================================================================
# ЗАГРУЗКА GLOVE EMBEDDINGS
# ============================================================================
def load_glove_embeddings(glove_path, vocab, embedding_dim=100):
    """Загрузка предобученных GloVe embeddings"""
    print(f"Загрузка GloVe embeddings из {glove_path}...")
    
    if not os.path.exists(glove_path):
        print("Скачивание GloVe 6B.100d...")
        os.system('wget http://nlp.stanford.edu/data/glove.6B.zip')
        os.system('unzip -q glove.6B.zip')
    
    embeddings_index = {}
    with open(glove_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Loading GloVe"):
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    found = 0
    
    for word, idx in vocab.items():
        if word in embeddings_index:
            embedding_matrix[idx] = embeddings_index[word]
            found += 1
        else:
            embedding_matrix[idx] = np.random.normal(scale=0.6, size=(embedding_dim,))
    
    print(f"Найдено embeddings для {found}/{vocab_size} слов ({found/vocab_size*100:.2f}%)")
    return embedding_matrix

GLOVE_PATH = 'glove.6B.100d.txt'
try:
    embedding_matrix = load_glove_embeddings(GLOVE_PATH, vocab, EMBEDDING_DIM)
    use_pretrained = True
except Exception as e:
    print(f"⚠ Не удалось загрузить GloVe: {e}")
    print("Используем случайную инициализацию")
    embedding_matrix = None
    use_pretrained = False

# ============================================================================
# PYTORCH DATASET
# ============================================================================
class NewsDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len=256):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        sequence = text_to_sequence(text, self.vocab, self.max_len)
        return torch.LongTensor(sequence), torch.LongTensor([label])

train_dataset = NewsDataset(X_train, y_train, vocab, MAX_LEN)
val_dataset = NewsDataset(X_val, y_val, vocab, MAX_LEN)
test_dataset = NewsDataset(X_test, y_test, vocab, MAX_LEN)

BATCH_SIZE = 16
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ============================================================================
# CNN МОДЕЛЬ
# ============================================================================
class CNNModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_filters=100, 
                 filter_sizes=[3, 4, 5], num_classes=2, dropout=0.3, 
                 embedding_matrix=None):
        super(CNNModel, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        if embedding_matrix is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(embedding_matrix))
            self.embedding.weight.requires_grad = True
        
        # Convolutional layers с разными размерами фильтров
        self.convs = nn.ModuleList([
            nn.Conv1d(embedding_dim, num_filters, kernel_size=fs)
            for fs in filter_sizes
        ])
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(len(filter_sizes) * num_filters, num_classes)
    
    def forward(self, x):
        # x shape: (batch_size, seq_len)
        embedded = self.embedding(x)  # (batch_size, seq_len, embedding_dim)
        
        # Conv1d ожидает (batch_size, embedding_dim, seq_len)
        embedded = embedded.permute(0, 2, 1)  # (batch_size, embedding_dim, seq_len)
        
        # Применяем свертки и max pooling
        conv_outputs = []
        for conv in self.convs:
            conv_out = conv(embedded)  # (batch_size, num_filters, seq_len - filter_size + 1)
            conv_out = torch.relu(conv_out)
            pooled = torch.max_pool1d(conv_out, kernel_size=conv_out.size(2))  # (batch_size, num_filters, 1)
            pooled = pooled.squeeze(2)  # (batch_size, num_filters)
            conv_outputs.append(pooled)
        
        # Объединяем выходы всех сверток
        concatenated = torch.cat(conv_outputs, dim=1)  # (batch_size, len(filter_sizes) * num_filters)
        
        concatenated = self.dropout(concatenated)
        output = self.fc(concatenated)  # (batch_size, num_classes)
        
        return output

cnn_model = CNNModel(
    vocab_size=vocab_size,
    embedding_dim=EMBEDDING_DIM,
    num_filters=100,
    filter_sizes=[3, 4, 5],
    num_classes=2,
    dropout=0.3,
    embedding_matrix=embedding_matrix if use_pretrained else None
).to(device)

print(f"\nCNN Model Parameters: {sum(p.numel() for p in cnn_model.parameters()):,}")

# ============================================================================
# ФУНКЦИИ ОБУЧЕНИЯ
# ============================================================================
def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for sequences, labels in tqdm(train_loader, desc="Training"):
        sequences = sequences.to(device)
        labels = labels.squeeze().to(device)
        
        optimizer.zero_grad()
        outputs = model(sequences)
        loss = criterion(outputs, labels)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    return total_loss / len(train_loader), 100 * correct / total

def evaluate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for sequences, labels in tqdm(val_loader, desc="Evaluating"):
            sequences = sequences.to(device)
            labels = labels.squeeze().to(device)
            
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(val_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    return avg_loss, accuracy, f1, all_preds, all_labels

# ============================================================================
# ОБУЧЕНИЕ
# ============================================================================
print("\n" + "=" * 60)
print("ОБУЧЕНИЕ CNN МОДЕЛИ")
print("=" * 60)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(cnn_model.parameters(), lr=2e-5)

num_epochs = 10
best_f1 = 0
patience = 3
patience_counter = 0

train_losses = []
val_losses = []
train_accs = []
val_accs = []
val_f1s = []

for epoch in range(num_epochs):
    print(f"\nEpoch {epoch+1}/{num_epochs}")
    
    train_loss, train_acc = train_epoch(cnn_model, train_loader, criterion, optimizer, device)
    val_loss, val_acc, val_f1, _, _ = evaluate(cnn_model, val_loader, criterion, device)
    
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accs.append(train_acc)
    val_accs.append(val_acc)
    val_f1s.append(val_f1)
    
    print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
    print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, Val F1: {val_f1:.4f}")
    
    if val_f1 > best_f1:
        best_f1 = val_f1
        patience_counter = 0
        torch.save(cnn_model.state_dict(), 'best_cnn_model.pth')
        print(f"✓ New best F1: {best_f1:.4f}, model saved")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping after {epoch+1} epochs")
            break

print("\n" + "=" * 60)
print(f"Лучший F1 на валидации: {best_f1:.4f}")
print("=" * 60)

# ============================================================================
# ОЦЕНКА НА ТЕСТОВОМ НАБОРЕ
# ============================================================================
cnn_model.load_state_dict(torch.load('best_cnn_model.pth'))

print("\nОценка CNN модели на тестовом наборе:")
test_loss, test_acc, test_f1, test_preds, test_labels = evaluate(
    cnn_model, test_loader, criterion, device
)

print(f"\nTest Results:")
print(f"  Loss: {test_loss:.4f}")
print(f"  Accuracy: {test_acc:.4f}")
print(f"  F1-Score: {test_f1:.4f}")

print("\nClassification Report:")
print(classification_report(test_labels, test_preds, target_names=['Real', 'Fake']))

# Confusion Matrix
cm = confusion_matrix(test_labels, test_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', 
            xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
plt.title('CNN - Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

cnn_results = {
    'test_loss': float(test_loss),
    'test_accuracy': float(test_acc),
    'test_f1': float(test_f1),
    'test_precision': float(precision_score(test_labels, test_preds, average='weighted')),
    'test_recall': float(recall_score(test_labels, test_preds, average='weighted'))
}

print(f"\nCNN Results: {cnn_results}")

