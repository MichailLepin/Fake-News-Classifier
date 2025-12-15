"""Скрипт для создания vocab локально из данных.

Этот скрипт можно использовать, если не удается сохранить vocab из Colab.
Он создаст vocab из тех же данных, что использовались в ноутбуках.
"""

import json
import os
import pandas as pd
import re
from collections import Counter
from pathlib import Path


def clean_text(text):
    """Очистка текста: lowercase, удаление URL, нормализация пробелов"""
    if pd.isna(text):
        return ""
    text = str(text)
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text


def build_vocab(texts, min_freq=2):
    """Построение словаря из текстов"""
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


def main():
    """Основная функция"""
    print("=" * 60)
    print("Создание vocab из данных")
    print("=" * 60)
    
    # Пути к данным
    # Вариант 1: Если данные уже скачаны локально
    data_path = input("Введите путь к данным Kaggle (или нажмите Enter для использования kagglehub): ").strip()
    
    if not data_path:
        try:
            import kagglehub
            print("\nЗагрузка данных через kagglehub...")
            path = kagglehub.dataset_download("clmentbisaillon/fake-and-real-news-dataset")
            fake_df = pd.read_csv(f"{path}/Fake.csv")
            true_df = pd.read_csv(f"{path}/True.csv")
        except ImportError:
            print("Ошибка: kagglehub не установлен.")
            print("Установите: pip install kagglehub")
            return
        except Exception as e:
            print(f"Ошибка при загрузке данных: {e}")
            return
    else:
        # Загрузка из локального пути
        try:
            fake_df = pd.read_csv(f"{data_path}/Fake.csv")
            true_df = pd.read_csv(f"{data_path}/True.csv")
        except Exception as e:
            print(f"Ошибка при загрузке данных: {e}")
            return
    
    print(f"✓ Fake news loaded: {fake_df.shape}")
    print(f"✓ True news loaded: {true_df.shape}")
    
    # Определение колонки с текстом
    text_col = None
    for col in fake_df.columns:
        if fake_df[col].dtype == 'object' and col.lower() in ['text', 'title', 'article']:
            text_col = col
            break
    if text_col is None:
        text_col = fake_df.select_dtypes(include=['object']).columns[0]
    
    print(f"\nИспользуется колонка: '{text_col}'")
    
    # Объединение данных
    combined_data = pd.concat([fake_df, true_df], ignore_index=True)
    
    # Очистка текста
    print("\nОчистка текста...")
    combined_data['text_cleaned'] = combined_data[text_col].apply(clean_text)
    
    # Удаление пустых текстов
    combined_data = combined_data[
        combined_data['text_cleaned'].notna() &
        (combined_data['text_cleaned'].str.len() > 0)
    ]
    
    # Разделение на train/val/test (как в ноутбуке)
    from sklearn.model_selection import train_test_split
    
    X = combined_data['text_cleaned'].values
    y = combined_data['label'].map({'fake': 1, 'real': 0}).values if 'label' in combined_data.columns else None
    
    # Первое разделение: train+val (80%) и test (20%)
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y if y is not None else X, test_size=0.2, random_state=42, 
        stratify=y if y is not None else None
    )
    
    # Второе разделение: train (64%) и val (16%)
    X_train, X_val, y_train_val_split, y_val = train_test_split(
        X_train_val, y_train_val if y is not None else X_train_val, 
        test_size=0.2, random_state=42,
        stratify=y_train_val if y is not None else None
    )
    
    print(f"\nРазделение данных:")
    print(f"  Train: {len(X_train):,} ({len(X_train)/len(combined_data)*100:.1f}%)")
    print(f"  Validation: {len(X_val):,} ({len(X_val)/len(combined_data)*100:.1f}%)")
    print(f"  Test: {len(X_test):,} ({len(X_test)/len(combined_data)*100:.1f}%)")
    
    # Построение vocab из тренировочных данных
    print("\nПостроение vocab из тренировочных данных...")
    vocab = build_vocab(X_train, min_freq=2)
    vocab_size = len(vocab)
    print(f"✓ Размер словаря: {vocab_size}")
    
    # Сохранение vocab
    vocab_dir = Path("vocab")
    vocab_dir.mkdir(exist_ok=True)
    vocab_path = vocab_dir / "vocab.json"
    
    with open(vocab_path, 'w', encoding='utf-8') as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)
    
    print(f"\n✓ Vocab сохранен в {vocab_path}")
    print(f"\nТеперь вы можете использовать этот vocab для бэкенда!")


if __name__ == "__main__":
    main()

