"""
Data Analysis and Integration Script for Fake News Classification Project
Analyzes ISOT/Kaggle and LIAR datasets, performs data integration, and generates report.
"""

import os
import zipfile
import pandas as pd
import numpy as np
import re
from pathlib import Path
import json
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Create directory structure
os.makedirs('data/isot_kaggle', exist_ok=True)
os.makedirs('data/liar', exist_ok=True)
os.makedirs('data/isot_kaggle/processed', exist_ok=True)
os.makedirs('data/liar/processed', exist_ok=True)

print("=" * 80)
print("FAKE NEWS CLASSIFICATION - DATA ANALYSIS & INTEGRATION")
print("=" * 80)

# ============================================================================
# STEP 1: Extract and organize files
# ============================================================================
print("\n[STEP 1] Organizing dataset files...")

# Extract ISOT/Kaggle datasets
if os.path.exists('Fake.csv.zip'):
    with zipfile.ZipFile('Fake.csv.zip', 'r') as zip_ref:
        zip_ref.extractall('data/isot_kaggle')
    print("  ✓ Extracted Fake.csv.zip")

if os.path.exists('True.csv.zip'):
    with zipfile.ZipFile('True.csv.zip', 'r') as zip_ref:
        zip_ref.extractall('data/isot_kaggle')
    print("  ✓ Extracted True.csv.zip")

# Extract and copy LIAR dataset
if os.path.exists('train.tsv.zip'):
    with zipfile.ZipFile('train.tsv.zip', 'r') as zip_ref:
        zip_ref.extractall('data/liar')
    print("  ✓ Extracted train.tsv.zip")

if os.path.exists('test.tsv'):
    import shutil
    shutil.copy('test.tsv', 'data/liar/test.tsv')
    print("  ✓ Copied test.tsv")

if os.path.exists('valid.tsv'):
    import shutil
    shutil.copy('valid.tsv', 'data/liar/valid.tsv')
    print("  ✓ Copied valid.tsv")

# ============================================================================
# STEP 2: Analyze ISOT/Kaggle Dataset
# ============================================================================
print("\n[STEP 2] Analyzing ISOT/Kaggle Dataset...")

isot_results = {}

# Load Fake.csv
fake_path = 'data/isot_kaggle/Fake.csv'
if os.path.exists(fake_path):
    print(f"\n  Loading {fake_path}...")
    try:
        fake_df = pd.read_csv(fake_path, encoding='utf-8')
    except:
        fake_df = pd.read_csv(fake_path, encoding='latin-1')
    
    print(f"  Shape: {fake_df.shape}")
    print(f"  Columns: {list(fake_df.columns)}")
    print(f"\n  First 3 rows:")
    print(fake_df.head(3).to_string())
    
    isot_results['fake'] = {
        'shape': fake_df.shape,
        'columns': list(fake_df.columns),
        'sample': fake_df.head(3).to_dict('records'),
        'dtypes': fake_df.dtypes.to_dict(),
        'null_counts': fake_df.isnull().sum().to_dict(),
        'duplicates': fake_df.duplicated().sum()
    }
    
    # Check for text columns
    text_cols = fake_df.select_dtypes(include=['object']).columns.tolist()
    if text_cols:
        for col in text_cols:
            lengths = fake_df[col].astype(str).str.len()
            isot_results['fake'][f'{col}_length_stats'] = {
                'mean': float(lengths.mean()),
                'min': int(lengths.min()),
                'max': int(lengths.max()),
                'median': float(lengths.median())
            }
else:
    print(f"  ✗ {fake_path} not found")
    fake_df = None

# Load True.csv
true_path = 'data/isot_kaggle/True.csv'
if os.path.exists(true_path):
    print(f"\n  Loading {true_path}...")
    try:
        true_df = pd.read_csv(true_path, encoding='utf-8')
    except:
        true_df = pd.read_csv(true_path, encoding='latin-1')
    
    print(f"  Shape: {true_df.shape}")
    print(f"  Columns: {list(true_df.columns)}")
    print(f"\n  First 3 rows:")
    print(true_df.head(3).to_string())
    
    isot_results['true'] = {
        'shape': true_df.shape,
        'columns': list(true_df.columns),
        'sample': true_df.head(3).to_dict('records'),
        'dtypes': true_df.dtypes.to_dict(),
        'null_counts': true_df.isnull().sum().to_dict(),
        'duplicates': true_df.duplicated().sum()
    }
    
    # Check for text columns
    text_cols = true_df.select_dtypes(include=['object']).columns.tolist()
    if text_cols:
        for col in text_cols:
            lengths = true_df[col].astype(str).str.len()
            isot_results['true'][f'{col}_length_stats'] = {
                'mean': float(lengths.mean()),
                'min': int(lengths.min()),
                'max': int(lengths.max()),
                'median': float(lengths.median())
            }
else:
    print(f"  ✗ {true_path} not found")
    true_df = None

# ============================================================================
# STEP 3: Analyze LIAR Dataset
# ============================================================================
print("\n[STEP 3] Analyzing LIAR Dataset...")

liar_results = {}

for split in ['train', 'test', 'valid']:
    tsv_path = f'data/liar/{split}.tsv'
    if os.path.exists(tsv_path):
        print(f"\n  Loading {tsv_path}...")
        liar_df = pd.read_csv(tsv_path, sep='\t', header=None)
        
        # LIAR dataset typically has these columns:
        # Column 0: ID, 1: Label, 2: Statement, 3: Subject, 4: Speaker, 
        # 5: Job title, 6: State info, 7: Party affiliation, 8-13: Context
        column_names = [
            'id', 'label', 'statement', 'subject', 'speaker', 
            'job_title', 'state_info', 'party_affiliation',
            'barely_true_counts', 'false_counts', 'half_true_counts', 
            'mostly_true_counts', 'pants_on_fire_counts', 'context'
        ]
        
        if len(liar_df.columns) >= len(column_names):
            liar_df.columns = column_names[:len(liar_df.columns)]
        else:
            # If columns don't match, use generic names
            liar_df.columns = [f'col_{i}' for i in range(len(liar_df.columns))]
            if len(liar_df.columns) >= 2:
                liar_df.rename(columns={liar_df.columns[1]: 'label', 
                                       liar_df.columns[2]: 'statement'}, inplace=True)
        
        print(f"  Shape: {liar_df.shape}")
        print(f"  Columns: {list(liar_df.columns)}")
        print(f"\n  First 3 rows:")
        print(liar_df.head(3).to_string())
        
        liar_results[split] = {
            'shape': liar_df.shape,
            'columns': list(liar_df.columns),
            'sample': liar_df.head(3).to_dict('records'),
            'dtypes': {str(k): str(v) for k, v in liar_df.dtypes.to_dict().items()},
            'null_counts': liar_df.isnull().sum().to_dict(),
            'duplicates': liar_df.duplicated().sum()
        }
        
        # Check label distribution
        if 'label' in liar_df.columns:
            label_counts = liar_df['label'].value_counts().to_dict()
            liar_results[split]['label_distribution'] = label_counts
            print(f"  Label distribution: {label_counts}")
        
        # Check statement length if exists
        if 'statement' in liar_df.columns:
            lengths = liar_df['statement'].astype(str).str.len()
            liar_results[split]['statement_length_stats'] = {
                'mean': float(lengths.mean()),
                'min': int(lengths.min()),
                'max': int(lengths.max()),
                'median': float(lengths.median())
            }
    else:
        print(f"  ✗ {tsv_path} not found")
        liar_results[split] = None

# ============================================================================
# STEP 4: Data Integration
# ============================================================================
print("\n[STEP 4] Performing Data Integration...")

def clean_text(text):
    """Clean text: lowercase, remove URLs, remove punctuation"""
    if pd.isna(text):
        return ""
    text = str(text)
    # Convert to lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text

# Process ISOT/Kaggle dataset
if fake_df is not None and true_df is not None:
    print("\n  Processing ISOT/Kaggle dataset...")
    
    # Identify text column (usually 'text' or 'title' or first text column)
    text_col = None
    for col in fake_df.columns:
        if fake_df[col].dtype == 'object' and col.lower() in ['text', 'title', 'article']:
            text_col = col
            break
    if text_col is None:
        # Use first text column
        text_col = fake_df.select_dtypes(include=['object']).columns[0]
    
    print(f"    Using '{text_col}' as text column")
    
    # Add label column
    fake_df['label'] = 'fake'
    true_df['label'] = 'real'
    
    # Combine datasets
    isot_combined = pd.concat([fake_df, true_df], ignore_index=True)
    
    # Clean text
    print("    Cleaning text...")
    isot_combined['text_cleaned'] = isot_combined[text_col].apply(clean_text)
    
    # Create binary label
    isot_combined['label_binary'] = isot_combined['label'].map({'fake': 1, 'real': 0})
    
    # Select relevant columns for processed dataset
    processed_cols = ['label', 'label_binary', 'text_cleaned']
    if 'title' in isot_combined.columns and text_col != 'title':
        processed_cols.insert(2, 'title')
    if 'subject' in isot_combined.columns:
        processed_cols.append('subject')
    if 'date' in isot_combined.columns:
        processed_cols.append('date')
    
    isot_processed = isot_combined[processed_cols].copy()
    isot_processed.rename(columns={'text_cleaned': 'text'}, inplace=True)
    
    # Save processed ISOT dataset
    output_path = 'data/isot_kaggle/processed/isot_processed.csv'
    isot_processed.to_csv(output_path, index=False, encoding='utf-8')
    print(f"    ✓ Saved processed data to {output_path}")
    print(f"    Processed shape: {isot_processed.shape}")
    print(f"    Label distribution: {isot_processed['label'].value_counts().to_dict()}")

# Process LIAR dataset
print("\n  Processing LIAR dataset...")

liar_dfs = []
for split in ['train', 'test', 'valid']:
    tsv_path = f'data/liar/{split}.tsv'
    if os.path.exists(tsv_path):
        liar_df = pd.read_csv(tsv_path, sep='\t', header=None)
        
        # Try to identify columns
        if len(liar_df.columns) >= 3:
            # Assume: col0=id, col1=label, col2=statement
            liar_df.columns = [f'col_{i}' for i in range(len(liar_df.columns))]
            if 'col_1' in liar_df.columns:
                liar_df.rename(columns={'col_1': 'label'}, inplace=True)
            if 'col_2' in liar_df.columns:
                liar_df.rename(columns={'col_2': 'statement'}, inplace=True)
        
        # Normalize labels to binary
        # LIAR has: pants-fire, false, barely-true, half-true, mostly-true, true
        # Map to: fake (pants-fire, false, barely-true) vs real (half-true, mostly-true, true)
        if 'label' in liar_df.columns:
            label_mapping = {
                'pants-fire': 'fake',
                'false': 'fake',
                'barely-true': 'fake',
                'half-true': 'real',
                'mostly-true': 'real',
                'true': 'real'
            }
            liar_df['label_normalized'] = liar_df['label'].map(label_mapping)
            liar_df['label_binary'] = liar_df['label_normalized'].map({'fake': 1, 'real': 0})
        
        # Clean text
        if 'statement' in liar_df.columns:
            print(f"    Cleaning {split} statements...")
            liar_df['text_cleaned'] = liar_df['statement'].apply(clean_text)
        
        # Select relevant columns
        processed_cols = []
        if 'label_normalized' in liar_df.columns:
            processed_cols.append('label_normalized')
        if 'label_binary' in liar_df.columns:
            processed_cols.append('label_binary')
        if 'text_cleaned' in liar_df.columns:
            processed_cols.append('text_cleaned')
        if 'subject' in liar_df.columns:
            processed_cols.append('subject')
        if 'speaker' in liar_df.columns:
            processed_cols.append('speaker')
        if 'party_affiliation' in liar_df.columns:
            processed_cols.append('party_affiliation')
        
        if processed_cols:
            liar_processed = liar_df[processed_cols].copy()
            if 'text_cleaned' in liar_processed.columns:
                liar_processed.rename(columns={'text_cleaned': 'text', 
                                            'label_normalized': 'label'}, inplace=True)
            
            # Save processed LIAR split
            output_path = f'data/liar/processed/liar_{split}_processed.csv'
            liar_processed.to_csv(output_path, index=False, encoding='utf-8')
            print(f"    ✓ Saved {split} processed data to {output_path}")
            print(f"    Processed shape: {liar_processed.shape}")
            if 'label' in liar_processed.columns:
                print(f"    Label distribution: {liar_processed['label'].value_counts().to_dict()}")
            
            liar_dfs.append(liar_processed)

# Combine all LIAR splits
if liar_dfs:
    liar_combined = pd.concat(liar_dfs, ignore_index=True)
    output_path = 'data/liar/processed/liar_all_processed.csv'
    liar_combined.to_csv(output_path, index=False, encoding='utf-8')
    print(f"\n    ✓ Saved combined LIAR data to {output_path}")
    print(f"    Combined shape: {liar_combined.shape}")
    if 'label' in liar_combined.columns:
        print(f"    Combined label distribution: {liar_combined['label'].value_counts().to_dict()}")

# ============================================================================
# STEP 5: Generate Summary Statistics
# ============================================================================
print("\n[STEP 5] Generating Summary Statistics...")

summary = {
    'isot_kaggle': isot_results,
    'liar': liar_results,
    'integration_summary': {
        'isot_processed_shape': isot_processed.shape if 'isot_processed' in locals() else None,
        'liar_combined_shape': liar_combined.shape if 'liar_combined' in locals() else None,
    }
}

# Save summary to JSON
with open('data/analysis_summary.json', 'w', encoding='utf-8') as f:
    json.dump(summary, f, indent=2, default=str)

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE!")
print("=" * 80)
print("\nProcessed files saved in:")
print("  - data/isot_kaggle/processed/")
print("  - data/liar/processed/")
print("\nSummary saved to: data/analysis_summary.json")

