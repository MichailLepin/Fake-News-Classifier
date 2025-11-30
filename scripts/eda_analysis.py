"""
Exploratory Data Analysis (EDA) Script
Generates comprehensive EDA analysis and exports data for JavaScript visualization
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# Create output directories
Path("reports/figures").mkdir(parents=True, exist_ok=True)
Path("reports/data").mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("EXPLORATORY DATA ANALYSIS (EDA)")
print("=" * 80)

# ============================================================================
# Load Data
# ============================================================================
print("\n[1] Loading processed datasets...")

isot_df = pd.read_csv('data/processed/isot_processed.csv')
liar_df = pd.read_csv('data/processed/liar_all_processed.csv')

print(f"  ISOT Dataset: {isot_df.shape[0]:,} records")
print(f"  LIAR Dataset: {liar_df.shape[0]:,} records")

# ============================================================================
# Basic Statistics
# ============================================================================
print("\n[2] Calculating basic statistics...")

def calculate_text_stats(df, text_col='text'):
    """Calculate text length statistics"""
    df = df.copy()
    df['char_count'] = df[text_col].astype(str).str.len()
    df['word_count'] = df[text_col].astype(str).str.split().str.len()
    df['sentence_count'] = df[text_col].astype(str).str.count(r'[.!?]+')
    return df

isot_df = calculate_text_stats(isot_df)
liar_df = calculate_text_stats(liar_df)

# Statistics by label
def get_stats_by_label(df):
    stats = {}
    for label in df['label'].unique():
        label_df = df[df['label'] == label]
        stats[label] = {
            'count': len(label_df),
            'char_count': {
                'mean': float(label_df['char_count'].mean()),
                'median': float(label_df['char_count'].median()),
                'std': float(label_df['char_count'].std()),
                'min': int(label_df['char_count'].min()),
                'max': int(label_df['char_count'].max())
            },
            'word_count': {
                'mean': float(label_df['word_count'].mean()),
                'median': float(label_df['word_count'].median()),
                'std': float(label_df['word_count'].std()),
                'min': int(label_df['word_count'].min()),
                'max': int(label_df['word_count'].max())
            }
        }
    return stats

isot_stats = get_stats_by_label(isot_df)
liar_stats = get_stats_by_label(liar_df)

print("\n  ISOT Statistics by Label:")
for label, stats in isot_stats.items():
    print(f"    {label.upper()}:")
    print(f"      Count: {stats['count']:,}")
    print(f"      Avg chars: {stats['char_count']['mean']:.1f}")
    print(f"      Avg words: {stats['word_count']['mean']:.1f}")

print("\n  LIAR Statistics by Label:")
for label, stats in liar_stats.items():
    print(f"    {label.upper()}:")
    print(f"      Count: {stats['count']:,}")
    print(f"      Avg chars: {stats['char_count']['mean']:.1f}")
    print(f"      Avg words: {stats['word_count']['mean']:.1f}")

# ============================================================================
# Label Distribution
# ============================================================================
print("\n[3] Analyzing label distribution...")

def plot_label_distribution(df, dataset_name, save_path):
    """Plot label distribution"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Count plot
    label_counts = df['label'].value_counts()
    colors = ['#ff6b6b' if x == 'fake' else '#51cf66' for x in label_counts.index]
    ax1.bar(label_counts.index, label_counts.values, color=colors, alpha=0.8)
    ax1.set_title(f'{dataset_name} - Label Distribution', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Label')
    ax1.set_ylabel('Count')
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for i, v in enumerate(label_counts.values):
        ax1.text(i, v, f'{v:,}', ha='center', va='bottom', fontweight='bold')
    
    # Pie chart
    ax2.pie(label_counts.values, labels=label_counts.index, autopct='%1.1f%%',
            colors=colors, startangle=90)
    ax2.set_title(f'{dataset_name} - Label Proportion', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Export data for JS
    return label_counts.to_dict()

isot_label_dist = plot_label_distribution(isot_df, 'ISOT/Kaggle', 'reports/figures/isot_label_distribution.png')
liar_label_dist = plot_label_distribution(liar_df, 'LIAR', 'reports/figures/liar_label_distribution.png')

# ============================================================================
# Text Length Analysis
# ============================================================================
print("\n[4] Analyzing text length distributions...")

def plot_text_length_distribution(df, dataset_name, save_path):
    """Plot text length distributions by label"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Character count
    for label in df['label'].unique():
        data = df[df['label'] == label]['char_count']
        axes[0, 0].hist(data, bins=50, alpha=0.6, label=label, density=True)
    axes[0, 0].set_title('Character Count Distribution', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Character Count')
    axes[0, 0].set_ylabel('Density')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    
    # Word count
    for label in df['label'].unique():
        data = df[df['label'] == label]['word_count']
        axes[0, 1].hist(data, bins=50, alpha=0.6, label=label, density=True)
    axes[0, 1].set_title('Word Count Distribution', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Word Count')
    axes[0, 1].set_ylabel('Density')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)
    
    # Box plots
    df_melted = df.melt(id_vars=['label'], value_vars=['char_count', 'word_count'],
                        var_name='metric', value_name='count')
    sns.boxplot(data=df_melted, x='metric', y='count', hue='label', ax=axes[1, 0])
    axes[1, 0].set_title('Text Length Comparison (Box Plot)', fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].grid(alpha=0.3)
    
    # Violin plot
    sns.violinplot(data=df_melted, x='metric', y='count', hue='label', ax=axes[1, 1])
    axes[1, 1].set_title('Text Length Comparison (Violin Plot)', fontsize=12, fontweight='bold')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].grid(alpha=0.3)
    
    plt.suptitle(f'{dataset_name} - Text Length Analysis', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

plot_text_length_distribution(isot_df, 'ISOT/Kaggle', 'reports/figures/isot_text_length.png')
plot_text_length_distribution(liar_df, 'LIAR', 'reports/figures/liar_text_length.png')

# ============================================================================
# Word Frequency Analysis
# ============================================================================
print("\n[5] Analyzing word frequencies...")

def get_top_words(df, label, n=20):
    """Get top N words for a label"""
    texts = df[df['label'] == label]['text'].astype(str).str.lower()
    all_words = []
    for text in texts:
        # Simple word tokenization
        words = re.findall(r'\b[a-z]+\b', text)
        all_words.extend(words)
    
    # Filter out very short words
    all_words = [w for w in all_words if len(w) > 2]
    
    word_counts = Counter(all_words)
    return dict(word_counts.most_common(n))

# Get top words for each dataset
isot_fake_words = get_top_words(isot_df, 'fake', 30)
isot_real_words = get_top_words(isot_df, 'real', 30)
liar_fake_words = get_top_words(liar_df, 'fake', 30)
liar_real_words = get_top_words(liar_df, 'real', 30)

print("\n  Top 10 words - ISOT Fake:", list(isot_fake_words.keys())[:10])
print("  Top 10 words - ISOT Real:", list(isot_real_words.keys())[:10])
print("  Top 10 words - LIAR Fake:", list(liar_fake_words.keys())[:10])
print("  Top 10 words - LIAR Real:", list(liar_real_words.keys())[:10])

def plot_top_words(word_dict, title, save_path):
    """Plot top words"""
    words = list(word_dict.keys())[:20]
    counts = list(word_dict.values())[:20]
    
    plt.figure(figsize=(12, 8))
    plt.barh(words[::-1], counts[::-1], color='steelblue', alpha=0.8)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Frequency')
    plt.ylabel('Words')
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

plot_top_words(isot_fake_words, 'ISOT - Top Words (Fake)', 'reports/figures/isot_fake_words.png')
plot_top_words(isot_real_words, 'ISOT - Top Words (Real)', 'reports/figures/isot_real_words.png')
plot_top_words(liar_fake_words, 'LIAR - Top Words (Fake)', 'reports/figures/liar_fake_words.png')
plot_top_words(liar_real_words, 'LIAR - Top Words (Real)', 'reports/figures/liar_real_words.png')

# ============================================================================
# Subject Analysis (ISOT only)
# ============================================================================
if 'subject' in isot_df.columns:
    print("\n[6] Analyzing subject distribution (ISOT)...")
    
    subject_label = pd.crosstab(isot_df['subject'], isot_df['label'])
    subject_label_pct = subject_label.div(subject_label.sum(axis=1), axis=0) * 100
    
    plt.figure(figsize=(14, 8))
    subject_label.plot(kind='bar', stacked=False, figsize=(14, 8))
    plt.title('ISOT - Subject Distribution by Label', fontsize=14, fontweight='bold')
    plt.xlabel('Subject')
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Label')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('reports/figures/isot_subject_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

# ============================================================================
# Export Data for JavaScript
# ============================================================================
print("\n[7] Exporting data for JavaScript visualization...")

# Prepare data for JS
js_data = {
    'isot': {
        'label_distribution': isot_label_dist,
        'statistics': isot_stats,
        'top_words': {
            'fake': isot_fake_words,
            'real': isot_real_words
        },
        'text_lengths': {
            'fake': {
                'char': isot_df[isot_df['label'] == 'fake']['char_count'].tolist()[:1000],  # Sample
                'word': isot_df[isot_df['label'] == 'fake']['word_count'].tolist()[:1000]
            },
            'real': {
                'char': isot_df[isot_df['label'] == 'real']['char_count'].tolist()[:1000],
                'word': isot_df[isot_df['label'] == 'real']['word_count'].tolist()[:1000]
            }
        }
    },
    'liar': {
        'label_distribution': liar_label_dist,
        'statistics': liar_stats,
        'top_words': {
            'fake': liar_fake_words,
            'real': liar_real_words
        },
        'text_lengths': {
            'fake': {
                'char': liar_df[liar_df['label'] == 'fake']['char_count'].tolist()[:1000],
                'word': liar_df[liar_df['label'] == 'fake']['word_count'].tolist()[:1000]
            },
            'real': {
                'char': liar_df[liar_df['label'] == 'real']['char_count'].tolist()[:1000],
                'word': liar_df[liar_df['label'] == 'real']['word_count'].tolist()[:1000]
            }
        }
    }
}

# Save JSON
with open('reports/data/eda_data.json', 'w', encoding='utf-8') as f:
    json.dump(js_data, f, indent=2, ensure_ascii=False)

# Save summary statistics
summary_stats = {
    'isot': isot_stats,
    'liar': liar_stats,
    'isot_label_dist': isot_label_dist,
    'liar_label_dist': liar_label_dist
}

with open('reports/data/summary_stats.json', 'w', encoding='utf-8') as f:
    json.dump(summary_stats, f, indent=2, ensure_ascii=False)

print("\n" + "=" * 80)
print("EDA ANALYSIS COMPLETE!")
print("=" * 80)
print("\nGenerated files:")
print("  - Visualizations: reports/figures/")
print("  - JSON data: reports/data/eda_data.json")
print("  - Summary stats: reports/data/summary_stats.json")


