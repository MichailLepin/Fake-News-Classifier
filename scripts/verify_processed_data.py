"""Quick verification script for processed datasets"""
import pandas as pd
import os

print("=" * 60)
print("PROCESSED DATA VERIFICATION")
print("=" * 60)

# Verify ISOT/Kaggle
print("\n[ISOT/Kaggle Processed]")
if os.path.exists('data/processed/isot_processed.csv'):
    df = pd.read_csv('data/processed/isot_processed.csv')
    print(f"  ✓ File exists")
    print(f"  Records: {len(df):,}")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Label distribution:")
    for label, count in df['label'].value_counts().items():
        print(f"    {label}: {count:,} ({count/len(df)*100:.1f}%)")
else:
    print("  ✗ File not found")

# Verify LIAR
print("\n[LIAR Combined Processed]")
if os.path.exists('data/processed/liar_all_processed.csv'):
    df2 = pd.read_csv('data/processed/liar_all_processed.csv')
    print(f"  ✓ File exists")
    print(f"  Records: {len(df2):,}")
    print(f"  Columns: {list(df2.columns)}")
    print(f"  Label distribution:")
    for label, count in df2['label'].value_counts().items():
        print(f"    {label}: {count:,} ({count/len(df2)*100:.1f}%)")
else:
    print("  ✗ File not found")

print("\n" + "=" * 60)
print("VERIFICATION COMPLETE!")
print("=" * 60)

