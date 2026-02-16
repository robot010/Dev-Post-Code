import pandas as pd
import numpy as np
import string
import random

np.random.seed(42)
random.seed(42)

print("Starting dataset generation...\n")

# ============================================================================
# Dataset 1: Large Dataset (10M rows) - For loading benchmarks
# ============================================================================
print("Generating Dataset 1: Large dataset (10M rows)...")
n_rows_large = 10_000_000

data_large = {
    'text_col': [''.join(np.random.choice(list(string.ascii_letters), 20)) for _ in range(n_rows_large)],
    'int_col': np.random.randint(low=0, high=1000000, size=n_rows_large)
}

df_large = pd.DataFrame(data_large)
df_large.to_csv("test_loading_10M.csv", index=False)
print(f"✓ Saved: test_loading_10M.csv ({len(df_large):,} rows)")
print(f"  Columns: {list(df_large.columns)}\n")

# ============================================================================
# Dataset 2: High Cardinality (1M rows) - Mostly unique strings
# ============================================================================
print("Generating Dataset 2: High cardinality (1M unique strings)...")
n_rows = 1_000_000

# Generate mostly unique strings (like IDs, emails, unique identifiers)
data_high_card = {
    'user_id': [f'USER_{i:08d}' for i in range(n_rows)],
    'email': [f'user{i}@example{i % 100}.com' for i in range(n_rows)],
    'session_token': [''.join(np.random.choice(list(string.ascii_letters + string.digits), 32)) for _ in range(n_rows)],
    'int_col': np.random.randint(low=0, high=1000, size=n_rows)
}

df_high_card = pd.DataFrame(data_high_card)
df_high_card.to_csv("test_high_cardinality_1M.csv", index=False)
print(f"✓ Saved: test_high_cardinality_1M.csv ({len(df_high_card):,} rows)")
print(f"  Columns: {list(df_high_card.columns)}")
print(f"  Unique values: {df_high_card['user_id'].nunique():,} user_ids, {df_high_card['email'].nunique():,} emails\n")

# ============================================================================
# Dataset 3: Low Cardinality (1M rows) - Repeated categorical strings
# ============================================================================
print("Generating Dataset 3: Low cardinality (repeated categories)...")

# Categories with realistic distributions
categories = ['Electronics', 'Clothing', 'Food', 'Books', 'Home & Garden', 'Sports', 'Toys', 'Health', 'Automotive', 'Music']
statuses = ['pending', 'completed', 'failed', 'cancelled']
regions = ['North', 'South', 'East', 'West', 'Central']
priorities = ['low', 'medium', 'high', 'critical']

data_low_card = {
    'category': np.random.choice(categories, size=n_rows),
    'status': np.random.choice(statuses, size=n_rows),
    'region': np.random.choice(regions, size=n_rows),
    'priority': np.random.choice(priorities, size=n_rows),
    'int_col': np.random.randint(low=0, high=1000, size=n_rows)
}

df_low_card = pd.DataFrame(data_low_card)
df_low_card.to_csv("test_low_cardinality_1M.csv", index=False)
print(f"✓ Saved: test_low_cardinality_1M.csv ({len(df_low_card):,} rows)")
print(f"  Columns: {list(df_low_card.columns)}")
print(f"  Unique values: {df_low_card['category'].nunique()} categories, {df_low_card['status'].nunique()} statuses\n")

# ============================================================================
# Dataset 4: Mixed String Lengths (1M rows)
# ============================================================================
print("Generating Dataset 4: Mixed string lengths...")

def generate_short_string():
    """2-5 character codes"""
    return ''.join(np.random.choice(list(string.ascii_uppercase), np.random.randint(2, 6)))

def generate_medium_string():
    """20-50 character names/addresses"""
    length = np.random.randint(20, 51)
    return ''.join(np.random.choice(list(string.ascii_letters + ' '), length)).strip()

def generate_long_string():
    """100-300 character descriptions/comments"""
    length = np.random.randint(100, 301)
    words = ['Lorem', 'ipsum', 'dolor', 'sit', 'amet', 'consectetur', 'adipiscing', 'elit',
             'sed', 'do', 'eiusmod', 'tempor', 'incididunt', 'ut', 'labore', 'et', 'dolore']
    return ' '.join(np.random.choice(words, size=length // 6))

data_mixed = {
    'short_code': [generate_short_string() for _ in range(n_rows)],
    'medium_name': [generate_medium_string() for _ in range(n_rows)],
    'long_description': [generate_long_string() for _ in range(n_rows)],
    'int_col': np.random.randint(low=0, high=1000, size=n_rows)
}

df_mixed = pd.DataFrame(data_mixed)
df_mixed.to_csv("test_mixed_lengths_1M.csv", index=False)
print(f"✓ Saved: test_mixed_lengths_1M.csv ({len(df_mixed):,} rows)")
print(f"  Columns: {list(df_mixed.columns)}")
print(f"  Avg lengths: short={df_mixed['short_code'].str.len().mean():.1f}, "
      f"medium={df_mixed['medium_name'].str.len().mean():.1f}, "
      f"long={df_mixed['long_description'].str.len().mean():.1f}\n")

# ============================================================================
# Dataset 5: With Nulls (1M rows) - Missing data scenarios
# ============================================================================
print("Generating Dataset 5: Dataset with null values...")

# Create base data then introduce nulls
data_with_nulls = {
    'name': [f'Name_{i}' if i % 5 != 0 else None for i in range(n_rows)],  # 20% nulls
    'address': [f'{i} Main St' if i % 10 != 0 else None for i in range(n_rows)],  # 10% nulls
    'notes': [f'Note {i}' if i % 3 != 0 else None for i in range(n_rows)],  # 33% nulls
    'int_col': np.random.randint(low=0, high=1000, size=n_rows)
}

df_nulls = pd.DataFrame(data_with_nulls)
df_nulls.to_csv("test_with_nulls_1M.csv", index=False)
print(f"✓ Saved: test_with_nulls_1M.csv ({len(df_nulls):,} rows)")
print(f"  Columns: {list(df_nulls.columns)}")
print(f"  Null percentages: name={df_nulls['name'].isna().mean()*100:.1f}%, "
      f"address={df_nulls['address'].isna().mean()*100:.1f}%, "
      f"notes={df_nulls['notes'].isna().mean()*100:.1f}%\n")

# ============================================================================
# Summary
# ============================================================================
print("="*70)
print("Dataset generation complete!")
print("="*70)
print("\nGenerated files:")
print("  1. test_loading_10M.csv           - 10M rows, for loading benchmarks")
print("  2. test_high_cardinality_1M.csv   - 1M rows, mostly unique strings")
print("  3. test_low_cardinality_1M.csv    - 1M rows, repeated categories")
print("  4. test_mixed_lengths_1M.csv      - 1M rows, varied string lengths")
print("  5. test_with_nulls_1M.csv         - 1M rows, with missing values")
print("\nReady for benchmarking!")