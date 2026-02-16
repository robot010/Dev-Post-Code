#!/usr/bin/env python3
"""
Comprehensive visualization of pandas 2 vs pandas 3 benchmark results.

Creates multiple plots showing memory savings, performance improvements,
and operation-specific comparisons.
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path

# Configuration
RESULTS_DIR = Path("results")
OUTPUT_DIR = Path("visualizations")
OUTPUT_DIR.mkdir(exist_ok=True)

# Set style
sns.set_theme(style="whitegrid")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10


def load_benchmark_results():
    """Load all benchmark result files"""
    results = {
        'pandas2': {},
        'pandas3': {}
    }

    for file in RESULTS_DIR.glob("test_*_pandas*.json"):
        with open(file, 'r') as f:
            data = json.load(f)

        dataset_name = data['metadata']['dataset']
        pandas_version = data['metadata']['pandas_version']

        version_key = 'pandas2' if pandas_version.startswith('2') else 'pandas3'
        results[version_key][dataset_name] = data

    return results


def plot_memory_comparison(results):
    """Create bar chart comparing memory usage across datasets"""
    data = []

    for dataset_name in results['pandas2'].keys():
        if dataset_name in results['pandas3']:
            pandas2_mem = results['pandas2'][dataset_name]['loading']['memory_mb']
            pandas3_mem = results['pandas3'][dataset_name]['loading']['memory_mb']

            data.append({
                'Dataset': dataset_name.replace('_', ' ').title(),
                'Version': 'Pandas 2.3',
                'Memory (MB)': pandas2_mem
            })
            data.append({
                'Dataset': dataset_name.replace('_', ' ').title(),
                'Version': 'Pandas 3.0',
                'Memory (MB)': pandas3_mem
            })

    df = pd.DataFrame(data)

    # Create plot
    fig, ax = plt.subplots(figsize=(14, 8))

    colors = ["#3498db", "#2ecc71"]
    bar_plot = sns.barplot(
        data=df,
        x='Dataset',
        y='Memory (MB)',
        hue='Version',
        palette=colors,
        ax=ax
    )

    # Add title and labels
    ax.set_title('Memory Usage: Pandas 2.3 vs Pandas 3.0 (PyArrow Strings)',
                 fontsize=16, pad=20, fontweight='bold')
    ax.set_ylabel('Memory Usage (MB)', fontsize=12)
    ax.set_xlabel('')
    ax.tick_params(axis='x', rotation=45)
    plt.xticks(rotation=45, ha='right')

    # Add value labels on bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.0f MB', padding=3, fontsize=9)

    # Calculate and add percentage savings inside pandas 3 bars
    datasets = df['Dataset'].unique()
    for i, dataset in enumerate(datasets):
        df_dataset = df[df['Dataset'] == dataset]
        pandas2_val = df_dataset[df_dataset['Version'] == 'Pandas 2.3']['Memory (MB)'].values[0]
        pandas3_val = df_dataset[df_dataset['Version'] == 'Pandas 3.0']['Memory (MB)'].values[0]

        reduction = ((pandas2_val - pandas3_val) / pandas2_val) * 100

        # Get the pandas3 bar for this dataset
        # ax.containers[0] = all Pandas 2.3 bars, ax.containers[1] = all Pandas 3.0 bars
        pandas3_bar = ax.containers[1][i]  # Get i-th bar from Pandas 3.0 container

        # Position text in the middle of the pandas 3 bar
        ax.text(
            pandas3_bar.get_x() + pandas3_bar.get_width() / 2,
            pandas3_bar.get_height() / 2,
            f'-{reduction:.1f}%',
            ha='center', va='center',
            color='white',
            fontweight='bold',
            fontsize=11
        )

    plt.tight_layout()
    output_file = OUTPUT_DIR / 'memory_comparison.png'
    plt.savefig(output_file, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    plt.close()


def plot_operation_performance(results):
    """Create heatmap showing operation execution times"""
    # Collect operation timing data
    operation_data = []

    for dataset_name in results['pandas2'].keys():
        if dataset_name not in results['pandas3']:
            continue

        pandas2_ops = {op['operation']: op.get('exec_time_sec', 0)
                      for op in results['pandas2'][dataset_name]['operations']
                      if op.get('success', False) and 'exec_time_sec' in op}

        pandas3_ops = {op['operation']: op.get('exec_time_sec', 0)
                      for op in results['pandas3'][dataset_name]['operations']
                      if op.get('success', False) and 'exec_time_sec' in op}

        # Calculate speedup for common operations
        for op_name in pandas2_ops.keys():
            if op_name in pandas3_ops and pandas3_ops[op_name] > 0:
                speedup = pandas2_ops[op_name] / pandas3_ops[op_name]
                operation_data.append({
                    'Dataset': dataset_name.replace('_', ' ').replace('test ', ''),
                    'Operation': op_name.replace('()', '').replace('"', ''),
                    'Speedup': speedup
                })

    if not operation_data:
        print("⚠ No operation data available for performance plot")
        return

    df = pd.DataFrame(operation_data)

    # Pivot for heatmap
    pivot_df = df.pivot_table(
        index='Operation',
        columns='Dataset',
        values='Speedup',
        aggfunc='mean'
    )

    # Create heatmap
    fig, ax = plt.subplots(figsize=(14, 10))

    # Custom colormap: red for slowdown (<1), white for same (1), green for speedup (>1)
    cmap = sns.diverging_palette(10, 130, as_cmap=True)

    sns.heatmap(
        pivot_df,
        annot=True,
        fmt='.2f',
        cmap=cmap,
        center=1.0,
        vmin=0.5,
        vmax=5.0,
        cbar_kws={'label': 'Speedup Factor (Pandas 3 vs Pandas 2)'},
        linewidths=0.5,
        ax=ax
    )

    ax.set_title('String Operation Performance: Pandas 3.0 Speedup vs Pandas 2.3\n'
                 '(Values > 1.0 mean Pandas 3 is faster)',
                 fontsize=14, pad=20, fontweight='bold')
    ax.set_xlabel('Dataset', fontsize=12)
    ax.set_ylabel('Operation', fontsize=12)

    plt.tight_layout()
    output_file = OUTPUT_DIR / 'operation_speedup_heatmap.png'
    plt.savefig(output_file, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    plt.close()


def plot_load_time_comparison(results):
    """Create comparison of load times"""
    data = []

    for dataset_name in results['pandas2'].keys():
        if dataset_name in results['pandas3']:
            pandas2_time = results['pandas2'][dataset_name]['loading']['load_time_sec']
            pandas3_time = results['pandas3'][dataset_name]['loading']['load_time_sec']

            data.append({
                'Dataset': dataset_name.replace('_', ' ').title(),
                'Version': 'Pandas 2.3',
                'Load Time (s)': pandas2_time
            })
            data.append({
                'Dataset': dataset_name.replace('_', ' ').title(),
                'Version': 'Pandas 3.0',
                'Load Time (s)': pandas3_time
            })

    df = pd.DataFrame(data)

    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))

    colors = ["#3498db", "#e74c3c"]
    sns.barplot(
        data=df,
        x='Dataset',
        y='Load Time (s)',
        hue='Version',
        palette=colors,
        ax=ax
    )

    ax.set_title('CSV Load Time Comparison: Pandas 2.3 vs Pandas 3.0',
                 fontsize=16, pad=20, fontweight='bold')
    ax.set_ylabel('Load Time (seconds)', fontsize=12)
    ax.set_xlabel('')
    plt.xticks(rotation=45, ha='right')

    # Add value labels
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2fs', padding=3, fontsize=9)

    plt.tight_layout()
    output_file = OUTPUT_DIR / 'load_time_comparison.png'
    plt.savefig(output_file, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    plt.close()


def plot_summary_metrics(results):
    """Create summary dashboard with key metrics"""
    # Calculate aggregate metrics
    total_mem_pandas2 = sum(r['loading']['memory_mb'] for r in results['pandas2'].values())
    total_mem_pandas3 = sum(r['loading']['memory_mb'] for r in results['pandas3'].values())
    mem_savings_pct = ((total_mem_pandas2 - total_mem_pandas3) / total_mem_pandas2) * 100

    # Calculate total CSV disk size
    csv_files = [
        'test_loading_10M.csv',
        'test_high_cardinality_1M.csv',
        'test_low_cardinality_1M.csv',
        'test_mixed_lengths_1M.csv',
        'test_with_nulls_1M.csv'
    ]
    total_disk_size_mb = 0
    for csv_file in csv_files:
        csv_path = Path(csv_file)
        if csv_path.exists():
            total_disk_size_mb += csv_path.stat().st_size / (1024**2)

    # Count successful operations
    total_ops_pandas2 = sum(len(r['operations']) for r in results['pandas2'].values())
    total_ops_pandas3 = sum(len(r['operations']) for r in results['pandas3'].values())

    # Calculate average speedup for operations
    all_speedups = []
    for dataset_name in results['pandas2'].keys():
        if dataset_name not in results['pandas3']:
            continue

        pandas2_ops = {op['operation']: op.get('exec_time_sec', 0)
                      for op in results['pandas2'][dataset_name]['operations']
                      if op.get('success', False) and 'exec_time_sec' in op}

        pandas3_ops = {op['operation']: op.get('exec_time_sec', 0)
                      for op in results['pandas3'][dataset_name]['operations']
                      if op.get('success', False) and 'exec_time_sec' in op}

        for op_name in pandas2_ops.keys():
            if op_name in pandas3_ops and pandas3_ops[op_name] > 0:
                speedup = pandas2_ops[op_name] / pandas3_ops[op_name]
                all_speedups.append(speedup)

    avg_speedup = np.mean(all_speedups) if all_speedups else 1.0

    # Create figure with subplots
    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

    # 1. Memory savings gauge
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.text(0.5, 0.5, f'{mem_savings_pct:.1f}%',
             ha='center', va='center',
             fontsize=48, fontweight='bold', color='#2ecc71')
    ax1.text(0.5, 0.15, 'Memory Savings',
             ha='center', va='center',
             fontsize=14)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.axis('off')

    # 2. Average speedup
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.text(0.5, 0.5, f'{avg_speedup:.2f}x',
             ha='center', va='center',
             fontsize=48, fontweight='bold', color='#3498db')
    ax2.text(0.5, 0.15, 'Avg Operation Speedup',
             ha='center', va='center',
             fontsize=14)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis('off')

    # 3. Total memory saved
    ax3 = fig.add_subplot(gs[0, 2])
    mem_saved_mb = total_mem_pandas2 - total_mem_pandas3
    ax3.text(0.5, 0.5, f'{mem_saved_mb:.0f} MB',
             ha='center', va='center',
             fontsize=48, fontweight='bold', color='#e67e22')
    ax3.text(0.5, 0.15, 'Total Memory Saved',
             ha='center', va='center',
             fontsize=14)
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.axis('off')

    # 4. Memory comparison bar (bottom left and center)
    ax4 = fig.add_subplot(gs[1, :2])
    categories = ['CSV Files\n(On Disk)', 'Pandas 2.3\n(In-Memory)', 'Pandas 3.0\n(In-Memory)']
    values = [total_disk_size_mb, total_mem_pandas2, total_mem_pandas3]
    colors_bar = ['#95a5a6', '#3498db', '#2ecc71']  # Gray for disk, blue for pandas 2, green for pandas 3

    bars = ax4.barh(categories, values, color=colors_bar)
    ax4.set_xlabel('Total Memory Usage (MB)', fontsize=12)
    ax4.set_title('Cumulative Memory Usage: Disk vs Pandas 2.3 In-Memory vs Pandas 3.0 In-Memory',
                  fontsize=11, fontweight='bold')

    # Add value labels with memory overhead info
    for i, bar in enumerate(bars):
        width = bar.get_width()
        label = f'{width:.0f} MB'

        # Add overhead percentage for in-memory versions
        if i == 1:  # Pandas 2.3
            overhead = (total_mem_pandas2 / total_disk_size_mb - 1) * 100 if total_disk_size_mb > 0 else 0
            label += f'  ({overhead:.1f}% overhead)'
        elif i == 2:  # Pandas 3.0
            overhead = (total_mem_pandas3 / total_disk_size_mb - 1) * 100 if total_disk_size_mb > 0 else 0
            label += f'  ({overhead:.1f}% overhead)'

        ax4.text(width + 30, bar.get_y() + bar.get_height()/2,
                label,
                ha='left', va='center', fontsize=11, fontweight='bold')

    # 5. Dataset count (bottom right)
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.text(0.5, 0.6, f'{len(results["pandas2"])}',
             ha='center', va='center',
             fontsize=48, fontweight='bold', color='#9b59b6')
    ax5.text(0.5, 0.3, 'Datasets Tested',
             ha='center', va='center',
             fontsize=14)
    ax5.text(0.5, 0.15, f'15+ string operations',
             ha='center', va='center',
             fontsize=10, style='italic', color='gray')
    ax5.set_xlim(0, 1)
    ax5.set_ylim(0, 1)
    ax5.axis('off')

    fig.suptitle('Pandas 3.0 Migration Benefits - Summary Dashboard',
                 fontsize=18, fontweight='bold', y=0.98)

    plt.tight_layout()
    output_file = OUTPUT_DIR / 'summary_dashboard.png'
    plt.savefig(output_file, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    plt.close()


def main():
    """Main execution"""
    print(f"\n{'='*70}")
    print("GENERATING COMPREHENSIVE VISUALIZATIONS")
    print(f"{'='*70}\n")

    # Load results
    print("Loading benchmark results...")
    results = load_benchmark_results()

    if not results['pandas2'] or not results['pandas3']:
        print("❌ No benchmark results found. Run benchmarks first.")
        return

    print(f"✓ Loaded results for {len(results['pandas2'])} datasets\n")

    # Generate plots
    print("Generating visualizations...\n")

    plot_summary_metrics(results)
    plot_memory_comparison(results)
    plot_load_time_comparison(results)
    plot_operation_performance(results)

    print(f"\n{'='*70}")
    print("ALL VISUALIZATIONS COMPLETE!")
    print(f"{'='*70}")
    print(f"\nOutput directory: {OUTPUT_DIR.absolute()}")
    print(f"Generated files:")
    for img in OUTPUT_DIR.glob("*.png"):
        print(f"  - {img.name}")


if __name__ == '__main__':
    main()
