#!/usr/bin/env python3
"""
Automated test runner for pandas 2 vs pandas 3 string operations comparison.

This script runs comprehensive benchmarks across multiple datasets using both
pandas 2 and pandas 3, then generates comparison reports.
"""

import subprocess
import json
from pathlib import Path
from datetime import datetime
import sys

# Configuration
PANDAS2_PYTHON = "/Users/botao/miniconda3/bin/python"
PANDAS3_PYTHON = "/Users/botao/miniconda3/envs/pandas3.0/bin/python"
BENCHMARK_SCRIPT = "string_ops_benchmark.py"
RESULTS_DIR = Path("results")

# Datasets to benchmark (in order of size for efficient testing)
DATASETS = [
    "test_low_cardinality_1M.csv",
    "test_high_cardinality_1M.csv",
    "test_with_nulls_1M.csv",
    "test_mixed_lengths_1M.csv",
    "test_loading_10M.csv",  # Largest dataset last
]

class BenchmarkRunner:
    """Orchestrates benchmark execution across multiple datasets and pandas versions"""

    def __init__(self):
        self.results_dir = RESULTS_DIR
        self.results_dir.mkdir(exist_ok=True)
        self.summary = {
            'run_timestamp': datetime.now().isoformat(),
            'datasets_tested': [],
            'pandas_versions': {},
            'results': []
        }

    def verify_setup(self):
        """Verify that all required files and environments exist"""
        print("Verifying setup...")

        # Check Python interpreters
        for name, python_path in [("Pandas 2", PANDAS2_PYTHON), ("Pandas 3", PANDAS3_PYTHON)]:
            if not Path(python_path).exists():
                print(f"❌ {name} Python not found at: {python_path}")
                return False

            # Get pandas version
            try:
                result = subprocess.run(
                    [python_path, "-c", "import pandas; print(pandas.__version__)"],
                    capture_output=True,
                    text=True,
                    check=True
                )
                version = result.stdout.strip()
                self.summary['pandas_versions'][name] = version
                print(f"  ✓ {name}: {version} at {python_path}")
            except subprocess.CalledProcessError as e:
                print(f"❌ Failed to get {name} version: {e}")
                return False

        # Check benchmark script
        if not Path(BENCHMARK_SCRIPT).exists():
            print(f"❌ Benchmark script not found: {BENCHMARK_SCRIPT}")
            return False
        print(f"  ✓ Benchmark script: {BENCHMARK_SCRIPT}")

        # Check datasets
        missing_datasets = []
        for dataset in DATASETS:
            if not Path(dataset).exists():
                missing_datasets.append(dataset)

        if missing_datasets:
            print(f"❌ Missing datasets: {', '.join(missing_datasets)}")
            return False

        print(f"  ✓ All {len(DATASETS)} datasets found")
        print()
        return True

    def run_benchmark(self, dataset, python_path, pandas_version_name):
        """
        Run benchmark for a single dataset with a specific pandas version

        Args:
            dataset: Path to CSV dataset
            python_path: Path to Python interpreter
            pandas_version_name: Human-readable name (e.g., "Pandas 2")

        Returns:
            dict: Benchmark results or None if failed
        """
        dataset_name = Path(dataset).stem
        print(f"  Running {pandas_version_name} on {dataset_name}...", end=' ')

        try:
            # Run the benchmark script
            result = subprocess.run(
                [python_path, BENCHMARK_SCRIPT, dataset, "--output-dir", str(self.results_dir)],
                capture_output=True,
                text=True,
                check=True,
                timeout=600  # 10 minute timeout
            )

            # Find and load the result file
            # The benchmark script creates files named: {dataset_name}_pandas{major_version}.json
            version_major = self.summary['pandas_versions'][pandas_version_name].split('.')[0]
            result_file = self.results_dir / f"{dataset_name}_pandas{version_major}.json"

            if result_file.exists():
                with open(result_file, 'r') as f:
                    results = json.load(f)
                print(f"✓ ({results['loading']['load_time_sec']:.2f}s, {results['loading']['memory_mb']:.1f} MB)")
                return results
            else:
                print(f"⚠ Result file not found: {result_file}")
                return None

        except subprocess.TimeoutExpired:
            print("✗ Timeout (>10 minutes)")
            return None
        except subprocess.CalledProcessError as e:
            print(f"✗ Error")
            print(f"    stdout: {e.stdout[:200]}")
            print(f"    stderr: {e.stderr[:200]}")
            return None
        except Exception as e:
            print(f"✗ Unexpected error: {e}")
            return None

    def run_all_benchmarks(self):
        """Run benchmarks for all datasets with both pandas versions"""
        print(f"\n{'='*70}")
        print("RUNNING ALL BENCHMARKS")
        print(f"{'='*70}")
        print(f"Datasets: {len(DATASETS)}")
        print(f"Pandas versions: {len(self.summary['pandas_versions'])}")
        print(f"Total benchmarks to run: {len(DATASETS) * 2}\n")

        for i, dataset in enumerate(DATASETS, 1):
            print(f"[{i}/{len(DATASETS)}] {dataset}")

            dataset_results = {
                'dataset': dataset,
                'pandas2': None,
                'pandas3': None
            }

            # Run with Pandas 2
            pandas2_results = self.run_benchmark(dataset, PANDAS2_PYTHON, "Pandas 2")
            if pandas2_results:
                dataset_results['pandas2'] = {
                    'load_time': pandas2_results['loading']['load_time_sec'],
                    'memory_mb': pandas2_results['loading']['memory_mb'],
                    'num_operations': len(pandas2_results['operations']),
                    'successful_operations': sum(1 for op in pandas2_results['operations'] if op.get('success', False))
                }

            # Run with Pandas 3
            pandas3_results = self.run_benchmark(dataset, PANDAS3_PYTHON, "Pandas 3")
            if pandas3_results:
                dataset_results['pandas3'] = {
                    'load_time': pandas3_results['loading']['load_time_sec'],
                    'memory_mb': pandas3_results['loading']['memory_mb'],
                    'num_operations': len(pandas3_results['operations']),
                    'successful_operations': sum(1 for op in pandas3_results['operations'] if op.get('success', False))
                }

            # Calculate improvements
            if dataset_results['pandas2'] and dataset_results['pandas3']:
                mem2 = dataset_results['pandas2']['memory_mb']
                mem3 = dataset_results['pandas3']['memory_mb']
                dataset_results['memory_reduction_pct'] = ((mem2 - mem3) / mem2 * 100) if mem2 > 0 else 0

                time2 = dataset_results['pandas2']['load_time']
                time3 = dataset_results['pandas3']['load_time']
                dataset_results['load_time_change_pct'] = ((time3 - time2) / time2 * 100) if time2 > 0 else 0

            self.summary['results'].append(dataset_results)
            print()

    def generate_summary_report(self):
        """Generate and display summary report"""
        print(f"\n{'='*70}")
        print("BENCHMARK SUMMARY REPORT")
        print(f"{'='*70}\n")

        print(f"Run timestamp: {self.summary['run_timestamp']}")
        print(f"Pandas 2 version: {self.summary['pandas_versions'].get('Pandas 2', 'N/A')}")
        print(f"Pandas 3 version: {self.summary['pandas_versions'].get('Pandas 3', 'N/A')}\n")

        print(f"{'Dataset':<35} {'Pandas 2':<15} {'Pandas 3':<15} {'Memory Savings':<15}")
        print(f"{'-'*35} {'-'*15} {'-'*15} {'-'*15}")

        total_mem2 = 0
        total_mem3 = 0

        for result in self.summary['results']:
            dataset_name = Path(result['dataset']).stem[:33]

            if result['pandas2'] and result['pandas3']:
                mem2 = result['pandas2']['memory_mb']
                mem3 = result['pandas3']['memory_mb']
                mem_reduction = result.get('memory_reduction_pct', 0)

                total_mem2 += mem2
                total_mem3 += mem3

                print(f"{dataset_name:<35} {mem2:>8.1f} MB    {mem3:>8.1f} MB    {mem_reduction:>8.1f}%")
            else:
                print(f"{dataset_name:<35} {'N/A':<15} {'N/A':<15} {'N/A':<15}")

        print(f"{'-'*35} {'-'*15} {'-'*15} {'-'*15}")

        if total_mem2 > 0:
            overall_reduction = ((total_mem2 - total_mem3) / total_mem2 * 100)
            print(f"{'TOTAL':<35} {total_mem2:>8.1f} MB    {total_mem3:>8.1f} MB    {overall_reduction:>8.1f}%")

        print(f"\n{'='*70}")
        print("DETAILED RESULTS BY DATASET")
        print(f"{'='*70}\n")

        for result in self.summary['results']:
            dataset_name = Path(result['dataset']).stem
            print(f"\n{dataset_name}:")

            if result['pandas2']:
                print(f"  Pandas 2: {result['pandas2']['memory_mb']:.1f} MB, "
                      f"{result['pandas2']['load_time']:.3f}s load, "
                      f"{result['pandas2']['successful_operations']}/{result['pandas2']['num_operations']} ops")

            if result['pandas3']:
                print(f"  Pandas 3: {result['pandas3']['memory_mb']:.1f} MB, "
                      f"{result['pandas3']['load_time']:.3f}s load, "
                      f"{result['pandas3']['successful_operations']}/{result['pandas3']['num_operations']} ops")

            if 'memory_reduction_pct' in result:
                print(f"  Improvement: {result['memory_reduction_pct']:.1f}% memory reduction, "
                      f"{result['load_time_change_pct']:+.1f}% load time change")

    def save_summary(self):
        """Save summary to JSON file"""
        summary_file = self.results_dir / f"benchmark_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(summary_file, 'w') as f:
            json.dump(self.summary, f, indent=2)
        print(f"\n✓ Summary saved to: {summary_file}")
        return summary_file

    def run(self):
        """Main execution flow"""
        print(f"\n{'='*70}")
        print("PANDAS 2 vs PANDAS 3 STRING OPERATIONS BENCHMARK")
        print(f"{'='*70}\n")

        # Verify setup
        if not self.verify_setup():
            print("\n❌ Setup verification failed. Please fix the issues above.")
            return False

        # Run all benchmarks
        self.run_all_benchmarks()

        # Generate reports
        self.generate_summary_report()
        self.save_summary()

        print(f"\n{'='*70}")
        print("ALL BENCHMARKS COMPLETE!")
        print(f"{'='*70}\n")
        print(f"Results directory: {self.results_dir.absolute()}")
        print(f"Individual result files: {len(list(self.results_dir.glob('*.json')))} JSON files")

        return True


def main():
    """Entry point"""
    runner = BenchmarkRunner()
    success = runner.run()
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
