import pandas as pd
import numpy as np
import time
import gc
import json
import argparse
from pathlib import Path
from datetime import datetime

class StringOpsBenchmark:
    """Comprehensive benchmark for pandas string operations"""

    def __init__(self, dataset_path, pandas_version=None):
        self.dataset_path = dataset_path
        self.dataset_name = Path(dataset_path).stem
        self.pandas_version = pandas_version or pd.__version__
        self.results = {
            'metadata': {
                'dataset': self.dataset_name,
                'pandas_version': self.pandas_version,
                'timestamp': datetime.now().isoformat()
            },
            'loading': {},
            'operations': []
        }
        self.df = None

    def _get_string_columns(self):
        """
        Get list of string columns that works for both pandas 2 (object) and pandas 3 (StringDtype)
        """
        string_cols = []
        for col in self.df.columns:
            dtype_str = str(self.df[col].dtype).lower()
            # Check for: object, str, string, stringdtype, string[pyarrow]
            if self.df[col].dtype == 'object' or dtype_str in ['str', 'string'] or 'string' in dtype_str:
                string_cols.append(col)
        return string_cols

    def load_data(self):
        """Load the dataset and measure loading metrics"""
        print(f"\n{'='*70}")
        print(f"Loading dataset: {self.dataset_name}")
        print(f"Pandas version: {self.pandas_version}")
        print(f"{'='*70}\n")

        gc.collect()
        start_time = time.perf_counter()
        self.df = pd.read_csv(self.dataset_path)
        load_time = time.perf_counter() - start_time

        memory_mb = self.df.memory_usage(deep=True).sum() / (1024**2)

        self.results['loading'] = {
            'load_time_sec': round(load_time, 4),
            'memory_mb': round(memory_mb, 2),
            'num_rows': len(self.df),
            'num_cols': len(self.df.columns),
            'columns': list(self.df.columns),
            'dtypes': {col: str(dtype) for col, dtype in self.df.dtypes.items()}
        }

        print(f"✓ Loaded {len(self.df):,} rows, {len(self.df.columns)} columns")
        print(f"  Load time: {load_time:.4f} seconds")
        print(f"  Memory usage: {memory_mb:.2f} MB")
        print(f"  Dtypes: {self.df.dtypes.to_dict()}\n")

        return self.df

    def benchmark_operation(self, operation_name, operation_func, column_name):
        """
        Benchmark a single string operation

        Args:
            operation_name: Name of the operation (e.g., "str.lower()")
            operation_func: Function that takes a Series and returns modified Series
            column_name: Name of the column to operate on
        """
        if column_name not in self.df.columns:
            print(f"⚠ Skipping {operation_name}: column '{column_name}' not found")
            return None

        print(f"Testing: {operation_name} on '{column_name}'...", end=' ')

        # Measure memory before
        gc.collect()
        mem_before = self.df.memory_usage(deep=True).sum() / (1024**2)

        # Run the operation and measure time
        start_time = time.perf_counter()
        try:
            result = operation_func(self.df[column_name])
            exec_time = time.perf_counter() - start_time

            # Measure memory after (create temporary df to measure result memory)
            temp_df = self.df.copy()
            temp_df[column_name] = result
            mem_after = temp_df.memory_usage(deep=True).sum() / (1024**2)
            del temp_df
            gc.collect()

            mem_change = mem_after - mem_before

            result_info = {
                'operation': operation_name,
                'column': column_name,
                'exec_time_sec': round(exec_time, 4),
                'memory_before_mb': round(mem_before, 2),
                'memory_after_mb': round(mem_after, 2),
                'memory_change_mb': round(mem_change, 2),
                'result_dtype': str(result.dtype),
                'success': True
            }

            print(f"✓ {exec_time:.4f}s (mem: {mem_change:+.2f} MB)")

        except Exception as e:
            result_info = {
                'operation': operation_name,
                'column': column_name,
                'success': False,
                'error': str(e)
            }
            print(f"✗ Error: {str(e)}")

        self.results['operations'].append(result_info)
        return result_info

    def run_transformation_tests(self):
        """Test string transformation operations"""
        print(f"\n{'─'*70}")
        print("1. TRANSFORMATION OPERATIONS")
        print(f"{'─'*70}")

        # Get first string column
        string_cols = self._get_string_columns()
        if not string_cols:
            print("⚠ No string columns found")
            return

        test_col = string_cols[0]

        # Define transformations
        transformations = [
            ('str.lower()', lambda s: s.str.lower()),
            ('str.upper()', lambda s: s.str.upper()),
            ('str.strip()', lambda s: s.str.strip()),
            ('str.title()', lambda s: s.str.title()),
            ('str.capitalize()', lambda s: s.str.capitalize()),
        ]

        for op_name, op_func in transformations:
            self.benchmark_operation(op_name, op_func, test_col)

    def run_pattern_matching_tests(self):
        """Test string pattern matching operations"""
        print(f"\n{'─'*70}")
        print("2. PATTERN MATCHING OPERATIONS")
        print(f"{'─'*70}")

        string_cols = self._get_string_columns()
        if not string_cols:
            return

        test_col = string_cols[0]

        pattern_ops = [
            ('str.contains("a")', lambda s: s.str.contains('a', na=False)),
            ('str.startswith("A")', lambda s: s.str.startswith('A', na=False)),
            ('str.endswith("z")', lambda s: s.str.endswith('z', na=False)),
            ('str.match("[A-Z]+")', lambda s: s.str.match('[A-Z]+', na=False)),
        ]

        for op_name, op_func in pattern_ops:
            self.benchmark_operation(op_name, op_func, test_col)

    def run_extraction_replacement_tests(self):
        """Test string extraction and replacement operations"""
        print(f"\n{'─'*70}")
        print("3. EXTRACTION & REPLACEMENT OPERATIONS")
        print(f"{'─'*70}")

        string_cols = self._get_string_columns()
        if not string_cols:
            return

        test_col = string_cols[0]

        extraction_ops = [
            ('str.replace("a", "X")', lambda s: s.str.replace('a', 'X', regex=False)),
            ('str.slice(0, 5)', lambda s: s.str.slice(0, 5)),
            ('str.len()', lambda s: s.str.len()),
        ]

        # Only test split on certain datasets
        if 'mixed' in self.dataset_name or 'high_cardinality' in self.dataset_name:
            extraction_ops.append(('str.split("@")', lambda s: s.str.split('@')))

        for op_name, op_func in extraction_ops:
            self.benchmark_operation(op_name, op_func, test_col)

    def run_aggregation_tests(self):
        """Test string aggregation operations"""
        print(f"\n{'─'*70}")
        print("4. AGGREGATION OPERATIONS")
        print(f"{'─'*70}")

        string_cols = self._get_string_columns()
        if not string_cols:
            return

        test_col = string_cols[0]

        # value_counts - special handling since it returns a different structure
        print(f"Testing: value_counts() on '{test_col}'...", end=' ')
        gc.collect()
        start_time = time.perf_counter()
        try:
            result = self.df[test_col].value_counts()
            exec_time = time.perf_counter() - start_time

            self.results['operations'].append({
                'operation': 'value_counts()',
                'column': test_col,
                'exec_time_sec': round(exec_time, 4),
                'result_size': len(result),
                'success': True
            })
            print(f"✓ {exec_time:.4f}s ({len(result):,} unique values)")
        except Exception as e:
            self.results['operations'].append({
                'operation': 'value_counts()',
                'column': test_col,
                'success': False,
                'error': str(e)
            })
            print(f"✗ Error: {str(e)}")

        # nunique
        print(f"Testing: nunique() on '{test_col}'...", end=' ')
        gc.collect()
        start_time = time.perf_counter()
        try:
            result = self.df[test_col].nunique()
            exec_time = time.perf_counter() - start_time

            self.results['operations'].append({
                'operation': 'nunique()',
                'column': test_col,
                'exec_time_sec': round(exec_time, 4),
                'result': int(result),
                'success': True
            })
            print(f"✓ {exec_time:.4f}s ({result:,} unique)")
        except Exception as e:
            self.results['operations'].append({
                'operation': 'nunique()',
                'column': test_col,
                'success': False,
                'error': str(e)
            })
            print(f"✗ Error: {str(e)}")

    def run_groupby_tests(self):
        """Test groupby operations on string columns"""
        print(f"\n{'─'*70}")
        print("5. GROUPBY OPERATIONS")
        print(f"{'─'*70}")

        string_cols = self._get_string_columns()
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()

        if not string_cols or not numeric_cols:
            print("⚠ Need both string and numeric columns for groupby tests")
            return

        group_col = string_cols[0]
        agg_col = numeric_cols[0]

        print(f"Testing: groupby('{group_col}').{agg_col}.mean()...", end=' ')
        gc.collect()
        start_time = time.perf_counter()
        try:
            result = self.df.groupby(group_col)[agg_col].mean()
            exec_time = time.perf_counter() - start_time

            self.results['operations'].append({
                'operation': f'groupby().mean()',
                'column': f'{group_col} -> {agg_col}',
                'exec_time_sec': round(exec_time, 4),
                'num_groups': len(result),
                'success': True
            })
            print(f"✓ {exec_time:.4f}s ({len(result):,} groups)")
        except Exception as e:
            self.results['operations'].append({
                'operation': 'groupby().mean()',
                'column': f'{group_col} -> {agg_col}',
                'success': False,
                'error': str(e)
            })
            print(f"✗ Error: {str(e)}")

    def run_all_benchmarks(self):
        """Run all benchmark tests"""
        self.load_data()

        self.run_transformation_tests()
        self.run_pattern_matching_tests()
        self.run_extraction_replacement_tests()
        self.run_aggregation_tests()
        self.run_groupby_tests()

        print(f"\n{'='*70}")
        print("BENCHMARK COMPLETE")
        print(f"{'='*70}")
        print(f"Total operations tested: {len(self.results['operations'])}")
        successful = sum(1 for op in self.results['operations'] if op.get('success', False))
        print(f"Successful: {successful}/{len(self.results['operations'])}")

        return self.results

    def save_results(self, output_dir='results'):
        """Save benchmark results to JSON file"""
        Path(output_dir).mkdir(exist_ok=True)

        output_file = Path(output_dir) / f"{self.dataset_name}_pandas{self.pandas_version.split('.')[0]}.json"

        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)

        print(f"\n✓ Results saved to: {output_file}")
        return output_file


def main():
    parser = argparse.ArgumentParser(description='Benchmark pandas string operations')
    parser.add_argument('dataset', help='Path to CSV dataset')
    parser.add_argument('--output-dir', default='results', help='Output directory for results')

    args = parser.parse_args()

    # Run benchmark
    benchmark = StringOpsBenchmark(args.dataset)
    benchmark.run_all_benchmarks()
    benchmark.save_results(args.output_dir)


if __name__ == '__main__':
    main()
