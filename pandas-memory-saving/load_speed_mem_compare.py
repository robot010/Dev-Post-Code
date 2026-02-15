import pandas as pd
import time
import gc
import argparse

filename = "test_strings_1M.csv"

def benchmark_load(name, **kwargs):
    # Clear memory before each run
    gc.collect()
    
    start_time = time.perf_counter()
    df = pd.read_csv(filename, **kwargs)
    end_time = time.perf_counter()
    
    # Calculate memory in MB
    memory_mb = df.memory_usage(deep=True).sum() / (1024**2)
    load_time = end_time - start_time
    
    print(f"--- {pd.__version__}: Loading {len(df)} rows, column name = {df.columns}---")
    print(f"Load Time: {load_time:.4f} seconds")
    print(f"Memory Usage: {memory_mb:.2f} MB")
    print(f"String Dtype: {df.iloc[:, 0].dtype}\n")
    
    return load_time, memory_mb

if __name__ == "__main__":

    print(f"Benchmarking file: {filename}\n")
    benchmark_load("Default (numpy object)")
