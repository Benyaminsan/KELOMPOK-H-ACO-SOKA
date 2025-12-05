#!/usr/bin/env python3
"""
plot_results.py

Membuat plot dari file CSV perbandingan:
- Jika CSV berisi kolom 'Execution_Time', 'Load_Balance', 'Makespan' langsung digunakan.
- Jika CSV seperti `compare_results.csv` (algorithm, run, makespan) maka
  script akan meng-agregasi (mean) makespan per algorithm dan mem-plotnya.

Usage:
  python plot_results.py --csv compare_results.csv
  python plot_results.py --csv compare_results.csv --outdir results_plots --show
"""
import argparse
import os
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

def ensure_outdir(path):
    Path(path).mkdir(parents=True, exist_ok=True)
    return path

def plot_bar(df, x_col, y_col, title, outpath, show=False):
    plt.figure(figsize=(8,5))
    ax = plt.bar(df[x_col], df[y_col], color='C0')
    plt.title(title)
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.xticks(rotation=25, ha='right')
    plt.tight_layout()
    plt.savefig(outpath)
    if show:
        plt.show()
    plt.close()

def main(csv_file: str, outdir: str = "plots", show: bool = False):
    csv_file = Path(csv_file)
    if not csv_file.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_file}")

    ensure_outdir(outdir)
    df = pd.read_csv(csv_file)

    # normalize column names (case-insensitive)
    cols = {c.lower(): c for c in df.columns}
    has_exec = 'execution_time' in cols
    has_load = 'load_balance' in cols
    has_makespan = 'makespan' in cols or 'makespan' in cols

    # Accept either 'algorithm' or 'Algorithm'
    alg_col = cols.get('algorithm', cols.get('Algorithm', None))
    if alg_col is None:
        raise ValueError("CSV must contain 'algorithm' column (case-insensitive).")

    # If CSV already aggregated with mean columns, use them directly
    # Otherwise, try to aggregate per algorithm
    grouped = df.groupby(alg_col)

    # Prepare aggregated dataframe
    agg = {}
    if has_exec:
        agg_exec_col = cols['execution_time']
        agg['Execution_Time'] = grouped[agg_exec_col].mean()
    if has_load:
        agg_load_col = cols['load_balance']
        agg['Load_Balance'] = grouped[agg_load_col].mean()
    # handle lowercase 'makespan' or any case variant
    makespan_key = None
    for k in cols:
        if k == 'makespan':
            makespan_key = cols[k]
            break
    if makespan_key:
        agg['Makespan'] = grouped[makespan_key].mean()

    if not agg:
        raise ValueError("CSV doesn't contain any of ['Execution_Time','Load_Balance','Makespan'] (case-insensitive).")

    agg_df = pd.DataFrame(agg)
    agg_df.index.name = 'Algorithm'
    agg_df = agg_df.reset_index()

    # Plot available metrics
    if 'Execution_Time' in agg_df.columns:
        plot_bar(agg_df, 'Algorithm', 'Execution_Time',
                 'Execution Time Comparison (mean)', os.path.join(outdir, 'execution_time_comparison.png'), show)
    if 'Load_Balance' in agg_df.columns:
        plot_bar(agg_df, 'Algorithm', 'Load_Balance',
                 'Load Balance Comparison (mean)', os.path.join(outdir, 'load_balance_comparison.png'), show)
    if 'Makespan' in agg_df.columns:
        plot_bar(agg_df, 'Algorithm', 'Makespan',
                 'Makespan Comparison (mean)', os.path.join(outdir, 'makespan_comparison.png'), show)

    print(f"Graphs saved to: {outdir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot comparison results CSV")
    parser.add_argument("--csv", "-c", default="compare_results.csv", help="input CSV file (default: compare_results.csv)")
    parser.add_argument("--outdir", "-o", default="plots", help="output directory to save PNGs")
    parser.add_argument("--show", action="store_true", help="show plots interactively (calls plt.show())")
    args = parser.parse_args()
    main(args.csv, args.outdir, args.show)
