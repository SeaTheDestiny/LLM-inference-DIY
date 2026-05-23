#!/usr/bin/env python3
"""
plot_results.py — Generate performance charts from bench_flash_attn results
===========================================================================
Usage: python plot_results.py [results.csv]

Reads CSV from bench_flash_attn.cu and generates:
  1. TFLOPS bar chart grouped by (N, d) for each kernel
  2. Speedup line chart vs N for each d
"""

import sys
import csv
import os

def load_csv(filepath):
    rows = []
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            row['N'] = int(row['N'])
            row['d'] = int(row['d'])
            row['time_ms'] = float(row['time_ms'])
            row['gflops'] = float(row['gflops'])
            rows.append(row)
    return rows

def try_plot_matplotlib(rows):
    """Try to plot with matplotlib. Falls back to ASCII chart if unavailable."""
    try:
        import matplotlib
        matplotlib.use('Agg')  # non-interactive backend
        import matplotlib.pyplot as plt
        import numpy as np
        return _plot_matplotlib(rows, plt, np)
    except ImportError:
        return _plot_ascii(rows)

def _plot_matplotlib(rows, plt, np):
    """Generate professional matplotlib charts."""
    
    kernels = sorted(set(r['kernel'] for r in rows))
    d_vals = sorted(set(r['d'] for r in rows))
    N_vals = sorted(set(r['N'] for r in rows))
    
    colors = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12', '#9B59B6']
    kernel_colors = dict(zip(kernels, colors[:len(kernels)]))
    
    fig, axes = plt.subplots(2, len(d_vals), figsize=(6 * len(d_vals), 11))
    if len(d_vals) == 1:
        axes = axes.reshape(2, 1)
    
    kernel_labels = {
        'finegrained_qk': 'Step1: FineGrained Q/K',
        'register_p': 'Step2: Register P',
        'async_doublebuf': 'Step3: Async+DoubleBuf',
        'final_optimized': 'Step4+5: Final Optimized',
    }
    
    # ===== Row 1: TFLOPS bar charts =====
    for di, d in enumerate(d_vals):
        ax = axes[0][di]
        data_d = [r for r in rows if r['d'] == d]
        
        x = np.arange(len(N_vals))
        width = 0.2
        bar_positions = []
        
        for ki, kernel in enumerate(kernels):
            vals = []
            for n in N_vals:
                found = [r for r in data_d if r['N'] == n and r['kernel'] == kernel]
                vals.append(found[0]['gflops'] / 1000.0 if found else 0)  # GFLOPS -> TFLOPS
            
            offset = (ki - len(kernels)/2 + 0.5) * width
            bars = ax.bar(x + offset, vals, width, 
                         label=kernel_labels.get(kernel, kernel),
                         color=kernel_colors[kernel], edgecolor='white', linewidth=0.5)
            bar_positions.append(x + offset)
        
        ax.set_title(f'd = {d}  |  TFLOPS Comparison', fontsize=13, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([f'N={n}' for n in N_vals])
        ax.set_ylabel('TFLOPS')
        ax.legend(fontsize=8, loc='upper left')
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for ki, kernel in enumerate(kernels):
            vals = []
            for n in N_vals:
                found = [r for r in data_d if r['N'] == n and r['kernel'] == kernel]
                vals.append(found[0]['gflops'] / 1000.0 if found else 0)
            for i, v in enumerate(vals):
                if v > 0:
                    ax.text(bar_positions[ki][i], v + max(vals)*0.02, 
                           f'{v:.1f}', ha='center', va='bottom', fontsize=7, rotation=90)
    
    # ===== Row 2: Speedup line charts =====
    for di, d in enumerate(d_vals):
        ax = axes[1][di]
        data_d = [r for r in rows if r['d'] == d]
        
        # Baseline: finegrained_qk at N=256
        baseline = None
        for r in data_d:
            if r['N'] == N_vals[0] and r['kernel'] == kernels[0]:
                baseline = r['time_ms']
                break
        
        for kernel in kernels:
            times = []
            for n in N_vals:
                found = [r for r in data_d if r['N'] == n and r['kernel'] == kernel]
                times.append(found[0]['time_ms'] if found else 0)
            
            if baseline:
                speedups = [baseline / t if t > 0 else 0 for t in times]
            else:
                speedups = times
            
            ax.plot(N_vals, speedups, 'o-', 
                   label=kernel_labels.get(kernel, kernel),
                   color=kernel_colors[kernel], linewidth=2, markersize=8)
        
        ax.set_title(f'd = {d}  |  Speedup vs N (baseline=N=256)', fontsize=13, fontweight='bold')
        ax.set_xlabel('Sequence Length N')
        ax.set_ylabel('Speedup (×)')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        ax.set_xscale('log', base=2)
        ax.set_xticks(N_vals)
        ax.set_xticklabels([str(n) for n in N_vals])
    
    plt.tight_layout(pad=2.0)
    outpath = os.path.join(os.path.dirname(__file__) or '.', 'bench_results.png')
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    print(f"\n[OK] Chart saved to: {outpath}")
    return True

def _plot_ascii(rows):
    """Fallback ASCII chart."""
    kernels = sorted(set(r['kernel'] for r in rows))
    N_vals = sorted(set(r['N'] for r in rows))
    d_vals = sorted(set(r['d'] for r in rows))
    
    print("\n" + "=" * 80)
    print("  FLASH ATTENTION BENCHMARK RESULTS")
    print("=" * 80)
    
    for d in d_vals:
        print(f"\n--- d = {d} ---")
        print(f"{'Kernel':<30}", end="")
        for n in N_vals:
            print(f"{'N='+str(n):>18}", end="")
        print()
        print("-" * (30 + 18 * len(N_vals)))
        
        for kernel in kernels:
            print(f"{kernel:<30}", end="")
            for n in N_vals:
                found = [r for r in rows if r['N']==n and r['d']==d and r['kernel']==kernel]
                if found:
                    r = found[0]
                    print(f"{r['time_ms']:10.4f}ms{r['gflops']:7.1f}G", end="")
                else:
                    print(f"{'N/A':>18}", end="")
            print()
    
    print("\n" + "-" * 80)
    print("Legend: ms=milliseconds, G=GFLOPS")
    print("For graphical charts, install matplotlib: pip install matplotlib")
    return False

if __name__ == '__main__':
    filepath = sys.argv[1] if len(sys.argv) > 1 else 'results.csv'
    
    if not os.path.exists(filepath):
        print(f"Error: {filepath} not found!")
        print("Run the benchmark first: ./bench_flash_attn > results.csv 2>log.txt")
        sys.exit(1)
    
    rows = load_csv(filepath)
    print(f"Loaded {len(rows)} benchmark results from {filepath}")
    
    try_plot_matplotlib(rows)
