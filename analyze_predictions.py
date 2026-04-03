#!/usr/bin/env python3
"""
Analyze prediction distribution from quick experiment.
"""
import sys
sys.path.append('.')
from elite_model_pipeline import EliteModelPipeline
from data_utils import get_data_range
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

def analyze_btc():
    """Analyze BTC predictions distribution."""
    symbol = 'BTC/USDT'
    start_date, end_date = get_data_range(symbol)
    # Use last 30 days
    quick_start = end_date - pd.Timedelta(days=30)
    
    print(f"Analyzing {symbol} from {quick_start} to {end_date}")
    
    pipeline = EliteModelPipeline(symbol)
    X, y = pipeline.create_full_dataset(quick_start, end_date)
    if X.empty:
        print("No data.")
        return
    
    print(f"Dataset shape: {X.shape}")
    
    # Run walk-forward validation
    initial_train_size = 20 * 24  # 20 days
    step_size = 48  # 2 days
    wf_results = pipeline.walk_forward_validation(X, y, initial_train_size, step_size)
    results_df = wf_results['results']
    xgb_results = results_df[results_df['model'] == 'xgb']
    
    if len(xgb_results) == 0:
        print("No XGBoost results.")
        return
    
    probs = xgb_results['prob']
    print(f"\nPrediction distribution for {symbol}:")
    print(f"Number of predictions: {len(probs)}")
    print(f"Mean probability: {probs.mean():.4f}")
    print(f"Std probability: {probs.std():.4f}")
    print(f"Min probability: {probs.min():.4f}")
    print(f"Max probability: {probs.max():.4f}")
    
    # Bin analysis
    bins = np.linspace(0, 1, 11)
    hist, bin_edges = np.histogram(probs, bins=bins)
    
    print("\nProbability distribution histogram:")
    for i in range(len(hist)):
        bin_low = bin_edges[i]
        bin_high = bin_edges[i+1]
        count = hist[i]
        percent = count / len(probs) * 100
        print(f"[{bin_low:.1f}, {bin_high:.1f}): {count:4d} ({percent:5.1f}%)")
    
    # Concentration in middle
    middle_mask = (probs >= 0.45) & (probs <= 0.55)
    middle_percent = middle_mask.sum() / len(probs) * 100
    print(f"\nPredictions in [0.45, 0.55]: {middle_mask.sum():d} ({middle_percent:.1f}%)")
    
    # Extreme predictions
    low_mask = probs < 0.3
    high_mask = probs > 0.7
    print(f"Predictions < 0.3: {low_mask.sum():d} ({low_mask.sum()/len(probs)*100:.1f}%)")
    print(f"Predictions > 0.7: {high_mask.sum():d} ({high_mask.sum()/len(probs)*100:.1f}%)")
    
    # Plot histogram
    plt.figure(figsize=(10, 6))
    plt.hist(probs, bins=20, edgecolor='black', alpha=0.7)
    plt.title(f'Predicted Probability Distribution - {symbol}')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Count')
    plt.axvline(0.5, color='red', linestyle='--', label='0.5')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('probability_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("\nPlot saved to probability_distribution.png")
    
    # Analyze features that might correlate with extreme predictions
    if not X.empty:
        # Get timestamps of extreme predictions
        extreme_mask = (probs < 0.3) | (probs > 0.7)
        if extreme_mask.any():
            extreme_timestamps = xgb_results.index[extreme_mask]
            print(f"\nExtreme prediction timestamps: {len(extreme_timestamps)}")
            
            # Get corresponding features
            # Note: X index may not align perfectly, need to merge
            # For simplicity, just note
            print("Extreme predictions exist - we should examine market states at those times.")
    
    return xgb_results, X

def analyze_multiple_assets():
    """Analyze multiple assets."""
    symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT']
    
    all_probs = []
    
    for sym in symbols:
        try:
            start_date, end_date = get_data_range(sym)
            quick_start = end_date - pd.Timedelta(days=30)
            
            pipeline = EliteModelPipeline(sym)
            X, y = pipeline.create_full_dataset(quick_start, end_date)
            if X.empty:
                continue
            
            initial_train_size = 20 * 24
            step_size = 48
            wf_results = pipeline.walk_forward_validation(X, y, initial_train_size, step_size)
            results_df = wf_results['results']
            xgb_results = results_df[results_df['model'] == 'xgb']
            
            if len(xgb_results) == 0:
                continue
            
            probs = xgb_results['prob']
            middle_percent = ((probs >= 0.45) & (probs <= 0.55)).sum() / len(probs) * 100
            
            print(f"{sym:9}: Mean={probs.mean():.4f}, Std={probs.std():.4f}, "
                  f"Middle[0.45-0.55]={middle_percent:.1f}%")
            
            all_probs.append(probs.values)
        except Exception as e:
            print(f"Error with {sym}: {e}")
            continue
    
    if all_probs:
        all_probs_flat = np.concatenate(all_probs)
        middle_percent = ((all_probs_flat >= 0.45) & (all_probs_flat <= 0.55)).sum() / len(all_probs_flat) * 100
        print(f"\nAll assets combined: {len(all_probs_flat)} predictions")
        print(f"Middle[0.45-0.55]: {middle_percent:.1f}%")
        print(f"Need <60% to pass: {'PASS' if middle_percent < 60 else 'FAIL'}")

if __name__ == '__main__':
    print("Analyzing prediction distributions...")
    analyze_btc()
    print("\n" + "="*80)
    print("Multiple assets analysis:")
    analyze_multiple_assets()