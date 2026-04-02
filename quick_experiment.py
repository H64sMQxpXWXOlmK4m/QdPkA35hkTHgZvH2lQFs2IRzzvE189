#!/usr/bin/env python3
"""
Quick experiment for all assets using last 30 days of data.
"""
import sys
sys.path.append('.')
from elite_model_pipeline import EliteModelPipeline
from data_utils import get_data_range
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def run_quick(symbol, days=30, train_ratio=0.7):
    """Run quick experiment for a symbol."""
    print(f"\n{'='*80}")
    print(f"QUICK EXPERIMENT: {symbol}")
    print(f"{'='*80}")
    
    # Get data range
    start_date, end_date = get_data_range(symbol)
    # Use last `days` days
    quick_start = end_date - pd.Timedelta(days=days)
    if quick_start < start_date:
        quick_start = start_date
    
    print(f"Data range: {quick_start} to {end_date}")
    print(f"Total days: {(end_date - quick_start).days}")
    
    # Initialize pipeline
    pipeline = EliteModelPipeline(symbol)
    
    # Run experiment with reduced initial train days
    initial_train_days = int((end_date - quick_start).days * train_ratio)
    if initial_train_days < 10:
        print(f"  Skipping: insufficient data")
        return None
    
    results = pipeline.run_experiment(
        start_date=quick_start,
        end_date=end_date,
        initial_train_days=initial_train_days
    )
    
    if not results:
        return None
    
    # Extract isotonic calibration metrics
    cal_res = results['calibration_results']['isotonic']
    metrics = cal_res['metrics']
    
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  ROC AUC: {metrics['roc_auc']:.4f}")
    print(f"  Brier: {metrics['brier']:.4f}")
    print(f"  ECE: {metrics['ece']:.4f}")
    
    return results

def main():
    symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'XRP/USDT', 'BNB/USDT', 'DOGE/USDT']
    
    all_results = {}
    
    for sym in symbols:
        try:
            results = run_quick(sym, days=30, train_ratio=0.7)
            if results:
                all_results[sym] = results
        except Exception as e:
            print(f"Error with {sym}: {e}")
            continue
    
    # Generate quick report
    if not all_results:
        print("No results.")
        return
    
    report = []
    report.append("=" * 80)
    report.append("QUICK EXPERIMENT RESULTS - LAST 30 DAYS")
    report.append("=" * 80)
    report.append(f"Generated: {pd.Timestamp.now()}")
    report.append("")
    
    # Table
    report.append("Asset     | Accuracy | ROC AUC | Brier   | ECE    | Samples")
    report.append("-" * 80)
    
    for sym, res in all_results.items():
        cal_res = res['calibration_results']['isotonic']
        metrics = cal_res['metrics']
        report.append(f"{sym:9} | {metrics['accuracy']:.4f} | {metrics['roc_auc']:.4f} | "
                     f"{metrics['brier']:.4f} | {metrics['ece']:.4f} | {metrics['n_samples']}")
    
    # Averages
    accs = [res['calibration_results']['isotonic']['metrics']['accuracy'] for res in all_results.values()]
    aucs = [res['calibration_results']['isotonic']['metrics']['roc_auc'] for res in all_results.values()]
    briers = [res['calibration_results']['isotonic']['metrics']['brier'] for res in all_results.values()]
    eces = [res['calibration_results']['isotonic']['metrics']['ece'] for res in all_results.values()]
    
    report.append("\nSUMMARY STATISTICS")
    report.append(f"Average Accuracy: {np.mean(accs):.4f} ± {np.std(accs):.4f}")
    report.append(f"Average ROC AUC: {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
    report.append(f"Average Brier: {np.mean(briers):.4f} ± {np.std(briers):.4f}")
    report.append(f"Average ECE: {np.mean(eces):.4f} ± {np.std(eces):.4f}")
    
    # Edge assessment
    brier_random = 0.25
    improvements = [(brier_random - b) / brier_random * 100 for b in briers]
    avg_improvement = np.mean(improvements)
    
    report.append(f"\nEDGE ASSESSMENT")
    report.append(f"Average Brier improvement over random: {avg_improvement:+.1f}%")
    if avg_improvement > 2:
        report.append("Verdict: POSITIVE EDGE DETECTED")
    elif avg_improvement > 0:
        report.append("Verdict: WEAK POSITIVE EDGE")
    else:
        report.append("Verdict: NO EDGE")
    
    # Feature importance consensus
    report.append("\nTOP FEATURES (consensus across assets)")
    report.append("-" * 80)
    
    feature_scores = {}
    for sym, res in all_results.items():
        imp_df = res['feature_importance']
        for _, row in imp_df.head(10).iterrows():
            feat = row['feature']
            importance = row['importance']
            feature_scores[feat] = feature_scores.get(feat, 0) + importance
    
    sorted_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)
    for feat, score in sorted_features[:10]:
        report.append(f"{feat:35}: {score:.4f}")
    
    report.append("\n" + "=" * 80)
    
    # Write report
    final = "\n".join(report)
    with open('QUICK_EXPERIMENT_REPORT.txt', 'w') as f:
        f.write(final)
    
    print("\n" + final)
    print("\nReport saved to QUICK_EXPERIMENT_REPORT.txt")

if __name__ == '__main__':
    main()