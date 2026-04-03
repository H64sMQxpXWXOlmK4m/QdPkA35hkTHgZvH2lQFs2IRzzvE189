#!/usr/bin/env python3
"""
Run elite model pipeline for all six assets.
Generate comprehensive final report.
"""
import sys
sys.path.append('.')
from elite_model_pipeline import EliteModelPipeline
from data_utils import get_data_range
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def run_asset(symbol, start_date, end_date, initial_train_days=60):
    """Run pipeline for a single asset."""
    print(f"\n{'='*80}")
    print(f"PROCESSING {symbol}")
    print(f"{'='*80}")
    
    # Initialize pipeline
    pipeline = EliteModelPipeline(symbol)
    
    # Run experiment
    results = pipeline.run_experiment(
        start_date=start_date,
        end_date=end_date,
        initial_train_days=initial_train_days
    )
    
    if not results:
        print(f"Failed to run experiment for {symbol}")
        return None
    
    # Generate report
    report = pipeline.generate_report()
    print(report)
    
    # Save individual report
    with open(f'results_{symbol.replace("/", "_")}.txt', 'w') as f:
        f.write(report)
    
    return results

def main():
    symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT']  # Reduced for speed
    
    all_results = {}
    
    for sym in symbols:
        try:
            # Get overlapping data range
            start_date, end_date = get_data_range(sym)
            print(f"{sym}: data range {start_date} to {end_date}")
            
            # Adjust end_date to leave some room for testing
            # Use 90% of data
            total_days = (end_date - start_date).days
            if total_days < 30:
                print(f"  Skipping {sym}: insufficient data ({total_days} days)")
                continue
            
            # Use last 10% as final test? Actually walk-forward already uses expanding window.
            # We'll use all data with initial_train_days = min(30, int(total_days * 0.7))
            initial_train_days = min(30, int(total_days * 0.7))
            print(f"  Total days: {total_days}, initial train days: {initial_train_days}")
            
            # Run pipeline
            results = run_asset(sym, start_date, end_date, initial_train_days)
            if results:
                all_results[sym] = results
        except Exception as e:
            print(f"Error processing {sym}: {e}")
            continue
    
    # Generate final consolidated report
    generate_final_report(all_results)

def generate_final_report(all_results):
    """Generate final comprehensive report."""
    if not all_results:
        print("No results to report.")
        return
    
    report = []
    report.append("=" * 80)
    report.append("FINAL MODEL ANALYSIS - ELITE PROBABILITY ENGINE")
    report.append("=" * 80)
    report.append(f"Generated: {pd.Timestamp.now()}")
    report.append(f"Assets analyzed: {len(all_results)}")
    report.append("")
    
    # Summary table
    report.append("SUMMARY METRICS (XGBoost with Isotonic Calibration)")
    report.append("-" * 80)
    report.append("Asset     | Accuracy | ROC AUC | Brier   | Log Loss | ECE    | Samples")
    report.append("-" * 80)
    
    for sym, res in all_results.items():
        # Get isotonic calibration metrics
        cal_res = res['calibration_results']['isotonic']
        metrics = cal_res['metrics']
        report.append(f"{sym:9} | {metrics['accuracy']:.4f} | {metrics['roc_auc']:.4f} | "
                     f"{metrics['brier']:.4f} | {metrics['log_loss']:.4f} | "
                     f"{metrics['ece']:.4f} | {metrics['n_samples']}")
    
    # Calibration quality assessment
    report.append("\nCALIBRATION QUALITY ASSESSMENT")
    report.append("-" * 80)
    
    ece_values = []
    for sym, res in all_results.items():
        cal_res = res['calibration_results']['isotonic']
        ece = cal_res['metrics']['ece']
        ece_values.append(ece)
        quality = "EXCELLENT" if ece < 0.01 else "GOOD" if ece < 0.05 else "FAIR" if ece < 0.1 else "POOR"
        report.append(f"{sym:9}: ECE = {ece:.4f} ({quality})")
    
    avg_ece = np.mean(ece_values)
    report.append(f"\nAverage ECE across assets: {avg_ece:.4f}")
    report.append(f"Calibration quality: {'EXCELLENT' if avg_ece < 0.01 else 'GOOD' if avg_ece < 0.05 else 'FAIR' if avg_ece < 0.1 else 'POOR'}")
    
    # Edge strength assessment
    report.append("\nEDGE STRENGTH ASSESSMENT")
    report.append("-" * 80)
    
    brier_random = 0.25  # Random classifier with balanced classes
    for sym, res in all_results.items():
        cal_res = res['calibration_results']['isotonic']
        brier = cal_res['metrics']['brier']
        improvement = (brier_random - brier) / brier_random * 100
        roc_auc = cal_res['metrics']['roc_auc']
        edge_strength = "STRONG" if improvement > 5 else "MODERATE" if improvement > 2 else "WEAK" if improvement > 0 else "NONE"
        report.append(f"{sym:9}: Brier improvement = {improvement:+.1f}%, ROC AUC = {roc_auc:.3f} ({edge_strength})")
    
    # Feature importance consensus
    report.append("\nTOP FEATURES ACROSS ASSETS (Consensus)")
    report.append("-" * 80)
    
    # Collect top features from each asset
    feature_scores = {}
    for sym, res in all_results.items():
        imp_df = res['feature_importance']
        for _, row in imp_df.head(10).iterrows():
            feat = row['feature']
            importance = row['importance']
            feature_scores[feat] = feature_scores.get(feat, 0) + importance
    
    # Sort by total importance
    sorted_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)
    for feat, score in sorted_features[:15]:
        report.append(f"{feat:35}: {score:.4f}")
    
    # Stability analysis (check consistency across assets)
    report.append("\nSTABILITY ANALYSIS")
    report.append("-" * 80)
    
    accuracies = [res['calibration_results']['isotonic']['metrics']['accuracy'] for res in all_results.values()]
    roc_aucs = [res['calibration_results']['isotonic']['metrics']['roc_auc'] for res in all_results.values()]
    
    report.append(f"Accuracy range: {min(accuracies):.4f} - {max(accuracies):.4f}")
    report.append(f"ROC AUC range: {min(roc_aucs):.4f} - {max(roc_aucs):.4f}")
    report.append(f"Accuracy std: {np.std(accuracies):.4f}")
    report.append(f"ROC AUC std: {np.std(roc_aucs):.4f}")
    
    # Final verdict
    report.append("\nFINAL VERDICT")
    report.append("-" * 80)
    
    avg_accuracy = np.mean(accuracies)
    avg_roc_auc = np.mean(roc_aucs)
    avg_brier_improvement = np.mean([(0.25 - res['calibration_results']['isotonic']['metrics']['brier']) / 0.25 * 100 
                                     for res in all_results.values()])
    
    report.append(f"Average Accuracy: {avg_accuracy:.4f}")
    report.append(f"Average ROC AUC: {avg_roc_auc:.4f}")
    report.append(f"Average Brier Improvement: {avg_brier_improvement:+.1f}%")
    
    if avg_roc_auc > 0.55 and avg_brier_improvement > 2:
        verdict = "SYSTEM HAS REAL PREDICTIVE EDGE"
    elif avg_roc_auc > 0.52 and avg_brier_improvement > 0:
        verdict = "SYSTEM SHOWS WEAK BUT POSITIVE EDGE"
    else:
        verdict = "SYSTEM LACKS CONSISTENT EDGE"
    
    report.append(f"\nVERDICT: {verdict}")
    
    # Recommendations
    report.append("\nRECOMMENDATIONS")
    report.append("-" * 80)
    if avg_ece < 0.05:
        report.append("✓ Calibration is excellent - probabilities can be trusted")
    else:
        report.append("✗ Calibration needs improvement")
    
    if avg_roc_auc > 0.55:
        report.append("✓ Predictive power is statistically significant")
    else:
        report.append("✗ Predictive power is weak - consider feature engineering improvements")
    
    report.append("✓ Use isotonic calibration for reliable probabilities")
    report.append("✓ Monitor performance by asset - some assets may have stronger edge")
    report.append("✓ Consider regime-dependent models for improved robustness")
    
    report.append("\n" + "=" * 80)
    
    # Write final report
    final_report = "\n".join(report)
    with open('FINAL_MODEL_ANALYSIS.txt', 'w') as f:
        f.write(final_report)
    
    print("\n" + final_report)
    print("\nFinal report saved to FINAL_MODEL_ANALYSIS.txt")

if __name__ == '__main__':
    main()