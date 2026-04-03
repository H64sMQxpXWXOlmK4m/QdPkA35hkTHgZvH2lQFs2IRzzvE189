#!/usr/bin/env python3
"""
Compute test-only metrics from proper regime predictions.
"""
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def main():
    # Load predictions
    df = pd.read_csv('proper_regime_predictions.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp')
    
    print(f"Loaded {len(df)} predictions")
    
    # For each regime, identify test set (last 20% chronologically)
    test_preds = []
    test_true = []
    test_regimes = []
    test_probs = []
    
    for regime in df['regime'].unique():
        regime_df = df[df['regime'] == regime].copy()
        regime_df = regime_df.sort_values('timestamp')
        
        # Determine split (80% train, 20% test)
        split_idx = int(len(regime_df) * 0.8)
        test_df = regime_df.iloc[split_idx:]
        
        test_preds.extend(test_df['pred'].values)
        test_true.extend(test_df['true'].values)
        test_regimes.extend(test_df['regime'].values)
        test_probs.extend(test_df['prob'].values)
    
    test_preds = np.array(test_preds)
    test_true = np.array(test_true)
    test_regimes = np.array(test_regimes)
    test_probs = np.array(test_probs)
    
    print(f"\nTest set size: {len(test_true)} ({len(test_true)/len(df)*100:.1f}% of total)")
    
    # Overall test metrics
    accuracy = (test_preds == test_true).mean()
    brier = ((test_probs - test_true) ** 2).mean()
    base_rate = test_true.mean()
    
    print(f"\nTEST SET METRICS (proper evaluation):")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Brier score: {brier:.4f}")
    print(f"  Base rate: {base_rate:.4f}")
    
    # Probability distribution
    bins = np.linspace(0, 1, 11)
    hist, _ = np.histogram(test_probs, bins=bins)
    
    print("\nProbability distribution (test set):")
    for i in range(len(hist)):
        bin_low = bins[i]
        bin_high = bins[i+1]
        percent = hist[i] / len(test_probs) * 100
        print(f"[{bin_low:.1f}, {bin_high:.1f}): {percent:5.1f}%")
    
    middle_percent = ((test_probs >= 0.45) & (test_probs <= 0.55)).sum() / len(test_probs) * 100
    print(f"\nPredictions in [0.45, 0.55]: {middle_percent:.1f}%")
    
    # Per-regime test metrics
    print("\nPer-regime test metrics:")
    for regime in np.unique(test_regimes):
        mask = test_regimes == regime
        if mask.sum() == 0:
            continue
        
        regime_probs = test_probs[mask]
        regime_true = test_true[mask]
        regime_preds = test_preds[mask]
        
        accuracy = (regime_preds == regime_true).mean()
        brier = ((regime_probs - regime_true) ** 2).mean()
        mean_prob = regime_probs.mean()
        
        print(f"  Regime {regime}: samples={mask.sum():3d}, accuracy={accuracy:.3f}, "
              f"brier={brier:.4f}, mean_prob={mean_prob:.3f}")
    
    # Generate updated report
    generate_report(test_probs, test_true, test_regimes, accuracy, brier, middle_percent)

def generate_report(test_probs, test_true, test_regimes, accuracy, brier, middle_percent):
    """Generate final test-only report."""
    report = []
    report.append("=" * 80)
    report.append("STATE-CONDITIONAL MODEL REPORT - TEST SET ONLY")
    report.append("=" * 80)
    report.append(f"Generated: {pd.Timestamp.now()}")
    report.append(f"Symbol: BTC/USDT")
    report.append(f"Test predictions: {len(test_probs)}")
    report.append("")
    
    # 1. Probability Distribution
    report.append("1. PROBABILITY DISTRIBUTION (TEST SET)")
    report.append("-" * 80)
    
    bins = np.linspace(0, 1, 11)
    hist, _ = np.histogram(test_probs, bins=bins)
    
    for i in range(len(hist)):
        bin_low = bins[i]
        bin_high = bins[i+1]
        percent = hist[i] / len(test_probs) * 100
        report.append(f"[{bin_low:.1f}, {bin_high:.1f}): {percent:5.1f}%")
    
    report.append(f"\nConcentration in [0.45, 0.55]: {middle_percent:.1f}%")
    report.append(f"Required: <60% ({"PASS" if middle_percent < 60 else "FAIL"})")
    
    # 2. Overall Test Performance
    report.append("\n2. OVERALL TEST PERFORMANCE")
    report.append("-" * 80)
    report.append(f"Accuracy: {accuracy:.4f}")
    report.append(f"Brier score: {brier:.4f}")
    report.append(f"Base rate: {test_true.mean():.4f}")
    report.append(f"Probability std: {test_probs.std():.4f}")
    
    # 3. Calibration Check
    report.append("\n3. CALIBRATION CHECK")
    report.append("-" * 80)
    
    # Simple mean prob vs actual rate
    mean_prob = test_probs.mean()
    actual_rate = test_true.mean()
    deviation = actual_rate - mean_prob
    report.append(f"Mean predicted probability: {mean_prob:.3f}")
    report.append(f"Actual up rate: {actual_rate:.3f}")
    report.append(f"Deviation: {deviation:.3f}")
    
    # 4. Edge Assessment
    report.append("\n4. EDGE ASSESSMENT")
    report.append("-" * 80)
    
    # Compare to random classifier
    brier_random = 0.25  # For balanced binary classification
    brier_improvement = (brier_random - brier) / brier_random * 100
    
    report.append(f"Brier improvement over random: {brier_improvement:+.1f}%")
    
    if accuracy > 0.5:
        edge = accuracy - max(test_true.mean(), 1 - test_true.mean())
        report.append(f"Accuracy edge over majority class: {edge:.4f}")
    
    # 5. Final Verdict
    report.append("\n5. FINAL VERDICT")
    report.append("-" * 80)
    
    has_dispersion = middle_percent < 60
    has_edge = accuracy > 0.5
    has_non_trivial = test_probs.std() > 0.1
    
    report.append(f"1. Probability dispersion: {middle_percent:.1f}% in middle (need <60%): {'✅ PASS' if has_dispersion else '❌ FAIL'}")
    report.append(f"2. Overall edge: accuracy={accuracy:.3f} (need >0.5): {'✅ PASS' if has_edge else '❌ FAIL'}")
    report.append(f"3. Non-trivial probabilities: std={test_probs.std():.3f} (need >0.1): {'✅ PASS' if has_non_trivial else '❌ FAIL'}")
    report.append("")
    
    if has_dispersion and has_edge and has_non_trivial:
        report.append("✅ SYSTEM SUCCESS: Model produces non-trivial probabilities with edge")
        report.append("   The conditional modeling approach works.")
    else:
        report.append("❌ SYSTEM FAILURE: Does not meet all requirements")
        if not has_edge:
            report.append("   Model lacks predictive edge on test set.")
    
    report.append("\n" + "=" * 80)
    
    # Write report
    with open('TEST_SET_REPORT.txt', 'w') as f:
        f.write("\n".join(report))
    
    print(f"\nTest set report saved to TEST_SET_REPORT.txt")

if __name__ == '__main__':
    main()