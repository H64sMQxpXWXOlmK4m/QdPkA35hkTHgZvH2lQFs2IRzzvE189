import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import warnings
warnings.filterwarnings('ignore')
import sys
sys.path.append('.')
from data_loader import DataLoader
from advanced_features import AdvancedFeatureEngineer
from calibration_improvement import (
    TemperatureScaling, BetaCalibration, evaluate_calibration, 
    compare_calibration_methods, plot_calibration_comparison
)

def load_model_and_data(symbol='BTC/USDT'):
    """Load saved model and compute validation predictions."""
    prefix = symbol.replace('/', '_')
    
    # Load artifacts
    with open(f'models/robust/{prefix}_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open(f'models/robust/{prefix}_feature_selector.pkl', 'rb') as f:
        selector = pickle.load(f)
    with open(f'models/robust/{prefix}_calibrator.pkl', 'rb') as f:
        calibrator = pickle.load(f)  # isotonic calibrator
    
    # Load features
    engineer = AdvancedFeatureEngineer(symbol, lookback_minutes=15)
    X, y = engineer.compute_all_features()
    
    # Temporal split (80% train, 20% test)
    split_idx = int(len(X) * 0.8)
    X_train = X.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    y_train = y.iloc[:split_idx]
    y_test = y.iloc[split_idx:]
    
    # Further split training into train/validation (80/20 of train)
    val_split = int(len(X_train) * 0.8)
    X_train_final = X_train.iloc[:val_split]
    X_val = X_train.iloc[val_split:]
    y_train_final = y_train.iloc[:val_split]
    y_val = y_train.iloc[val_split:]
    
    # Apply feature selection
    X_train_selected = selector.transform(X_train_final)
    X_val_selected = selector.transform(X_val)
    X_test_selected = selector.transform(X_test)
    
    # Get predictions
    y_train_pred = model.predict_proba(X_train_selected)[:, 1]
    y_val_pred = model.predict_proba(X_val_selected)[:, 1]
    y_test_pred = model.predict_proba(X_test_selected)[:, 1]
    
    # Apply saved calibrator (isotonic)
    y_val_cal = calibrator.predict(y_val_pred.reshape(-1, 1)).flatten()
    y_test_cal = calibrator.predict(y_test_pred.reshape(-1, 1)).flatten()
    
    return {
        'model': model,
        'selector': selector,
        'calibrator': calibrator,
        'X_train': X_train_final,
        'y_train': y_train_final,
        'X_val': X_val,
        'y_val': y_val,
        'X_test': X_test,
        'y_test': y_test,
        'y_train_pred': y_train_pred,
        'y_val_pred': y_val_pred,
        'y_test_pred': y_test_pred,
        'y_val_cal': y_val_cal,
        'y_test_cal': y_test_cal,
    }

def evaluate_existing_calibration(data):
    """Evaluate current calibration."""
    print("=== Existing Calibration Evaluation ===")
    
    # Validation set
    val_metrics = evaluate_calibration(data['y_val'].values, data['y_val_cal'])
    print(f"Validation set:")
    print(f"  ECE: {val_metrics['ece']:.4f}")
    print(f"  Brier: {val_metrics['brier']:.4f}")
    print(f"  Log loss: {val_metrics['log_loss']:.4f}")
    
    # Test set
    test_metrics = evaluate_calibration(data['y_test'].values, data['y_test_cal'])
    print(f"\nTest set:")
    print(f"  ECE: {test_metrics['ece']:.4f}")
    print(f"  Brier: {test_metrics['brier']:.4f}")
    print(f"  Log loss: {test_metrics['log_loss']:.4f}")
    
    # Tail calibration
    for region, stats in test_metrics['tail_stats'].items():
        print(f"  {region}: pred={stats['mean_pred']:.3f}, actual={stats['mean_actual']:.3f}, count={stats['count']}")
    
    return val_metrics, test_metrics

def compare_calibration_methods_on_validation(data):
    """Compare different calibration methods."""
    print("\n=== Comparing Calibration Methods ===")
    
    # Use validation set for training calibrators
    # We'll split validation into train_cal/val_cal
    y_val = data['y_val'].values
    y_val_pred = data['y_val_pred']
    
    n = len(y_val)
    split = int(0.7 * n)
    y_cal_train_true = y_val[:split]
    y_cal_train_pred = y_val_pred[:split]
    y_cal_val_true = y_val[split:]
    y_cal_val_pred = y_val_pred[split:]
    
    # Compare methods
    results = compare_calibration_methods(
        y_cal_train_pred, y_cal_train_true,
        y_cal_val_pred, y_cal_val_true
    )
    
    print("\nCalibration Method Performance (validation):")
    for name, res in results.items():
        print(f"\n{name}:")
        print(f"  ECE: {res['ece']:.4f}")
        print(f"  Brier: {res['brier']:.4f}")
        print(f"  Log loss: {res['log_loss']:.4f}")
        if 'tail_stats' in res:
            for region, stats in res['tail_stats'].items():
                print(f"  {region}: pred={stats['mean_pred']:.3f}, actual={stats['mean_actual']:.3f}")
    
    # Plot comparison
    plot_calibration_comparison(results, y_cal_val_true, y_cal_val_pred)
    
    return results

def regime_specific_calibration(data, regime_feats):
    """
    Test calibration separately for different regimes.
    """
    print("\n=== Regime-Specific Calibration ===")
    
    # Align regime features with validation set
    aligned_idx = data['X_val'].index.intersection(regime_feats.index)
    if len(aligned_idx) == 0:
        print("No aligned regime features")
        return
    
    y_val_aligned = data['y_val'].loc[aligned_idx].values
    y_pred_aligned = pd.Series(data['y_val_pred'], index=data['X_val'].index).loc[aligned_idx].values
    regime_aligned = regime_feats.loc[aligned_idx]
    
    # Analyze by volatility regime
    regimes = {
        'high_vol': regime_aligned['vol_high'] == 1,
        'low_vol': regime_aligned['vol_low'] == 1,
        'medium_vol': regime_aligned['vol_medium'] == 1,
    }
    
    regime_results = {}
    for regime_name, mask in regimes.items():
        if mask.sum() > 20:
            y_regime = y_val_aligned[mask]
            p_regime = y_pred_aligned[mask]
            
            # Fit temperature scaling per regime
            temp = TemperatureScaling()
            temp.fit(p_regime, y_regime)
            p_cal = temp.predict(p_regime)
            
            metrics = evaluate_calibration(y_regime, p_cal)
            regime_results[regime_name] = {
                'samples': len(y_regime),
                'temperature': temp.temperature,
                'ece': metrics['ece'],
                'brier': metrics['brier'],
            }
    
    print("\nRegime-specific temperature scaling:")
    for regime, res in regime_results.items():
        print(f"  {regime}: samples={res['samples']}, T={res['temperature']:.3f}, ECE={res['ece']:.4f}, Brier={res['brier']:.4f}")
    
    return regime_results

def main():
    symbol = 'BTC/USDT'
    print(f"Evaluating calibration for {symbol}")
    
    # 1. Load model and data
    data = load_model_and_data(symbol)
    print(f"Loaded model with {data['X_train'].shape[1]} features")
    print(f"Train samples: {len(data['X_train'])}")
    print(f"Validation samples: {len(data['X_val'])}")
    print(f"Test samples: {len(data['X_test'])}")
    
    # 2. Evaluate existing calibration
    val_metrics, test_metrics = evaluate_existing_calibration(data)
    
    # 3. Compare calibration methods
    cal_results = compare_calibration_methods_on_validation(data)
    
    # 4. Regime-specific calibration (load regime features)
    try:
        from regime_detection import RegimeDetector
        loader = DataLoader(data_dir='data/deep')
        df_hourly = loader.load_asset(symbol, timeframe='1h')
        detector = RegimeDetector(window_bars=168)
        regime_feats = detector.compute_regime_features(df_hourly)
        regime_results = regime_specific_calibration(data, regime_feats)
    except Exception as e:
        print(f"Could not compute regime-specific calibration: {e}")
    
    # 5. Select best calibration method and apply to test set
    print("\n=== Best Calibration Method ===")
    # Find method with lowest ECE on validation
    best_method = min(cal_results.items(), key=lambda x: x[1]['ece'])
    best_name, best_res = best_method
    print(f"Best method: {best_name} (ECE={best_res['ece']:.4f})")
    
    # Apply to test set
    calibrator = best_res['calibrator']['predict']
    y_test_cal_best = calibrator(data['y_test_pred'])
    test_metrics_best = evaluate_calibration(data['y_test'].values, y_test_cal_best)
    
    print(f"Test set with {best_name}:")
    print(f"  ECE: {test_metrics_best['ece']:.4f}")
    print(f"  Brier: {test_metrics_best['brier']:.4f}")
    print(f"  Log loss: {test_metrics_best['log_loss']:.4f}")
    
    # Compare with existing isotonic
    print(f"\nImprovement over existing isotonic:")
    print(f"  ECE: {test_metrics['ece']:.4f} -> {test_metrics_best['ece']:.4f} ({test_metrics['ece'] - test_metrics_best['ece']:.4f})")
    print(f"  Brier: {test_metrics['brier']:.4f} -> {test_metrics_best['brier']:.4f} ({test_metrics['brier'] - test_metrics_best['brier']:.4f})")
    
    # Plot final calibration curve
    from sklearn.calibration import calibration_curve
    prob_true, prob_pred = calibration_curve(data['y_test'].values, y_test_cal_best, n_bins=10)
    
    plt.figure(figsize=(8,6))
    plt.plot(prob_pred, prob_true, marker='o', label=f'{best_name} (ECE={test_metrics_best["ece"]:.3f})')
    plt.plot([0,1], [0,1], 'k--', label='Perfect')
    plt.xlabel('Mean predicted probability')
    plt.ylabel('Fraction of positives')
    plt.title(f'Final Calibration Curve ({symbol})')
    plt.legend()
    plt.grid(True)
    plt.savefig('final_calibration_curve.png')
    plt.close()
    
    print("\n=== Evaluation Complete ===")

if __name__ == '__main__':
    main()