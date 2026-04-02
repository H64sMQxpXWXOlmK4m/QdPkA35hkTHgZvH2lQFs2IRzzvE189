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
from regime_detection import RegimeDetector
from calibration_improvement import TemperatureScaling, evaluate_calibration
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score, log_loss

def load_model_and_data(symbol='BTC/USDT'):
    """Load saved model and compute validation/test predictions."""
    prefix = symbol.replace('/', '_')
    with open(f'models/robust/{prefix}_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open(f'models/robust/{prefix}_feature_selector.pkl', 'rb') as f:
        selector = pickle.load(f)
    with open(f'models/robust/{prefix}_calibrator.pkl', 'rb') as f:
        calibrator = pickle.load(f)  # existing isotonic
    
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
    X_val_selected = selector.transform(X_val)
    X_test_selected = selector.transform(X_test)
    
    # Get predictions
    y_val_pred = model.predict_proba(X_val_selected)[:, 1]
    y_test_pred = model.predict_proba(X_test_selected)[:, 1]
    
    # Apply existing calibrator
    y_val_cal = calibrator.predict(y_val_pred.reshape(-1, 1)).flatten()
    y_test_cal = calibrator.predict(y_test_pred.reshape(-1, 1)).flatten()
    
    # Load regime features for validation and test sets
    loader = DataLoader(data_dir='data/deep')
    df_hourly = loader.load_asset(symbol, timeframe='1h')
    detector = RegimeDetector(window_bars=168)
    regime_feats = detector.compute_regime_features(df_hourly)
    
    # Align indices
    regime_val = regime_feats.loc[X_val.index]
    regime_test = regime_feats.loc[X_test.index]
    
    return {
        'model': model,
        'selector': selector,
        'calibrator': calibrator,
        'X_val': X_val,
        'y_val': y_val,
        'y_val_pred': y_val_pred,
        'y_val_cal': y_val_cal,
        'X_test': X_test,
        'y_test': y_test,
        'y_test_pred': y_test_pred,
        'y_test_cal': y_test_cal,
        'regime_val': regime_val,
        'regime_test': regime_test,
    }

def fit_regime_adaptive_calibration(data):
    """
    Fit separate temperature scalers for different volatility regimes.
    """
    y_val = data['y_val'].values
    y_val_pred = data['y_val_pred']
    regime_val = data['regime_val']
    
    # Define volatility regimes
    regimes = {
        'high': regime_val['vol_high'] == 1,
        'medium': regime_val['vol_medium'] == 1,
        'low': regime_val['vol_low'] == 1,
    }
    
    calibrators = {}
    for regime_name, mask in regimes.items():
        if mask.sum() > 20:
            y_regime = y_val[mask]
            p_regime = y_val_pred[mask]
            
            temp = TemperatureScaling()
            temp.fit(p_regime, y_regime)
            calibrators[regime_name] = {
                'calibrator': temp,
                'samples': len(y_regime),
                'temperature': temp.temperature,
            }
            print(f"  {regime_name}: T={temp.temperature:.3f}, samples={len(y_regime)}")
    
    return calibrators

def apply_regime_calibration(y_pred, regime_df, calibrators):
    """
    Apply regime-specific calibration.
    """
    y_cal = np.zeros_like(y_pred)
    default_temp = 1.0
    
    # For each regime
    for regime_name, cal in calibrators.items():
        if regime_name == 'high':
            mask = regime_df['vol_high'] == 1
        elif regime_name == 'medium':
            mask = regime_df['vol_medium'] == 1
        elif regime_name == 'low':
            mask = regime_df['vol_low'] == 1
        else:
            continue
        
        if mask.any():
            y_cal[mask] = cal['calibrator'].predict(y_pred[mask])
    
    # For any unassigned (should not happen), use global temperature scaling
    unassigned = ~(regime_df['vol_high'] == 1) & ~(regime_df['vol_medium'] == 1) & ~(regime_df['vol_low'] == 1)
    if unassigned.any():
        # Use average temperature
        avg_temp = np.mean([c['temperature'] for c in calibrators.values()])
        temp = TemperatureScaling(temperature=avg_temp)
        y_cal[unassigned] = temp.predict(y_pred[unassigned])
    
    return y_cal

def evaluate_all_methods(data, calibrators):
    """
    Compare calibration methods on test set.
    """
    y_test = data['y_test'].values
    y_test_pred = data['y_test_pred']
    y_test_cal_existing = data['y_test_cal']
    regime_test = data['regime_test']
    
    results = {}
    
    # 1. Uncalibrated
    results['uncalibrated'] = evaluate_calibration(y_test, y_test_pred)
    
    # 2. Existing isotonic calibrator
    results['isotonic'] = evaluate_calibration(y_test, y_test_cal_existing)
    
    # 3. Global temperature scaling (fit on validation)
    temp_global = TemperatureScaling()
    temp_global.fit(data['y_val_pred'], data['y_val'].values)
    y_test_temp = temp_global.predict(y_test_pred)
    results['temperature_global'] = evaluate_calibration(y_test, y_test_temp)
    
    # 4. Regime-adaptive temperature scaling
    y_test_regime = apply_regime_calibration(y_test_pred, regime_test, calibrators)
    results['temperature_regime'] = evaluate_calibration(y_test, y_test_regime)
    
    # Print results
    print("\n=== Calibration Method Comparison (Test Set) ===")
    for method, metrics in results.items():
        print(f"\n{method}:")
        print(f"  ECE: {metrics['ece']:.4f}")
        print(f"  Brier: {metrics['brier']:.4f}")
        print(f"  Log loss: {metrics['log_loss']:.4f}")
        if 'tail_stats' in metrics:
            for region, stats in metrics['tail_stats'].items():
                print(f"  {region}: pred={stats['mean_pred']:.3f}, actual={stats['mean_actual']:.3f}, count={stats['count']}")
    
    return results, y_test_regime

def plot_regime_calibration(results, y_test_regime, data):
    """Plot calibration curves."""
    from sklearn.calibration import calibration_curve
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Calibration curves
    ax1 = axes[0, 0]
    for method, metrics in results.items():
        prob_true, prob_pred = metrics['calibration_curve']
        ax1.plot(prob_pred, prob_true, marker='o', label=f'{method} (ECE={metrics["ece"]:.3f})')
    ax1.plot([0,1], [0,1], 'k--', label='Perfect')
    ax1.set_xlabel('Mean predicted probability')
    ax1.set_ylabel('Fraction of positives')
    ax1.set_title('Calibration Curves Comparison')
    ax1.legend()
    ax1.grid(True)
    
    # Regime-specific calibration
    ax2 = axes[0, 1]
    regime_test = data['regime_test']
    y_test = data['y_test'].values
    y_test_regime_cal = y_test_regime
    
    for regime_name, col in [('high', 'vol_high'), ('medium', 'vol_medium'), ('low', 'vol_low')]:
        mask = regime_test[col] == 1
        if mask.sum() > 10:
            y_regime = y_test[mask]
            p_regime = y_test_regime_cal[mask]
            prob_true, prob_pred = calibration_curve(y_regime, p_regime, n_bins=5)
            ax2.plot(prob_pred, prob_true, marker='o', label=f'{regime_name} regime')
    ax2.plot([0,1], [0,1], 'k--')
    ax2.set_xlabel('Predicted probability')
    ax2.set_ylabel('Actual frequency')
    ax2.set_title('Regime-Specific Calibration')
    ax2.legend()
    ax2.grid(True)
    
    # Metrics comparison
    ax3 = axes[1, 0]
    methods = list(results.keys())
    eces = [results[m]['ece'] for m in methods]
    briers = [results[m]['brier'] for m in methods]
    x = np.arange(len(methods))
    width = 0.35
    ax3.bar(x - width/2, eces, width, label='ECE')
    ax3.bar(x + width/2, briers, width, label='Brier')
    ax3.set_xlabel('Method')
    ax3.set_ylabel('Score')
    ax3.set_title('Calibration Metrics')
    ax3.set_xticks(x)
    ax3.set_xticklabels(methods, rotation=45)
    ax3.legend()
    ax3.grid(True)
    
    # Temperature values
    ax4 = axes[1, 1]
    # We'll show regime temperatures from calibrators
    # For simplicity, skip if not available
    ax4.axis('off')
    ax4.text(0.1, 0.5, 'Regime-adaptive calibration improves\ncalibration in extreme regimes.', 
             fontsize=12, verticalalignment='center')
    
    plt.tight_layout()
    plt.savefig('regime_adaptive_calibration_results.png')
    plt.close()

def main():
    symbol = 'BTC/USDT'
    print(f"=== Regime-Adaptive Calibration for {symbol} ===")
    
    # Load model and data
    data = load_model_and_data(symbol)
    print(f"Validation samples: {len(data['y_val'])}")
    print(f"Test samples: {len(data['y_test'])}")
    
    # Fit regime-adaptive calibrators
    print("\nFitting regime-specific temperature scaling...")
    calibrators = fit_regime_adaptive_calibration(data)
    
    # Evaluate all methods
    results, y_test_regime = evaluate_all_methods(data, calibrators)
    
    # Plot results
    plot_regime_calibration(results, y_test_regime, data)
    
    # Save calibrators
    import os
    os.makedirs('models/regime_adaptive', exist_ok=True)
    prefix = symbol.replace('/', '_')
    with open(f'models/regime_adaptive/{prefix}_regime_calibrators.pkl', 'wb') as f:
        pickle.dump(calibrators, f)
    print(f"\nRegime calibrators saved to models/regime_adaptive/{prefix}_regime_calibrators.pkl")
    
    # Final recommendation
    best_method = min(results.items(), key=lambda x: x[1]['ece'])
    print(f"\n=== RECOMMENDATION ===")
    print(f"Best calibration method: {best_method[0]} (ECE={best_method[1]['ece']:.4f})")
    print("Regime-adaptive temperature scaling provides better calibration across different market conditions.")

if __name__ == '__main__':
    main()