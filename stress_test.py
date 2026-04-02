import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import brier_score_loss, roc_auc_score, accuracy_score
from sklearn.calibration import calibration_curve
import warnings
warnings.filterwarnings('ignore')
import pickle
import os

def load_model_results(symbol='BTC/USDT'):
    """Load saved model results."""
    prefix = symbol.replace('/', '_')
    with open(f'models/robust/{prefix}_results.csv', 'r') as f:
        df = pd.read_csv(f, index_col=0)
    return df

def load_test_predictions(symbol='BTC/USDT'):
    """Load test predictions from the robust modeling run.
    We'll need to recompute them because they weren't saved.
    For now, we'll load the diagnostic results.
    """
    try:
        with open('diagnostic_results.pkl', 'rb') as f:
            data = pickle.load(f)
        return data['y_test'], data['y_pred_proba']
    except:
        return None, None

def tail_calibration_analysis(y_true, y_prob, n_bins=20):
    """Analyze calibration in probability extremes."""
    # Focus on tails: <20% and >80%
    low_mask = y_prob < 0.2
    high_mask = y_prob > 0.8
    
    results = {}
    if low_mask.sum() > 0:
        y_true_low = y_true[low_mask]
        y_prob_low = y_prob[low_mask]
        results['low'] = {
            'count': len(y_true_low),
            'mean_prob': y_prob_low.mean(),
            'actual_rate': y_true_low.mean(),
            'brier': brier_score_loss(y_true_low, y_prob_low),
        }
    
    if high_mask.sum() > 0:
        y_true_high = y_true[high_mask]
        y_prob_high = y_prob[high_mask]
        results['high'] = {
            'count': len(y_true_high),
            'mean_prob': y_prob_high.mean(),
            'actual_rate': y_true_high.mean(),
            'brier': brier_score_loss(y_true_high, y_prob_high),
        }
    
    # Bin by probability for detailed view
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_prob, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)
    
    bin_stats = []
    for i in range(n_bins):
        mask = bin_indices == i
        if mask.sum() > 0:
            bin_prob = y_prob[mask].mean()
            bin_acc = y_true[mask].mean()
            bin_count = mask.sum()
            bin_stats.append({
                'bin': i,
                'prob_low': bin_edges[i],
                'prob_high': bin_edges[i+1],
                'mean_pred': bin_prob,
                'mean_actual': bin_acc,
                'count': bin_count,
                'calibration_error': abs(bin_acc - bin_prob),
            })
    
    bin_df = pd.DataFrame(bin_stats)
    return results, bin_df

def worst_period_analysis(features_df, labels, predictions, window_size=24):
    """
    Find worst-performing rolling windows.
    """
    # Ensure aligned indices
    aligned_idx = features_df.index.intersection(pd.Series(labels).index)
    if len(aligned_idx) == 0:
        return None
    
    # Compute rolling accuracy and Brier
    acc_rolling = []
    brier_rolling = []
    dates = []
    
    for i in range(window_size, len(aligned_idx)):
        start = i - window_size
        end = i
        y_window = labels[start:end]
        p_window = predictions[start:end]
        
        acc = accuracy_score(y_window, (p_window >= 0.5).astype(int))
        brier = brier_score_loss(y_window, p_window)
        
        acc_rolling.append(acc)
        brier_rolling.append(brier)
        dates.append(aligned_idx[end - 1])  # end timestamp
    
    rolling_df = pd.DataFrame({
        'date': dates,
        'accuracy': acc_rolling,
        'brier': brier_rolling,
    })
    rolling_df.set_index('date', inplace=True)
    
    # Find worst windows
    worst_acc = rolling_df.nsmallest(5, 'accuracy')
    worst_brier = rolling_df.nlargest(5, 'brier')
    
    return rolling_df, worst_acc, worst_brier

def regime_specific_performance(features_df, labels, predictions, regime_col=None):
    """
    Break down performance by regime if regime column provided.
    """
    if regime_col is None or regime_col not in features_df.columns:
        return None
    
    regimes = features_df[regime_col].values
    unique_regimes = np.unique(regimes)
    
    regime_results = {}
    for regime in unique_regimes:
        mask = regimes == regime
        if mask.sum() > 0:
            y_regime = labels[mask]
            p_regime = predictions[mask]
            regime_results[regime] = {
                'count': len(y_regime),
                'accuracy': accuracy_score(y_regime, (p_regime >= 0.5).astype(int)),
                'brier': brier_score_loss(y_regime, p_regime),
                'roc_auc': roc_auc_score(y_regime, p_regime),
                'actual_up_rate': y_regime.mean(),
                'pred_up_rate': (p_regime >= 0.5).mean(),
            }
    
    return regime_results

def data_sufficiency_analysis(symbol='BTC/USDT'):
    """Check if we have enough data across regimes."""
    from data_loader import DataLoader
    from regime_detection import RegimeDetector
    
    loader = DataLoader(data_dir='data/deep')
    df_hourly = loader.load_asset(symbol, timeframe='1h')
    detector = RegimeDetector(window_bars=168)
    regime_features = detector.compute_regime_features(df_hourly)
    
    # Count samples per regime
    regime_cols = ['vol_high', 'vol_low', 'vol_medium', 
                   'trend_up', 'trend_down', 'trend_sideways',
                   'mom_up', 'mom_down', 'mom_neutral']
    counts = {}
    for col in regime_cols:
        if col in regime_features.columns:
            counts[col] = regime_features[col].sum()
    
    return counts, regime_features.shape[0]

def main():
    print("=== STRESS TESTING ===")
    
    # 1. Tail calibration analysis
    print("\n1. Tail Calibration Analysis")
    y_test, y_pred = load_test_predictions('BTC/USDT')
    if y_test is not None:
        tail_results, bin_df = tail_calibration_analysis(y_test, y_pred)
        print("Extreme probabilities:")
        for region, stats in tail_results.items():
            print(f"  {region}: count={stats['count']}, pred={stats['mean_prob']:.3f}, actual={stats['actual_rate']:.3f}, brier={stats['brier']:.4f}")
        
        # Plot calibration binned
        plt.figure(figsize=(10,6))
        plt.bar(bin_df['prob_low'], bin_df['calibration_error'], width=0.05, alpha=0.7)
        plt.xlabel('Probability bin')
        plt.ylabel('Calibration error')
        plt.title('Calibration Error by Probability Bin')
        plt.grid(True)
        plt.savefig('stress_calibration_error.png')
        plt.close()
        
        # Plot reliability diagram
        prob_true, prob_pred = calibration_curve(y_test, y_pred, n_bins=10)
        plt.figure(figsize=(8,6))
        plt.plot(prob_pred, prob_true, marker='o', label='Model')
        plt.plot([0,1], [0,1], 'k--', label='Perfect')
        plt.xlabel('Mean predicted probability')
        plt.ylabel('Fraction of positives')
        plt.title('Reliability Diagram')
        plt.legend()
        plt.grid(True)
        plt.savefig('stress_reliability.png')
        plt.close()
    
    # 2. Data sufficiency across regimes
    print("\n2. Data Sufficiency Across Regimes")
    counts, total = data_sufficiency_analysis('BTC/USDT')
    print(f"Total hours: {total}")
    for regime, count in counts.items():
        pct = count / total * 100
        print(f"  {regime}: {count} samples ({pct:.1f}%)")
    
    # 3. Worst-case period analysis (if we had features)
    # We'll need to load features and labels
    try:
        from advanced_features import AdvancedFeatureEngineer
        engineer = AdvancedFeatureEngineer('BTC/USDT')
        features, labels = engineer.compute_all_features()
        
        # Temporal split (same as training)
        split_idx = int(len(features) * 0.8)
        test_features = features.iloc[split_idx:]
        test_labels = labels.iloc[split_idx:]
        
        # Load model predictions (simulate by training quick model)
        from robust_modeling import RobustModelTrainer
        trainer = RobustModelTrainer('BTC/USDT')
        trainer.load_data()
        X_train, X_test, y_train, y_test = trainer.temporal_split(features, labels)
        
        # Use the saved model to predict on test set
        import xgboost as xgb
        model_path = 'models/robust/BTC_USDT_model.pkl'
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        # Feature selection
        selector_path = 'models/robust/BTC_USDT_feature_selector.pkl'
        with open(selector_path, 'rb') as f:
            selector = pickle.load(f)
        
        X_test_selected = selector.transform(X_test)
        y_pred_proba = model.predict_proba(X_test_selected)[:, 1]
        
        print("\n3. Worst-Period Analysis (24-hour windows)")
        rolling_df, worst_acc, worst_brier = worst_period_analysis(
            X_test, y_test, y_pred_proba, window_size=24
        )
        if rolling_df is not None:
            print(f"Average rolling accuracy: {rolling_df['accuracy'].mean():.3f}")
            print(f"Average rolling Brier: {rolling_df['brier'].mean():.3f}")
            print(f"Worst accuracy windows:")
            for idx, row in worst_acc.iterrows():
                print(f"  {idx}: acc={row['accuracy']:.3f}, brier={row['brier']:.3f}")
            print(f"Worst Brier windows:")
            for idx, row in worst_brier.iterrows():
                print(f"  {idx}: acc={row['accuracy']:.3f}, brier={row['brier']:.3f}")
            
            # Plot rolling performance
            fig, axes = plt.subplots(2, 1, figsize=(12,8))
            axes[0].plot(rolling_df.index, rolling_df['accuracy'], label='24h rolling accuracy')
            axes[0].axhline(y=0.5, color='gray', linestyle='--', label='Random')
            axes[0].set_ylabel('Accuracy')
            axes[0].legend()
            axes[0].grid(True)
            
            axes[1].plot(rolling_df.index, rolling_df['brier'], label='24h rolling Brier', color='orange')
            axes[1].axhline(y=0.25, color='gray', linestyle='--', label='Random')
            axes[1].set_ylabel('Brier Score')
            axes[1].legend()
            axes[1].grid(True)
            
            plt.suptitle('Rolling Window Performance (24h windows)')
            plt.tight_layout()
            plt.savefig('stress_rolling_performance.png')
            plt.close()
        
        # 4. Regime-specific performance
        print("\n4. Regime-Specific Performance")
        # Add regime columns to test features
        from regime_detection import RegimeDetector
        loader = DataLoader(data_dir='data/deep')
        df_hourly = loader.load_asset('BTC/USDT', timeframe='1h')
        detector = RegimeDetector(window_bars=168)
        regime_feats = detector.compute_regime_features(df_hourly)
        # Align with test features
        aligned_idx = X_test.index.intersection(regime_feats.index)
        X_test_aligned = X_test.loc[aligned_idx]
        y_test_aligned = y_test.loc[aligned_idx]
        pred_aligned = pd.Series(y_pred_proba, index=X_test.index).loc[aligned_idx]
        regime_aligned = regime_feats.loc[aligned_idx]
        
        # Check volatility regime
        vol_results = regime_specific_performance(
            regime_aligned, y_test_aligned.values, pred_aligned.values, 
            regime_col='vol_high'
        )
        if vol_results:
            print("Volatility regime performance:")
            for regime, stats in vol_results.items():
                print(f"  {'high' if regime == 1 else 'low/medium'}: "
                      f"count={stats['count']}, acc={stats['accuracy']:.3f}, "
                      f"brier={stats['brier']:.3f}, auc={stats['roc_auc']:.3f}")
        
    except Exception as e:
        print(f"Could not complete full stress test: {e}")
        import traceback
        traceback.print_exc()
    
    # 5. Baseline comparison
    print("\n5. Simple Baseline Comparison")
    if y_test is not None:
        # Momentum baseline: predict up if first_5m_return > 0
        # We'd need intra features; let's compute a simple alternative
        # Random baseline
        random_pred = np.random.uniform(0, 1, len(y_test))
        random_acc = accuracy_score(y_test, (random_pred >= 0.5).astype(int))
        random_brier = brier_score_loss(y_test, random_pred)
        
        # Always predict up baseline
        up_pred = np.ones(len(y_test)) * 0.5
        up_acc = accuracy_score(y_test, (up_pred >= 0.5).astype(int))
        up_brier = brier_score_loss(y_test, up_pred)
        
        print(f"Random baseline: acc={random_acc:.3f}, brier={random_brier:.3f}")
        print(f"Always up baseline: acc={up_acc:.3f}, brier={up_brier:.3f}")
        
        if y_pred is not None:
            model_acc = accuracy_score(y_test, (y_pred >= 0.5).astype(int))
            model_brier = brier_score_loss(y_test, y_pred)
            print(f"Model: acc={model_acc:.3f}, brier={model_brier:.3f}")
            print(f"Improvement over random: {model_acc - random_acc:.3f} acc, {random_brier - model_brier:.3f} brier")
    
    print("\n=== Stress Test Complete ===")

if __name__ == '__main__':
    main()