import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score
import warnings
warnings.filterwarnings('ignore')
import sys
sys.path.append('.')
from data_loader import DataLoader
from advanced_features import AdvancedFeatureEngineer
from regime_detection import RegimeDetector
import pickle

def analyze_regime_performance(symbol='BTC/USDT'):
    """
    Analyze model performance across different market regimes.
    """
    print(f"=== Regime Analysis for {symbol} ===")
    
    # 1. Load features and labels
    engineer = AdvancedFeatureEngineer(symbol, lookback_minutes=15)
    X, y = engineer.compute_all_features()
    
    # 2. Load saved model
    prefix = symbol.replace('/', '_')
    with open(f'models/robust/{prefix}_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open(f'models/robust/{prefix}_feature_selector.pkl', 'rb') as f:
        selector = pickle.load(f)
    with open(f'models/robust/{prefix}_calibrator.pkl', 'rb') as f:
        calibrator = pickle.load(f)
    
    # 3. Load regime features
    loader = DataLoader(data_dir='data/deep')
    df_hourly = loader.load_asset(symbol, timeframe='1h')
    detector = RegimeDetector(window_bars=168)
    regime_feats = detector.compute_regime_features(df_hourly)
    
    # 4. Align indices
    aligned_idx = X.index.intersection(regime_feats.index)
    X_aligned = X.loc[aligned_idx]
    y_aligned = y.loc[aligned_idx]
    regime_aligned = regime_feats.loc[aligned_idx]
    
    # 5. Temporal split (same as training)
    split_idx = int(len(X_aligned) * 0.8)
    X_train = X_aligned.iloc[:split_idx]
    X_test = X_aligned.iloc[split_idx:]
    y_train = y_aligned.iloc[:split_idx]
    y_test = y_aligned.iloc[split_idx:]
    regime_test = regime_aligned.iloc[split_idx:]
    
    # 6. Predict on test set
    X_test_selected = selector.transform(X_test)
    y_pred_proba = model.predict_proba(X_test_selected)[:, 1]
    # Calibrate
    y_pred_proba_cal = calibrator.predict(y_pred_proba.reshape(-1, 1)).flatten()
    
    # 7. Analyze by regime
    regime_cols = ['vol_high', 'vol_low', 'vol_medium', 
                   'trend_up', 'trend_down', 'trend_sideways',
                   'mom_up', 'mom_down', 'mom_neutral']
    
    regime_results = {}
    for col in regime_cols:
        if col not in regime_test.columns:
            continue
        
        # Binary regime (1 = in regime, 0 = not)
        mask = regime_test[col] == 1
        if mask.sum() > 10:
            y_regime = y_test[mask]
            p_regime = y_pred_proba_cal[mask]
            
            regime_results[col] = {
                'count': len(y_regime),
                'accuracy': accuracy_score(y_regime, (p_regime >= 0.5).astype(int)),
                'brier': brier_score_loss(y_regime, p_regime),
                'roc_auc': roc_auc_score(y_regime, p_regime),
                'actual_up_rate': y_regime.mean(),
                'pred_up_rate': (p_regime >= 0.5).mean(),
            }
    
    # Print results
    print("\nPerformance by Regime (test set):")
    for regime, stats in regime_results.items():
        print(f"\n{regime}:")
        print(f"  Samples: {stats['count']}")
        print(f"  Accuracy: {stats['accuracy']:.3f}")
        print(f"  Brier: {stats['brier']:.3f}")
        print(f"  ROC AUC: {stats['roc_auc']:.3f}")
        print(f"  Actual up rate: {stats['actual_up_rate']:.3f}")
        print(f"  Predicted up rate: {stats['pred_up_rate']:.3f}")
    
    # 8. Analyze early momentum signal by regime
    print("\n=== Early Momentum Signal by Regime ===")
    # Get first_5m_return column
    first_5m_cols = [col for col in X_test.columns if 'first_5m_return' in col]
    if first_5m_cols:
        first_5m_col = first_5m_cols[0]
        early_returns = X_test[first_5m_col]
        
        # Combine with regimes
        analysis_df = pd.DataFrame({
            'early_return': early_returns,
            'hour_up': y_test,
            'vol_high': regime_test['vol_high'] if 'vol_high' in regime_test.columns else 0,
            'vol_low': regime_test['vol_low'] if 'vol_low' in regime_test.columns else 0,
        })
        
        # Define regimes
        analysis_df['vol_regime'] = 'medium'
        analysis_df.loc[analysis_df['vol_high'] == 1, 'vol_regime'] = 'high'
        analysis_df.loc[analysis_df['vol_low'] == 1, 'vol_regime'] = 'low'
        
        # Analyze correlation by regime
        for regime in ['high', 'medium', 'low']:
            subset = analysis_df[analysis_df['vol_regime'] == regime]
            if len(subset) > 10:
                corr = np.corrcoef(subset['early_return'], subset['hour_up'])[0,1]
                print(f"Volatility {regime}:")
                print(f"  Samples: {len(subset)}")
                print(f"  Correlation (early_return vs hour_up): {corr:.3f}")
                print(f"  Early return mean: {subset['early_return'].mean():.5f}")
                print(f"  Hour up rate: {subset['hour_up'].mean():.3f}")
                
                # Logistic regression
                from sklearn.linear_model import LogisticRegression
                X_regime = subset['early_return'].values.reshape(-1,1)
                y_regime = subset['hour_up'].values
                lr = LogisticRegression(C=1e9, solver='lbfgs')
                lr.fit(X_regime, y_regime)
                coef = lr.coef_[0][0]
                print(f"  Logistic coefficient: {coef:.3f}")
    
    # 9. Plot regime performance
    fig, axes = plt.subplots(2, 2, figsize=(12,10))
    
    # Accuracy by regime
    regimes = list(regime_results.keys())
    accuracies = [regime_results[r]['accuracy'] for r in regimes]
    axes[0,0].bar(regimes, accuracies)
    axes[0,0].set_title('Accuracy by Regime')
    axes[0,0].set_ylabel('Accuracy')
    axes[0,0].tick_params(axis='x', rotation=45)
    
    # Brier by regime
    briers = [regime_results[r]['brier'] for r in regimes]
    axes[0,1].bar(regimes, briers, color='orange')
    axes[0,1].axhline(y=0.25, color='gray', linestyle='--', label='Random')
    axes[0,1].set_title('Brier Score by Regime')
    axes[0,1].set_ylabel('Brier Score')
    axes[0,1].tick_params(axis='x', rotation=45)
    axes[0,1].legend()
    
    # ROC AUC by regime
    aucs = [regime_results[r]['roc_auc'] for r in regimes]
    axes[1,0].bar(regimes, aucs, color='green')
    axes[1,0].axhline(y=0.5, color='gray', linestyle='--', label='Random')
    axes[1,0].set_title('ROC AUC by Regime')
    axes[1,0].set_ylabel('ROC AUC')
    axes[1,0].tick_params(axis='x', rotation=45)
    axes[1,0].legend()
    
    # Sample counts
    counts = [regime_results[r]['count'] for r in regimes]
    axes[1,1].bar(regimes, counts, color='purple')
    axes[1,1].set_title('Sample Count by Regime')
    axes[1,1].set_ylabel('Count')
    axes[1,1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(f'regime_performance_{symbol.replace("/", "_")}.png')
    plt.close()
    
    return regime_results

def regime_specific_modeling(symbol='BTC/USDT'):
    """
    Train separate models for different regimes.
    """
    print(f"\n=== Regime-Specific Modeling for {symbol} ===")
    
    # Load features, labels, regimes
    engineer = AdvancedFeatureEngineer(symbol, lookback_minutes=15)
    X, y = engineer.compute_all_features()
    
    loader = DataLoader(data_dir='data/deep')
    df_hourly = loader.load_asset(symbol, timeframe='1h')
    detector = RegimeDetector(window_bars=168)
    regime_feats = detector.compute_regime_features(df_hourly)
    
    # Align indices
    aligned_idx = X.index.intersection(regime_feats.index)
    X_aligned = X.loc[aligned_idx]
    y_aligned = y.loc[aligned_idx]
    regime_aligned = regime_feats.loc[aligned_idx]
    
    # Split (temporal)
    split_idx = int(len(X_aligned) * 0.8)
    X_train = X_aligned.iloc[:split_idx]
    X_test = X_aligned.iloc[split_idx:]
    y_train = y_aligned.iloc[:split_idx]
    y_test = y_aligned.iloc[split_idx:]
    regime_train = regime_aligned.iloc[:split_idx]
    regime_test = regime_aligned.iloc[split_idx:]
    
    # Define regimes of interest
    regimes = {
        'high_vol': regime_train['vol_high'] == 1,
        'low_vol': regime_train['vol_low'] == 1,
        'medium_vol': regime_train['vol_medium'] == 1,
        'trending': (regime_train['trend_up'] == 1) | (regime_train['trend_down'] == 1),
        'sideways': regime_train['trend_sideways'] == 1,
    }
    
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.feature_selection import SelectKBest, mutual_info_classif
    
    regime_models = {}
    
    for regime_name, mask in regimes.items():
        if mask.sum() < 50:
            continue
        
        X_regime = X_train[mask]
        y_regime = y_train[mask]
        
        # Feature selection
        selector = SelectKBest(mutual_info_classif, k=min(20, X_regime.shape[1]))
        X_train_sel = selector.fit_transform(X_regime, y_regime)
        
        # Scale
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_sel)
        
        # Train
        model = LogisticRegression(C=1.0, penalty='l2', max_iter=1000, random_state=42)
        model.fit(X_train_scaled, y_regime)
        
        # Test on corresponding regime in test set
        # Determine test mask (same regime definition)
        if regime_name == 'high_vol':
            test_mask = regime_test['vol_high'] == 1
        elif regime_name == 'low_vol':
            test_mask = regime_test['vol_low'] == 1
        elif regime_name == 'medium_vol':
            test_mask = regime_test['vol_medium'] == 1
        elif regime_name == 'trending':
            test_mask = (regime_test['trend_up'] == 1) | (regime_test['trend_down'] == 1)
        elif regime_name == 'sideways':
            test_mask = regime_test['trend_sideways'] == 1
        else:
            test_mask = pd.Series(False, index=regime_test.index)
        
        if test_mask.sum() > 10:
            X_test_regime = X_test[test_mask]
            y_test_regime = y_test[test_mask]
            
            X_test_sel = selector.transform(X_test_regime)
            X_test_scaled = scaler.transform(X_test_sel)
            
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            
            metrics = {
                'train_samples': len(X_regime),
                'test_samples': len(X_test_regime),
                'accuracy': accuracy_score(y_test_regime, (y_pred_proba >= 0.5).astype(int)),
                'brier': brier_score_loss(y_test_regime, y_pred_proba),
                'roc_auc': roc_auc_score(y_test_regime, y_pred_proba),
            }
            
            regime_models[regime_name] = {
                'model': model,
                'selector': selector,
                'scaler': scaler,
                'metrics': metrics,
            }
            
            print(f"\n{regime_name}:")
            print(f"  Train samples: {metrics['train_samples']}")
            print(f"  Test samples: {metrics['test_samples']}")
            print(f"  Accuracy: {metrics['accuracy']:.3f}")
            print(f"  Brier: {metrics['brier']:.3f}")
            print(f"  ROC AUC: {metrics['roc_auc']:.3f}")
    
    # Compare with global model
    print("\n=== Comparison with Global Model ===")
    # Load saved model metrics
    prefix = symbol.replace('/', '_')
    results_df = pd.read_csv(f'models/robust/{prefix}_results.csv', index_col=0)
    global_acc = results_df.loc['test', 'accuracy']
    global_brier = results_df.loc['test', 'brier']
    print(f"Global model: accuracy={global_acc:.3f}, brier={global_brier:.3f}")
    
    return regime_models

if __name__ == '__main__':
    # Run regime analysis
    regime_results = analyze_regime_performance('BTC/USDT')
    
    # Run regime-specific modeling
    regime_models = regime_specific_modeling('BTC/USDT')