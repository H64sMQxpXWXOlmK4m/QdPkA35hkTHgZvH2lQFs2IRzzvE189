import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, roc_auc_score, brier_score_loss, 
    log_loss, precision_score, recall_score, f1_score
)
from sklearn.calibration import calibration_curve
import warnings
warnings.filterwarnings('ignore')
import pickle
import os
import sys
sys.path.append('.')
from data_loader import DataLoader
from advanced_features import AdvancedFeatureEngineer
from regime_detection import RegimeDetector
import xgboost as xgb
from sklearn.feature_selection import SelectKBest, mutual_info_classif
import time

def load_saved_model(symbol):
    """Load saved model artifacts."""
    prefix = symbol.replace('/', '_')
    with open(f'models/robust/{prefix}_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open(f'models/robust/{prefix}_feature_selector.pkl', 'rb') as f:
        selector = pickle.load(f)
    with open(f'models/robust/{prefix}_calibrator.pkl', 'rb') as f:
        calibrator = pickle.load(f)
    return model, selector, calibrator

def temporal_train_test_split(X, y, train_ratio=0.7):
    """Split by time."""
    split_idx = int(len(X) * train_ratio)
    X_train = X.iloc[:split_idx]
    y_train = y.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    y_test = y.iloc[split_idx:]
    return X_train, X_test, y_train, y_test

def walk_forward_validation(X, y, train_ratio=0.5, step=0.1):
    """
    Walk-forward validation: expand training window, test on next chunk.
    Returns performance over time.
    """
    n = len(X)
    train_size = int(n * train_ratio)
    step_size = int(n * step)
    
    performances = []
    for start_test in range(train_size, n - step_size, step_size):
        X_train = X.iloc[:start_test]
        y_train = y.iloc[:start_test]
        X_test = X.iloc[start_test:start_test + step_size]
        y_test = y.iloc[start_test:start_test + step_size]
        
        if len(X_test) < 10 or len(np.unique(y_test)) < 2:
            continue
        
        # Train a simple model (logistic regression) for speed
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        
        # Feature selection
        selector = SelectKBest(mutual_info_classif, k=min(20, X_train.shape[1]))
        X_train_sel = selector.fit_transform(X_train, y_train)
        X_test_sel = selector.transform(X_test)
        
        # Scale
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_sel)
        X_test_scaled = scaler.transform(X_test_sel)
        
        # Train
        model = LogisticRegression(C=1.0, penalty='l2', max_iter=1000, random_state=42)
        model.fit(X_train_scaled, y_train)
        
        # Predict
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        
        # Metrics
        metrics = {
            'train_end': start_test,
            'test_start': start_test,
            'test_end': start_test + step_size,
            'accuracy': accuracy_score(y_test, (y_pred_proba >= 0.5).astype(int)),
            'brier': brier_score_loss(y_test, y_pred_proba),
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'log_loss': log_loss(y_test, y_pred_proba),
        }
        performances.append(metrics)
    
    return pd.DataFrame(performances)

def regime_shift_test(X, y, regime_col, regime_value):
    """
    Test performance when regime changes.
    Train on regime != value, test on regime == value.
    """
    # Get regime features
    from regime_detection import RegimeDetector
    loader = DataLoader(data_dir='data/deep')
    df_hourly = loader.load_asset('BTC/USDT', timeframe='1h')
    detector = RegimeDetector(window_bars=168)
    regime_feats = detector.compute_regime_features(df_hourly)
    
    # Align indices
    aligned_idx = X.index.intersection(regime_feats.index)
    X_aligned = X.loc[aligned_idx]
    y_aligned = y.loc[aligned_idx]
    regime_aligned = regime_feats.loc[aligned_idx]
    
    if regime_col not in regime_aligned.columns:
        print(f"Regime column {regime_col} not found")
        return None
    
    # Split
    train_mask = regime_aligned[regime_col] != regime_value
    test_mask = regime_aligned[regime_col] == regime_value
    
    if train_mask.sum() == 0 or test_mask.sum() == 0:
        print(f"Not enough data for regime shift test")
        return None
    
    X_train = X_aligned[train_mask]
    y_train = y_aligned[train_mask]
    X_test = X_aligned[test_mask]
    y_test = y_aligned[test_mask]
    
    # Train model
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.feature_selection import SelectKBest, mutual_info_classif
    
    selector = SelectKBest(mutual_info_classif, k=min(20, X_train.shape[1]))
    X_train_sel = selector.fit_transform(X_train, y_train)
    X_test_sel = selector.transform(X_test)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_sel)
    X_test_scaled = scaler.transform(X_test_sel)
    
    model = LogisticRegression(C=1.0, penalty='l2', max_iter=1000, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    metrics = {
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'accuracy': accuracy_score(y_test, (y_pred_proba >= 0.5).astype(int)),
        'brier': brier_score_loss(y_test, y_pred_proba),
        'roc_auc': roc_auc_score(y_test, y_pred_proba),
        'log_loss': log_loss(y_test, y_pred_proba),
    }
    
    return metrics

def label_shuffling_test(X, y, n_permutations=100):
    """
    Test if model performance is significantly better than chance.
    Shuffle labels and see how often random performance matches ours.
    """
    # Compute real performance
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    
    # Simple split (not temporal for this test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, shuffle=False  # still temporal
    )
    
    selector = SelectKBest(mutual_info_classif, k=min(20, X_train.shape[1]))
    X_train_sel = selector.fit_transform(X_train, y_train)
    X_test_sel = selector.transform(X_test)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_sel)
    X_test_scaled = scaler.transform(X_test_sel)
    
    model = LogisticRegression(C=1.0, penalty='l2', max_iter=1000, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    real_auc = roc_auc_score(y_test, y_pred_proba)
    
    # Permutation test
    permuted_aucs = []
    for i in range(n_permutations):
        # Shuffle labels in training set only
        y_train_perm = y_train.sample(frac=1, random_state=i).values
        
        # Retrain
        model_perm = LogisticRegression(C=1.0, penalty='l2', max_iter=1000, random_state=42)
        model_perm.fit(X_train_scaled, y_train_perm)
        
        # Predict on same test set (labels not shuffled)
        y_pred_perm = model_perm.predict_proba(X_test_scaled)[:, 1]
        perm_auc = roc_auc_score(y_test, y_pred_perm)
        permuted_aucs.append(perm_auc)
    
    permuted_aucs = np.array(permuted_aucs)
    p_value = (permuted_aucs >= real_auc).mean()
    
    return {
        'real_auc': real_auc,
        'permuted_mean': permuted_aucs.mean(),
        'permuted_std': permuted_aucs.std(),
        'p_value': p_value,
        'significant_05': p_value < 0.05,
        'significant_01': p_value < 0.01,
    }

def data_period_test(X, y, train_periods, test_periods):
    """
    Test performance across different time periods.
    train_periods, test_periods: list of (start_idx, end_idx) tuples.
    """
    results = []
    for train_start, train_end in train_periods:
        for test_start, test_end in test_periods:
            X_train = X.iloc[train_start:train_end]
            y_train = y.iloc[train_start:train_end]
            X_test = X.iloc[test_start:test_end]
            y_test = y.iloc[test_start:test_end]
            
            if len(X_train) < 100 or len(X_test) < 20:
                continue
            
            # Train simple model
            from sklearn.linear_model import LogisticRegression
            from sklearn.preprocessing import StandardScaler
            from sklearn.feature_selection import SelectKBest, mutual_info_classif
            
            selector = SelectKBest(mutual_info_classif, k=min(20, X_train.shape[1]))
            X_train_sel = selector.fit_transform(X_train, y_train)
            X_test_sel = selector.transform(X_test)
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_sel)
            X_test_scaled = scaler.transform(X_test_sel)
            
            model = LogisticRegression(C=1.0, penalty='l2', max_iter=1000, random_state=42)
            model.fit(X_train_scaled, y_train)
            
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            
            metrics = {
                'train_period': f"{train_start}-{train_end}",
                'test_period': f"{test_start}-{test_end}",
                'train_samples': len(X_train),
                'test_samples': len(X_test),
                'accuracy': accuracy_score(y_test, (y_pred_proba >= 0.5).astype(int)),
                'brier': brier_score_loss(y_test, y_pred_proba),
                'roc_auc': roc_auc_score(y_test, y_pred_proba),
            }
            results.append(metrics)
    
    return pd.DataFrame(results)

def main():
    print("=== RIGOROUS VALIDATION ===")
    
    symbol = 'BTC/USDT'
    
    # 1. Load features
    print("\n1. Loading features...")
    engineer = AdvancedFeatureEngineer(symbol, lookback_minutes=15)
    X, y = engineer.compute_all_features()
    print(f"Features shape: {X.shape}, Labels shape: {y.shape}")
    
    # 2. Walk-forward validation
    print("\n2. Walk-forward validation...")
    wf_results = walk_forward_validation(X, y, train_ratio=0.6, step=0.1)
    if len(wf_results) > 0:
        print(f"Number of walk-forward folds: {len(wf_results)}")
        print(f"Mean accuracy: {wf_results['accuracy'].mean():.3f} ± {wf_results['accuracy'].std():.3f}")
        print(f"Mean Brier: {wf_results['brier'].mean():.3f} ± {wf_results['brier'].std():.3f}")
        print(f"Mean AUC: {wf_results['roc_auc'].mean():.3f} ± {wf_results['roc_auc'].std():.3f}")
        
        # Plot performance over time
        plt.figure(figsize=(10,6))
        plt.plot(wf_results.index, wf_results['accuracy'], label='Accuracy', marker='o')
        plt.plot(wf_results.index, wf_results['roc_auc'], label='ROC AUC', marker='s')
        plt.xlabel('Fold index')
        plt.ylabel('Metric')
        plt.title('Walk-Forward Performance')
        plt.legend()
        plt.grid(True)
        plt.savefig('walk_forward_performance.png')
        plt.close()
    
    # 3. Regime shift test
    print("\n3. Regime shift test...")
    # Test training on non-high-vol, testing on high-vol
    shift_result = regime_shift_test(X, y, regime_col='vol_high', regime_value=1)
    if shift_result:
        print(f"Train samples: {shift_result['train_samples']}")
        print(f"Test samples: {shift_result['test_samples']}")
        print(f"Accuracy (train on non-high-vol, test on high-vol): {shift_result['accuracy']:.3f}")
        print(f"Brier: {shift_result['brier']:.3f}")
        print(f"AUC: {shift_result['roc_auc']:.3f}")
    
    # 4. Label shuffling test
    print("\n4. Label shuffling test...")
    perm_test = label_shuffling_test(X, y, n_permutations=50)
    print(f"Real AUC: {perm_test['real_auc']:.3f}")
    print(f"Permuted mean AUC: {perm_test['permuted_mean']:.3f} ± {perm_test['permuted_std']:.3f}")
    print(f"p-value: {perm_test['p_value']:.3f}")
    print(f"Significant at 0.05: {perm_test['significant_05']}")
    print(f"Significant at 0.01: {perm_test['significant_01']}")
    
    # 5. Data period test
    print("\n5. Data period test...")
    n = len(X)
    periods = [
        (0, n//3),           # first third
        (n//3, 2*n//3),      # second third  
        (2*n//3, n),         # last third
    ]
    period_results = data_period_test(X, y, periods, periods)
    if len(period_results) > 0:
        print("Cross-period performance:")
        print(period_results[['train_period', 'test_period', 'accuracy', 'brier', 'roc_auc']].to_string())
    
    # 6. Simple baseline comparison
    print("\n6. Baseline comparison...")
    # Random guessing
    random_pred = np.random.uniform(0, 1, len(y))
    random_acc = accuracy_score(y, (random_pred >= 0.5).astype(int))
    random_brier = brier_score_loss(y, random_pred)
    
    # Always predict up (since up rate ~0.5)
    up_pred = np.ones(len(y)) * 0.5
    up_acc = accuracy_score(y, (up_pred >= 0.5).astype(int))
    up_brier = brier_score_loss(y, up_pred)
    
    # Momentum baseline: predict up if first_5m_return > 0 (need to compute)
    # We'll approximate with a simple logistic regression on first_5m_return only
    from sklearn.linear_model import LogisticRegression
    # Find first_5m_return column
    first_5m_cols = [col for col in X.columns if 'first_5m_return' in col]
    if first_5m_cols:
        first_5m_col = first_5m_cols[0]
        X_simple = X[[first_5m_col]].values
        # Temporal split
        split_idx = int(len(X) * 0.7)
        X_train_simple = X_simple[:split_idx]
        y_train_simple = y[:split_idx]
        X_test_simple = X_simple[split_idx:]
        y_test_simple = y[split_idx:]
        
        lr_simple = LogisticRegression(C=1e9, solver='lbfgs')
        lr_simple.fit(X_train_simple, y_train_simple)
        y_pred_simple = lr_simple.predict_proba(X_test_simple)[:, 1]
        
        simple_acc = accuracy_score(y_test_simple, (y_pred_simple >= 0.5).astype(int))
        simple_brier = brier_score_loss(y_test_simple, y_pred_simple)
        simple_auc = roc_auc_score(y_test_simple, y_pred_simple)
        
        print(f"Momentum baseline (first_5m_return only):")
        print(f"  Accuracy: {simple_acc:.3f}")
        print(f"  Brier: {simple_brier:.3f}")
        print(f"  AUC: {simple_auc:.3f}")
    
    print(f"Random baseline: acc={random_acc:.3f}, brier={random_brier:.3f}")
    print(f"Always up baseline: acc={up_acc:.3f}, brier={up_brier:.3f}")
    
    # Compare with our model's expected performance
    # Load saved model performance
    try:
        prefix = symbol.replace('/', '_')
        results_df = pd.read_csv(f'models/robust/{prefix}_results.csv', index_col=0)
        test_acc = results_df.loc['test', 'accuracy']
        test_brier = results_df.loc['test', 'brier']
        print(f"Saved model performance: acc={test_acc:.3f}, brier={test_brier:.3f}")
        print(f"Improvement over random brier: {random_brier - test_brier:.3f}")
    except:
        pass
    
    print("\n=== Validation Complete ===")

if __name__ == '__main__':
    main()