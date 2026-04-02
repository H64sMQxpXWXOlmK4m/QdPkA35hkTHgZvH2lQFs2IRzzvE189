import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    brier_score_loss, log_loss, roc_auc_score,
    accuracy_score, precision_score, recall_score, f1_score
)
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from scipy.stats import binom, beta
import warnings
warnings.filterwarnings('ignore')

# Load results from previous runs
from data_loader import DataLoader
from intra_hour_features import IntraHourFeatureEngineer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

def expected_calibration_error(y_true, y_prob, n_bins=10):
    """Compute Expected Calibration Error."""
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_prob, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)
    
    ece = 0
    for i in range(n_bins):
        mask = bin_indices == i
        if np.sum(mask) > 0:
            bin_prob = y_prob[mask].mean()
            bin_acc = y_true[mask].mean()
            ece += np.abs(bin_acc - bin_prob) * np.sum(mask)
    ece /= len(y_true)
    return ece

def plot_calibration_detailed(y_true, y_prob, n_bins=10, title='Calibration'):
    """Detailed calibration plot with confidence intervals."""
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins)
    bin_counts = np.histogram(y_prob, bins=n_bins)[0]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Calibration curve with confidence intervals (binomial proportion CI)
    ax1.plot(prob_pred, prob_true, marker='o', label='Model')
    ax1.plot([0,1], [0,1], linestyle='--', color='gray', label='Perfect')
    ax1.set_xlabel('Mean predicted probability')
    ax1.set_ylabel('Fraction of positives')
    ax1.set_title(title)
    ax1.legend()
    ax1.grid(True)
    
    # Add confidence intervals (Wilson score interval)
    for i in range(len(prob_pred)):
        n = bin_counts[i]
        p = prob_true[i]
        if n > 0:
            # Wilson score interval
            z = 1.96  # 95% CI
            denominator = 1 + z**2 / n
            center = (p + z**2 / (2*n)) / denominator
            half_width = z * np.sqrt(p*(1-p)/n + z**2/(4*n**2)) / denominator
            ax1.fill_between([prob_pred[i] - 0.01, prob_pred[i] + 0.01],
                             [center - half_width, center - half_width],
                             [center + half_width, center + half_width],
                             alpha=0.3, color='blue')
    
    # Reliability histogram
    ax2.hist(y_prob, bins=n_bins, edgecolor='black', alpha=0.7)
    ax2.set_xlabel('Predicted probability')
    ax2.set_ylabel('Count')
    ax2.set_title('Probability Distribution')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'detailed_calibration_{title.replace(" ", "_")}.png')
    plt.close()
    
    return prob_true, prob_pred, bin_counts

def evaluate_over_time(features_df, labels, model, scaler, window_size=100, step=50):
    """
    Rolling window evaluation.
    """
    X = features_df.values
    y = labels.values
    X_scaled = scaler.transform(X)
    
    n = len(X)
    metrics_over_time = []
    for start in range(0, n - window_size, step):
        end = start + window_size
        X_train = X_scaled[:end]
        y_train = y[:end]
        X_test = X_scaled[end:min(end + step, n)]
        y_test = y[end:min(end + step, n)]
        
        if len(X_test) == 0 or len(np.unique(y_test)) < 2:
            continue
        
        # Train on expanding window
        model_copy = LogisticRegression(C=1.0, penalty='l2', max_iter=1000, random_state=42)
        model_copy.fit(X_train, y_train)
        y_pred_proba = model_copy.predict_proba(X_test)[:, 1]
        
        # Metrics
        metrics = {
            'start_idx': start,
            'end_idx': end,
            'accuracy': accuracy_score(y_test, (y_pred_proba >= 0.5).astype(int)),
            'brier': brier_score_loss(y_test, y_pred_proba),
            'log_loss': log_loss(y_test, y_pred_proba),
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'ece': expected_calibration_error(y_test, y_pred_proba),
        }
        metrics_over_time.append(metrics)
    
    return pd.DataFrame(metrics_over_time)

def load_btc_data():
    """Load BTC features and labels."""
    from data_loader import DataLoader
    from intra_hour_features import IntraHourFeatureEngineer
    loader = DataLoader()
    symbol = 'BTC/USDT'
    panel = loader.get_hourly_panel([symbol])
    prefix = symbol.replace('/', '_')
    df_hourly = pd.DataFrame()
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df_hourly[col] = panel[col][f'{prefix}_{col}']
    df_minute = loader.load_asset(symbol, timeframe='1m')
    engineer = IntraHourFeatureEngineer(df_minute)
    labels = (df_hourly['close'] > df_hourly['open']).astype(int)
    intra_feats, labels_aligned = engineer.align_with_labels(labels)
    # combine with hourly features (simplified)
    # We'll just use intra features for now
    return intra_feats, labels_aligned

def main():
    print("=== Diagnostic Analysis ===")
    
    # Load BTC data
    print("Loading BTC data...")
    features, labels = load_btc_data()
    print(f"Features shape: {features.shape}")
    
    # Train/test split
    split_idx = int(len(features) * 0.7)
    X_train = features.iloc[:split_idx]
    y_train = labels.iloc[:split_idx]
    X_test = features.iloc[split_idx:]
    y_test = labels.iloc[split_idx:]
    
    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train logistic regression
    model = LogisticRegression(C=1.0, penalty='l2', max_iter=1000, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Predict
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    # Metrics
    print("\n=== Test Set Metrics ===")
    print(f"Accuracy: {accuracy_score(y_test, (y_pred_proba >= 0.5).astype(int)):.4f}")
    print(f"Brier score: {brier_score_loss(y_test, y_pred_proba):.4f}")
    print(f"Log loss: {log_loss(y_test, y_pred_proba):.4f}")
    print(f"ROC AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")
    print(f"ECE (10 bins): {expected_calibration_error(y_test, y_pred_proba):.4f}")
    
    # Detailed calibration plot
    print("\nGenerating detailed calibration plot...")
    plot_calibration_detailed(y_test, y_pred_proba, title='BTC Intra-hour Model')
    
    # Rolling evaluation
    print("\nRunning rolling evaluation...")
    # Combine train+test for rolling
    X_full = pd.concat([X_train, X_test])
    y_full = pd.concat([y_train, y_test])
    rolling_metrics = evaluate_over_time(X_full, y_full, model, scaler, window_size=200, step=50)
    print(f"Rolling windows: {len(rolling_metrics)}")
    print("Average rolling metrics:")
    print(rolling_metrics[['accuracy', 'brier', 'log_loss', 'roc_auc', 'ece']].mean())
    
    # Plot performance over time
    plt.figure(figsize=(10,6))
    plt.plot(rolling_metrics['accuracy'], label='Accuracy')
    plt.plot(rolling_metrics['brier'], label='Brier score')
    plt.plot(rolling_metrics['roc_auc'], label='ROC AUC')
    plt.xlabel('Window index')
    plt.ylabel('Metric')
    plt.title('Model Performance Over Time (Rolling Windows)')
    plt.legend()
    plt.grid(True)
    plt.savefig('performance_over_time.png')
    plt.close()
    
    # Analyze by predicted probability bins
    print("\n=== Calibration by Probability Bins ===")
    n_bins = 10
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_pred_proba, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)
    
    for i in range(n_bins):
        mask = bin_indices == i
        if np.sum(mask) > 0:
            bin_prob = y_pred_proba[mask].mean()
            bin_acc = y_test.iloc[mask].mean()
            count = np.sum(mask)
            print(f"Bin {i+1}: pred={bin_prob:.3f}, actual={bin_acc:.3f}, count={count}")
    
    # Save detailed results
    results = {
        'y_test': y_test.values,
        'y_pred_proba': y_pred_proba,
        'rolling_metrics': rolling_metrics,
    }
    import pickle
    with open('diagnostic_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    print("\nDiagnostic results saved to diagnostic_results.pkl")

if __name__ == '__main__':
    main()