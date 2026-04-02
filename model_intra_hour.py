import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, brier_score_loss
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
from data_loader import DataLoader
from intra_hour_features import IntraHourFeatureEngineer

def evaluate_intra_hour_model(symbol, lookback_minutes=15, test_size=0.2):
    """
    Train and evaluate model using intra-hour features.
    """
    loader = DataLoader(data_dir='data/raw')
    # Load minute data
    df_minute = loader.load_asset(symbol, timeframe='1m')
    # Load hourly labels
    df_hourly = loader.load_asset(symbol, timeframe='1h')
    labels = (df_hourly['close'] > df_hourly['open']).astype(int)
    
    # Generate features
    engineer = IntraHourFeatureEngineer(df_minute)
    features, labels_aligned = engineer.align_with_labels(labels)
    if features.empty:
        print(f"No features for {symbol}")
        return None
    
    # Temporal split
    n = len(features)
    split_idx = int(n * (1 - test_size))
    X = features.values
    y = labels_aligned.values
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train logistic regression
    model = LogisticRegression(C=1.0, penalty='l2', max_iter=1000, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Predict
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)
    
    # Metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred_proba),
        'brier_score': brier_score_loss(y_test, y_pred_proba),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
    }
    
    # Calibration curve
    prob_true, prob_pred = calibration_curve(y_test, y_pred_proba, n_bins=10)
    metrics['calibration_curve'] = (prob_true, prob_pred)
    
    # Feature importances (coefficients)
    metrics['coef'] = model.coef_[0]
    metrics['features'] = features.columns.tolist()
    
    return metrics

def plot_calibration(prob_true, prob_pred, symbol):
    plt.figure(figsize=(8,6))
    plt.plot(prob_pred, prob_true, marker='o', label='Model')
    plt.plot([0,1], [0,1], linestyle='--', color='gray', label='Perfect')
    plt.xlabel('Mean predicted probability')
    plt.ylabel('Fraction of positives')
    plt.title(f'Calibration Curve for {symbol} (Intra-hour)')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'calibration_intra_hour_{symbol.replace("/", "_")}.png')
    plt.close()

if __name__ == '__main__':
    from sklearn.metrics import precision_score, recall_score, f1_score
    symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'XRP/USDT', 'BNB/USDT', 'DOGE/USDT']
    all_metrics = {}
    for sym in symbols:
        print(f"\n=== {sym} ===")
        metrics = evaluate_intra_hour_model(sym, lookback_minutes=15, test_size=0.2)
        if metrics is None:
            continue
        all_metrics[sym] = metrics
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"ROC AUC: {metrics['roc_auc']:.4f}")
        print(f"Brier score: {metrics['brier_score']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1: {metrics['f1']:.4f}")
        # plot calibration
        plot_calibration(*metrics['calibration_curve'], sym)
    
    # Summary
    print("\n=== Summary ===")
    summary_df = pd.DataFrame(all_metrics).T
    print(summary_df[['accuracy', 'roc_auc', 'brier_score', 'precision', 'recall', 'f1']])
    
    # Save summary
    summary_df.to_csv('intra_hour_model_results.csv')