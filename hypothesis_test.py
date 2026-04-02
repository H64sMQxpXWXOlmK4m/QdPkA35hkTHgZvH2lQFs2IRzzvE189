import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from data_loader import DataLoader
from intra_hour_features import IntraHourFeatureEngineer
import warnings
warnings.filterwarnings('ignore')

def test_early_overreaction(symbol='BTC/USDT', lookback_minutes=15):
    """
    Test hypothesis: early price moves distort probabilities.
    Compute probability of hour up conditional on early return quintiles.
    """
    loader = DataLoader(data_dir='data/deep')
    # Hourly data
    panel = loader.get_hourly_panel([symbol])
    prefix = symbol.replace('/', '_')
    df_hourly = pd.DataFrame()
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df_hourly[col] = panel[col][f'{prefix}_{col}']
    # Minute data
    df_minute = loader.load_asset(symbol, timeframe='1m')
    
    engineer = IntraHourFeatureEngineer(df_minute)
    hourly_labels = (df_hourly['close'] > df_hourly['open']).astype(int)
    intra_feats, labels_aligned = engineer.align_with_labels(hourly_labels)
    
    # Merge first_5m_return with labels
    df = pd.concat([intra_feats['first_5m_return'], labels_aligned], axis=1)
    df.columns = ['first_5m_return', 'hour_up']
    df = df.dropna()
    
    print(f"Number of hours: {len(df)}")
    print(f"Overall probability hour up: {df['hour_up'].mean():.3f}")
    
    # Quintiles of early return
    df['early_quintile'] = pd.qcut(df['first_5m_return'], q=5, labels=False)
    # Extreme quintiles (0 = most negative, 4 = most positive)
    df['early_extreme'] = pd.cut(df['first_5m_return'], bins=[-np.inf, -0.001, 0.001, np.inf], labels=['neg', 'neutral', 'pos'])
    
    # Conditional probabilities
    print("\n=== Conditional Probabilities ===")
    print("By quintile (0=most negative, 4=most positive):")
    for q in range(5):
        subset = df[df['early_quintile'] == q]
        prob = subset['hour_up'].mean()
        mean_ret = subset['first_5m_return'].mean()
        print(f"Quintile {q}: early return={mean_ret:.5f}, P(up)={prob:.3f}, count={len(subset)}")
    
    print("\nBy extreme categories:")
    for cat in ['neg', 'neutral', 'pos']:
        subset = df[df['early_extreme'] == cat]
        prob = subset['hour_up'].mean()
        mean_ret = subset['first_5m_return'].mean()
        print(f"{cat}: early return={mean_ret:.5f}, P(up)={prob:.3f}, count={len(subset)}")
    
    # Compute reversal effect: if early return negative, does probability revert?
    # i.e., probability of hour up > 0.5 when early negative?
    early_neg = df[df['first_5m_return'] < -0.001]
    early_pos = df[df['first_5m_return'] > 0.001]
    print(f"\nEarly negative (< -0.1%): P(up) = {early_neg['hour_up'].mean():.3f}, count={len(early_neg)}")
    print(f"Early positive (> 0.1%): P(up) = {early_pos['hour_up'].mean():.3f}, count={len(early_pos)}")
    
    # Plot
    plt.figure(figsize=(10,6))
    # Scatter
    plt.scatter(df['first_5m_return'], df['hour_up'], alpha=0.3, s=10, label='Hours')
    # Rolling mean
    df_sorted = df.sort_values('first_5m_return')
    rolling_mean = df_sorted['hour_up'].rolling(window=50, center=True).mean()
    plt.plot(df_sorted['first_5m_return'], rolling_mean, color='red', linewidth=3, label='Rolling mean (window=50)')
    plt.xlabel('First 5-minute return')
    plt.ylabel('Hour Up (1) / Down (0)')
    plt.title(f'{symbol}: Early Return vs Hour Direction')
    plt.axhline(y=df['hour_up'].mean(), color='gray', linestyle='--', label='Overall mean')
    plt.legend()
    plt.grid(True)
    plt.savefig('hypothesis_early_vs_hour.png')
    plt.close()
    
    # Logistic regression of hour_up ~ first_5m_return
    from sklearn.linear_model import LogisticRegression
    X = df['first_5m_return'].values.reshape(-1,1)
    y = df['hour_up'].values
    lr = LogisticRegression(C=1e9, solver='lbfgs')
    lr.fit(X, y)
    coef = lr.coef_[0][0]
    intercept = lr.intercept_[0]
    print(f"\nLogistic regression: hour_up ~ first_5m_return")
    print(f"Coefficient: {coef:.3f}")
    print(f"Intercept: {intercept:.3f}")
    # Predict probabilities across range
    x_range = np.linspace(df['first_5m_return'].min(), df['first_5m_return'].max(), 100)
    prob_range = lr.predict_proba(x_range.reshape(-1,1))[:,1]
    plt.figure(figsize=(10,6))
    plt.plot(x_range, prob_range, color='blue', linewidth=2, label='Logistic fit')
    plt.scatter(df['first_5m_return'], df['hour_up'], alpha=0.3, s=10)
    plt.xlabel('First 5-minute return')
    plt.ylabel('P(Hour Up)')
    plt.title(f'{symbol}: Logistic Regression')
    plt.grid(True)
    plt.legend()
    plt.savefig('hypothesis_logistic_fit.png')
    plt.close()
    
    return df

if __name__ == '__main__':
    df = test_early_overreaction('BTC/USDT')
    print("\nHypothesis test completed.")