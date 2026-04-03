#!/usr/bin/env python3
"""
Rule-based regime detection and conditional modeling.
"""
import sys
sys.path.append('.')
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import calibration_curve
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt

class RegimeDetector:
    """Detect market regimes using rule-based thresholds."""
    
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.hourly_df = None
        self.regimes = None
        
    def load_hourly_data(self, start_date: pd.Timestamp = None, end_date: pd.Timestamp = None):
        """Load hourly data from CSV."""
        file_path = f'data/deep/{self.symbol.replace("/", "_")}_1h.csv'
        try:
            df = pd.read_csv(file_path)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            df.sort_index(inplace=True)
            
            # Filter date range if provided
            if start_date is not None:
                df = df[df.index >= start_date]
            if end_date is not None:
                df = df[df.index <= end_date]
            
            self.hourly_df = df
            print(f"Loaded {len(df)} hourly candles from {df.index[0]} to {df.index[-1]}")
            return df
        except Exception as e:
            print(f"Failed to load {file_path}: {e}")
            return None
    
    def compute_regime_features(self, lookback: int = 24):
        """Compute regime features using rolling windows."""
        df = self.hourly_df.copy()
        
        # Ensure enough data
        if len(df) < lookback * 2:
            raise ValueError(f"Need at least {lookback*2} samples, got {len(df)}")
        
        # 1. Volatility: ATR (Average True Range) over lookback
        # True Range = max(high-low, abs(high-prev_close), abs(low-prev_close))
        df['prev_close'] = df['close'].shift(1)
        df['tr1'] = df['high'] - df['low']
        df['tr2'] = abs(df['high'] - df['prev_close'])
        df['tr3'] = abs(df['low'] - df['prev_close'])
        df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
        df['atr'] = df['tr'].rolling(window=lookback).mean()
        
        # 2. Trend: linear regression slope over lookback
        def compute_slope(series):
            if len(series) < lookback:
                return np.nan
            X = np.arange(len(series)).reshape(-1, 1)
            y = series.values
            model = LinearRegression()
            model.fit(X, y)
            return model.coef_[0]
        
        # Use close prices for trend
        df['trend_slope'] = df['close'].rolling(window=lookback).apply(compute_slope, raw=False)
        
        # Normalize slope by price level to get % slope
        df['trend_pct'] = df['trend_slope'] / df['close'].rolling(window=lookback).mean()
        
        # 3. Volume: z-score relative to lookback
        df['volume_mean'] = df['volume'].rolling(window=lookback).mean()
        df['volume_std'] = df['volume'].rolling(window=lookback).std()
        df['volume_z'] = (df['volume'] - df['volume_mean']) / df['volume_std'].replace(0, 1)
        
        # 4. Price position within recent range
        df['high_24h'] = df['high'].rolling(window=lookback).max()
        df['low_24h'] = df['low'].rolling(window=lookback).min()
        df['range_position'] = (df['close'] - df['low_24h']) / (df['high_24h'] - df['low_24h']).replace(0, 1)
        
        # 5. Momentum: 12h return
        df['return_12h'] = df['close'].pct_change(12)
        
        # Drop NaN rows from rolling calculations
        df.dropna(subset=['atr', 'trend_pct', 'volume_z', 'range_position', 'return_12h'], inplace=True)
        
        return df
    
    def assign_regimes(self, df: pd.DataFrame):
        """Assign regime labels based on thresholds."""
        # Determine thresholds using percentiles of historical data
        # Use expanding window percentiles to avoid lookahead
        regimes = []
        
        for i in range(len(df)):
            if i < 100:  # Need some history for percentiles
                regimes.append(0)  # unknown
                continue
            
            # Use data up to i (excluding future)
            historical = df.iloc[:i+1]
            
            # Compute percentiles
            atr_50 = historical['atr'].quantile(0.5)
            trend_33 = historical['trend_pct'].quantile(0.33)
            trend_66 = historical['trend_pct'].quantile(0.66)
            vol_z_50 = historical['volume_z'].quantile(0.5)
            
            current = df.iloc[i]
            
            # Determine volatility regime
            vol_regime = 'high' if current['atr'] > atr_50 else 'low'
            
            # Determine trend regime
            if current['trend_pct'] > trend_66:
                trend_regime = 'up'
            elif current['trend_pct'] < trend_33:
                trend_regime = 'down'
            else:
                trend_regime = 'neutral'
            
            # Determine volume regime
            vol_z_regime = 'high' if current['volume_z'] > vol_z_50 else 'low'
            
            # Encode as integer (0-11)
            # We'll use simple scheme: focus on vol + trend first
            regime_map = {
                ('high', 'up'): 0,
                ('high', 'neutral'): 1,
                ('high', 'down'): 2,
                ('low', 'up'): 3,
                ('low', 'neutral'): 4,
                ('low', 'down'): 5,
            }
            
            regime_id = regime_map.get((vol_regime, trend_regime), 0)
            regimes.append(regime_id)
        
        df['regime'] = regimes
        
        # Remove unknown regimes at start
        df = df[df['regime'] != 0].copy()
        
        print("\nRegime distribution:")
        regime_counts = df['regime'].value_counts().sort_index()
        for regime_id, count in regime_counts.items():
            percent = count / len(df) * 100
            print(f"  Regime {regime_id}: {count:4d} samples ({percent:5.1f}%)")
        
        # Map regime IDs to names
        regime_names = {
            0: 'HighVol_UpTrend',
            1: 'HighVol_Neutral',
            2: 'HighVol_DownTrend',
            3: 'LowVol_UpTrend',
            4: 'LowVol_Neutral',
            5: 'LowVol_DownTrend'
        }
        
        df['regime_name'] = df['regime'].map(regime_names)
        
        self.regimes = df
        return df
    
    def analyze_regimes(self):
        """Analyze regime characteristics."""
        if self.regimes is None:
            raise ValueError("Must assign regimes first.")
        
        results = []
        
        for regime_id in sorted(self.regimes['regime'].unique()):
            mask = self.regimes['regime'] == regime_id
            subset = self.regimes[mask]
            
            # Target: close > open (1 if up, 0 if down)
            target = (subset['close'] > subset['open']).astype(int)
            up_rate = target.mean()
            
            stats = {
                'regime': regime_id,
                'regime_name': subset['regime_name'].iloc[0],
                'samples': len(subset),
                'percent': len(subset) / len(self.regimes) * 100,
                'up_rate': up_rate,
                'predictability': abs(up_rate - 0.5),
                'mean_atr': subset['atr'].mean(),
                'mean_trend': subset['trend_pct'].mean(),
                'mean_volume_z': subset['volume_z'].mean(),
                'mean_range_pos': subset['range_position'].mean(),
                'mean_return_12h': subset['return_12h'].mean(),
            }
            results.append(stats)
        
        return pd.DataFrame(results)

class ConditionalModel:
    """Train separate models per regime."""
    
    def __init__(self):
        self.models = {}
        self.regime_stats = {}
        
    def create_features(self, df: pd.DataFrame):
        """Create simple features for prediction."""
        # Use basic features (could be expanded)
        features = pd.DataFrame(index=df.index)
        
        # Price features
        features['open'] = df['open']
        features['high'] = df['high']
        features['low'] = df['low']
        features['close'] = df['close']
        
        # Derived features (using only past)
        features['prev_return'] = df['close'].pct_change(1)
        features['prev_volume'] = df['volume'].shift(1)
        features['hour_of_day'] = df.index.hour
        features['day_of_week'] = df.index.dayofweek
        
        # Add regime features already computed
        features['atr'] = df['atr']
        features['trend_pct'] = df['trend_pct']
        features['volume_z'] = df['volume_z']
        features['range_position'] = df['range_position']
        features['return_12h'] = df['return_12h']
        
        # Drop NaN rows
        features.dropna(inplace=True)
        
        # Target: current hour up (close > open)
        target = (df.loc[features.index, 'close'] > df.loc[features.index, 'open']).astype(int)
        
        return features, target
    
    def train_per_regime(self, df: pd.DataFrame):
        """Train separate XGBoost model for each regime."""
        results = {}
        
        for regime_id in sorted(df['regime'].unique()):
            mask = df['regime'] == regime_id
            regime_df = df[mask].copy()
            
            if len(regime_df) < 50:  # Need minimum samples
                print(f"  Regime {regime_id}: skipping, only {len(regime_df)} samples")
                continue
            
            # Create features and target
            X, y = self.create_features(regime_df)
            
            if len(X) < 30:
                continue
            
            # Split chronologically (80% train, 20% test)
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
            
            # Train XGBoost
            model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                eval_metric='logloss',
                use_label_encoder=False
            )
            
            model.fit(X_train, y_train)
            
            # Predict on test set
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]
            
            # Store model and results
            self.models[regime_id] = {
                'model': model,
                'X_train': X_train,
                'y_train': y_train,
                'X_test': X_test,
                'y_test': y_test,
                'y_pred': y_pred,
                'y_prob': y_prob
            }
            
            # Compute metrics
            accuracy = (y_pred == y_test).mean()
            up_rate = y_test.mean()
            brier = ((y_prob - y_test) ** 2).mean()
            
            # Calibration curve
            prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=5)
            ece = np.mean(np.abs(prob_true - prob_pred))
            
            results[regime_id] = {
                'samples_train': len(X_train),
                'samples_test': len(X_test),
                'accuracy': accuracy,
                'up_rate': up_rate,
                'brier': brier,
                'ece': ece,
                'mean_prob': y_prob.mean(),
                'std_prob': y_prob.std()
            }
            
            print(f"  Regime {regime_id}: test samples={len(X_test)}, accuracy={accuracy:.3f}, "
                  f"brier={brier:.4f}, ECE={ece:.4f}")
        
        self.regime_stats = results
        return results
    
    def predict_all(self, df: pd.DataFrame):
        """Make predictions for entire dataset using regime-specific models."""
        all_probs = []
        all_true = []
        all_regimes = []
        
        for regime_id, model_data in self.models.items():
            # Get regime subset
            mask = df['regime'] == regime_id
            regime_df = df[mask].copy()
            
            if len(regime_df) == 0:
                continue
            
            # Create features
            X, y = self.create_features(regime_df)
            
            if len(X) == 0:
                continue
            
            # Predict using regime-specific model
            model = model_data['model']
            y_prob = model.predict_proba(X)[:, 1]
            
            all_probs.extend(y_prob)
            all_true.extend(y.values)
            all_regimes.extend([regime_id] * len(y))
        
        return np.array(all_probs), np.array(all_true), np.array(all_regimes)

def main():
    """Run regime-based modeling."""
    symbol = 'BTC/USDT'
    
    print("=" * 80)
    print(f"REGIME-BASED MODELING - {symbol}")
    print("=" * 80)
    
    # Step 1: Load data
    detector = RegimeDetector(symbol)
    df = detector.load_hourly_data()
    if df is None:
        return
    
    # Use last 90 days for sufficient regimes
    end_date = df.index[-1]
    start_date = end_date - pd.Timedelta(days=90)
    df = df[(df.index >= start_date) & (df.index <= end_date)]
    print(f"Using {len(df)} samples from {start_date.date()} to {end_date.date()}")
    
    # Step 2: Compute regime features
    print("\nComputing regime features...")
    df_features = detector.compute_regime_features(lookback=24)
    
    # Step 3: Assign regimes
    print("\nAssigning regimes...")
    df_regimes = detector.assign_regimes(df_features)
    
    # Step 4: Analyze regimes
    print("\nAnalyzing regime characteristics...")
    regime_analysis = detector.analyze_regimes()
    print(regime_analysis.to_string())
    
    # Step 5: Train conditional models
    print("\nTraining conditional models per regime...")
    conditional = ConditionalModel()
    results = conditional.train_per_regime(df_regimes)
    
    # Step 6: Generate predictions
    print("\nGenerating predictions for all regimes...")
    all_probs, all_true, all_regimes = conditional.predict_all(df_regimes)
    
    if len(all_probs) == 0:
        print("No predictions generated.")
        return
    
    # Step 7: Analyze prediction distribution
    print("\n" + "=" * 80)
    print("PREDICTION DISTRIBUTION ANALYSIS")
    print("=" * 80)
    
    print(f"Total predictions: {len(all_probs)}")
    print(f"Mean probability: {all_probs.mean():.4f}")
    print(f"Std probability: {all_probs.std():.4f}")
    
    # Bin analysis
    bins = np.linspace(0, 1, 11)
    hist, _ = np.histogram(all_probs, bins=bins)
    
    print("\nProbability distribution histogram:")
    for i in range(len(hist)):
        bin_low = bins[i]
        bin_high = bins[i+1]
        count = hist[i]
        percent = count / len(all_probs) * 100
        print(f"[{bin_low:.1f}, {bin_high:.1f}): {count:4d} ({percent:5.1f}%)")
    
    # Concentration in middle
    middle_mask = (all_probs >= 0.45) & (all_probs <= 0.55)
    middle_percent = middle_mask.sum() / len(all_probs) * 100
    print(f"\nPredictions in [0.45, 0.55]: {middle_mask.sum():d} ({middle_percent:.1f}%)")
    
    # Extreme predictions
    low_mask = all_probs < 0.3
    high_mask = all_probs > 0.7
    print(f"Predictions < 0.3: {low_mask.sum():d} ({low_mask.sum()/len(all_probs)*100:.1f}%)")
    print(f"Predictions > 0.7: {high_mask.sum():d} ({high_mask.sum()/len(all_probs)*100:.1f}%)")
    
    # Evaluate overall performance
    overall_accuracy = ((all_probs >= 0.5) == all_true).mean()
    overall_brier = ((all_probs - all_true) ** 2).mean()
    
    print(f"\nOverall metrics:")
    print(f"  Accuracy: {overall_accuracy:.4f}")
    print(f"  Brier score: {overall_brier:.4f}")
    print(f"  Base rate: {all_true.mean():.4f}")
    
    # Per-regime performance
    print("\nPer-regime performance:")
    for regime_id in np.unique(all_regimes):
        mask = all_regimes == regime_id
        if mask.sum() == 0:
            continue
        
        regime_probs = all_probs[mask]
        regime_true = all_true[mask]
        
        accuracy = ((regime_probs >= 0.5) == regime_true).mean()
        brier = ((regime_probs - regime_true) ** 2).mean()
        mean_prob = regime_probs.mean()
        
        print(f"  Regime {regime_id}: samples={mask.sum():3d}, accuracy={accuracy:.3f}, "
              f"brier={brier:.4f}, mean_prob={mean_prob:.3f}")
    
    # Save results
    results_df = pd.DataFrame({
        'timestamp': df_regimes.index[:len(all_probs)],
        'regime': all_regimes,
        'true': all_true,
        'prob': all_probs,
        'pred': (all_probs >= 0.5).astype(int)
    })
    results_df.to_csv('regime_predictions.csv', index=False)
    
    # Generate report
    generate_report(regime_analysis, results, all_probs, all_true, all_regimes)
    
    print("\n" + "=" * 80)
    print("Analysis complete. Results saved to regime_predictions.csv and report.txt")
    print("=" * 80)

def generate_report(regime_analysis, results, all_probs, all_true, all_regimes):
    """Generate comprehensive report."""
    report = []
    report.append("=" * 80)
    report.append("STATE-CONDITIONAL MODEL REPORT")
    report.append("=" * 80)
    report.append(f"Generated: {pd.Timestamp.now()}")
    report.append(f"Symbol: BTC/USDT")
    report.append(f"Total predictions: {len(all_probs)}")
    report.append("")
    
    # 1. Market State Definitions
    report.append("1. MARKET STATE DEFINITIONS")
    report.append("-" * 80)
    report.append("States based on volatility (high/low) and trend (up/neutral/down):")
    report.append("  0: HighVol_UpTrend")
    report.append("  1: HighVol_Neutral")
    report.append("  2: HighVol_DownTrend")
    report.append("  3: LowVol_UpTrend")
    report.append("  4: LowVol_Neutral")
    report.append("  5: LowVol_DownTrend")
    report.append("")
    
    # 2. State-Level Metrics
    report.append("2. STATE-LEVEL METRICS")
    report.append("-" * 80)
    report.append("Regime | Samples | Up Rate | Predictability | Description")
    report.append("-" * 80)
    
    for _, row in regime_analysis.iterrows():
        predictability = abs(row['up_rate'] - 0.5)
        report.append(f"{row['regime']:6} | {row['samples']:7} | {row['up_rate']:.3f}   | {predictability:.3f}         | {row['regime_name']}")
    
    report.append("")
    
    # 3. Model Performance per State
    report.append("3. MODEL PERFORMANCE PER STATE")
    report.append("-" * 80)
    report.append("Regime | Test Samples | Accuracy | Brier | ECE | Mean Prob | Std Prob")
    report.append("-" * 80)
    
    for regime_id, stats in results.items():
        report.append(f"{regime_id:6} | {stats['samples_test']:11} | {stats['accuracy']:.3f}    | {stats['brier']:.4f} | {stats['ece']:.4f} | {stats['mean_prob']:.3f}   | {stats['std_prob']:.3f}")
    
    report.append("")
    
    # 4. Probability Distribution
    report.append("4. PROBABILITY DISTRIBUTION")
    report.append("-" * 80)
    
    bins = np.linspace(0, 1, 11)
    hist, _ = np.histogram(all_probs, bins=bins)
    
    for i in range(len(hist)):
        bin_low = bins[i]
        bin_high = bins[i+1]
        percent = hist[i] / len(all_probs) * 100
        report.append(f"[{bin_low:.1f}, {bin_high:.1f}): {percent:5.1f}%")
    
    middle_percent = ((all_probs >= 0.45) & (all_probs <= 0.55)).sum() / len(all_probs) * 100
    report.append(f"\nConcentration in [0.45, 0.55]: {middle_percent:.1f}%")
    report.append(f"Required: <60% ({"PASS" if middle_percent < 60 else "FAIL"})")
    
    # 5. Calibration per State
    report.append("\n5. CALIBRATION PER STATE")
    report.append("-" * 80)
    
    for regime_id in np.unique(all_regimes):
        mask = all_regimes == regime_id
        if mask.sum() < 20:
            continue
        
        regime_probs = all_probs[mask]
        regime_true = all_true[mask]
        
        # Simple calibration check: mean prob vs actual rate
        mean_prob = regime_probs.mean()
        actual_rate = regime_true.mean()
        deviation = actual_rate - mean_prob
        
        report.append(f"Regime {regime_id}: Mean prob={mean_prob:.3f}, Actual rate={actual_rate:.3f}, Deviation={deviation:.3f}")
    
    # 6. Edge Concentration
    report.append("\n6. EDGE CONCENTRATION")
    report.append("-" * 80)
    
    # Find regimes with strongest edge (highest accuracy over base rate)
    edge_by_regime = {}
    for regime_id in np.unique(all_regimes):
        mask = all_regimes == regime_id
        if mask.sum() < 20:
            continue
        
        regime_probs = all_probs[mask]
        regime_true = all_true[mask]
        accuracy = ((regime_probs >= 0.5) == regime_true).mean()
        base_rate = regime_true.mean()
        edge = accuracy - max(base_rate, 1-base_rate)  # Edge over majority class
        edge_by_regime[regime_id] = edge
    
    if edge_by_regime:
        sorted_regimes = sorted(edge_by_regime.items(), key=lambda x: x[1], reverse=True)
        report.append("Regimes ranked by edge strength:")
        for regime_id, edge in sorted_regimes:
            report.append(f"  Regime {regime_id}: edge={edge:.4f}")
    
    # 7. Final Verdict
    report.append("\n7. FINAL VERDICT")
    report.append("-" * 80)
    
    # Check key requirements
    overall_accuracy = ((all_probs >= 0.5) == all_true).mean()
    middle_percent = ((all_probs >= 0.45) & (all_probs <= 0.55)).sum() / len(all_probs) * 100
    has_dispersion = middle_percent < 60
    has_edge = overall_accuracy > 0.5
    
    report.append(f"1. Probability dispersion: {middle_percent:.1f}% in middle (need <60%): {'✅ PASS' if has_dispersion else '❌ FAIL'}")
    report.append(f"2. Overall edge: accuracy={overall_accuracy:.3f} (need >0.5): {'✅ PASS' if has_edge else '❌ FAIL'}")
    report.append(f"3. Non-trivial probabilities: mean std={all_probs.std():.3f}")
    report.append("")
    
    if has_dispersion and has_edge:
        report.append("✅ SYSTEM SUCCESS: Model produces non-trivial probabilities with edge")
    else:
        report.append("❌ SYSTEM FAILURE: Does not meet requirements")
    
    report.append("\n" + "=" * 80)
    
    # Write report
    with open('STATE_CONDITIONAL_MODEL_REPORT.txt', 'w') as f:
        f.write("\n".join(report))
    
    print("\nReport saved to STATE_CONDITIONAL_MODEL_REPORT.txt")

if __name__ == '__main__':
    main()