import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

from advanced_features import AdvancedFeatureEngineer
from data_loader import DataLoader

class RealTimePredictor:
    def __init__(self, symbol: str, model_dir: str = 'models/robust'):
        self.symbol = symbol
        self.model_dir = model_dir
        self.prefix = symbol.replace('/', '_')
        self.load_artifacts()
        self.feature_engineer = None
    
    def load_artifacts(self):
        """Load model, scaler, feature selector, calibrator."""
        prefix = self.prefix
        with open(f'{self.model_dir}/{prefix}_model.pkl', 'rb') as f:
            self.model = pickle.load(f)
        with open(f'{self.model_dir}/{prefix}_scaler.pkl', 'rb') as f:
            self.scaler = pickle.load(f)
        with open(f'{self.model_dir}/{prefix}_feature_selector.pkl', 'rb') as f:
            self.feature_selector = pickle.load(f)
        with open(f'{self.model_dir}/{prefix}_calibrator.pkl', 'rb') as f:
            self.calibrator = pickle.load(f)
        # Load selected feature names
        with open(f'{self.model_dir}/{prefix}_selected_features.txt', 'r') as f:
            self.selected_features = [line.strip() for line in f if line.strip()]
        print(f"Loaded model for {self.symbol} with {len(self.selected_features)} features")
    
    def compute_features_for_hour(self, hour_start: pd.Timestamp, lookback_minutes: int = 15):
        """
        Compute features for a specific hour start.
        Returns DataFrame with single row (all original features).
        """
        # We'll reuse AdvancedFeatureEngineer but compute for all hours and select the one we need.
        # This is inefficient but okay for demonstration.
        # We'll cache the engineer to avoid reloading data multiple times.
        if self.feature_engineer is None:
            self.feature_engineer = AdvancedFeatureEngineer(self.symbol, lookback_minutes=lookback_minutes)
        
        # Compute all features (may already be cached)
        features_all, _ = self.feature_engineer.compute_all_features()
        # If hour_start is in features_all index, return that row
        if hour_start in features_all.index:
            row = features_all.loc[[hour_start]]
            return row
        else:
            raise ValueError(f"No features for hour {hour_start}. Available range: {features_all.index.min()} to {features_all.index.max()}")
    
    def predict_probability(self, hour_start: pd.Timestamp) -> float:
        """
        Predict probability that hour closes higher than opens.
        """
        # Compute features
        X_raw = self.compute_features_for_hour(hour_start)
        # Ensure column order matches training (should be same)
        # Apply feature selection
        X_selected = self.feature_selector.transform(X_raw)
        # Note: scaler is not needed for XGBoost (scale invariant) and was not used in training.
        # Predict uncalibrated probability
        prob_uncal = self.model.predict_proba(X_selected)[:, 1][0]
        # Calibrate
        prob_cal = self.calibrator.predict(prob_uncal.reshape(1, -1))[0]
        return prob_cal
    
    def predict_latest(self) -> dict:
        """
        Predict for the latest hour that has at least 15 minutes of data.
        Returns dict with prediction and metadata.
        """
        # Load minute data to see latest timestamp
        loader = DataLoader(data_dir='data/deep')
        df_min = loader.load_asset(self.symbol, timeframe='1m')
        latest_minute = df_min.index.max()
        hour_start = latest_minute.replace(minute=0, second=0, microsecond=0)
        print(f"Latest minute: {latest_minute}")
        print(f"Predicting for hour starting: {hour_start}")
        
        # Ensure we have at least 15 minutes of data in this hour
        minute_slice = df_min.loc[hour_start: hour_start + pd.Timedelta(minutes=59)]
        if len(minute_slice) < 15:
            # fallback to previous hour
            hour_start -= pd.Timedelta(hours=1)
            print(f"Insufficient data, using previous hour: {hour_start}")
        
        prob = self.predict_probability(hour_start)
        
        # Get open price
        df_hourly = loader.load_asset(self.symbol, timeframe='1h')
        open_price = df_hourly.loc[hour_start, 'open']
        
        result = {
            'symbol': self.symbol,
            'hour_start': hour_start,
            'predicted_probability': prob,
            'open_price': open_price,
            'direction': 'UP' if prob >= 0.5 else 'DOWN',
            'confidence': abs(prob - 0.5) * 2,  # 0 to 1
        }
        
        # If hour has already closed, we can compare with actual
        if hour_start + pd.Timedelta(hours=1) <= df_hourly.index.max():
            close_price = df_hourly.loc[hour_start, 'close']
            actual_up = close_price > open_price
            result['actual_close'] = close_price
            result['actual_direction'] = 'UP' if actual_up else 'DOWN'
            result['correct'] = (prob >= 0.5) == actual_up
        
        return result

def main():
    import sys
    symbol = sys.argv[1] if len(sys.argv) > 1 else 'BTC/USDT'
    print(f"Real-time prediction for {symbol}")
    predictor = RealTimePredictor(symbol)
    result = predictor.predict_latest()
    
    print("\n=== Prediction ===")
    print(f"Symbol: {result['symbol']}")
    print(f"Hour start: {result['hour_start']}")
    print(f"Open price: {result['open_price']:.4f}")
    print(f"Predicted probability of higher close: {result['predicted_probability']:.3f}")
    print(f"Direction: {result['direction']}")
    print(f"Confidence: {result['confidence']:.3f}")
    if 'actual_close' in result:
        print(f"Actual close: {result['actual_close']:.4f}")
        print(f"Actual direction: {result['actual_direction']}")
        print(f"Prediction correct: {result['correct']}")
    print()

if __name__ == '__main__':
    main()