import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')
import sys
sys.path.append('.')
from data_loader import DataLoader
from advanced_features import AdvancedFeatureEngineer
from regime_detection import RegimeDetector
from calibration_improvement import TemperatureScaling

class ElitePredictor:
    def __init__(self, symbol='BTC/USDT'):
        self.symbol = symbol
        self.prefix = symbol.replace('/', '_')
        self.load_artifacts()
        self.feature_engineer = None
    
    def load_artifacts(self):
        """Load model, selector, and regime calibrators."""
        # Load robust model artifacts
        with open(f'models/robust/{self.prefix}_model.pkl', 'rb') as f:
            self.model = pickle.load(f)
        with open(f'models/robust/{self.prefix}_feature_selector.pkl', 'rb') as f:
            self.selector = pickle.load(f)
        
        # Load regime calibrators
        try:
            with open(f'models/regime_adaptive/{self.prefix}_regime_calibrators.pkl', 'rb') as f:
                self.regime_calibrators = pickle.load(f)
        except:
            print("Warning: No regime calibrators found, using global temperature scaling.")
            self.regime_calibrators = None
        
        # Load existing isotonic calibrator (optional)
        try:
            with open(f'models/robust/{self.prefix}_calibrator.pkl', 'rb') as f:
                self.isotonic_calibrator = pickle.load(f)
        except:
            self.isotonic_calibrator = None
        
        print(f"Loaded model for {self.symbol}")
    
    def compute_features_for_hour(self, hour_start: pd.Timestamp, lookback_minutes: int = 15):
        """Compute features for a specific hour."""
        if self.feature_engineer is None:
            self.feature_engineer = AdvancedFeatureEngineer(self.symbol, lookback_minutes=lookback_minutes)
        
        features_all, _ = self.feature_engineer.compute_all_features()
        if hour_start in features_all.index:
            return features_all.loc[[hour_start]]
        else:
            raise ValueError(f"No features for hour {hour_start}")
    
    def detect_regime(self, hour_start: pd.Timestamp):
        """Detect volatility regime for the hour."""
        loader = DataLoader(data_dir='data/deep')
        df_hourly = loader.load_asset(self.symbol, timeframe='1h')
        detector = RegimeDetector(window_bars=168)
        regime_feats = detector.compute_regime_features(df_hourly)
        
        if hour_start in regime_feats.index:
            regime_row = regime_feats.loc[hour_start]
            # Determine which volatility regime is active
            if regime_row['vol_high'] == 1:
                return 'high'
            elif regime_row['vol_low'] == 1:
                return 'low'
            elif regime_row['vol_medium'] == 1:
                return 'medium'
            else:
                return 'medium'  # default
        else:
            return 'medium'  # default
    
    def predict_probability(self, hour_start: pd.Timestamp):
        """
        Predict probability that hour closes higher than opens,
        with regime-adaptive calibration.
        """
        # 1. Compute features
        X_raw = self.compute_features_for_hour(hour_start)
        
        # 2. Apply feature selection
        X_selected = self.selector.transform(X_raw)
        
        # 3. Get raw model probability
        prob_raw = self.model.predict_proba(X_selected)[:, 1][0]
        
        # 4. Detect regime
        regime = self.detect_regime(hour_start)
        
        # 5. Apply calibration
        if self.regime_calibrators and regime in self.regime_calibrators:
            calibrator = self.regime_calibrators[regime]['calibrator']
            prob_cal = calibrator.predict(np.array([prob_raw]))[0]
            calibration_method = f'temperature_{regime}'
        elif self.isotonic_calibrator is not None:
            prob_cal = self.isotonic_calibrator.predict(prob_raw.reshape(1, -1))[0]
            calibration_method = 'isotonic'
        else:
            prob_cal = prob_raw
            calibration_method = 'uncalibrated'
        
        # 6. Get open price
        loader = DataLoader(data_dir='data/deep')
        df_hourly = loader.load_asset(self.symbol, timeframe='1h')
        open_price = df_hourly.loc[hour_start, 'open']
        
        result = {
            'symbol': self.symbol,
            'hour_start': hour_start,
            'open_price': open_price,
            'regime': regime,
            'prob_raw': prob_raw,
            'prob_cal': prob_cal,
            'calibration_method': calibration_method,
            'direction': 'UP' if prob_cal >= 0.5 else 'DOWN',
            'confidence': abs(prob_cal - 0.5) * 2,
        }
        
        # If hour has already closed, add actual outcome
        if hour_start + pd.Timedelta(hours=1) <= df_hourly.index.max():
            close_price = df_hourly.loc[hour_start, 'close']
            actual_up = close_price > open_price
            result['close_price'] = close_price
            result['actual_direction'] = 'UP' if actual_up else 'DOWN'
            result['correct'] = (prob_cal >= 0.5) == actual_up
        
        return result
    
    def predict_latest(self):
        """Predict for the latest hour with at least 15 minutes of data."""
        loader = DataLoader(data_dir='data/deep')
        df_min = loader.load_asset(self.symbol, timeframe='1m')
        latest_minute = df_min.index.max()
        hour_start = latest_minute.replace(minute=0, second=0, microsecond=0)
        
        # Ensure we have at least 15 minutes of data in this hour
        minute_slice = df_min.loc[hour_start: hour_start + pd.Timedelta(minutes=59)]
        if len(minute_slice) < 15:
            hour_start -= pd.Timedelta(hours=1)
            print(f"Insufficient data, using previous hour: {hour_start}")
        
        return self.predict_probability(hour_start)

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Elite prediction with regime-adaptive calibration')
    parser.add_argument('--symbol', default='BTC/USDT', help='Trading symbol')
    args = parser.parse_args()
    
    predictor = ElitePredictor(args.symbol)
    result = predictor.predict_latest()
    
    print("\n" + "="*60)
    print("ELITE PREDICTION WITH REGIME-ADAPTIVE CALIBRATION")
    print("="*60)
    print(f"Symbol: {result['symbol']}")
    print(f"Hour start: {result['hour_start']}")
    print(f"Open price: {result['open_price']:.4f}")
    print(f"Volatility regime: {result['regime']}")
    print(f"Raw model probability: {result['prob_raw']:.3f}")
    print(f"Calibrated probability: {result['prob_cal']:.3f}")
    print(f"Calibration method: {result['calibration_method']}")
    print(f"Direction: {result['direction']}")
    print(f"Confidence: {result['confidence']:.3f}")
    
    if 'close_price' in result:
        print(f"Actual close: {result['close_price']:.4f}")
        print(f"Actual direction: {result['actual_direction']}")
        print(f"Prediction correct: {result['correct']}")
    
    print("\nRegime-specific temperature scaling:")
    if predictor.regime_calibrators:
        for regime, cal in predictor.regime_calibrators.items():
            print(f"  {regime}: T={cal['temperature']:.3f}, samples={cal['samples']}")
    
    print("\n" + "="*60)

if __name__ == '__main__':
    main()