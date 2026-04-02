import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.calibration import IsotonicRegression
from sklearn.metrics import (
    accuracy_score, roc_auc_score, brier_score_loss, log_loss,
    precision_score, recall_score, f1_score
)
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')
import pickle
import os
import sys
sys.path.append('.')
from enhanced_features import EnhancedFeatureEngineer
from regime_detection import RegimeDetector
from data_loader import DataLoader
from calibration_improvement import TemperatureScaling, BetaCalibration

class EliteModelTrainer:
    def __init__(self, symbol: str, lookback_minutes: int = 15,
                 test_size: float = 0.2, random_state: int = 42):
        self.symbol = symbol
        self.lookback = lookback_minutes
        self.test_size = test_size
        self.random_state = random_state
        self.features_df = None
        self.labels = None
        self.imputer = None
        self.scaler = StandardScaler()
        self.feature_selector = None
        self.model = None
        self.calibrator = None
        self.regime_calibrators = {}  # per-regime calibrators
        self.results = {}
    
    def load_extended_features(self):
        """
        Load features for extended period (2 years hourly + intra features with imputation).
        """
        print("Loading extended features...")
        engineer = EnhancedFeatureEngineer(self.symbol, lookback_minutes=self.lookback)
        
        # Get simplified features (hourly + regime) for all available hours
        simple_features, simple_labels = engineer.compute_simplified_features()
        print(f"Simplified features shape: {simple_features.shape}")
        
        # Get enhanced intra features (only where minute data exists)
        enhanced_features, enhanced_labels = engineer.compute_all_features_enhanced()
        print(f"Enhanced features shape: {enhanced_features.shape}")
        
        # Align indices: we want all hours from simplified features
        # For hours missing intra features, we'll impute later
        all_features = simple_features.copy()
        
        # Add intra features as additional columns (with NaN where missing)
        # Identify intra feature columns (excluding those already in simple)
        intra_cols = [col for col in enhanced_features.columns 
                      if col not in all_features.columns]
        
        # Merge intra features using concat (handles duplicate column names)
        if intra_cols:
            intra_subset = enhanced_features[intra_cols]
            all_features = pd.concat([all_features, intra_subset], axis=1)
        
        # Labels: use simplified labels (all hours)
        labels = simple_labels
        
        # Add missing indicator for intra features
        intra_missing = all_features[intra_cols].isna().any(axis=1) if intra_cols else pd.Series(False, index=all_features.index)
        all_features['intra_missing'] = intra_missing.astype(float)
        
        print(f"Combined features shape: {all_features.shape}")
        print(f"Intra features missing for {intra_missing.sum()} out of {len(all_features)} hours")
        
        self.features_df = all_features
        self.labels = labels
        return all_features, labels
    
    def temporal_split(self, X: pd.DataFrame, y: pd.Series):
        """Split by time."""
        n = len(X)
        split_idx = int(n * (1 - self.test_size))
        X_train = X.iloc[:split_idx]
        y_train = y.iloc[:split_idx]
        X_test = X.iloc[split_idx:]
        y_test = y.iloc[split_idx:]
        return X_train, X_test, y_train, y_test
    
    def impute_missing(self, X_train, X_test):
        """Impute missing values using median of training data."""
        self.imputer = SimpleImputer(strategy='median', add_indicator=True)
        X_train_imputed = self.imputer.fit_transform(X_train)
        X_test_imputed = self.imputer.transform(X_test)
        
        # Get feature names after imputation
        feature_names = list(X_train.columns)
        indicator_features = [f'missing_{i}' for i in range(len(self.imputer.indicator_.features_))]
        feature_names += indicator_features
        
        return X_train_imputed, X_test_imputed, feature_names
    
    def select_features(self, X_train, y_train, X_test, k=30):
        """Select top k features using mutual information."""
        selector = SelectKBest(mutual_info_classif, k=min(k, X_train.shape[1]))
        X_train_selected = selector.fit_transform(X_train, y_train)
        X_test_selected = selector.transform(X_test)
        
        if hasattr(selector, 'get_support'):
            selected_mask = selector.get_support()
            # If we have feature names from imputer, use them
            if hasattr(self, 'feature_names'):
                selected_features = [self.feature_names[i] for i in range(len(selected_mask)) if selected_mask[i]]
            else:
                selected_features = [f'feature_{i}' for i in range(len(selected_mask)) if selected_mask[i]]
        else:
            selected_features = []
        
        self.feature_selector = selector
        return X_train_selected, X_test_selected, selected_features
    
    def train_xgboost(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train XGBoost with hyperparameter tuning using time-series CV.
        """
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 4, 5],
            'learning_rate': [0.01, 0.05, 0.1],
            'subsample': [0.7, 0.8, 0.9],
            'colsample_bytree': [0.7, 0.8, 0.9],
            'reg_alpha': [0, 0.1, 1],
            'reg_lambda': [1, 5, 10],
        }
        
        # Use time-series cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        best_score = -np.inf
        best_params = None
        
        # Simple grid search (small)
        for n_estimators in param_grid['n_estimators'][:2]:
            for max_depth in param_grid['max_depth'][:2]:
                for lr in param_grid['learning_rate'][:2]:
                    # Cross-validation
                    scores = []
                    for train_idx, val_idx in tscv.split(X_train):
                        X_tr, X_v = X_train[train_idx], X_train[val_idx]
                        y_tr, y_v = y_train[train_idx], y_train[val_idx]
                        
                        model = xgb.XGBClassifier(
                            n_estimators=n_estimators,
                            max_depth=max_depth,
                            learning_rate=lr,
                            objective='binary:logistic',
                            eval_metric='logloss',
                            random_state=self.random_state,
                            n_jobs=-1,
                        )
                        model.fit(X_tr, y_tr)
                        y_pred = model.predict_proba(X_v)[:, 1]
                        score = -log_loss(y_v, y_pred)  # maximize negative log loss
                        scores.append(score)
                    
                    avg_score = np.mean(scores)
                    if avg_score > best_score:
                        best_score = avg_score
                        best_params = {
                            'n_estimators': n_estimators,
                            'max_depth': max_depth,
                            'learning_rate': lr,
                        }
        
        # Train final model with best params
        model = xgb.XGBClassifier(
            **best_params,
            objective='binary:logistic',
            eval_metric='logloss',
            random_state=self.random_state,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)
        
        self.model = model
        print(f"Best params: {best_params}")
        return model
    
    def calibrate_per_regime(self, X_val, y_val, regime_feats):
        """
        Calibrate separately for different regimes.
        """
        # Align indices
        aligned_idx = pd.Series(range(len(X_val))).index  # simple index
        # We need regime features aligned with validation set
        # For simplicity, we'll calibrate globally for now
        # TODO: implement per-regime calibration
        self.calibrator = IsotonicRegression(out_of_bounds='clip')
        y_pred = self.model.predict_proba(X_val)[:, 1]
        self.calibrator.fit(y_pred, y_val)
        
        # Also fit temperature scaling as fallback
        self.temp_scaler = TemperatureScaling()
        self.temp_scaler.fit(y_pred, y_val)
        
        return self.calibrator
    
    def evaluate_regime_performance(self, X, y, regime_feats):
        """Evaluate performance across regimes."""
        results = {}
        
        # Define regimes
        regimes = {
            'high_vol': regime_feats['vol_high'] == 1,
            'low_vol': regime_feats['vol_low'] == 1,
            'medium_vol': regime_feats['vol_medium'] == 1,
            'trending': (regime_feats['trend_up'] == 1) | (regime_feats['trend_down'] == 1),
            'sideways': regime_feats['trend_sideways'] == 1,
        }
        
        for name, mask in regimes.items():
            if mask.sum() > 10:
                X_regime = X[mask]
                y_regime = y[mask]
                
                if len(X_regime) == 0:
                    continue
                
                y_pred = self.model.predict_proba(X_regime)[:, 1]
                if self.calibrator is not None:
                    y_pred_cal = self.calibrator.predict(y_pred)
                else:
                    y_pred_cal = y_pred
                
                results[name] = {
                    'samples': len(y_regime),
                    'accuracy': accuracy_score(y_regime, (y_pred_cal >= 0.5).astype(int)),
                    'brier': brier_score_loss(y_regime, y_pred_cal),
                    'roc_auc': roc_auc_score(y_regime, y_pred_cal),
                    'actual_up_rate': y_regime.mean(),
                    'pred_up_rate': (y_pred_cal >= 0.5).mean(),
                }
        
        return results
    
    def run_pipeline(self):
        """Full training pipeline."""
        print(f"=== Elite Modeling for {self.symbol} ===")
        
        # 1. Load features
        X, y = self.load_extended_features()
        
        # 2. Temporal split
        X_train, X_test, y_train, y_test = self.temporal_split(X, y)
        print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
        
        # 3. Impute missing values
        X_train_imputed, X_test_imputed, feature_names = self.impute_missing(X_train, X_test)
        self.feature_names = feature_names
        
        # 4. Scale features
        X_train_scaled = self.scaler.fit_transform(X_train_imputed)
        X_test_scaled = self.scaler.transform(X_test_imputed)
        
        # 5. Feature selection
        X_train_selected, X_test_selected, selected_features = self.select_features(
            X_train_scaled, y_train.values, X_test_scaled, k=40
        )
        print(f"Selected {len(selected_features)} features")
        print("Top features:", selected_features[:10])
        
        # 6. Split train into train/validation for calibration
        val_size = int(0.2 * len(X_train_selected))
        X_train_final = X_train_selected[:-val_size]
        X_val = X_train_selected[-val_size:]
        y_train_final = y_train.values[:-val_size]
        y_val = y_train.values[-val_size:]
        
        # 7. Train XGBoost
        print("Training XGBoost...")
        model = self.train_xgboost(X_train_final, y_train_final, X_val, y_val)
        
        # 8. Calibrate
        print("Calibrating...")
        self.calibrate_per_regime(X_val, y_val, None)  # TODO: add regime features
        
        # 9. Evaluate on test set
        print("Evaluating...")
        y_test_pred = model.predict_proba(X_test_selected)[:, 1]
        y_test_cal = self.calibrator.predict(y_test_pred)
        
        test_metrics = {
            'accuracy': accuracy_score(y_test, (y_test_cal >= 0.5).astype(int)),
            'brier': brier_score_loss(y_test, y_test_cal),
            'roc_auc': roc_auc_score(y_test, y_test_cal),
            'log_loss': log_loss(y_test, y_test_cal),
        }
        
        # 10. Evaluate on training set (check overfitting)
        y_train_pred = model.predict_proba(X_train_selected)[:, 1]
        y_train_cal = self.calibrator.predict(y_train_pred)
        
        train_metrics = {
            'accuracy': accuracy_score(y_train_final, (y_train_cal >= 0.5).astype(int)),
            'brier': brier_score_loss(y_train_final, y_train_cal),
            'roc_auc': roc_auc_score(y_train_final, y_train_cal),
            'log_loss': log_loss(y_train_final, y_train_cal),
        }
        
        # 11. Regime performance (need regime features for test set)
        try:
            loader = DataLoader(data_dir='data/deep')
            df_hourly = loader.load_asset(self.symbol, timeframe='1h')
            detector = RegimeDetector(window_bars=168)
            regime_feats = detector.compute_regime_features(df_hourly)
            # Align with test set
            aligned_idx = X_test.index.intersection(regime_feats.index)
            regime_test = regime_feats.loc[aligned_idx]
            # Need to map back to X_test_selected indices... skip for now
        except Exception as e:
            print(f"Could not compute regime performance: {e}")
        
        self.results = {
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'selected_features': selected_features,
            'model': model,
            'calibrator': self.calibrator,
            'scaler': self.scaler,
            'imputer': self.imputer,
            'feature_selector': self.feature_selector,
        }
        
        print("\n=== Results ===")
        print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
        print(f"Test ROC AUC: {test_metrics['roc_auc']:.4f}")
        print(f"Test Brier: {test_metrics['brier']:.4f}")
        print(f"Test Log Loss: {test_metrics['log_loss']:.4f}")
        
        # Plot calibration curve
        self.plot_calibration(y_test, y_test_cal, 'Test Set')
        
        return self.results
    
    def plot_calibration(self, y_true, y_pred, title):
        """Plot calibration curve."""
        from sklearn.calibration import calibration_curve
        prob_true, prob_pred = calibration_curve(y_true, y_pred, n_bins=10)
        
        plt.figure(figsize=(8,6))
        plt.plot(prob_pred, prob_true, marker='o', label='Model')
        plt.plot([0,1], [0,1], 'k--', label='Perfect')
        plt.xlabel('Mean predicted probability')
        plt.ylabel('Fraction of positives')
        plt.title(f'Calibration Curve: {title}')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'elite_calibration_{self.symbol.replace("/", "_")}.png')
        plt.close()
    
    def save_artifacts(self, dir_path='models/elite'):
        """Save model artifacts."""
        os.makedirs(dir_path, exist_ok=True)
        prefix = self.symbol.replace('/', '_')
        
        with open(f'{dir_path}/{prefix}_model.pkl', 'wb') as f:
            pickle.dump(self.model, f)
        with open(f'{dir_path}/{prefix}_scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)
        with open(f'{dir_path}/{prefix}_imputer.pkl', 'wb') as f:
            pickle.dump(self.imputer, f)
        with open(f'{dir_path}/{prefix}_feature_selector.pkl', 'wb') as f:
            pickle.dump(self.feature_selector, f)
        with open(f'{dir_path}/{prefix}_calibrator.pkl', 'wb') as f:
            pickle.dump(self.calibrator, f)
        
        # Save results
        results_df = pd.DataFrame({
            'train': self.results['train_metrics'],
            'test': self.results['test_metrics']
        }).T
        results_df.to_csv(f'{dir_path}/{prefix}_results.csv')
        
        # Save selected features
        with open(f'{dir_path}/{prefix}_selected_features.txt', 'w') as f:
            for feat in self.results['selected_features']:
                f.write(f"{feat}\n")
        
        print(f"Artifacts saved to {dir_path}")

def main():
    symbol = 'BTC/USDT'
    trainer = EliteModelTrainer(symbol, lookback_minutes=15, test_size=0.2)
    results = trainer.run_pipeline()
    trainer.save_artifacts()

if __name__ == '__main__':
    main()