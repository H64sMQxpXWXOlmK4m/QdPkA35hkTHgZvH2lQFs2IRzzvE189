import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif, VarianceThreshold
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, roc_auc_score, brier_score_loss, log_loss,
    precision_score, recall_score, f1_score, confusion_matrix,
    classification_report
)
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
# import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')
import pickle
import os
from typing import Tuple, List, Dict, Any

from advanced_features import AdvancedFeatureEngineer

class RobustModelTrainer:
    def __init__(self, symbol: str, lookback_minutes: int = 15, 
                 test_size: float = 0.2, random_state: int = 42):
        self.symbol = symbol
        self.lookback = lookback_minutes
        self.test_size = test_size
        self.random_state = random_state
        self.features_df = None
        self.labels = None
        self.scaler = StandardScaler()
        self.feature_selector = None
        self.model = None
        self.calibrator = None
        self.results = {}
    
    def load_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Load features and labels using AdvancedFeatureEngineer."""
        engineer = AdvancedFeatureEngineer(self.symbol, lookback_minutes=self.lookback)
        features, labels = engineer.compute_all_features()
        self.features_df = features
        self.labels = labels
        return features, labels
    
    def temporal_split(self, X: pd.DataFrame, y: pd.Series) -> Tuple:
        """Split data temporally."""
        n = len(X)
        split_idx = int(n * (1 - self.test_size))
        X_train = X.iloc[:split_idx]
        y_train = y.iloc[:split_idx]
        X_test = X.iloc[split_idx:]
        y_test = y.iloc[split_idx:]
        return X_train, X_test, y_train, y_test
    
    def expanding_window_split(self, X: pd.DataFrame, y: pd.Series, n_splits: int = 5):
        """
        Generate expanding window splits for time series.
        Yields (train_idx, val_idx) for each fold.
        """
        n = len(X)
        min_train_size = int(n * 0.3)  # at least 30% for training
        step = (n - min_train_size) // n_splits
        for i in range(n_splits):
            train_end = min_train_size + i * step
            val_start = train_end
            val_end = min(train_end + step, n)
            if val_end <= val_start:
                continue
            train_idx = list(range(train_end))
            val_idx = list(range(val_start, val_end))
            yield train_idx, val_idx
    
    def select_features(self, X_train: pd.DataFrame, y_train: pd.Series, 
                        X_test: pd.DataFrame, method: str = 'mutual_info', k: int = 30):
        """
        Select top k features using mutual information or variance.
        Returns selected X_train, X_test, and feature selector.
        """
        if method == 'mutual_info':
            selector = SelectKBest(mutual_info_classif, k=min(k, X_train.shape[1]))
            selector.fit(X_train, y_train)
        elif method == 'variance':
            selector = VarianceThreshold(threshold=0.01)
            selector.fit(X_train)
            # after variance threshold, maybe still many features
            # we'll keep all remaining
        else:
            raise ValueError("method must be 'mutual_info' or 'variance'")
        
        X_train_selected = selector.transform(X_train)
        X_test_selected = selector.transform(X_test)
        # get selected feature names
        if hasattr(selector, 'get_support'):
            selected_mask = selector.get_support()
            selected_features = X_train.columns[selected_mask].tolist()
        else:
            selected_features = X_train.columns.tolist()
        
        self.feature_selector = selector
        return X_train_selected, X_test_selected, selected_features
    
    def train_xgboost(self, X_train: np.ndarray, y_train: np.ndarray,
                      X_val: np.ndarray = None, y_val: np.ndarray = None):
        """
        Train XGBoost with hyperparameter tuning using time series cross-validation.
        X_val, y_val are ignored for hyperparameter search (used later for calibration).
        """
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [2, 3, 4, 5],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0],
            'reg_alpha': [0, 0.1, 1],
            'reg_lambda': [0.1, 1, 10],
        }
        
        # Use randomized search with time series cross-validation
        tscv = list(self.expanding_window_split(
            pd.DataFrame(X_train), pd.Series(y_train), n_splits=3
        ))
        # Convert to sklearn's split format
        cv = [(train, val) for train, val in tscv]
        
        model = xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric='logloss',
            random_state=self.random_state,
        )
        search = RandomizedSearchCV(
            model, param_grid, n_iter=20, cv=cv,
            scoring='neg_log_loss', random_state=self.random_state, n_jobs=-1,
            verbose=0
        )
        search.fit(X_train, y_train)
        model = search.best_estimator_
        
        self.model = model
        return model
    
    def calibrate(self, X_cal: np.ndarray, y_cal: np.ndarray, method: str = 'isotonic'):
        """
        Calibrate model probabilities using Platt scaling or isotonic regression.
        """
        if self.model is None:
            raise ValueError("Model not trained yet.")
        
        # Get uncalibrated probabilities
        y_pred_proba = self.model.predict_proba(X_cal)[:, 1]
        
        if method == 'platt':
            calibrator = LogisticRegression(C=1e9, solver='lbfgs')
            calibrator.fit(y_pred_proba.reshape(-1, 1), y_cal)
        elif method == 'isotonic':
            calibrator = IsotonicRegression(out_of_bounds='clip')
            calibrator.fit(y_pred_proba, y_cal)
        else:
            raise ValueError("method must be 'platt' or 'isotonic'")
        
        self.calibrator = calibrator
        return calibrator
    
    def predict_proba(self, X: np.ndarray, calibrated: bool = True) -> np.ndarray:
        """Predict probabilities, optionally calibrated."""
        if self.model is None:
            raise ValueError("Model not trained yet.")
        
        y_pred = self.model.predict_proba(X)[:, 1]
        if calibrated and self.calibrator is not None:
            y_pred = self.calibrator.predict(y_pred.reshape(-1, 1)).flatten()
        return y_pred
    
    def evaluate(self, X: np.ndarray, y: np.ndarray, calibrated: bool = True) -> Dict[str, Any]:
        """Evaluate model on given data."""
        y_pred_proba = self.predict_proba(X, calibrated=calibrated)
        y_pred = (y_pred_proba >= 0.5).astype(int)
        
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'roc_auc': roc_auc_score(y, y_pred_proba),
            'brier': brier_score_loss(y, y_pred_proba),
            'log_loss': log_loss(y, y_pred_proba),
            'precision': precision_score(y, y_pred, zero_division=0),
            'recall': recall_score(y, y_pred, zero_division=0),
            'f1': f1_score(y, y_pred, zero_division=0),
        }
        
        # Calibration curve
        from sklearn.calibration import calibration_curve
        prob_true, prob_pred = calibration_curve(y, y_pred_proba, n_bins=10)
        metrics['calibration_curve'] = (prob_true, prob_pred)
        
        # Expected Calibration Error
        def expected_calibration_error(y_true, y_prob, n_bins=10):
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
        
        metrics['ece'] = expected_calibration_error(y, y_pred_proba)
        
        # Confusion matrix
        metrics['confusion_matrix'] = confusion_matrix(y, y_pred)
        
        return metrics
    
    def run_pipeline(self, feature_selection_k: int = 30, calibration_method: str = 'isotonic'):
        """
        Full training pipeline:
        1. Load data
        2. Temporal split
        3. Feature scaling
        4. Feature selection
        5. Train XGBoost with validation early stopping
        6. Calibrate on validation set
        7. Evaluate on test set
        """
        print(f"=== Running pipeline for {self.symbol} ===")
        
        # 1. Load data
        X, y = self.load_data()
        print(f"Data shape: {X.shape}")
        
        # 2. Temporal split
        X_train_raw, X_test_raw, y_train, y_test = self.temporal_split(X, y)
        print(f"Train shape: {X_train_raw.shape}, Test shape: {X_test_raw.shape}")
        
        # 3. Scale features
        X_train_scaled = self.scaler.fit_transform(X_train_raw)
        X_test_scaled = self.scaler.transform(X_test_raw)
        
        # 4. Feature selection (using mutual info on training only)
        X_train_selected, X_test_selected, selected_features = self.select_features(
            X_train_raw, y_train, X_test_raw, method='mutual_info', k=feature_selection_k
        )
        print(f"Selected {len(selected_features)} features")
        print("Top 10 features:", selected_features[:10])
        
        # 5. Split training into train/validation for early stopping
        val_size = int(0.2 * len(X_train_selected))
        X_train_final = X_train_selected[:-val_size]
        X_val = X_train_selected[-val_size:]
        y_train_final = y_train.iloc[:-val_size]
        y_val = y_train.iloc[-val_size:]
        
        # 6. Train XGBoost with early stopping
        print("Training XGBoost with early stopping...")
        model = self.train_xgboost(X_train_final, y_train_final.values, X_val, y_val.values)
        print(f"Best iteration: {model.best_iteration if hasattr(model, 'best_iteration') else 'N/A'}")
        
        # 7. Calibrate on validation set
        print(f"Calibrating with {calibration_method}...")
        self.calibrate(X_val, y_val.values, method=calibration_method)
        
        # 8. Evaluate on test set
        print("Evaluating on test set...")
        test_metrics = self.evaluate(X_test_selected, y_test.values, calibrated=True)
        
        # 9. Evaluate on training set (for overfitting check)
        train_metrics = self.evaluate(X_train_selected, y_train.values, calibrated=True)
        
        self.results = {
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'selected_features': selected_features,
            'model': self.model,
            'calibrator': self.calibrator,
            'scaler': self.scaler,
            'feature_selector': self.feature_selector,
        }
        
        print("\n=== Results ===")
        print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
        print(f"Test ROC AUC: {test_metrics['roc_auc']:.4f}")
        print(f"Test Brier: {test_metrics['brier']:.4f}")
        print(f"Test Log Loss: {test_metrics['log_loss']:.4f}")
        print(f"Test ECE: {test_metrics['ece']:.4f}")
        
        # Plot calibration curve
        self.plot_calibration_curve(test_metrics['calibration_curve'], self.symbol)
        
        return self.results
    
    def plot_calibration_curve(self, cal_curve, symbol):
        prob_true, prob_pred = cal_curve
        plt.figure(figsize=(8,6))
        plt.plot(prob_pred, prob_true, marker='o', label='Model')
        plt.plot([0,1], [0,1], linestyle='--', color='gray', label='Perfect')
        plt.xlabel('Mean predicted probability')
        plt.ylabel('Fraction of positives')
        plt.title(f'Calibration Curve for {symbol} (Robust Model)')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'calibration_robust_{symbol.replace("/", "_")}.png')
        plt.close()
    
    def save_artifacts(self, dir_path: str = 'models/robust'):
        """Save model artifacts."""
        os.makedirs(dir_path, exist_ok=True)
        prefix = self.symbol.replace('/', '_')
        
        with open(f'{dir_path}/{prefix}_model.pkl', 'wb') as f:
            pickle.dump(self.model, f)
        with open(f'{dir_path}/{prefix}_scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)
        with open(f'{dir_path}/{prefix}_feature_selector.pkl', 'wb') as f:
            pickle.dump(self.feature_selector, f)
        with open(f'{dir_path}/{prefix}_calibrator.pkl', 'wb') as f:
            pickle.dump(self.calibrator, f)
        
        # Save results summary
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

def run_all_assets():
    symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'XRP/USDT', 'BNB/USDT', 'DOGE/USDT']
    all_results = {}
    for sym in symbols:
        print(f"\n{'='*60}")
        trainer = RobustModelTrainer(sym, lookback_minutes=15, test_size=0.2)
        try:
            results = trainer.run_pipeline(feature_selection_k=30, calibration_method='isotonic')
            all_results[sym] = results['test_metrics']
            trainer.save_artifacts()
        except Exception as e:
            print(f"Failed for {sym}: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary table
    print("\n" + "="*60)
    print("SUMMARY ACROSS ASSETS")
    print("="*60)
    summary_df = pd.DataFrame(all_results).T
    print(summary_df[['accuracy', 'roc_auc', 'brier', 'log_loss', 'ece']])
    
    # Save summary
    summary_df.to_csv('models/robust/summary_all_assets.csv')
    return summary_df

if __name__ == '__main__':
    # Test with BTC only
    trainer = RobustModelTrainer('BTC/USDT', lookback_minutes=15, test_size=0.2)
    results = trainer.run_pipeline(feature_selection_k=30, calibration_method='isotonic')
    trainer.save_artifacts()