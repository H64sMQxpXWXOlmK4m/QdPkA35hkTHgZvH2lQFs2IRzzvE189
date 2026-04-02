import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score, roc_auc_score, brier_score_loss,
    precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict, Any
import warnings
warnings.filterwarnings('ignore')

from feature_engineering import FeatureEngineer
from data_loader import DataLoader

class SingleAssetModel:
    def __init__(self, symbol: str, test_size: float = 0.2):
        self.symbol = symbol
        self.test_size = test_size
        self.scaler = StandardScaler()
        self.model = LogisticRegression(C=1.0, penalty='l2', max_iter=1000, random_state=42)
        self.features_df = None
        self.labels = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
    
    def load_data(self):
        """Load hourly data for symbol."""
        loader = DataLoader()
        panel = loader.get_hourly_panel([self.symbol])
        df = pd.DataFrame()
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = panel[col].iloc[:, 0]  # single column
        return df
    
    def prepare_features_labels(self):
        """Generate features and labels."""
        df = self.load_data()
        eng = FeatureEngineer(df)
        feats, labels = eng.generate()
        self.features_df = feats
        self.labels = labels
        return feats, labels
    
    def train_test_split(self):
        """Split data temporally."""
        n = len(self.features_df)
        split_idx = int(n * (1 - self.test_size))
        X = self.features_df.values
        y = self.labels.values
        self.X_train = X[:split_idx]
        self.y_train = y[:split_idx]
        self.X_test = X[split_idx:]
        self.y_test = y[split_idx:]
        # scale features
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def train(self):
        """Train model."""
        self.model.fit(self.X_train, self.y_train)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities."""
        return self.model.predict_proba(X)[:, 1]
    
    def evaluate(self) -> Dict[str, Any]:
        """Evaluate model on test set."""
        y_pred_proba = self.predict_proba(self.X_test)
        y_pred = (y_pred_proba >= 0.5).astype(int)
        results = {
            'accuracy': accuracy_score(self.y_test, y_pred),
            'roc_auc': roc_auc_score(self.y_test, y_pred_proba),
            'brier_score': brier_score_loss(self.y_test, y_pred_proba),
            'precision': precision_score(self.y_test, y_pred, zero_division=0),
            'recall': recall_score(self.y_test, y_pred, zero_division=0),
            'f1': f1_score(self.y_test, y_pred, zero_division=0),
        }
        # calibration buckets
        from sklearn.calibration import calibration_curve
        prob_true, prob_pred = calibration_curve(self.y_test, y_pred_proba, n_bins=10)
        results['calibration_curve'] = (prob_true, prob_pred)
        return results
    
    def run_pipeline(self):
        """Full pipeline."""
        self.prepare_features_labels()
        self.train_test_split()
        self.train()
        results = self.evaluate()
        return results

def plot_calibration_curve(prob_true, prob_pred, symbol):
    """Plot calibration curve."""
    plt.figure(figsize=(8,6))
    plt.plot(prob_pred, prob_true, marker='o', label='Model')
    plt.plot([0,1], [0,1], linestyle='--', color='gray', label='Perfect')
    plt.xlabel('Mean predicted probability')
    plt.ylabel('Fraction of positives')
    plt.title(f'Calibration Curve for {symbol}')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'calibration_{symbol.replace("/", "_")}.png')
    plt.close()

if __name__ == '__main__':
    symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'XRP/USDT', 'BNB/USDT', 'DOGE/USDT']
    all_results = {}
    for sym in symbols:
        print(f"\n=== Training model for {sym} ===")
        model = SingleAssetModel(sym, test_size=0.2)
        try:
            results = model.run_pipeline()
            all_results[sym] = results
            print(f"Accuracy: {results['accuracy']:.4f}")
            print(f"ROC AUC: {results['roc_auc']:.4f}")
            print(f"Brier score: {results['brier_score']:.4f}")
            print(f"Precision: {results['precision']:.4f}")
            print(f"Recall: {results['recall']:.4f}")
            print(f"F1: {results['f1']:.4f}")
            # plot calibration
            prob_true, prob_pred = results['calibration_curve']
            plot_calibration_curve(prob_true, prob_pred, sym)
        except Exception as e:
            print(f"Error for {sym}: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary table
    print("\n=== Summary ===")
    summary_df = pd.DataFrame(all_results).T
    print(summary_df[['accuracy', 'roc_auc', 'brier_score', 'precision', 'recall', 'f1']])