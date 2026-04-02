import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# Models
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
import xgboost as xgb
import lightgbm as lgb

# Calibration
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression as PlattScaler

# Metrics
from sklearn.metrics import accuracy_score, roc_auc_score, brier_score_loss, log_loss
from sklearn.model_selection import TimeSeriesSplit

# Feature engineer
from elite_feature_engineer import EliteFeatureEngineer

class EliteModelPipeline:
    """
    Elite modeling pipeline with walk-forward validation and calibration.
    """
    
    def __init__(self, symbol: str, data_dir: str = 'data/deep'):
        self.symbol = symbol
        self.data_dir = data_dir
        self.engineer = EliteFeatureEngineer(symbol, data_dir)
        
        # Models to compare (focus on XGBoost and Gradient Boosting)
        self.models = {
            'xgb': xgb.XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                eval_metric='logloss',
                use_label_encoder=False
            ),
            'gbm': GradientBoostingClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                random_state=42
            )
        }
        
        # Calibration methods
        self.calibration_methods = {
            'none': None,
            'platt': 'sigmoid',
            'isotonic': 'isotonic'
        }
        
        # Results storage
        self.results = {}
        
    def create_full_dataset(self, start_date: pd.Timestamp, end_date: pd.Timestamp) -> Tuple[pd.DataFrame, pd.Series]:
        """Create full feature dataset for date range."""
        return self.engineer.create_dataset(start_date, end_date)
    
    def walk_forward_validation(self, X: pd.DataFrame, y: pd.Series,
                                initial_train_size: int = 1000,
                                step_size: int = 24) -> Dict[str, Any]:
        """
        Perform walk-forward validation with expanding window.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Feature matrix with datetime index.
        y : pd.Series
            Target series.
        initial_train_size : int
            Number of samples for initial training.
        step_size : int
            Number of samples to move forward each step.
            
        Returns:
        --------
        Dict with predictions, metrics, and feature importance.
        """
        # Ensure chronological order
        X = X.sort_index()
        y = y.sort_index()
        
        n_samples = len(X)
        if initial_train_size >= n_samples:
            raise ValueError(f"initial_train_size {initial_train_size} >= total samples {n_samples}")
        
        # Prepare storage
        all_preds = []
        all_probs = []
        all_true = []
        all_indices = []
        all_models = []
        
        # Walk forward
        train_start = 0
        train_end = initial_train_size
        
        while train_end < n_samples:
            test_start = train_end
            test_end = min(test_start + step_size, n_samples)
            
            if test_start >= test_end:
                break
            
            # Split
            X_train = X.iloc[train_start:train_end]
            y_train = y.iloc[train_start:train_end]
            X_test = X.iloc[test_start:test_end]
            y_test = y.iloc[test_start:test_end]
            
            # Train each model
            for model_name, model in self.models.items():
                # Train
                model.fit(X_train, y_train)
                
                # Predict
                y_pred = model.predict(X_test)
                y_prob = model.predict_proba(X_test)[:, 1]
                
                # Store with model identifier
                for i in range(len(X_test)):
                    all_preds.append(y_pred[i])
                    all_probs.append(y_prob[i])
                    all_true.append(y_test.iloc[i])
                    all_indices.append(X_test.index[i])
                    all_models.append(model_name)
            
            # Expand training window
            train_end = test_end
        
        # Convert to DataFrame
        results_df = pd.DataFrame({
            'index': all_indices,
            'model': all_models,
            'true': all_true,
            'pred': all_preds,
            'prob': all_probs
        })
        results_df.set_index('index', inplace=True)
        
        return {
            'results': results_df,
            'metrics': self._compute_metrics(results_df['true'], results_df['pred'], results_df['prob'])
        }
    
    def _compute_metrics(self, y_true: pd.Series, y_pred: pd.Series, y_prob: pd.Series) -> Dict:
        """Compute comprehensive metrics."""
        if len(y_true) == 0:
            return {}
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'roc_auc': roc_auc_score(y_true, y_prob),
            'brier': brier_score_loss(y_true, y_prob),
            'log_loss': log_loss(y_true, y_prob),
            'up_rate': y_true.mean(),
            'n_samples': len(y_true)
        }
        
        # Calibration metrics (ECE)
        ece = self._compute_ece(y_true, y_prob, n_bins=10)
        metrics['ece'] = ece
        
        return metrics
    
    def _compute_ece(self, y_true: pd.Series, y_prob: pd.Series, n_bins: int = 10) -> float:
        """Compute Expected Calibration Error."""
        bins = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(y_prob, bins) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)
        
        ece = 0
        for i in range(n_bins):
            mask = bin_indices == i
            if np.sum(mask) > 0:
                bin_prob = np.mean(y_prob[mask])
                bin_acc = np.mean(y_true[mask])
                bin_weight = np.sum(mask) / len(y_true)
                ece += bin_weight * np.abs(bin_acc - bin_prob)
        
        return ece
    
    def calibrate_probabilities(self, y_true: pd.Series, y_prob: pd.Series, 
                                method: str = 'platt') -> pd.Series:
        """
        Calibrate probabilities using specified method.
        
        Parameters:
        -----------
        y_true : pd.Series
            True labels.
        y_prob : pd.Series
            Predicted probabilities.
        method : str
            'platt', 'isotonic', or 'none'.
            
        Returns:
        --------
        Calibrated probabilities.
        """
        if method == 'none' or len(y_true) < 10:
            return y_prob
        
        y_true = np.array(y_true)
        y_prob = np.array(y_prob)
        
        if method == 'platt':
            # Platt scaling (logistic regression)
            platt = LogisticRegression(C=1e10, solver='lbfgs')
            platt.fit(y_prob.reshape(-1, 1), y_true)
            calibrated = platt.predict_proba(y_prob.reshape(-1, 1))[:, 1]
        
        elif method == 'isotonic':
            # Isotonic regression
            iso = IsotonicRegression(out_of_bounds='clip')
            calibrated = iso.fit_transform(y_prob, y_true)
        
        else:
            raise ValueError(f"Unknown calibration method: {method}")
        
        return pd.Series(calibrated, index=pd.RangeIndex(len(calibrated)))
    
    def evaluate_calibration(self, y_true: pd.Series, y_prob: pd.Series, 
                             n_bins: int = 10) -> Dict:
        """
        Evaluate calibration with reliability diagram.
        
        Returns:
        --------
        Dict with calibration curve data and metrics.
        """
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins)
        
        # Compute calibration metrics
        ece = self._compute_ece(y_true, y_prob, n_bins)
        mce = 0  # max calibration error
        for i in range(len(prob_true)):
            mce = max(mce, np.abs(prob_true[i] - prob_pred[i]))
        
        return {
            'prob_true': prob_true,
            'prob_pred': prob_pred,
            'ece': ece,
            'mce': mce,
            'bin_counts': np.histogram(y_prob, bins=np.linspace(0, 1, n_bins + 1))[0]
        }
    
    def run_experiment(self, start_date: pd.Timestamp, end_date: pd.Timestamp,
                       initial_train_days: int = 30) -> Dict[str, Any]:
        """
        Run complete experiment: dataset creation, walk-forward validation,
        calibration, and evaluation.
        
        Parameters:
        -----------
        start_date, end_date : pd.Timestamp
            Date range for entire dataset.
        initial_train_days : int
            Number of days for initial training (converted to hours).
            
        Returns:
        --------
        Dict with results, metrics, calibration info.
        """
        print(f"Creating dataset for {self.symbol} from {start_date} to {end_date}")
        X, y = self.create_full_dataset(start_date, end_date)
        
        if X.empty:
            print("No data created.")
            return {}
        
        print(f"Dataset shape: {X.shape}")
        print(f"Target up rate: {y.mean():.3f}")
        
        # Convert initial_train_days to hours
        initial_train_size = initial_train_days * 24
        step_size = 48  # 2 day steps (reduce computational load)
        
        # Walk-forward validation
        print("Running walk-forward validation...")
        wf_results = self.walk_forward_validation(X, y, initial_train_size, step_size)
        
        results_df = wf_results['results']
        # Use XGBoost predictions only
        xgb_results = results_df[results_df['model'] == 'xgb']
        if len(xgb_results) == 0:
            print("No XGBoost predictions found.")
            return {}
        
        metrics = self._compute_metrics(xgb_results['true'], xgb_results['pred'], xgb_results['prob'])
        
        print("\nOverall Metrics (XGBoost):")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")
        
        # Calibration experiment
        print("\nCalibration experiment...")
        calibration_results = {}
        for method_name in ['none', 'platt', 'isotonic']:
            calibrated_probs = self.calibrate_probabilities(
                xgb_results['true'], xgb_results['prob'], method=method_name
            )
            cal_metrics = self._compute_metrics(
                xgb_results['true'], 
                (calibrated_probs >= 0.5).astype(int),
                calibrated_probs
            )
            calibration_results[method_name] = {
                'metrics': cal_metrics,
                'probs': calibrated_probs
            }
            print(f"  {method_name:8} - ECE: {cal_metrics['ece']:.4f}, Brier: {cal_metrics['brier']:.4f}")
        
        # Feature importance (using last trained model)
        print("\nComputing feature importance...")
        feature_importance = self._compute_feature_importance(X, y)
        
        # Store results
        self.results = {
            'symbol': self.symbol,
            'X_shape': X.shape,
            'y_shape': y.shape,
            'overall_metrics': metrics,
            'calibration_results': calibration_results,
            'feature_importance': feature_importance,
            'results_df': results_df,
            'xgb_results': xgb_results
        }
        
        return self.results
    
    def _compute_feature_importance(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Compute feature importance using last trained XGBoost model."""
        # Train on full dataset for importance
        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.05,
            random_state=42,
            use_label_encoder=False
        )
        model.fit(X, y)
        
        # Get feature importance
        importance = model.feature_importances_
        feature_names = X.columns
        
        # Create DataFrame sorted by importance
        imp_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return imp_df
    
    def generate_report(self) -> str:
        """Generate detailed report of experiment results."""
        if not self.results:
            return "No results available. Run experiment first."
        
        report = []
        report.append("=" * 80)
        report.append(f"ELITE MODEL REPORT - {self.symbol}")
        report.append("=" * 80)
        
        # Dataset info
        report.append("\nDATASET")
        report.append(f"Features: {self.results['X_shape'][1]}")
        report.append(f"Samples: {self.results['X_shape'][0]}")
        report.append(f"Target up rate: {self.results['results_df']['true'].mean():.3f}")
        
        # Overall metrics
        report.append("\nOVERALL METRICS")
        for k, v in self.results['overall_metrics'].items():
            if isinstance(v, float):
                report.append(f"{k:15}: {v:.4f}")
            else:
                report.append(f"{k:15}: {v}")
        
        # Calibration comparison
        report.append("\nCALIBRATION COMPARISON")
        report.append("Method     | Accuracy | ROC AUC | Brier   | Log Loss | ECE")
        report.append("-" * 80)
        for method, res in self.results['calibration_results'].items():
            m = res['metrics']
            report.append(f"{method:10} | {m['accuracy']:.4f} | {m['roc_auc']:.4f} | "
                         f"{m['brier']:.4f} | {m['log_loss']:.4f} | {m['ece']:.4f}")
        
        # Top features
        report.append("\nTOP 10 FEATURES BY IMPORTANCE")
        top_features = self.results['feature_importance'].head(10)
        for _, row in top_features.iterrows():
            report.append(f"{row['feature']:30}: {row['importance']:.4f}")
        
        return "\n".join(report)