import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

class TemperatureScaling:
    """
    Temperature scaling for probability calibration.
    Learns a single temperature parameter T > 0.
    """
    def __init__(self, temperature=1.0):
        self.temperature = temperature
    
    def fit(self, y_prob, y_true):
        """
        Find optimal temperature by minimizing log loss.
        y_prob: predicted probabilities (0-1)
        y_true: true labels (0/1)
        """
        # Convert probabilities to logits (with clipping)
        eps = 1e-12
        y_prob_clipped = np.clip(y_prob, eps, 1-eps)
        logits = np.log(y_prob_clipped / (1 - y_prob_clipped))
        
        # Optimize temperature
        def loss(t):
            # Temperature must be positive
            t = np.exp(t)  # optimize in log space
            scaled_logits = logits / t
            scaled_probs = 1 / (1 + np.exp(-scaled_logits))
            return log_loss(y_true, scaled_probs)
        
        # Initial guess: log(T) = 0 => T = 1
        result = minimize(loss, x0=0.0, method='Nelder-Mead')
        optimal_log_t = result.x[0]
        self.temperature = np.exp(optimal_log_t)
        return self
    
    def predict(self, y_prob):
        """Scale probabilities using learned temperature."""
        eps = 1e-12
        y_prob_clipped = np.clip(y_prob, eps, 1-eps)
        logits = np.log(y_prob_clipped / (1 - y_prob_clipped))
        scaled_logits = logits / self.temperature
        scaled_probs = 1 / (1 + np.exp(-scaled_logits))
        return scaled_probs

class BetaCalibration:
    """
    Beta calibration (two-parameter): p_cal = sigmoid(a * logit(p) + b)
    Equivalent to temperature scaling with bias.
    """
    def __init__(self, a=1.0, b=0.0):
        self.a = a
        self.b = b
    
    def fit(self, y_prob, y_true):
        """Fit parameters a, b using logistic regression on logits."""
        eps = 1e-12
        y_prob_clipped = np.clip(y_prob, eps, 1-eps)
        logits = np.log(y_prob_clipped / (1 - y_prob_clipped)).reshape(-1, 1)
        
        # Logistic regression (no intercept, we add bias term)
        from sklearn.linear_model import LogisticRegression
        lr = LogisticRegression(C=1e9, solver='lbfgs', fit_intercept=True)
        lr.fit(logits, y_true)
        
        self.a = lr.coef_[0][0]
        self.b = lr.intercept_[0]
        return self
    
    def predict(self, y_prob):
        eps = 1e-12
        y_prob_clipped = np.clip(y_prob, eps, 1-eps)
        logits = np.log(y_prob_clipped / (1 - y_prob_clipped))
        scaled_logits = self.a * logits + self.b
        scaled_probs = 1 / (1 + np.exp(-scaled_logits))
        return scaled_probs

def evaluate_calibration(y_true, y_prob, n_bins=10):
    """Compute calibration metrics."""
    from sklearn.calibration import calibration_curve
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins)
    
    # Expected Calibration Error
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
    
    # Brier score
    brier = brier_score_loss(y_true, y_prob)
    
    # Log loss
    loss = log_loss(y_true, y_prob)
    
    # Tail calibration (prob < 0.2, > 0.8)
    low_mask = y_prob < 0.2
    high_mask = y_prob > 0.8
    
    tail_stats = {}
    if low_mask.sum() > 0:
        tail_stats['low'] = {
            'count': low_mask.sum(),
            'mean_pred': y_prob[low_mask].mean(),
            'mean_actual': y_true[low_mask].mean(),
            'brier': brier_score_loss(y_true[low_mask], y_prob[low_mask]),
        }
    if high_mask.sum() > 0:
        tail_stats['high'] = {
            'count': high_mask.sum(),
            'mean_pred': y_prob[high_mask].mean(),
            'mean_actual': y_true[high_mask].mean(),
            'brier': brier_score_loss(y_true[high_mask], y_prob[high_mask]),
        }
    
    return {
        'ece': ece,
        'brier': brier,
        'log_loss': loss,
        'tail_stats': tail_stats,
        'calibration_curve': (prob_true, prob_pred),
    }

def compare_calibration_methods(y_train_prob, y_train_true, y_val_prob, y_val_true):
    """
    Compare different calibration methods.
    """
    methods = {}
    
    # 1. No calibration (baseline)
    methods['none'] = {'predict': lambda x: x}
    
    # 2. Isotonic regression
    iso = IsotonicRegression(out_of_bounds='clip')
    iso.fit(y_train_prob, y_train_true)
    methods['isotonic'] = {'predict': iso.predict}
    
    # 3. Temperature scaling
    temp = TemperatureScaling()
    temp.fit(y_train_prob, y_train_true)
    methods['temperature'] = {'predict': temp.predict}
    
    # 4. Beta calibration (Platt scaling with bias)
    beta = BetaCalibration()
    beta.fit(y_train_prob, y_train_true)
    methods['beta'] = {'predict': beta.predict}
    
    # Evaluate on validation set
    results = {}
    for name, method in methods.items():
        y_cal = method['predict'](y_val_prob)
        metrics = evaluate_calibration(y_val_true, y_cal)
        results[name] = metrics
        results[name]['calibrator'] = method
    
    return results

def plot_calibration_comparison(results, y_val_true, y_val_prob):
    """Plot calibration curves for comparison."""
    from sklearn.calibration import calibration_curve
    
    plt.figure(figsize=(12, 10))
    
    # Calibration curves
    ax1 = plt.subplot(2, 2, 1)
    for name, res in results.items():
        prob_true, prob_pred = res['calibration_curve']
        ax1.plot(prob_pred, prob_true, marker='o', label=f'{name} (ECE={res["ece"]:.3f})')
    ax1.plot([0,1], [0,1], 'k--', label='Perfect')
    ax1.set_xlabel('Mean predicted probability')
    ax1.set_ylabel('Fraction of positives')
    ax1.set_title('Calibration Curves')
    ax1.legend()
    ax1.grid(True)
    
    # Reliability diagram (binned)
    ax2 = plt.subplot(2, 2, 2)
    bin_edges = np.linspace(0, 1, 21)
    for name, res in results.items():
        y_cal = res['calibrator']['predict'](y_val_prob)
        bin_means = []
        bin_actuals = []
        for i in range(len(bin_edges)-1):
            mask = (y_cal >= bin_edges[i]) & (y_cal < bin_edges[i+1])
            if mask.sum() > 0:
                bin_means.append(y_cal[mask].mean())
                bin_actuals.append(y_val_true[mask].mean())
        ax2.plot(bin_means, bin_actuals, marker='o', label=name)
    ax2.plot([0,1], [0,1], 'k--')
    ax2.set_xlabel('Predicted probability')
    ax2.set_ylabel('Actual frequency')
    ax2.set_title('Reliability Diagram')
    ax2.legend()
    ax2.grid(True)
    
    # Metrics comparison
    ax3 = plt.subplot(2, 2, 3)
    names = list(results.keys())
    eces = [results[n]['ece'] for n in names]
    briers = [results[n]['brier'] for n in names]
    x = np.arange(len(names))
    width = 0.35
    ax3.bar(x - width/2, eces, width, label='ECE')
    ax3.bar(x + width/2, briers, width, label='Brier')
    ax3.set_xlabel('Calibration method')
    ax3.set_ylabel('Score')
    ax3.set_title('Calibration Metrics')
    ax3.set_xticks(x)
    ax3.set_xticklabels(names, rotation=45)
    ax3.legend()
    ax3.grid(True)
    
    # Tail calibration
    ax4 = plt.subplot(2, 2, 4)
    tail_data = []
    for name, res in results.items():
        if 'low' in res['tail_stats']:
            tail_data.append({
                'method': name,
                'region': 'low',
                'pred': res['tail_stats']['low']['mean_pred'],
                'actual': res['tail_stats']['low']['mean_actual'],
            })
        if 'high' in res['tail_stats']:
            tail_data.append({
                'method': name,
                'region': 'high',
                'pred': res['tail_stats']['high']['mean_pred'],
                'actual': res['tail_stats']['high']['mean_actual'],
            })
    if tail_data:
        tail_df = pd.DataFrame(tail_data)
        for region in ['low', 'high']:
            subset = tail_df[tail_df['region'] == region]
            if not subset.empty:
                ax4.bar(subset['method'] + '_' + region, subset['pred'], alpha=0.6, label=f'{region} pred')
                ax4.bar(subset['method'] + '_' + region, subset['actual'], alpha=0.6, label=f'{region} actual', width=0.4)
        ax4.set_xlabel('Method & Region')
        ax4.set_ylabel('Probability')
        ax4.set_title('Tail Calibration')
        ax4.legend()
        ax4.grid(True)
    
    plt.tight_layout()
    plt.savefig('calibration_comparison.png')
    plt.close()

def test_with_existing_model():
    """Test calibration methods with existing model predictions."""
    # We need validation set predictions
    # For now, we'll generate synthetic example
    # Later integrate with actual model
    print("Testing calibration methods...")
    
    # Generate synthetic predictions (overconfident)
    np.random.seed(42)
    n = 1000
    y_true = np.random.binomial(1, 0.5, n)
    # True probabilities (unknown)
    true_probs = 0.3 + 0.4 * np.random.rand(n)
    # Model predictions with systematic bias
    y_pred = true_probs + 0.1 * np.random.randn(n)
    # Overconfidence: push away from 0.5
    y_pred = np.clip(y_pred + 0.2 * (y_pred - 0.5), 0.05, 0.95)
    
    # Split
    split = int(0.7 * n)
    y_train_true, y_val_true = y_true[:split], y_true[split:]
    y_train_prob, y_val_prob = y_pred[:split], y_pred[split:]
    
    # Compare methods
    results = compare_calibration_methods(y_train_prob, y_train_true, y_val_prob, y_val_true)
    
    print("\nCalibration Results:")
    for name, res in results.items():
        print(f"\n{name}:")
        print(f"  ECE: {res['ece']:.4f}")
        print(f"  Brier: {res['brier']:.4f}")
        print(f"  Log loss: {res['log_loss']:.4f}")
        if 'tail_stats' in res:
            for region, stats in res['tail_stats'].items():
                print(f"  {region}: pred={stats['mean_pred']:.3f}, actual={stats['mean_actual']:.3f}, count={stats['count']}")
    
    # Plot
    plot_calibration_comparison(results, y_val_true, y_val_prob)
    print("\nPlot saved to calibration_comparison.png")
    
    return results

if __name__ == '__main__':
    # Test with synthetic data first
    results = test_with_existing_model()
    
    # TODO: Integrate with actual model predictions
    # Need to load validation set predictions from saved model