import pandas as pd
import numpy as np
from robust_modeling import RobustModelTrainer
import warnings
warnings.filterwarnings('ignore')

symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'XRP/USDT', 'BNB/USDT', 'DOGE/USDT']
all_results = {}

for sym in symbols:
    print(f"\n{'='*60}")
    print(f"Processing {sym}")
    print('='*60)
    try:
        trainer = RobustModelTrainer(sym, lookback_minutes=15, test_size=0.2)
        results = trainer.run_pipeline(feature_selection_k=30, calibration_method='isotonic')
        all_results[sym] = results['test_metrics']
        trainer.save_artifacts()
    except Exception as e:
        print(f"Failed for {sym}: {e}")
        import traceback
        traceback.print_exc()

# Summary
print("\n" + "="*60)
print("SUMMARY ACROSS ASSETS")
print("="*60)
summary_df = pd.DataFrame(all_results).T
print(summary_df[['accuracy', 'roc_auc', 'brier', 'log_loss', 'ece']])

# Save summary
import os
os.makedirs('models/robust', exist_ok=True)
summary_df.to_csv('models/robust/summary_all_assets.csv')
print("\nSummary saved to models/robust/summary_all_assets.csv")