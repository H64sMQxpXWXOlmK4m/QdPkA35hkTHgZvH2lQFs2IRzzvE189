#!/usr/bin/env python3
"""Test the elite model pipeline."""
import sys
sys.path.append('.')
from elite_model_pipeline import EliteModelPipeline
import pandas as pd
import numpy as np

def test_basic():
    print("Testing EliteModelPipeline with BTC/USDT...")
    
    # Initialize pipeline
    pipeline = EliteModelPipeline('BTC/USDT')
    
    # Define date range (10 days)
    start_date = pd.Timestamp('2026-03-01 00:00:00')
    end_date = pd.Timestamp('2026-03-11 23:00:00')
    
    print(f"Date range: {start_date} to {end_date}")
    
    # Run experiment
    results = pipeline.run_experiment(
        start_date=start_date,
        end_date=end_date,
        initial_train_days=5
    )
    
    if not results:
        print("Experiment failed.")
        return
    
    # Print report
    report = pipeline.generate_report()
    print(report)
    
    # Save results to file
    with open('test_pipeline_results.txt', 'w') as f:
        f.write(report)
    
    print("\nTest completed. Results saved to test_pipeline_results.txt")

if __name__ == '__main__':
    test_basic()