import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def create_sample_data(n_experiments=200):
    """Create sample experiment data for demonstration"""
    np.random.seed(42)  # For reproducible results
    random.seed(42)
    
    # Define possible values
    tuning_methods = ['grid_search', 'random_search', 'bayesian', 'hyperband', 'optuna']
    datasets = ['iris', 'wine', 'breast_cancer', 'digits', 'boston_housing']
    models = ['random_forest', 'svm', 'xgboost', 'neural_network', 'logistic_regression']
    
    # Generate data
    data = []
    
    for i in range(n_experiments):
        method = np.random.choice(tuning_methods)
        dataset = np.random.choice(datasets)
        model = np.random.choice(models)
        
        # Generate realistic performance scores based on method and model
        base_score = {
            'random_forest': 0.85,
            'svm': 0.82,
            'xgboost': 0.88,
            'neural_network': 0.86,
            'logistic_regression': 0.80
        }[model]
        
        method_modifier = {
            'grid_search': 0.02,
            'random_search': 0.01,
            'bayesian': 0.03,
            'hyperband': 0.015,
            'optuna': 0.025
        }[method]
        
        dataset_modifier = {
            'iris': 0.05,
            'wine': 0.03,
            'breast_cancer': 0.02,
            'digits': -0.02,
            'boston_housing': -0.03
        }[dataset]
        
        # Add some random variation
        score = base_score + method_modifier + dataset_modifier + np.random.normal(0, 0.05)
        score = np.clip(score, 0.5, 0.99)  # Keep scores realistic
        
        # Generate time based on method complexity
        base_time = {
            'grid_search': 300,
            'random_search': 150,
            'bayesian': 200,
            'hyperband': 120,
            'optuna': 180
        }[method]
        
        time_sec = base_time + np.random.exponential(50)
        
        # Generate memory usage
        base_memory = {
            'random_forest': 500_000_000,
            'svm': 200_000_000,
            'xgboost': 400_000_000,
            'neural_network': 800_000_000,
            'logistic_regression': 100_000_000
        }[model]
        
        memory_bytes = int(base_memory + np.random.normal(0, base_memory * 0.2))
        memory_bytes = max(memory_bytes, 50_000_000)  # Minimum 50MB
        
        # Create timestamp
        created_at = datetime.now() - timedelta(days=np.random.randint(0, 30))
        
        data.append({
            'id': i + 1,
            'tuning_method': method,
            'dataset': dataset,
            'model': model,
            'best_score': round(score, 4),
            'time_sec': round(time_sec, 2),
            'memory_bytes': memory_bytes,
            'created_at': created_at
        })
    
    df = pd.DataFrame(data)
    
    # Add some missing values randomly (realistic scenario)
    missing_indices = np.random.choice(df.index, size=int(0.05 * len(df)), replace=False)
    df.loc[missing_indices, 'memory_bytes'] = np.nan
    
    return df
