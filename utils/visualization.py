import pandas as pd

def get_best_model_per_dataset(df):
    """
    Get the best model performance per dataset
    
    Args:
        df (pd.DataFrame): DataFrame containing experiment results
        
    Returns:
        pd.DataFrame: DataFrame with best model scores per dataset
    """
    try:
        if not all(col in df.columns for col in ['dataset', 'model', 'best_score']):
            raise ValueError("Required columns missing: dataset, model, best_score")
        
        return (
            df.groupby(['dataset', 'model'])['best_score']
              .mean()
              .reset_index()
              .sort_values(['dataset', 'best_score'], ascending=[True, False])
        )
    except Exception as e:
        print(f"Error in get_best_model_per_dataset: {e}")
        return pd.DataFrame()

def get_best_combo_per_dataset(df):
    """
    Get the best model-tuning method combination per dataset
    
    Args:
        df (pd.DataFrame): DataFrame containing experiment results
        
    Returns:
        pd.DataFrame: DataFrame with best combinations per dataset
    """
    try:
        required_cols = ['dataset', 'model', 'tuning_method', 'best_score']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"Required columns missing: {required_cols}")
        
        combo = (
            df.groupby(['dataset', 'model', 'tuning_method'])['best_score']
              .mean()
              .reset_index()
        )
        
        # Get the best combination for each dataset
        best_combo = combo.loc[combo.groupby('dataset')['best_score'].idxmax()]
        return best_combo.sort_values('best_score', ascending=False)
        
    except Exception as e:
        print(f"Error in get_best_combo_per_dataset: {e}")
        return pd.DataFrame()
