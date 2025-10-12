import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import scipy.stats as stats
from scipy.stats import f_oneway, ttest_ind, kruskal, mannwhitneyu, shapiro, levene
import warnings
warnings.filterwarnings('ignore')

def create_plots(df):
    """Create basic analysis plots"""
    plots = {}
    
    try:
        # Score vs Method plot
        if 'tuning_method' in df.columns and 'best_score' in df.columns:
            fig = px.box(df, x='tuning_method', y='best_score', 
                        title='Score Distribution by Tuning Method',
                        color='tuning_method')
            fig.update_layout(xaxis_tickangle=-45, height=500)
            plots['score_vs_method'] = fig
        
        # Time vs Method plot
        if 'tuning_method' in df.columns and 'time_sec' in df.columns:
            fig = px.box(df, x='tuning_method', y='time_sec',
                        title='Time Distribution by Tuning Method',
                        color='tuning_method')
            fig.update_layout(xaxis_tickangle=-45, height=500)
            plots['time_vs_method'] = fig
        
        # Memory vs Method plot
        if 'tuning_method' in df.columns and 'memory_bytes' in df.columns:
            fig = px.box(df, x='tuning_method', y='memory_bytes',
                        title='Memory Usage by Tuning Method',
                        color='tuning_method')
            fig.update_layout(xaxis_tickangle=-45, height=500)
            plots['memory_vs_method'] = fig
        
        # Score vs Time efficiency plot
        if 'best_score' in df.columns and 'time_sec' in df.columns:
            fig = px.scatter(df, x='time_sec', y='best_score',
                           color='tuning_method' if 'tuning_method' in df.columns else None,
                           title='Score vs Time Efficiency',
                           hover_data=['tuning_method'] if 'tuning_method' in df.columns else None)
            plots['score_vs_time'] = fig
        
        # Performance heatmap
        if all(col in df.columns for col in ['tuning_method', 'best_score']):
            if 'dataset' in df.columns:
                pivot_data = df.pivot_table(
                    index='tuning_method', 
                    columns='dataset', 
                    values='best_score', 
                    aggfunc='mean'
                )
                fig = px.imshow(pivot_data, 
                               title='Performance Heatmap: Method vs Dataset',
                               color_continuous_scale='Viridis')
                plots['performance_heatmap'] = fig
        
    except Exception as e:
        print(f"Error creating plots: {e}")
    
    return plots

def run_statistical_tests(df, metric='best_score', alpha=0.05):
    """Run statistical tests on the data"""
    results = {}
    
    try:
        if metric not in df.columns:
            return {'error': f"Column '{metric}' not found in data"}
        
        if 'tuning_method' not in df.columns:
            return {'error': "Column 'tuning_method' not found in data"}
        
        # Group data by tuning method
        groups = [group[metric].dropna() for name, group in df.groupby('tuning_method')]
        group_names = [name for name, group in df.groupby('tuning_method')]
        
        if len(groups) < 2:
            return {'error': "Need at least 2 groups for statistical testing"}
        
        # Overall test (ANOVA or Kruskal-Wallis)
        try:
            # Test for normality first
            normal_groups = []
            for group in groups:
                if len(group) >= 3:  # Need at least 3 samples for Shapiro-Wilk
                    _, p_val = shapiro(group)
                    normal_groups.append(p_val > alpha)
                else:
                    normal_groups.append(False)
            
            if all(normal_groups) and len(groups) > 1:
                # Use ANOVA
                stat, p_val = f_oneway(*groups)
                test_name = "One-way ANOVA"
            else:
                # Use Kruskal-Wallis
                stat, p_val = kruskal(*groups)
                test_name = "Kruskal-Wallis"
            
            results['overall_test'] = {
                'test': test_name,
                'statistic': stat,
                'p_value': p_val,
                'significant': p_val < alpha,
                'interpretation': f"{'Significant' if p_val < alpha else 'No significant'} difference between groups (α={alpha})"
            }
        except Exception as e:
            results['overall_test'] = {'error': str(e)}
        
        # Pairwise tests
        pairwise_results = []
        try:
            for i in range(len(groups)):
                for j in range(i + 1, len(groups)):
                    group1, group2 = groups[i], groups[j]
                    name1, name2 = group_names[i], group_names[j]
                    
                    if len(group1) > 0 and len(group2) > 0:
                        # Use t-test or Mann-Whitney U test
                        if normal_groups[i] and normal_groups[j]:
                            stat, p_val = ttest_ind(group1, group2)
                            test = "t-test"
                        else:
                            stat, p_val = mannwhitneyu(group1, group2, alternative='two-sided')
                            test = "Mann-Whitney U"
                        
                        pairwise_results.append({
                            'Group1': name1,
                            'Group2': name2,
                            'Test': test,
                            'Statistic': round(stat, 4),
                            'P-value': round(p_val, 6),
                            'Significant': p_val < alpha
                        })
            
            results['pairwise_tests'] = pairwise_results
        except Exception as e:
            results['pairwise_tests'] = {'error': str(e)}
        
        # Descriptive statistics
        try:
            desc_stats = {}
            for name, group in zip(group_names, groups):
                if len(group) > 0:
                    desc_stats[name] = {
                        'Count': len(group),
                        'Mean': round(np.mean(group), 4),
                        'Std': round(np.std(group, ddof=1), 4),
                        'Min': round(np.min(group), 4),
                        'Max': round(np.max(group), 4),
                        'Median': round(np.median(group), 4)
                    }
            results['descriptive_stats'] = desc_stats
        except Exception as e:
            results['descriptive_stats'] = {'error': str(e)}
        
        # Effect sizes (Cohen's d for pairwise comparisons)
        try:
            effect_sizes = []
            for i in range(len(groups)):
                for j in range(i + 1, len(groups)):
                    group1, group2 = groups[i], groups[j]
                    name1, name2 = group_names[i], group_names[j]
                    
                    if len(group1) > 1 and len(group2) > 1:
                        # Cohen's d
                        pooled_std = np.sqrt(((len(group1) - 1) * np.var(group1, ddof=1) + 
                                            (len(group2) - 1) * np.var(group2, ddof=1)) / 
                                           (len(group1) + len(group2) - 2))
                        if pooled_std > 0:
                            cohens_d = (np.mean(group1) - np.mean(group2)) / pooled_std
                            effect_sizes.append({
                                'Group1': name1,
                                'Group2': name2,
                                'Cohens_d': round(cohens_d, 4),
                                'Effect_Size': 'Small' if abs(cohens_d) < 0.5 else 'Medium' if abs(cohens_d) < 0.8 else 'Large'
                            })
            
            results['effect_sizes'] = effect_sizes
        except Exception as e:
            results['effect_sizes'] = {'error': str(e)}
        
        # Normality tests for each group
        try:
            normality_results = {}
            for name, group in zip(group_names, groups):
                if len(group) >= 3:
                    stat, p_val = shapiro(group)
                    normality_results[name] = {
                        'statistic': round(stat, 4),
                        'p_value': round(p_val, 6),
                        'normal': p_val > alpha
                    }
                else:
                    normality_results[name] = {'error': 'Insufficient data for normality test'}
            
            results['normality_tests'] = normality_results
        except Exception as e:
            results['normality_tests'] = {'error': str(e)}
        
    except Exception as e:
        results['error'] = str(e)
    
    return results

def generate_summary_report(df):
    """Generate a summary report of the analysis"""
    try:
        summary = {
            'total_experiments': len(df),
            'columns': list(df.columns),
            'data_types': df.dtypes.to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'numeric_summary': {},
            'categorical_summary': {}
        }
        
        # Numeric columns summary
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            summary['numeric_summary'][col] = {
                'mean': float(df[col].mean()) if not df[col].isna().all() else None,
                'std': float(df[col].std()) if not df[col].isna().all() else None,
                'min': float(df[col].min()) if not df[col].isna().all() else None,
                'max': float(df[col].max()) if not df[col].isna().all() else None
            }
        
        # Categorical columns summary
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            summary['categorical_summary'][col] = {
                'unique_values': int(df[col].nunique()),
                'most_common': str(df[col].mode().iloc[0]) if not df[col].empty else None
            }
        
        return summary
        
    except Exception as e:
        return {'error': str(e)}

def create_model_performance_plots(df):
    """Create model performance comparison plots"""
    plots = {}
    
    try:
        if 'model' not in df.columns:
            return plots
        
        # Model performance overview
        if 'best_score' in df.columns:
            fig = px.box(df, x='model', y='best_score',
                        title='Model Performance Comparison',
                        color='model')
            fig.update_layout(xaxis_tickangle=-45, height=500)
            plots['model_performance'] = fig
        
        # Model performance by dataset
        if 'dataset' in df.columns and 'best_score' in df.columns:
            fig = px.box(df, x='dataset', y='best_score', color='model',
                        title='Model Performance by Dataset')
            fig.update_layout(xaxis_tickangle=-45, height=500)
            plots['model_by_dataset'] = fig
        
        # Model efficiency (score vs time)
        if all(col in df.columns for col in ['model', 'best_score', 'time_sec']):
            fig = px.scatter(df, x='time_sec', y='best_score', color='model',
                           title='Model Efficiency: Score vs Time',
                           hover_data=['tuning_method'] if 'tuning_method' in df.columns else None)
            plots['model_efficiency'] = fig
        
    except Exception as e:
        print(f"Error creating model performance plots: {e}")
    
    return plots

def create_dataset_analysis_plots(df):
    """Create dataset-specific analysis plots"""
    plots = {}
    
    try:
        if 'dataset' not in df.columns:
            return plots
        
        # Dataset difficulty analysis
        if 'best_score' in df.columns:
            dataset_stats = df.groupby('dataset')['best_score'].agg(['mean', 'std']).reset_index()
            fig = px.scatter(dataset_stats, x='mean', y='std', text='dataset',
                           title='Dataset Difficulty Analysis (Mean vs Variability)',
                           labels={'mean': 'Mean Score', 'std': 'Score Variability'})
            fig.update_traces(textposition="middle right")
            plots['dataset_difficulty'] = fig
        
        # Score distributions by dataset
        if 'best_score' in df.columns:
            fig = px.violin(df, x='dataset', y='best_score',
                          title='Score Distributions by Dataset',
                          box=True)
            fig.update_layout(xaxis_tickangle=-45, height=500)
            plots['dataset_distribution'] = fig
        
        # Dataset vs method heatmap
        if 'tuning_method' in df.columns and 'best_score' in df.columns:
            pivot_data = df.pivot_table(
                index='dataset', 
                columns='tuning_method', 
                values='best_score', 
                aggfunc='mean'
            )
            fig = px.imshow(pivot_data,
                           title='Dataset vs Tuning Method Performance',
                           color_continuous_scale='Viridis')
            plots['dataset_method_heatmap'] = fig
        
    except Exception as e:
        print(f"Error creating dataset analysis plots: {e}")
    
    return plots

def create_resource_efficiency_plots(df):
    """Create resource efficiency plots"""
    plots = {}
    
    try:
        # Resource usage overview
        if all(col in df.columns for col in ['time_sec', 'memory_bytes']):
            fig = px.scatter(df, x='time_sec', y='memory_bytes',
                           color='tuning_method' if 'tuning_method' in df.columns else None,
                           title='Resource Usage: Time vs Memory',
                           hover_data=['best_score'] if 'best_score' in df.columns else None)
            plots['resource_usage'] = fig
        
        # Efficiency scatter (score vs resources)
        if all(col in df.columns for col in ['best_score', 'time_sec']):
            # Create a clean dataframe for the scatter plot
            df_clean = df.copy()
            
            # Use memory_bytes for size only if it has non-null values
            size_col = None
            if 'memory_bytes' in df.columns and df['memory_bytes'].notna().any():
                # Fill NaN values with the mean to avoid plot errors
                df_clean['memory_bytes_clean'] = df_clean['memory_bytes'].fillna(df_clean['memory_bytes'].mean())
                size_col = 'memory_bytes_clean'
            
            fig = px.scatter(df_clean, x='time_sec', y='best_score',
                           size=size_col,
                           color='tuning_method' if 'tuning_method' in df.columns else None,
                           title='Efficiency Analysis: Score vs Time' + (' (bubble size = memory)' if size_col else ''))
            plots['efficiency_scatter'] = fig
        
        # Scalability analysis
        if 'tuning_method' in df.columns and 'time_sec' in df.columns:
            method_stats = df.groupby('tuning_method')['time_sec'].agg(['mean', 'std']).reset_index()
            fig = px.bar(method_stats, x='tuning_method', y='mean',
                        error_y='std',
                        title='Average Time by Tuning Method')
            fig.update_layout(xaxis_tickangle=-45)
            plots['scalability_analysis'] = fig
        
    except Exception as e:
        print(f"Error creating resource efficiency plots: {e}")
    
    return plots

def create_convergence_analysis_plots(df):
    """Create convergence analysis plots"""
    plots = {}
    
    try:
        # Learning curves (if iteration data available)
        if 'best_score' in df.columns and 'tuning_method' in df.columns:
            fig = px.line(df.groupby(['tuning_method']).cumcount().to_frame('iteration').join(df),
                         x='iteration', y='best_score', color='tuning_method',
                         title='Learning Curves by Tuning Method')
            plots['learning_curves'] = fig
        
        # Convergence speed
        if all(col in df.columns for col in ['time_sec', 'best_score']):
            df_conv = df.copy()
            df_conv['cummax_score'] = df_conv.groupby('tuning_method')['best_score'].cummax()
            fig = px.line(df_conv, x='time_sec', y='cummax_score',
                         color='tuning_method',
                         title='Convergence Speed: Best Score Over Time')
            plots['convergence_speed'] = fig
        
        # Stability analysis
        if 'tuning_method' in df.columns and 'best_score' in df.columns:
            stability_stats = df.groupby('tuning_method')['best_score'].agg(['mean', 'std']).reset_index()
            fig = px.scatter(stability_stats, x='mean', y='std',
                           text='tuning_method',
                           title='Stability Analysis: Mean vs Standard Deviation')
            fig.update_traces(textposition="middle right")
            plots['stability_analysis'] = fig
        
    except Exception as e:
        print(f"Error creating convergence analysis plots: {e}")
    
    return plots

def create_statistical_deep_dive_plots(df):
    """Create statistical deep dive plots"""
    plots = {}
    
    try:
        # Distribution analysis
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            fig = make_subplots(rows=len(numeric_cols), cols=1,
                               subplot_titles=[f'Distribution of {col}' for col in numeric_cols])
            
            for i, col in enumerate(numeric_cols, 1):
                fig.add_trace(
                    go.Histogram(x=df[col], name=col, nbinsx=30),
                    row=i, col=1
                )
            
            fig.update_layout(height=300*len(numeric_cols), title_text="Distribution Analysis")
            plots['distribution_analysis'] = fig
        
        # Correlation matrix
        if len(numeric_cols) > 1:
            corr_matrix = df[numeric_cols].corr()
            fig = px.imshow(corr_matrix,
                           title='Correlation Matrix',
                           color_continuous_scale='RdBu_r')
            plots['correlation_matrix'] = fig
        
        # Outlier detection (box plots)
        if len(numeric_cols) > 0:
            fig = make_subplots(rows=1, cols=len(numeric_cols),
                               subplot_titles=list(numeric_cols))
            
            for i, col in enumerate(numeric_cols, 1):
                fig.add_trace(
                    go.Box(y=df[col], name=col),
                    row=1, col=i
                )
            
            fig.update_layout(title_text="Outlier Detection")
            plots['outlier_detection'] = fig
        
        # Feature importance (if applicable)
        if 'best_score' in df.columns and len(numeric_cols) > 1:
            correlations = df[numeric_cols].corr()['best_score'].abs().sort_values(ascending=True)
            fig = px.bar(y=correlations.index, x=correlations.values,
                        title='Feature Importance (Correlation with Best Score)',
                        orientation='h')
            plots['feature_importance'] = fig
        
    except Exception as e:
        print(f"Error creating statistical deep dive plots: {e}")
    
    return plots

def test_model_dataset_significance(df, alpha=0.05):
    """Test statistical significance between model-dataset combinations"""
    results = {}
    
    try:
        if not all(col in df.columns for col in ['dataset', 'model', 'best_score']):
            return {'error': 'Required columns (dataset, model, best_score) not found'}
        
        # Group by dataset and run tests for each
        dataset_results = []
        
        for dataset in df['dataset'].unique():
            dataset_df = df[df['dataset'] == dataset]
            models = dataset_df['model'].unique()
            
            if len(models) < 2:
                continue
            
            # Get model groups
            model_groups = [dataset_df[dataset_df['model'] == model]['best_score'].dropna() 
                           for model in models]
            model_names = [model for model in models]
            
            # Test normality
            normal_groups = []
            for group in model_groups:
                if len(group) >= 3:
                    _, p_val = shapiro(group)
                    normal_groups.append(p_val > alpha)
                else:
                    normal_groups.append(False)
            
            # Choose appropriate test
            if len(model_groups) > 1:
                if all(normal_groups):
                    # Use ANOVA
                    stat, p_val = f_oneway(*model_groups)
                    test_name = "ANOVA"
                else:
                    # Use Kruskal-Wallis
                    stat, p_val = kruskal(*model_groups)
                    test_name = "Kruskal-Wallis"
                
                # Calculate effect size (eta-squared for ANOVA-like measure)
                all_scores = dataset_df['best_score'].values
                group_means = [np.mean(g) for g in model_groups]
                grand_mean = np.mean(all_scores)
                
                ss_between = sum(len(g) * (np.mean(g) - grand_mean)**2 for g in model_groups)
                ss_total = sum((score - grand_mean)**2 for score in all_scores)
                eta_squared = ss_between / ss_total if ss_total > 0 else 0
                
                dataset_results.append({
                    'dataset': dataset,
                    'test': test_name,
                    'statistic': round(stat, 4),
                    'p_value': round(p_val, 6),
                    'significant': p_val < alpha,
                    'eta_squared': round(eta_squared, 4),
                    'num_models': len(models)
                })
        
        results['dataset_tests'] = dataset_results
        
        # Pairwise comparisons for significant datasets
        pairwise_results = []
        for dataset_result in dataset_results:
            if dataset_result['significant']:
                dataset = dataset_result['dataset']
                dataset_df = df[df['dataset'] == dataset]
                models = dataset_df['model'].unique()
                
                for i in range(len(models)):
                    for j in range(i + 1, len(models)):
                        model1, model2 = models[i], models[j]
                        group1 = dataset_df[dataset_df['model'] == model1]['best_score'].dropna()
                        group2 = dataset_df[dataset_df['model'] == model2]['best_score'].dropna()
                        
                        if len(group1) > 0 and len(group2) > 0:
                            # Use Mann-Whitney U test for pairwise comparison
                            stat, p_val = mannwhitneyu(group1, group2, alternative='two-sided')
                            
                            # Cohen's d effect size
                            pooled_std = np.sqrt(((len(group1) - 1) * np.var(group1, ddof=1) + 
                                                (len(group2) - 1) * np.var(group2, ddof=1)) / 
                                               (len(group1) + len(group2) - 2))
                            if pooled_std > 0:
                                cohens_d = (np.mean(group1) - np.mean(group2)) / pooled_std
                            else:
                                cohens_d = 0
                            
                            pairwise_results.append({
                                'dataset': dataset,
                                'model1': model1,
                                'model2': model2,
                                'mean1': round(np.mean(group1), 4),
                                'mean2': round(np.mean(group2), 4),
                                'p_value': round(p_val, 6),
                                'significant': p_val < alpha,
                                'cohens_d': round(cohens_d, 4),
                                'effect_size': 'Small' if abs(cohens_d) < 0.5 else 'Medium' if abs(cohens_d) < 0.8 else 'Large'
                            })
        
        results['pairwise_comparisons'] = pairwise_results
        
        # Summary statistics
        sig_count = sum(1 for r in dataset_results if r.get('significant', False))
        results['summary'] = {
            'total_datasets': len(dataset_results),
            'significant_datasets': sig_count,
            'significance_rate': round(sig_count / len(dataset_results), 3) if len(dataset_results) > 0 else 0
        }
        
    except Exception as e:
        results['error'] = str(e)
    
    return results

def compare_resource_modes(df, mode='memory_usage'):
    """Compare different resource usage modes"""
    try:
        results = {}
        
        if mode == 'memory_usage' and 'memory_bytes' in df.columns:
            if 'tuning_method' in df.columns:
                fig = px.box(df, x='tuning_method', y='memory_bytes',
                            title='Memory Usage Comparison')
                results['plot'] = fig
                
                summary = df.groupby('tuning_method')['memory_bytes'].agg(['mean', 'std', 'min', 'max']).round(2)
                results['summary'] = summary.to_dict('index')
        
        elif mode == 'time_efficiency' and 'time_sec' in df.columns:
            if 'tuning_method' in df.columns:
                fig = px.box(df, x='tuning_method', y='time_sec',
                            title='Time Efficiency Comparison')
                results['plot'] = fig
                
                summary = df.groupby('tuning_method')['time_sec'].agg(['mean', 'std', 'min', 'max']).round(2)
                results['summary'] = summary.to_dict('index')
        
        elif mode == 'combined_analysis':
            if all(col in df.columns for col in ['time_sec', 'memory_bytes', 'best_score']):
                # Create efficiency score
                df_copy = df.copy()
                df_copy['efficiency'] = df_copy['best_score'] / (df_copy['time_sec'] * df_copy['memory_bytes'] / 1e9)
                
                fig = px.scatter(df_copy, x='time_sec', y='memory_bytes',
                               color='efficiency', size='best_score',
                               title='Combined Resource Analysis')
                results['plot'] = fig
                
                if 'tuning_method' in df.columns:
                    summary = df_copy.groupby('tuning_method')['efficiency'].agg(['mean', 'std']).round(6)
                    results['summary'] = summary.to_dict('index')
        
        return results
        
    except Exception as e:
        return {'error': str(e)}
