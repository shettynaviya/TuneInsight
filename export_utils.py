import pandas as pd
import numpy as np
import json
import io
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from analysis_utils import create_plots, run_statistical_tests

def export_to_excel(df):
    """Export data to Excel format"""
    try:
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            # Main data
            df.to_excel(writer, sheet_name='Experiment_Data', index=False)
            
            # Summary statistics
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                summary_stats = df[numeric_cols].describe()
                summary_stats.to_excel(writer, sheet_name='Summary_Statistics')
            
            # Group analysis
            if 'tuning_method' in df.columns and 'best_score' in df.columns:
                group_stats = df.groupby('tuning_method')['best_score'].agg(['count', 'mean', 'std', 'min', 'max'])
                group_stats.to_excel(writer, sheet_name='Method_Comparison')
        
        output.seek(0)
        return output.getvalue()
        
    except Exception as e:
        print(f"Error exporting to Excel: {e}")
        return None

def export_to_csv(df):
    """Export data to CSV format"""
    try:
        return df.to_csv(index=False)
    except Exception as e:
        print(f"Error exporting to CSV: {e}")
        return None

def export_plots_to_html(df):
    """Export plots to HTML format"""
    try:
        plots = create_plots(df)
        
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Experiment Analysis Plots</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .plot-container { margin-bottom: 40px; }
                h1 { color: #1f77b4; }
                h2 { color: #17becf; }
            </style>
        </head>
        <body>
            <h1>Experiment Analysis Report</h1>
            <p>Generated on: {}</p>
        """.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        
        for plot_name, fig in plots.items():
            html_content += f"""
            <div class="plot-container">
                <h2>{plot_name.replace('_', ' ').title()}</h2>
                <div id="{plot_name}"></div>
                <script>
                    Plotly.newPlot('{plot_name}', {fig.to_json()});
                </script>
            </div>
            """
        
        html_content += """
        </body>
        </html>
        """
        
        return html_content
        
    except Exception as e:
        print(f"Error exporting plots to HTML: {e}")
        return None

def export_summary_to_json(df):
    """Export summary statistics to JSON"""
    try:
        summary = {
            'metadata': {
                'export_timestamp': datetime.now().isoformat(),
                'total_experiments': len(df),
                'columns': list(df.columns)
            },
            'numeric_summary': {},
            'categorical_summary': {},
            'missing_data': df.isnull().sum().to_dict()
        }
        
        # Numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            summary['numeric_summary'][col] = {
                'count': int(df[col].count()),
                'mean': float(df[col].mean()) if not df[col].isna().all() else None,
                'std': float(df[col].std()) if not df[col].isna().all() else None,
                'min': float(df[col].min()) if not df[col].isna().all() else None,
                'max': float(df[col].max()) if not df[col].isna().all() else None,
                'median': float(df[col].median()) if not df[col].isna().all() else None
            }
        
        # Categorical columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            value_counts = df[col].value_counts().to_dict()
            summary['categorical_summary'][col] = {
                'unique_count': int(df[col].nunique()),
                'value_counts': {str(k): int(v) for k, v in value_counts.items()},
                'most_common': str(df[col].mode().iloc[0]) if not df[col].empty else None
            }
        
        return json.dumps(summary, indent=2)
        
    except Exception as e:
        print(f"Error exporting summary to JSON: {e}")
        return None

def generate_custom_report(df, report_type='executive_summary', include_plots=True, include_stats=True):
    """Generate a custom report"""
    try:
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Custom Analysis Report - {report_type.replace('_', ' ').title()}</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }}
                .header {{ background: linear-gradient(90deg, #1f77b4, #17becf); color: white; padding: 20px; border-radius: 5px; }}
                .section {{ margin: 20px 0; padding: 15px; border-left: 4px solid #1f77b4; background-color: #f8f9fa; }}
                .metric {{ display: inline-block; margin: 10px; padding: 15px; background: white; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
                .plot-container {{ margin: 20px 0; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #1f77b4; color: white; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Experiment Analysis Report</h1>
                <p>Report Type: {report_type.replace('_', ' ').title()}</p>
                <p>Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
            </div>
        """
        
        # Executive Summary
        if report_type == 'executive_summary':
            html_content += f"""
            <div class="section">
                <h2>Executive Summary</h2>
                <div class="metric">
                    <strong>Total Experiments:</strong> {len(df)}
                </div>
            """
            
            if 'tuning_method' in df.columns:
                html_content += f"""
                <div class="metric">
                    <strong>Tuning Methods:</strong> {df['tuning_method'].nunique()}
                </div>
                """
            
            if 'best_score' in df.columns:
                best_score = df['best_score'].max()
                avg_score = df['best_score'].mean()
                html_content += f"""
                <div class="metric">
                    <strong>Best Score:</strong> {best_score:.4f}
                </div>
                <div class="metric">
                    <strong>Average Score:</strong> {avg_score:.4f}
                </div>
                """
            
            html_content += "</div>"
        
        # Include statistical analysis
        if include_stats and 'best_score' in df.columns and 'tuning_method' in df.columns:
            stats_results = run_statistical_tests(df)
            if 'overall_test' in stats_results and 'error' not in stats_results['overall_test']:
                html_content += f"""
                <div class="section">
                    <h2>Statistical Analysis</h2>
                    <p><strong>Test:</strong> {stats_results['overall_test']['test']}</p>
                    <p><strong>P-value:</strong> {stats_results['overall_test']['p_value']:.6f}</p>
                    <p><strong>Result:</strong> {stats_results['overall_test']['interpretation']}</p>
                </div>
                """
        
        # Include plots
        if include_plots:
            plots = create_plots(df)
            for plot_name, fig in plots.items():
                html_content += f"""
                <div class="section">
                    <h2>{plot_name.replace('_', ' ').title()}</h2>
                    <div id="{plot_name}" class="plot-container"></div>
                    <script>
                        Plotly.newPlot('{plot_name}', {fig.to_json()});
                    </script>
                </div>
                """
        
        html_content += """
        </body>
        </html>
        """
        
        return html_content
        
    except Exception as e:
        print(f"Error generating custom report: {e}")
        return None
