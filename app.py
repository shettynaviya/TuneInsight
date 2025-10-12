import streamlit as st
import pandas as pd
import numpy as np
from analysis_utils import (
    create_plots, 
    run_statistical_tests, 
    generate_summary_report,
    create_model_performance_plots,
    create_dataset_analysis_plots,
    create_resource_efficiency_plots,
    create_convergence_analysis_plots,
    create_statistical_deep_dive_plots,
    compare_resource_modes
)
from export_utils import (
    export_to_excel,
    export_to_csv,
    export_plots_to_html,
    export_summary_to_json,
    generate_custom_report
)
from sample_data import create_sample_data
from database import initialize_database, get_all_experiments, add_experiment
from utils.visualization import get_best_model_per_dataset, get_best_combo_per_dataset
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json

# Page configuration
st.set_page_config(
    page_title="Advanced Experiment Analysis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .section-header {
        background: linear-gradient(90deg, #1f77b4, #17becf);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def load_data():
    """Load data from uploaded file, database, or create sample data"""
    # Initialize database
    initialize_database()
    
    data_source = st.radio(
        "Select data source:",
        ["Upload CSV", "Use Database", "Sample Data"],
        horizontal=True
    )
    
    if data_source == "Upload CSV":
        uploaded_file = st.file_uploader(
            "Upload your experiment results CSV file", 
            type=['csv'],
            help="Upload a CSV file containing your experiment results with columns like tuning_method, best_score, time_sec, etc."
        )
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.success(f"✅ Data loaded successfully! Found {len(df)} experiments.")
                return df, False
            except Exception as e:
                st.error(f"Error loading data: {str(e)}")
                return None, False
        else:
            st.warning("Please upload a CSV file to proceed.")
            return None, False
    
    elif data_source == "Use Database":
        try:
            df = get_all_experiments()
            if len(df) > 0:
                st.success(f"✅ Database data loaded successfully! Found {len(df)} experiments.")
                return df, False
            else:
                st.warning("No experiments found in database. Please add some data or use sample data.")
                return None, False
        except Exception as e:
            st.error(f"Error loading database data: {str(e)}")
            return None, False
    
    else:  # Sample Data
        df = create_sample_data()
        st.info(f"📊 Using sample data with {len(df)} experiments for demonstration.")
        return df, True

def display_data_overview(df):
    """Display overview of the loaded data"""
    st.markdown('<div class="section-header"><h2>📋 Data Overview</h2></div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Total Experiments", len(df))
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        methods = df['tuning_method'].nunique() if 'tuning_method' in df.columns else 0
        st.metric("Tuning Methods", methods)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        datasets = df['dataset'].nunique() if 'dataset' in df.columns else 0
        st.metric("Datasets", datasets)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        models = df['model'].nunique() if 'model' in df.columns else 0
        st.metric("Models", models)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Overall Best Performing Model Summary Card
    if 'model' in df.columns and 'best_score' in df.columns:
        summary = df.groupby('model')['best_score'].mean().sort_values(ascending=False)
        if not summary.empty:
            col5, col6, col7, col8 = st.columns(4)
            with col5:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Overall Best Performing Model", summary.index[0], f"{summary.iloc[0]:.3f}")
                st.markdown('</div>', unsafe_allow_html=True)
    
    # Data preview
    with st.expander("🔍 Data Preview", expanded=False):
        st.dataframe(df.head(10), use_container_width=True)
    
    # Column information
    with st.expander("📊 Column Information", expanded=False):
        col_info = pd.DataFrame({
            'Column': df.columns,
            'Data Type': df.dtypes.astype(str),
            'Non-Null Count': df.count(),
            'Null Count': df.isnull().sum()
        })
        st.dataframe(col_info, use_container_width=True)

def display_basic_analysis(df):
    """Display basic analysis plots - EXISTING SECTION 1"""
    st.markdown('<div class="section-header"><h2>📈 Basic Performance Analysis</h2></div>', unsafe_allow_html=True)
    
    plots = create_plots(df)
    
    if plots:
        # Create tabs for different plot types
        plot_tabs = st.tabs(["Score Analysis", "Time Analysis", "Memory Analysis", "Efficiency", "Heatmap"])
        
        with plot_tabs[0]:
            if 'score_vs_method' in plots:
                st.plotly_chart(plots['score_vs_method'], use_container_width=True)
            else:
                st.warning("Score vs Method plot not available. Check if 'tuning_method' and 'best_score' columns exist.")
        
        with plot_tabs[1]:
            if 'time_vs_method' in plots:
                st.plotly_chart(plots['time_vs_method'], use_container_width=True)
            else:
                st.warning("Time analysis plot not available. Check if 'time_sec' column exists.")
        
        with plot_tabs[2]:
            if 'memory_vs_method' in plots:
                st.plotly_chart(plots['memory_vs_method'], use_container_width=True)
            else:
                st.warning("Memory analysis plot not available. Check if 'memory_bytes' column exists.")
        
        with plot_tabs[3]:
            if 'score_vs_time' in plots:
                st.plotly_chart(plots['score_vs_time'], use_container_width=True)
            else:
                st.warning("Efficiency plot not available.")
        
        with plot_tabs[4]:
            if 'performance_heatmap' in plots:
                st.plotly_chart(plots['performance_heatmap'], use_container_width=True)
            else:
                st.warning("Performance heatmap not available.")
    else:
        st.error("No plots could be generated. Please check your data format.")

def display_statistical_analysis(df):
    """Display statistical analysis - EXISTING SECTION 2"""
    st.markdown('<div class="section-header"><h2>🔬 Statistical Analysis</h2></div>', unsafe_allow_html=True)
    
    # Metric selection
    available_metrics = [col for col in df.columns if df[col].dtype in ['int64', 'float64']]
    
    if not available_metrics:
        st.error("No numeric columns found for statistical analysis.")
        return
    
    selected_metric = st.selectbox(
        "Select metric for analysis:",
        available_metrics,
        index=available_metrics.index('best_score') if 'best_score' in available_metrics else 0
    )
    
    alpha = st.slider("Significance level (α)", 0.01, 0.10, 0.05, 0.01)
    
    # Run statistical tests
    statistical_results = run_statistical_tests(df, metric=selected_metric, alpha=alpha)
    
    if 'error' in statistical_results:
        st.error(f"Statistical analysis failed: {statistical_results['error']}")
        return
    
    # Display results in tabs
    stats_tabs = st.tabs(["Overall Test", "Pairwise Comparisons", "Descriptive Stats", "Effect Sizes", "Normality Tests"])
    
    with stats_tabs[0]:
        if 'overall_test' in statistical_results:
            overall = statistical_results['overall_test']
            if 'error' not in overall:
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Test Statistic", f"{overall['statistic']:.4f}")
                    st.metric("P-value", f"{overall['p_value']:.6f}")
                with col2:
                    significance = "Significant" if overall['significant'] else "Not Significant"
                    st.metric("Result", significance)
                    st.info(overall['interpretation'])
            else:
                st.error(overall['error'])
    
    with stats_tabs[1]:
        if 'pairwise_tests' in statistical_results:
            pairwise_df = pd.DataFrame(statistical_results['pairwise_tests'])
            if not pairwise_df.empty:
                st.dataframe(pairwise_df, use_container_width=True)
            else:
                st.warning("No pairwise test results available.")
    
    with stats_tabs[2]:
        if 'descriptive_stats' in statistical_results:
            desc_df = pd.DataFrame(statistical_results['descriptive_stats']).T
            st.dataframe(desc_df, use_container_width=True)
    
    with stats_tabs[3]:
        if 'effect_sizes' in statistical_results:
            effect_df = pd.DataFrame(statistical_results['effect_sizes'])
            if not effect_df.empty:
                st.dataframe(effect_df, use_container_width=True)
            else:
                st.warning("No effect size calculations available.")
    
    with stats_tabs[4]:
        if 'normality_tests' in statistical_results:
            normality_results = statistical_results['normality_tests']
            for method, result in normality_results.items():
                if 'error' not in result:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.write(f"**{method}**")
                    with col2:
                        st.write(f"P-value: {result['p_value']:.6f}")
                    with col3:
                        normal_status = "✅ Normal" if result['normal'] else "❌ Not Normal"
                        st.write(normal_status)
                else:
                    st.error(f"{method}: {result['error']}")

def display_model_performance_analysis(df):
    """Display model performance comparison - RENAMED SECTION"""
    st.markdown('<div class="section-header"><h2>🤖 Model–Dataset Comparative Analysis</h2></div>', unsafe_allow_html=True)
    
    if 'model' not in df.columns:
        st.warning("Model performance analysis requires a 'model' column in your data.")
        return
    
    plots = create_model_performance_plots(df)
    
    if plots:
        model_tabs = st.tabs(["Performance Overview", "By Dataset", "Efficiency Analysis"])
        
        with model_tabs[0]:
            if 'model_performance' in plots:
                st.plotly_chart(plots['model_performance'], use_container_width=True)
                
                # Model ranking table
                if 'best_score' in df.columns:
                    model_stats = df.groupby('model')['best_score'].agg(['mean', 'std', 'count']).round(4)
                    model_stats = model_stats.sort_values('mean', ascending=False)
                    st.subheader("📊 Model Performance Ranking")
                    st.dataframe(model_stats, use_container_width=True)
        
        with model_tabs[1]:
            if 'model_by_dataset' in plots:
                st.plotly_chart(plots['model_by_dataset'], use_container_width=True)
        
        with model_tabs[2]:
            if 'model_efficiency' in plots:
                st.plotly_chart(plots['model_efficiency'], use_container_width=True)
                
                # Efficiency metrics
                if all(col in df.columns for col in ['time_sec', 'best_score']):
                    df_efficiency = df.copy()
                    df_efficiency['score_per_second'] = df_efficiency['best_score'] / df_efficiency['time_sec']
                    efficiency_stats = df_efficiency.groupby('model')['score_per_second'].agg(['mean', 'std']).round(6)
                    efficiency_stats = efficiency_stats.sort_values('mean', ascending=False)
                    st.subheader("⚡ Model Efficiency Ranking (Score/Time)")
                    st.dataframe(efficiency_stats, use_container_width=True)

def display_dataset_analysis(df):
    """Display dataset-specific analysis - EXISTING SECTION"""
    st.markdown('<div class="section-header"><h2>🗂️ Dataset Analysis</h2></div>', unsafe_allow_html=True)
    
    if 'dataset' not in df.columns:
        st.warning("Dataset analysis requires a 'dataset' column in your data.")
        return
    
    plots = create_dataset_analysis_plots(df)
    
    if plots:
        dataset_tabs = st.tabs(["Difficulty Analysis", "Score Distributions", "Method Performance"])
        
        with dataset_tabs[0]:
            if 'dataset_difficulty' in plots:
                st.plotly_chart(plots['dataset_difficulty'], use_container_width=True)
                st.info("💡 **Interpretation**: Datasets in the upper-right quadrant are more challenging (high variability), while those in the lower-left are more predictable.")
        
        with dataset_tabs[1]:
            if 'dataset_distribution' in plots:
                st.plotly_chart(plots['dataset_distribution'], use_container_width=True)
        
        with dataset_tabs[2]:
            if 'dataset_method_heatmap' in plots:
                st.plotly_chart(plots['dataset_method_heatmap'], use_container_width=True)

def display_tuning_method_analysis(df):
    """Display tuning method analysis - RENAMED SECTION"""
    st.markdown('<div class="section-header"><h2>🔧 Model vs Tuning Method Analysis</h2></div>', unsafe_allow_html=True)
    
    plots = create_resource_efficiency_plots(df)
    
    if plots:
        resource_tabs = st.tabs(["Resource Usage", "Efficiency Metrics", "Scalability Analysis"])
        
        with resource_tabs[0]:
            if 'resource_usage' in plots:
                st.plotly_chart(plots['resource_usage'], use_container_width=True)
        
        with resource_tabs[1]:
            if 'efficiency_scatter' in plots:
                st.plotly_chart(plots['efficiency_scatter'], use_container_width=True)
        
        with resource_tabs[2]:
            if 'scalability_analysis' in plots:
                st.plotly_chart(plots['scalability_analysis'], use_container_width=True)

def display_convergence_analysis(df):
    """Display convergence analysis - EXISTING SECTION"""
    st.markdown('<div class="section-header"><h2>📈 Convergence Analysis</h2></div>', unsafe_allow_html=True)
    
    plots = create_convergence_analysis_plots(df)
    
    if plots:
        convergence_tabs = st.tabs(["Learning Curves", "Convergence Speed", "Stability Analysis"])
        
        with convergence_tabs[0]:
            if 'learning_curves' in plots:
                st.plotly_chart(plots['learning_curves'], use_container_width=True)
        
        with convergence_tabs[1]:
            if 'convergence_speed' in plots:
                st.plotly_chart(plots['convergence_speed'], use_container_width=True)
        
        with convergence_tabs[2]:
            if 'stability_analysis' in plots:
                st.plotly_chart(plots['stability_analysis'], use_container_width=True)

def display_statistical_deep_dive(df):
    """Display statistical deep dive - EXISTING SECTION"""
    st.markdown('<div class="section-header"><h2>🔬 Statistical Deep Dive</h2></div>', unsafe_allow_html=True)
    
    plots = create_statistical_deep_dive_plots(df)
    
    if plots:
        deep_dive_tabs = st.tabs(["Distribution Analysis", "Correlation Matrix", "Outlier Detection", "Feature Importance"])
        
        with deep_dive_tabs[0]:
            if 'distribution_analysis' in plots:
                st.plotly_chart(plots['distribution_analysis'], use_container_width=True)
        
        with deep_dive_tabs[1]:
            if 'correlation_matrix' in plots:
                st.plotly_chart(plots['correlation_matrix'], use_container_width=True)
        
        with deep_dive_tabs[2]:
            if 'outlier_detection' in plots:
                st.plotly_chart(plots['outlier_detection'], use_container_width=True)
        
        with deep_dive_tabs[3]:
            if 'feature_importance' in plots:
                st.plotly_chart(plots['feature_importance'], use_container_width=True)

def display_best_model_tuning_analysis(df):
    """Display best model and tuning method per dataset - NEW SECTION"""
    st.markdown('<div class="section-header"><h2>🏆 Best Model & Tuning Method per Dataset</h2></div>', unsafe_allow_html=True)
    
    # Check if required columns exist
    required_cols = ['dataset', 'model', 'tuning_method', 'best_score']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        st.warning(f"This analysis requires the following columns: {', '.join(missing_cols)}")
        return
    
    try:
        # Get analysis data
        best_models = get_best_model_per_dataset(df)
        best_combos = get_best_combo_per_dataset(df)
        
        # Heatmap: Dataset vs Model
        fig1 = px.density_heatmap(
            best_models, x='dataset', y='model', z='best_score',
            color_continuous_scale='Viridis',
            title="Model Performance Across Datasets",
            labels={'best_score': 'Average Score'}
        )
        fig1.update_layout(height=500)
        st.plotly_chart(fig1, use_container_width=True)
        
        # Bar Chart: Best Combination per Dataset
        fig2 = px.bar(
            best_combos, x='dataset', y='best_score', color='model',
            text='tuning_method', title="Best Model–Tuning Combination per Dataset",
            labels={'best_score': 'Best Score', 'dataset': 'Dataset'}
        )
        fig2.update_traces(textposition='outside')
        fig2.update_layout(height=500)
        st.plotly_chart(fig2, use_container_width=True)
        
        # Summary Table
        st.subheader("📋 Best Combinations Summary")
        display_df = best_combos[['dataset', 'model', 'tuning_method', 'best_score']].copy()
        display_df['best_score'] = display_df['best_score'].round(4)
        st.dataframe(display_df, use_container_width=True)
        
        # Highlight top performer
        if not best_combos.empty:
            top = best_combos.loc[best_combos['best_score'].idxmax()]
            st.success(
                f"🎯 **Overall Best**: {top['model']} tuned with {top['tuning_method']} "
                f"on {top['dataset']} dataset → Score: {top['best_score']:.4f}"
            )
        
    except Exception as e:
        st.error(f"Error in best model analysis: {str(e)}")

def display_resource_comparison(df):
    """Display resource comparison analysis - EXISTING SECTION"""
    st.markdown('<div class="section-header"><h2>💻 Resource Comparison Analysis</h2></div>', unsafe_allow_html=True)
    
    resource_mode = st.selectbox(
        "Select resource comparison mode:",
        ["Memory Usage", "Time Efficiency", "Combined Analysis"]
    )
    
    comparison_results = compare_resource_modes(df, mode=resource_mode.lower().replace(" ", "_"))
    
    if comparison_results:
        if 'plot' in comparison_results:
            st.plotly_chart(comparison_results['plot'], use_container_width=True)
        
        if 'summary' in comparison_results:
            st.subheader("📊 Resource Analysis Summary")
            summary_df = pd.DataFrame(comparison_results['summary'])
            st.dataframe(summary_df, use_container_width=True)

def display_export_options(df):
    """Display export options - EXISTING SECTION"""
    st.markdown('<div class="section-header"><h2>📥 Export & Reports</h2></div>', unsafe_allow_html=True)
    
    export_tabs = st.tabs(["Quick Export", "Custom Report", "Summary"])
    
    with export_tabs[0]:
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📊 Export to Excel"):
                excel_data = export_to_excel(df)
                if excel_data:
                    st.download_button(
                        label="Download Excel File",
                        data=excel_data,
                        file_name=f"experiment_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
            
            if st.button("📄 Export to CSV"):
                csv_data = export_to_csv(df)
                if csv_data:
                    st.download_button(
                        label="Download CSV File",
                        data=csv_data,
                        file_name=f"experiment_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
        
        with col2:
            if st.button("📈 Export Plots to HTML"):
                html_data = export_plots_to_html(df)
                if html_data:
                    st.download_button(
                        label="Download HTML Report",
                        data=html_data,
                        file_name=f"plots_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                        mime="text/html"
                    )
            
            if st.button("🔄 Export Summary to JSON"):
                json_data = export_summary_to_json(df)
                if json_data:
                    st.download_button(
                        label="Download JSON Summary",
                        data=json_data,
                        file_name=f"analysis_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
    
    with export_tabs[1]:
        report_type = st.selectbox("Select report type:", ["Executive Summary", "Technical Report", "Comparison Report"])
        include_plots = st.checkbox("Include visualizations", value=True)
        include_stats = st.checkbox("Include statistical analysis", value=True)
        
        if st.button("Generate Custom Report"):
            custom_report = generate_custom_report(df, report_type.lower().replace(" ", "_"), include_plots, include_stats)
            if custom_report:
                st.download_button(
                    label="Download Custom Report",
                    data=custom_report,
                    file_name=f"custom_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                    mime="text/html"
                )
    
    with export_tabs[2]:
        summary_report = generate_summary_report(df)
        if summary_report and 'error' not in summary_report:
            st.json(summary_report)
        else:
            st.error("Could not generate summary report.")

def main():
    """Main application function"""
    st.title("🔬 Advanced Experiment Analysis Dashboard")
    st.markdown("Comprehensive analysis of hyperparameter tuning experiments with statistical insights and performance comparisons.")
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")
    
    # Load data
    df, is_sample = load_data()
    
    if df is not None:
        # Display data overview
        display_data_overview(df)
        
        # Create main analysis sections
        st.markdown("---")
        
        # Section 1: Basic Analysis
        display_basic_analysis(df)
        
        st.markdown("---")
        
        # Section 2: Statistical Analysis
        display_statistical_analysis(df)
        
        st.markdown("---")
        
        # Section 3: Model Performance Analysis (renamed)
        display_model_performance_analysis(df)
        
        st.markdown("---")
        
        # Section 4: Dataset Analysis
        display_dataset_analysis(df)
        
        st.markdown("---")
        
        # Section 5: Tuning Method Analysis (renamed)
        display_tuning_method_analysis(df)
        
        st.markdown("---")
        
        # Section 6: Best Model & Tuning Method Analysis (NEW)
        display_best_model_tuning_analysis(df)
        
        st.markdown("---")
        
        # Section 7: Convergence Analysis
        display_convergence_analysis(df)
        
        st.markdown("---")
        
        # Section 8: Statistical Deep Dive
        display_statistical_deep_dive(df)
        
        st.markdown("---")
        
        # Section 9: Resource Comparison
        display_resource_comparison(df)
        
        st.markdown("---")
        
        # Section 10: Export Options
        display_export_options(df)
        
        # Footer
        st.markdown("---")
        st.markdown("*Dashboard generated with ❤️ using Streamlit*")
    
    else:
        st.info("👆 Please load your data using one of the options above to begin analysis.")

if __name__ == "__main__":
    main()
