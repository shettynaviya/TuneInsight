# Overview

This is a Streamlit-based experiment analysis dashboard designed to analyze and visualize hyperparameter tuning efficiency across different machine learning models, datasets, and optimization methods. The application allows users to track, compare, and export experimental results with comprehensive statistical analysis and interactive visualizations.

The system provides tools for:
- Tracking ML experiment metrics (accuracy scores, execution time, memory usage)
- Comparing hyperparameter tuning methods (grid search, random search, Bayesian optimization, etc.)
- Statistical analysis and hypothesis testing
- Interactive data visualization using Plotly
- Multiple export formats (Excel, CSV, PDF, HTML, JSON)

# User Preferences

Preferred communication style: Simple, everyday language.

# System Architecture

## Application Structure

**Framework**: Streamlit web application with modular Python components
- `app.py` - Main application entry point and UI orchestration
- `analysis_utils.py` - Core analysis and visualization logic
- `export_utils.py` - Data export functionality (Excel, CSV, PDF, JSON, HTML)
- `database.py` - SQLite database operations and data persistence
- `sample_data.py` - Sample data generation for testing/demonstration
- `utils/visualization.py` - Specialized visualization helper functions

**Design Pattern**: Modular separation of concerns with utility-based architecture
- UI layer (Streamlit components) separated from business logic
- Reusable utility functions for analysis, visualization, and export
- Context manager pattern for database connections

## Data Layer

**Database**: SQLite with single table schema
- Table: `experiments`
- Core columns: `id`, `tuning_method`, `best_score`, `time_sec`, `memory_bytes`, `created_at`
- Extended columns: `dataset`, `model` (added via ALTER TABLE if missing)
- Connection management via context managers for proper resource handling

**Data Model**: Experimental tracking for ML hyperparameter tuning
- Each record represents one tuning experiment
- Captures: method used, performance score, resource consumption (time/memory)
- Supports multiple datasets and model types for comparative analysis

**Dynamic Schema Evolution**: Database automatically adds missing columns (`dataset`, `model`) on initialization to support backward compatibility

## Visualization Architecture

**Library**: Plotly (plotly.express and plotly.graph_objects)
- Interactive charts: box plots, scatter plots, heatmaps
- Subplots for multi-dimensional analysis
- Export capabilities to HTML and static images

**Visualization Types**:
1. Performance comparisons (score distributions by method/model/dataset)
2. Resource efficiency analysis (time/memory usage)
3. Statistical deep dives (convergence analysis, hypothesis testing)
4. Best model/combination identification per dataset

**Key Design Decision**: Plotly chosen over matplotlib for interactivity and modern aesthetics, enabling users to explore data dynamically in web interface

## Analysis Engine

**Statistical Methods**: scipy.stats for hypothesis testing
- Parametric tests: t-test, ANOVA, Levene's test
- Non-parametric tests: Mann-Whitney U, Kruskal-Wallis
- Normality testing: Shapiro-Wilk
- Automatic test selection based on data distribution

**Analysis Functions**:
- `create_plots()` - Generate standard visualization suite
- `run_statistical_tests()` - Execute hypothesis testing
- `generate_summary_report()` - Compile comprehensive analysis
- Specialized analysis: model performance, dataset comparisons, resource efficiency, convergence trends

**Comparative Analysis**:
- `get_best_model_per_dataset()` - Identify top-performing models by dataset
- `get_best_combo_per_dataset()` - Find optimal model-tuning method pairs
- Resource mode comparisons and performance trend analysis

## Export System

**Formats Supported**:
- **Excel** (.xlsx): Multi-sheet workbooks with data, statistics, and grouped analysis
- **CSV**: Simple tabular export for maximum compatibility
- **PDF**: ReportLab-based professional reports with tables and visualizations
- **HTML**: Interactive Plotly charts embedded in web pages
- **JSON**: Structured data export for programmatic consumption

**Export Architecture**:
- Separate utility module (`export_utils.py`) for export logic
- In-memory file generation using `io.BytesIO` for efficient streaming
- Custom report generation combining multiple data views

**PDF Generation**: ReportLab library for layout control, tables, and chart embedding via temporary image files

## Sample Data Generation

**Purpose**: Demonstration and testing without real experiment data
- Generates realistic experiment results with configurable size
- Simulates 5 tuning methods × 5 datasets × 5 models
- Uses seeded randomization for reproducibility
- Applies realistic performance modifiers based on method/model/dataset combinations

# External Dependencies

## Core Libraries

**Web Framework**:
- `streamlit` - Web application framework and UI components

**Data Processing**:
- `pandas` - DataFrame operations and data manipulation
- `numpy` - Numerical computations and array operations

**Visualization**:
- `plotly` (plotly.express, plotly.graph_objects) - Interactive charting
- plotly.subplots - Multi-panel visualizations

**Statistical Analysis**:
- `scipy.stats` - Statistical testing (ANOVA, t-tests, Kruskal-Wallis, Mann-Whitney, Shapiro-Wilk, Levene's test)

**Export & Reporting**:
- `openpyxl` - Excel file generation (via pandas ExcelWriter)
- `reportlab` - PDF document generation with tables and styling

**Database**:
- `sqlite3` (Python standard library) - Local database storage
- No external database server required

**Utilities**:
- `datetime` - Timestamp handling
- `json` - JSON serialization for exports
- `io` - In-memory file operations
- `tempfile` - Temporary file handling for PDF chart embedding
- `contextlib` - Context manager utilities for database connections

## Service Architecture

**Deployment Model**: Self-contained application
- No external API dependencies
- No third-party authentication services
- All processing occurs locally within the application
- Database file stored in application directory

**Data Sources**:
- User-uploaded experiment data
- Sample data generator (built-in)
- SQLite database (persistent storage)
- Bulk import capability for CSV/Excel files