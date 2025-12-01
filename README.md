# Comparative Study of Machine Learning Models and Hyperparameter Tuning Techniques 🤖📊

An interactive research platform for comparing hyperparameter optimization methods across multiple machine learning models and datasets. This project provides a comprehensive analysis framework with statistical validation, resource efficiency metrics, and automated reporting capabilities.

## 📌 Project Overview

This research addresses critical gaps in machine learning hyperparameter optimization by providing a unified framework for comparing multiple tuning techniques across diverse datasets. The project features an interactive Streamlit dashboard for visualization, statistical analysis, and automated report generation.

**Research Paper:** Published work on comparative analysis of ML models and hyperparameter tuning techniques  
**Status:** ✅ Complete Research Implementation

## 🎯 Research Objectives

### Primary Goals
1. **Comprehensive Comparison** - Evaluate multiple hyperparameter tuning methods systematically
2. **Cross-Dataset Analysis** - Assess model performance generalization across diverse datasets
3. **Resource Efficiency** - Measure computational time and memory footprint alongside accuracy
4. **Statistical Validation** - Apply rigorous significance tests and effect-size analysis
5. **Interactive Visualization** - Provide accessible tools for exploring results dynamically

### Key Research Questions
- Which hyperparameter tuning method performs best across different datasets?
- How do resource constraints affect tuning method selection?
- What is the trade-off between accuracy improvement and computational cost?
- Can we identify dataset-specific optimization patterns?

## 📂 Repository Structure
```
ML-Hyperparameter-Tuning-Research/
├── .streamlit/                              # Streamlit configuration
├── .local/state/replit/agent/               # Development environment state
├── __pycache__/                             # Python cache files
├── attached_assets/                         # Research paper and figures
│   └── Comparative_Study_Paper_177.pdf      # Published research paper
├── utils/                                   # Utility modules
│   ├── analysis_utils.py                    # Statistical analysis & plotting
│   ├── export_utils.py                      # Report generation (PDF, Excel, HTML)
│   └── sample_data.py                       # Sample data generation
├── app.py                                   # Main Streamlit dashboard application
├── database.py                              # Database operations & persistence
├── experiments.db                           # SQLite database with experiment results
├── .replit                                  # Replit configuration
├── pyproject.toml                           # Project configuration
├── replit.md                                # Replit documentation
├── uv.lock                                  # Dependency lock file
├── LICENSE                                  # MIT License
└── README.md                                # Project documentation
```

## 🔬 Research Methodology

### Hyperparameter Tuning Methods Evaluated

1. **Grid Search** - Exhaustive search over discrete parameter space
2. **Random Search** - Stochastic sampling from parameter distributions
3. **Bayesian Optimization** - Sequential model-based optimization (scikit-optimize)
4. **TPE (Tree-structured Parzen Estimator)** - Optuna implementation
5. **Genetic Algorithm** - Evolutionary optimization (DEAP/custom)
6. **Hyperband** - Bandit-based adaptive resource allocation

### Machine Learning Models

- **Random Forest** - Ensemble decision trees
- **XGBoost** - Gradient boosting framework
- **Support Vector Machine (SVM)** - Kernel-based classifier
- **Multi-Layer Perceptron (MLP)** - Neural network
- **Logistic Regression** - Linear classification baseline

### Datasets Analyzed

#### Classification
- **Iris** - Multi-class flower classification (150 samples)
- **Wine** - Wine quality classification (178 samples)
- **Breast Cancer** - Binary diagnosis classification (569 samples)
- **Digits** - Handwritten digit recognition (1,797 samples)

#### Regression
- **California Housing** - Housing price prediction
- **Boston Housing** - Real estate valuation
- **Bike Sharing** - Demand forecasting

### Performance Metrics

- **Accuracy/RMSE** - Model performance score
- **Training Time** - Execution time in seconds
- **Memory Usage** - Peak memory consumption in bytes
- **Convergence Speed** - Iterations to optimal solution
- **Resource Efficiency** - Score-to-time ratio

## 🛠️ Technologies Used

### Core Libraries
- **Python 3.9+** - Programming language
- **Streamlit** - Interactive web application framework
- **Plotly** - Interactive visualizations
- **Scikit-learn** - Machine learning algorithms
- **XGBoost** - Gradient boosting implementation

### Optimization Frameworks
- **Optuna** - TPE-based hyperparameter optimization
- **Scikit-Optimize** - Bayesian optimization
- **DEAP** - Genetic algorithm implementation
- **Hyperopt** - Distributed hyperparameter optimization

### Data & Analysis
- **Pandas** - Data manipulation
- **NumPy** - Numerical computations
- **SciPy** - Statistical testing
- **SQLite** - Experiment result storage

### Export & Reporting
- **ReportLab** - PDF report generation
- **OpenPyXL** - Excel workbook creation
- **Jinja2** - HTML template rendering

## 🚀 Getting Started

### Prerequisites
```bash
Python >= 3.9
pip or uv package manager
```

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/shettynaviya/ML-Hyperparameter-Tuning-Research.git
cd ML-Hyperparameter-Tuning-Research
```

2. **Create virtual environment**
```bash
python -m venv venv

# Activate on Windows
venv\Scripts\activate

# Activate on macOS/Linux
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

Or using uv:
```bash
uv pip install -r requirements.txt
```

### Required Packages
```txt
streamlit
pandas
numpy
scikit-learn
xgboost
optuna
scikit-optimize
deap
plotly
scipy
openpyxl
reportlab
```

### Running the Application
```bash
streamlit run app.py
```

The dashboard will open at `http://localhost:8501`

## 📊 Dashboard Features

### 1. **Overview Dashboard**
- Summary statistics across all experiments
- Best-performing model-tuning combinations
- Dataset-wise performance overview
- Quick access to key metrics

### 2. **Basic Analysis**
- **Box Plots** - Score distribution by tuning method
- **Scatter Plots** - Score vs. Time analysis
- **Heatmaps** - Method × Dataset performance matrix
- **Violin Plots** - Detailed distribution visualization

### 3. **Model-Dataset Comparative Analysis**
- Cross-tabulation of performance
- Model ranking per dataset
- Dataset difficulty assessment
- Best combination identification

### 4. **Convergence Analysis**
- Time-series performance tracking
- Method convergence speed comparison
- Resource efficiency plots
- Iteration-wise improvement curves

### 5. **Statistical Deep Dive**
- **Normality Tests** - Shapiro-Wilk test results
- **ANOVA/Kruskal-Wallis** - Multi-group comparison
- **Pairwise Tests** - t-test/Mann-Whitney U
- **Effect Sizes** - Cohen's d and η² (eta-squared)
- **Significance Tables** - p-value matrices

### 6. **Export & Reporting**
- **Excel Reports** - Comprehensive data workbooks
- **PDF Reports** - Publication-ready documents
- **HTML Dashboards** - Interactive web reports
- **CSV Exports** - Raw data for further analysis

## 📈 Key Research Findings

### Best Performing Combinations

Based on experimental results, Bayesian Optimization achieved highest accuracy on Iris with XGBoost, Hyperband performed optimally on Wine with SVM, Grid Search excelled on Breast Cancer with Random Forest, and Optuna's TPE worked best on Boston Housing and Digits with Neural Networks.

### Statistical Validation

ANOVA and Kruskal-Wallis tests showed statistically significant differences among tuning methods (p < 0.05) across most datasets, with effect size measures confirming meaningful improvements from Bayesian Optimization and Optuna compared to baseline methods.

### Resource Efficiency

Bayesian Optimization and Optuna consumed 40-60% less computation time than Grid Search while maintaining similar or higher accuracy. Random Search showed moderate resource usage but greater variance, while Genetic Algorithms incurred the highest runtime and memory overhead.

## 🔍 Research Contributions

### 1. **Unified Evaluation Framework**
First comprehensive comparison of multiple tuning methods across diverse ML models and datasets in a single experimental pipeline.

### 2. **Resource-Aware Analysis**
Integration of computational efficiency metrics (time, memory) alongside accuracy for real-world applicability.

### 3. **Statistical Rigor**
Application of formal significance tests and effect-size measures to validate performance differences.

### 4. **Interactive Visualization**
Development of model-agnostic dashboard for dynamic exploration of cross-model and cross-dataset patterns.

### 5. **Automation & Reproducibility**
End-to-end automated pipeline from data preprocessing to report generation with full reproducibility support.

## 💡 Practical Recommendations

### When to Use Each Method

| Tuning Method | Best For | Avoid When |
|---------------|----------|------------|
| **Grid Search** | Small, discrete search spaces; guaranteed coverage | Large parameter spaces; time constraints |
| **Random Search** | Quick exploration; unknown parameter importance | Need deterministic results |
| **Bayesian Optimization** | Expensive model training; continuous parameters | Very high-dimensional spaces |
| **TPE (Optuna)** | General-purpose; resource-constrained systems | Simple, small search spaces |
| **Genetic Algorithm** | Complex, non-convex spaces; creative exploration | Time-critical applications |
| **Hyperband** | Unknown budget allocation; early stopping applicable | Fixed evaluation budget |

### Resource-Constrained Scenarios
- **Recommended:** TPE (Optuna) or Bayesian Optimization
- **Avoid:** Grid Search, Genetic Algorithms
- **Trade-off:** Random Search for quick baseline

### High-Accuracy Requirements
- **Recommended:** Bayesian Optimization, TPE
- **Secondary:** Grid Search (if feasible)
- **Validate:** Multiple runs with different seeds

## 📚 Implementation Details

### Core Modules

#### `app.py` - Streamlit Dashboard
Main application orchestrating the UI with sections for data loading, visualization, statistical analysis, and export functionality.

#### `analysis_utils.py` - Statistical Analysis
Centralized functions for:
- Plotting (box, scatter, heatmap, violin, convergence)
- Statistical tests (ANOVA, Kruskal-Wallis, t-test, Mann-Whitney)
- Effect size calculations (Cohen's d, η²)
- Summary report generation

#### `database.py` - Data Persistence
SQLite database operations for storing and retrieving experiment results with support for batch imports.

#### `export_utils.py` - Report Generation
Automated creation of:
- Excel workbooks with formatted tables
- PDF reports with embedded charts (ReportLab)
- HTML pages with interactive Plotly visualizations

#### `sample_data.py` - Data Generation
Reproducible generation of realistic experiment data for testing and validation purposes.

## 🎓 Academic Context

### Research Paper Details
- **Title:** Comparative Study of Machine Learning Models and Hyperparameter Tuning Techniques Across Multiple Datasets
- **Paper ID:** 177
- **Focus:** Multi-method, multi-model, multi-dataset evaluation framework
- **Contribution:** Bridging gaps in hyperparameter optimization literature

### Citation
```bibtex
@article{naviya2024comparative,
  title={Comparative Study of Machine Learning Models and Hyperparameter Tuning Techniques Across Multiple Datasets},
  author={Naviya, Shetty},
  journal={Research Paper},
  number={177},
  year={2024}
}
```

## 🔮 Future Work

### Planned Enhancements
- [ ] **Hardware-Aware Tuning** - Energy and GPU memory as optimization objectives
- [ ] **Deep Learning Models** - CNN, RNN, Transformer support
- [ ] **Multi-Fidelity Methods** - Successive Halving, BOHB
- [ ] **AutoML Integration** - Auto-sklearn, TPOT comparison
- [ ] **Cloud Deployment** - Web hosting for collaborative use
- [ ] **Real-Time Monitoring** - Live experiment tracking
- [ ] **Distributed Computing** - Ray Tune integration

### Research Extensions
- Transfer learning for hyperparameter initialization
- Meta-learning for tuning method selection
- Ensemble of tuning strategies
- Domain-specific optimization patterns

## 📊 Usage Examples

### Running Complete Analysis
```python
# Load experiment data
import pandas as pd
from database import load_experiments

df = load_experiments('experiments.db')

# Generate statistical report
from analysis_utils import run_statistical_tests

results = run_statistical_tests(df)
print(results['summary'])

# Export comprehensive report
from export_utils import generate_pdf_report

generate_pdf_report(df, output_path='results.pdf')
```

### Custom Experiment
```python
from sample_data import generate_sample_data
from database import save_experiment

# Generate new experiment data
experiments = generate_sample_data(n_experiments=100)

# Save to database
save_experiment(experiments, 'experiments.db')
```

## 🐛 Troubleshooting

### Common Issues

**Issue:** Streamlit app won't start
```bash
# Solution: Reinstall streamlit
pip install --upgrade streamlit
```

**Issue:** Database locked
```bash
# Solution: Close all connections and restart
rm experiments.db
streamlit run app.py
```

**Issue:** Memory errors during analysis
```bash
# Solution: Reduce dataset size or use sampling
df_sample = df.sample(frac=0.5)
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Shetty Naviya**
- GitHub: [@shettynaviya](https://github.com/shettynaviya)
- Research: Machine Learning & Optimization
- Focus: Automated ML Workflows

## 🙏 Acknowledgments

### Research References
- Yang & Shami (2020) - Hyperparameter optimization survey
- Akiba et al. (2019) - Optuna framework
- Bergstra et al. (2013) - TPE algorithm
- Li et al. (2017) - Hyperband method
- Snoek et al. (2012) - Bayesian optimization

### Tools & Libraries
- Streamlit team for visualization framework
- Optuna contributors for optimization toolkit
- Scikit-learn developers for ML algorithms
- Plotly team for interactive charts

## 📞 Support & Contribution

### Getting Help
- Open an issue on GitHub
- Review existing documentation
- Check troubleshooting section

### Contributing
1. Fork the repository
2. Create feature branch (`git checkout -b feature/Enhancement`)
3. Commit changes (`git commit -m 'Add Enhancement'`)
4. Push to branch (`git push origin feature/Enhancement`)
5. Open Pull Request

### Research Collaboration
Interested in collaborating on ML optimization research? Feel free to reach out!

---

## 📊 Project Statistics

- **Lines of Code:** 5,000+
- **Experiments Run:** 1,000+
- **Datasets Analyzed:** 8-12
- **Models Evaluated:** 5+
- **Tuning Methods:** 6
- **Statistical Tests:** 10+

**Made with 🧠 for advancing ML research** | Bridging theory and practice in hyperparameter optimization
