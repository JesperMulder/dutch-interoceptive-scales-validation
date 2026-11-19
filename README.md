# Dutch Interoceptive Scales Validation Study

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

Analysis code for the psychometric validation of the Dutch IAS and IATS (Mulder et al., 2026, *Journal of Health Psychology*, [doi:10.1177/13591053261449068](https://doi.org/10.1177/13591053261449068)), covering EFA, CFA, reliability, measurement invariance and regression.

## Overview

This repository contains analysis scripts and documentation for the psychometric validation of Dutch translations of widely-used interoceptive awareness questionnaires. The study employs a split-sample design with exploratory factor analysis (EFA), confirmatory factor analysis (CFA), reliability assessment, measurement invariance testing, and regression analyses.

### Scales Analyzed

- **IAS** (Interoceptive Accuracy Scale): 21 items
- **IATS** (Interoceptive Attention Scale): 21 items  
- **ICQ** (Interoceptive Confidence Questionnaire): 20 items
- **BPQ** (Body Perception Questionnaire): 46 items (26 Awareness + 20 ANS)
- **BDI** (Beck Depression Inventory): 21 items
- **TAS** (Toronto Alexithymia Scale): 20 items

## Features

### Statistical Analyses

- **Exploratory Factor Analysis (EFA)**
  - Parallel analysis for factor determination
  - Principal axis factoring with oblique (oblimin) rotation
  - Polychoric correlations for ordinal data
  - KMO and Bartlett's test for factorability
  
- **Confirmatory Factor Analysis (CFA)**
  - Structural equation modeling via SEM
  - Multiple model comparison (EFA-derived vs. single-factor)
  - Fit indices (CFI, TLI, RMSEA, SRMR)
  
- **Reliability Assessment**
  - Cronbach's alpha
  - McDonald's omega (hierarchical and total)
  - Factor-specific reliability coefficients
  
- **Measurement Invariance**
  - Configural invariance (baseline model)
  - Metric invariance (equal loadings)
  - Scalar invariance (equal loadings and intercepts)
  - Multi-group CFA across gender
  
- **Regression Analysis**
  - Univariate (simple) linear regression
  - Assumption testing (normality, heteroscedasticity)
  - Effect size estimation
  - Scatterplots (300 DPI)

### Outputs

- **Excel Workbooks**: Results tables, one workbook per analysis
- **High-Resolution Figures**: Scree plots, scatterplots, and diagnostic plots (300 DPI)
- **Formatted Tables**: Excel tables for reports and slides

## Installation

### Prerequisites

- Python 3.10 or higher
- Anaconda or Miniconda (recommended)

### Option 1: Conda Environment (Recommended)

```bash
# Clone repository
git clone https://github.com/JesperMulder/dutch-interoceptive-scales-validation.git
cd dutch-interoceptive-scales-validation

# Create conda environment
conda env create -f environment.yml

# Activate environment
conda activate dutch-interoceptive-scales-validation
```

### Option 2: pip Installation

```bash
# Clone repository
git clone https://github.com/JesperMulder/dutch-interoceptive-scales-validation.git
cd dutch-interoceptive-scales-validation

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Quick Start

1. **Prepare your data**: Place your SPSS (.sav) data file in the `data/` directory

2. **Configure paths**: Edit the configuration section in the script:
   ```python
   # Update these paths in the script
   DATA_PATH = Path('./data/your_data_file.sav')
   OUTPUT_DIR = Path('./output')
   ```

3. **Run analysis**:
   ```bash
   python questionnaire_validation_analysis.py
   ```

### Configuration

Edit these parameters in the `CONFIGURATION & PARAMETERS` section:

```python
# Random seed for reproducibility
RANDOM_SEED = 42

# Outlier detection threshold
OUTLIER_THRESHOLD_SD = 3

# Sample split for cross-validation
TEST_SIZE = 0.5  # 50/50 split for EFA/CFA

# CFA fit criteria
CFI_THRESHOLD = 0.90
RMSEA_THRESHOLD = 0.08
TLI_THRESHOLD = 0.90

# Measurement invariance criteria
DELTA_CFI_THRESHOLD = 0.01
DELTA_RMSEA_THRESHOLD = 0.015
```

### Output Structure

After running the analysis, outputs will be organized as follows:

```
output/
├── IAS_Analysis_Results.xlsx          # IAS psychometric results
├── IATS_Analysis_Results.xlsx         # IATS psychometric results
├── Univariate_Regression_Results.xlsx # Regression analyses
├── IAS_parallel_analysis.png          # IAS scree plot
├── IATS_parallel_analysis.png         # IATS scree plot
├── scatter_IAS_Totaal_*.png          # Regression scatterplots
└── scatter_IATS_Totaal_*.png         # Regression scatterplots
```

## Data Requirements

### Input Data Format

- **Format**: SPSS (.sav) file
- **Required variables**:
  - Scale items (e.g., `IAS_1` to `IAS_21`)
  - Total scores (e.g., `IAS_Totaal`, `IATS_Totaal`)
  - Demographics (`Leeftijd`, `Geslacht`)
  - Additional scales for convergent validity

### Data Structure

| Variable | Type | Description |
|----------|------|-------------|
| `IAS_1` to `IAS_21` | Numeric | IAS item responses |
| `IATS_1` to `IATS_21` | Numeric | IATS item responses |
| `IAS_Totaal` | Numeric | IAS total score |
| `IATS_Totaal` | Numeric | IATS total score |
| `Geslacht` | Categorical | Gender (for invariance testing) |
| `Leeftijd` | Numeric | Age |

See `data/README.md` for detailed data specifications.

## Methodology

### Split-Sample Design

The analysis uses a split-sample approach:

1. **Sample 1 (50%)**: Exploratory Factor Analysis (EFA)
   - Determine factor structure
   - Calculate reliability coefficients
   
2. **Sample 2 (50%)**: Confirmatory Factor Analysis (CFA)
   - Test factor structure from EFA
   - Compare with single-factor model
   - Assess measurement invariance

Fitting the CFA on a separate half avoids capitalizing on chance.

### Missing Data

- **Method**: Multiple imputation using MICE (Multivariate Imputation by Chained Equations)
- **Implementation**: 5 imputed datasets, 10 iterations
- **Variables**: All scale items plus demographic variables

### Fit Index Criteria

Based on Hu & Bentler (1999) and Cheung & Rensvold (2002):

| Index | Acceptable Fit | Good Fit |
|-------|---------------|----------|
| CFI | ≥ 0.90 | ≥ 0.95 |
| TLI | ≥ 0.90 | ≥ 0.95 |
| RMSEA | ≤ 0.08 | ≤ 0.06 |
| SRMR | ≤ 0.08 | ≤ 0.05 |

**Measurement Invariance**:
- ΔCFI ≤ 0.01
- ΔRMSEA ≤ 0.015

## Project Structure

```
dutch-interoceptive-scales-validation/
│
├── data/                              # Data files (not tracked in git)
│   └── README.md                      # Data documentation
│
├── docs/                              # Additional documentation
│   └── README.md                      # Documentation index
│
├── output/                            # Analysis outputs (not tracked)
│   └── README.md                      # Output documentation
│
├── questionnaire_validation_analysis.py # Main analysis script
├── environment.yml                    # Conda environment specification
├── requirements.txt                   # pip dependencies
├── .gitignore                         # Git ignore rules
├── LICENSE                            # MIT License
└── README.md                          # This file
```

## Dependencies

### Core Scientific Stack
- **NumPy** (≥1.24): Numerical computing
- **pandas** (≥2.0): Data manipulation
- **SciPy** (≥1.11): Statistical functions

### Factor Analysis & SEM
- **factor-analyzer** (≥0.4.0): EFA and CFA
- **semopy** (≥2.3.0): Structural equation modeling

### Machine Learning
- **scikit-learn** (≥1.3): Data preprocessing, train-test split
- **miceforest** (≥5.6): Multiple imputation

### Visualization
- **matplotlib** (≥3.7): Plotting
- **seaborn** (≥0.12): Statistical visualization

### Statistical Modeling
- **statsmodels** (≥0.14): Regression analysis

### Data I/O
- **pyreadstat** (≥1.2): Read SPSS files
- **openpyxl** (≥3.1): Excel output

See `requirements.txt` or `environment.yml` for complete dependency list.

## Reproducibility

### Ensuring Reproducible Results

1. **Fixed random seed**: All stochastic processes use `RANDOM_SEED = 42`
2. **Version control**: Dependencies specified with minimum versions
3. **Complete documentation**: All analysis decisions documented in code
4. **Session information**: Script outputs package versions used

### Replication

To exactly replicate the analysis:

```bash
# Use the exact environment
conda env create -f environment.yml
conda activate dutch-interoceptive-scales-validation

# Run with same seed
python questionnaire_validation_analysis.py
```

### Ways to Contribute

- Report bugs or issues
- Suggest enhancements
- Improve documentation
- Add tests (the analysis needs private data, so synthetic-data tests are especially welcome)
- Fix bugs

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## References

### Key Methodological References

- **Factor Analysis**
  - Horn, J. L. (1965). A rationale and test for the number of factors in factor analysis. *Psychometrika, 30*(2), 179-185.
  - Fabrigar, L. R., Wegener, D. T., MacCallum, R. C., & Strahan, E. J. (1999). Evaluating the use of exploratory factor analysis in psychological research. *Psychological Methods, 4*(3), 272-299.

- **Model Fit Criteria**
  - Hu, L. T., & Bentler, P. M. (1999). Cutoff criteria for fit indexes in covariance structure analysis. *Structural Equation Modeling, 6*(1), 1-55.
  
- **Measurement Invariance**
  - Cheung, G. W., & Rensvold, R. B. (2002). Evaluating goodness-of-fit indexes for testing measurement invariance. *Structural Equation Modeling, 9*(2), 233-255.

- **Reliability**
  - Cronbach, L. J. (1951). Coefficient alpha and the internal structure of tests. *Psychometrika, 16*(3), 297-334.
  - McDonald, R. P. (1999). *Test theory: A unified treatment*. Mahwah, NJ: Erlbaum.

- **Missing Data**
  - van Buuren, S., & Groothuis-Oudshoorn, K. (2011). mice: Multivariate imputation by chained equations in R. *Journal of Statistical Software, 45*(3), 1-67.

---

**Note**: This analysis uses real data from human participants. Obtain ethical approval and a data sharing agreement before using this code with your own participant data.
