# \# Dutch Interoceptive Scales Validation

# 

# Validation of Dutch translations of the Interoceptive Accuracy Scale (IAS) and Interoceptive Attention Scale (IATS).

# 

# \## Overview

# 

# This repository contains the analysis code and documentation for validating Dutch translations of two important interoceptive awareness questionnaires:

# 

# \- \*\*Interoceptive Accuracy Scale (IAS)\*\*: Measures perceived ability to accurately detect internal bodily signals

# \- \*\*Interoceptive Attention Scale (IATS)\*\*: Measures attention paid to internal bodily signals

# 

# \*\*Citation\*\*:

# Mulder, J., Elferink-Gemser, M. T., de Vries, J. D., & Kiefte-de Jong, J. C. (2025). Reliability and Validity of the Dutch Interoceptive Accuracy Scale and Interoceptive Attention Scale. medRxiv. https://doi.org/https://doi.org/10.1101/2025.05.06.25326009 

# 

# \*\*Analyses Performed\*\*:

# \- Principal Component Analysis (PCA) with varimax rotation

# \- Internal consistency analysis (Cronbach's α)

# \- Confirmatory Factor Analysis (CFA) for 1-4 factor solutions

# \- Regression analyses with related questionnaires

# 

# \## Repository Structure

# 

# ```

# dutch-interoceptive-scales-validation/

# ├── README.md                 # This file

# ├── LICENSE                   # MIT License

# ├── requirements.txt          # Python dependencies

# ├── questionnaire_validation_analysis.py              # Main analysis script

# ├── data/                    # Data files (not included in public repo)

# │   └── README.md           # Data format documentation

# ├── output/                  # Analysis results and plots

# │   └── README.md           # Output documentation

# └── docs/                    # Additional documentation

# &nbsp;   └── README.md           # Documentation index

# ```

# 

# \## Quick Start

# 

# \### Prerequisites

# 

# \- Python 3.8 or higher

# \- Required packages (see `requirements.txt`)

# 

# \### Installation

# 

# 1\. Clone this repository:

# ```bash

# git clone https://github.com/\[your-username]/dutch-interoceptive-scales-validation.git

# cd dutch-interoceptive-scales-validation

# ```

# 

# 2\. Install required packages:

# ```bash

# pip install -r requirements.txt

# ```

# 

# 3\. Place your data file in the `data/` folder and update the `DATA\_PATH` in `analysis.py`

# 

# \### Running the Analysis

# 

# ```bash

# python analysis.py

# ```

# 

# \## Key Variables

# 

# The analysis focuses on these essential variables:

# \- \*\*Participant ID\*\*: `Persoon\_ID`

# \- \*\*Demographics\*\*: `Leeftijd` (Age), `Geslacht` (Gender)

# \- \*\*Target Scales\*\*: `IAS` (21 items), `IATS` (21 items)

# \- \*\*Validation Measures\*\*: `BPQ`, `ICQ`, `BDI`, `TAS` total scores

# 

# \## Analysis Pipeline

# 

# 1\. \*\*Data Preparation\*\*

# &nbsp;  - Load and clean data

# &nbsp;  - Remove outliers (Z-score > 3)

# &nbsp;  - Calculate descriptive statistics

# 

# 2\. \*\*Principal Component Analysis\*\*

# &nbsp;  - Test factorability (KMO, Bartlett's test)

# &nbsp;  - Extract factors using Kaiser criterion

# &nbsp;  - Apply varimax rotation

# &nbsp;  - Generate scree plots

# 

# 3\. \*\*Internal Consistency\*\*

# &nbsp;  - Calculate Cronbach's α for overall scales

# &nbsp;  - Test α for different factor structures (1-4 factors)

# 

# 4\. \*\*Confirmatory Factor Analysis\*\*

# &nbsp;  - Test 1-4 factor models using train/test split

# &nbsp;  - Calculate fit indices (χ², CFI, TLI, RMSEA)

# &nbsp;  - Evaluate model fit quality

# 

# 5\. \*\*Regression Analysis\*\*

# &nbsp;  - Univariate regressions between IAS/IATS and validation measures

# &nbsp;  - Report standardized coefficients and effect sizes

# 

# \## Output

# 

# The script generates:

# \- Console output with detailed results

# \- Scree plots saved to `output/` folder

# \- Comprehensive statistical summaries

# 

# \## Data Format

# 

# See `data/README.md` for detailed information about the expected data format and variable naming conventions.

# 

# \## Contributing

# 

# This repository contains the analysis code for a published study. For questions or suggestions, please open an issue.

# 

# \## License

# 

# This project is licensed under the MIT License - see the \[LICENSE](LICENSE) file for details.

# 

# \## Citation

# 

# If you use this code in your research, please cite:

# 

# Mulder, J., Elferink-Gemser, M. T., de Vries, J. D., & Kiefte-de Jong, J. C. (2025). Reliability and Validity of the Dutch Interoceptive Accuracy Scale and Interoceptive Attention Scale. medRxiv. https://doi.org/https://doi.org/10.1101/2025.05.06.25326009

