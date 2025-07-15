# \# Documentation

# 

# This folder contains additional documentation for the Dutch Interoceptive Scales Validation project.

# 

# \## Documentation Structure

# 

# ```

# docs/

# ├── README.md              # This file - documentation index

# ```

# 

# \## Available Documentation

# 

# \### Main Project Documentation

# \- \*\*\[Main README](../README.md)\*\*: Project overview, quick start guide, and key findings

# \- \*\*\[Data Documentation](../data/README.md)\*\*: Data format requirements and variable descriptions

# \- \*\*\[Output Documentation](../output/README.md)\*\*: Understanding analysis results and output files

# 

# \### Analysis Pipeline

# 

# The validation study follows a systematic approach:

# 

# 1\. \*\*Questionnaire Translation\*\*: Dutch translations of IAS and IATS

# 2\. \*\*Data Collection\*\*: Data collected through research organization Flycatcher (https://www.flycatcher.eu/nl/)

# 3\. \*\*Statistical Validation\*\*: Multi-step validation using PCA, internal consistency, CFA, and criterion validity

# 4\. \*\*Interpretation\*\*: Evaluation of psychometric properties

# 

# \## Questionnaire Information

# 

# \### Interoceptive Accuracy Scale (IAS)

# \- \*\*Original Authors\*\*: Murphy, J., Brewer, R., Plans, D., Khalsa, S. S., Catmur, C., \& Bird, G. (2020). Testing the independence of self-reported interoceptive accuracy and attention. Quarterly journal of experimental psychology (2006), 73(1), 115–133. https://doi.org/10.1177/1747021819879826

# \- \*\*Purpose\*\*: Measures perceived ability to accurately detect internal bodily signals

# \- \*\*Items\*\*: 21 items

# \- \*\*Response Scale\*\*: 1 = Strongly disagree, 5 = Strongly agree

# \- \*\*Scoring\*\*: Higher scores indicate greater perceived interoceptive accuracy

# 

# \### Interoceptive Attention Scale (IATS)

# \- \*\*Original Authors\*\*: Gabriele, E., Spooner, R., Brewer, R., \& Murphy, J. (2022). Dissociations between self-reported interoceptive accuracy and attention: Evidence from the Interoceptive Attention Scale. Biological psychology, 168, 108243. https://doi.org/10.1016/j.biopsycho.2021.108243

# \- \*\*Purpose\*\*: Measures attention paid to internal bodily signals

# \- \*\*Items\*\*: 21 items

# \- \*\*Response Scale\*\*: 1 = Strongly disagree, 5 = Strongly agree

# \- \*\*Scoring\*\*: Higher scores indicate greater attention to bodily signals

# 

# \### Validation Questionnaires

# 

# \#### Body Perception Questionnaire (BPQ)

# \- \*\*Purpose\*\*: Measures awareness of normal bodily functions

# \- \*\*Used for\*\*: Convergent validity

# 

# \#### Interoceptive Confusion Questionnaire (ICQ)

# \- \*\*Purpose\*\*: Measures confusion about bodily signals

# \- \*\*Used for\*\*: Discriminant validity

# 

# \#### Beck Depression Inventory (BDI)

# \- \*\*Purpose\*\*: Measures depressive symptoms

# \- \*\*Used for\*\*: Testing known-groups validity

# 

# \#### Toronto Alexithymia Scale (TAS)

# \- \*\*Purpose\*\*: Measures difficulty identifying and describing emotions

# \- \*\*Used for\*\*: Discriminant validity

# 

# \## Statistical Methods

# 

# \### Principal Component Analysis (PCA)

# \- \*\*Purpose\*\*: Explore underlying factor structure

# \- \*\*Method\*\*: Eigenvalue > 1 criterion with varimax rotation

# \- \*\*Output\*\*: Factor loadings and variance explained

# 

# \### Internal Consistency Analysis

# \- \*\*Method\*\*: Cronbach's alpha calculation

# \- \*\*Tested\*\*: Overall scales and factor-specific reliability

# \- \*\*Benchmark\*\*: α > 0.70 for acceptable reliability

# 

# \### Confirmatory Factor Analysis (CFA)

# \- \*\*Purpose\*\*: Test theoretical factor structures

# \- \*\*Models Tested\*\*: 1-factor, 2-factor, 3-factor, and 4-factor solutions

# \- \*\*Fit Indices\*\*: χ², CFI, TLI, RMSEA

# \- \*\*Cross-validation\*\*: Train/test split (50/50)

# 

# \### Criterion Validity

# \- \*\*Method\*\*: Univariate regression analyses

# \- \*\*Predictors\*\*: IAS and IATS total scores

# \- \*\*Outcomes\*\*: BPQ, ICQ, BDI, TAS total scores

# 

# \## Technical Implementation

# 

# \### Software Requirements

# \- \*\*Python\*\*: Version 3.8 or higher

# \- \*\*Key Packages\*\*: pandas, numpy, scikit-learn, factor-analyzer, semopy

# \- \*\*Statistical Environment\*\*: Suitable for reproducible research

# 

# \### Analysis Features

# \- \*\*Reproducibility\*\*: Fixed random seeds for replicable results

# \- \*\*Error Handling\*\*: Robust error checking throughout pipeline

# \- \*\*Visualization\*\*: Automatic generation of scree plots

# \- \*\*Documentation\*\*: Comprehensive code documentation

# 

# \## Interpretation Guidelines

# 

# \### Factor Structure

# \- \*\*Kaiser Criterion\*\*: Factors with eigenvalues > 1

# \- \*\*Scree Plot\*\*: Visual inspection for optimal number of factors

# \- \*\*Rotation\*\*: Varimax rotation for interpretable factors

# 

# \### Model Fit Evaluation

# \- \*\*Good Fit\*\*: CFI/TLI > 0.95, RMSEA < 0.06

# \- \*\*Acceptable Fit\*\*: CFI/TLI > 0.90, RMSEA < 0.08

# \- \*\*Poor Fit\*\*: Below acceptable thresholds

# 

# \### Effect Size Interpretation

# \- \*\*Small\*\*: r = 0.10

# \- \*\*Medium\*\*: r = 0.30  

# \- \*\*Large\*\*: r = 0.50

# 

# \## Citation Guidelines

# 

# \### Citing This Work

# ```

# Mulder, J., Elferink-Gemser, M. T., de Vries, J. D., \& Kiefte-de Jong, J. C. (2025). Reliability and Validity of the Dutch Interoceptive Accuracy Scale and Interoceptive Attention Scale. medRxiv. https://doi.org/https://doi.org/10.1101/2025.05.06.25326009



# ```

