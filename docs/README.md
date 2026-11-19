# Documentation

This directory contains additional documentation, supplementary materials, and methodological details for the Dutch Interoceptive Scales Validation study.

## Contents

### Methodological Documentation

- **Statistical Methods**: Detailed description of all statistical procedures
- **Decision Rules**: Rationale for analytical decisions
- **Validation Procedures**: Steps taken to ensure analysis quality

### Interpretation Guides

- **Factor Analysis Results**: How to interpret EFA and CFA outputs
- **Fit Indices**: Guidelines for evaluating model fit
- **Reliability Coefficients**: Interpreting Cronbach's alpha and McDonald's omega
- **Measurement Invariance**: Understanding invariance testing results

## Key Methodological Decisions

### Factor Analysis

#### Why Parallel Analysis?

**Parallel Analysis** (Horn, 1965) is considered the gold standard for determining the number of factors because:

1. **Reduces over-extraction**: Compared to Kaiser's criterion (eigenvalues >1)
2. **Empirical comparison**: Compares real data to random data
3. **Recommended practice**: Widely endorsed in psychometric literature

**Implementation**:
- 1,000 random datasets generated
- 95th percentile of random eigenvalues used as threshold
- Factors retained when real eigenvalues > random eigenvalues

#### Why Oblique Rotation?

**Oblimin rotation** (oblique) is preferred over orthogonal rotation because:

1. **Realistic**: Psychological constructs typically correlate
2. **Flexible**: Allows factors to correlate if data support it
3. **Informative**: Provides both pattern and structure matrices
4. **Conservative**: If factors are truly uncorrelated, oblique ≈ orthogonal

**Alternative**: Varimax rotation assumes orthogonal (uncorrelated) factors, which is often unrealistic for psychological constructs.

#### Why Principal Axis Factoring?

**Principal Axis Factoring** (common factor extraction) is preferred over PCA because:

1. **True factor analysis**: Extracts common variance only
2. **Error variance**: Separates error variance from common variance
3. **Theoretical alignment**: Aligns with common factor model
4. **Publication standards**: Widely accepted in psychometric research

**Note**: PCA is a data reduction technique, not true factor analysis.

### Confirmatory Factor Analysis

#### Fit Index Cutoffs

Based on **Hu & Bentler (1999)**, we use:

| Index | Acceptable | Good | Rationale |
|-------|-----------|------|-----------|
| CFI | ≥0.90 | ≥0.95 | Incremental fit relative to baseline |
| TLI | ≥0.90 | ≥0.95 | Penalizes model complexity |
| RMSEA | ≤0.08 | ≤0.06 | Parsimony-adjusted badness-of-fit |
| SRMR | ≤0.08 | ≤0.05 | Standardized residual correlations |

**Important**: 
- No single index defines "good fit"
- Consider multiple indices together
- Theoretical meaningfulness trumps statistical fit
- Sample size affects some indices (e.g., χ²)

#### Model Comparison

We test multiple models:

1. **EFA-derived structure**: Empirically determined from Sample 1
2. **Single-factor model**: Theoretical baseline comparison
3. **Competing models**: If theoretically justified

**Decision rule**: Prefer simpler model unless more complex model shows substantially better fit (ΔCFI >0.01, ΔRMSEA >0.015)

### Measurement Invariance

#### Why Test Invariance?

**Measurement invariance** ensures scales measure the same construct across groups:

1. **Fairness**: Prevents group bias in measurement
2. **Comparability**: Allows meaningful group comparisons
3. **Validity**: Supports construct validity across populations

#### Invariance Levels

**Configural invariance** (baseline):
- Same factor structure across groups
- Parameters estimated freely per group
- Establishes baseline for comparisons

**Metric invariance** (weak):
- Equal factor loadings across groups
- Tests if items relate to factors equivalently
- Required for meaningful comparisons of relationships

**Scalar invariance** (strong):
- Equal factor loadings AND intercepts
- Tests if items have same meaning across groups
- Required for meaningful mean comparisons

#### Decision Criteria

Based on **Cheung & Rensvold (2002)**:

- **ΔCFI ≤ 0.01**: Invariance supported
- **ΔRMSEA ≤ 0.015**: Invariance supported

**Both criteria** should be met to support invariance at each level.

### Reliability

#### Cronbach's Alpha

**Strengths**:
- Widely used and understood
- Assesses internal consistency
- Easy to interpret

**Limitations**:
- Assumes tau-equivalent model (equal loadings)
- Underestimates reliability when loadings vary
- Affected by number of items

**Interpretation**:
- α ≥ 0.90: Excellent
- α ≥ 0.80: Good
- α ≥ 0.70: Acceptable
- α < 0.70: Questionable

#### McDonald's Omega

**Advantages** over alpha:
- Based on factor analysis
- Accounts for different loadings
- More accurate reliability estimate
- Preferred in modern psychometrics

**Types**:
- **ω_total**: Overall reliability
- **ω_hierarchical**: Reliability due to general factor

**Interpretation**: Same thresholds as alpha

**When to use**:
- Always report omega alongside alpha
- Prefer omega when loadings vary substantially
- Required for multidimensional scales

### Missing Data

#### Multiple Imputation (MICE)

**Why imputation?**
- Preserves sample size
- Reduces bias vs. listwise deletion
- Properly accounts for uncertainty

**Parameters**:
- **Method**: MICE (Multivariate Imputation by Chained Equations)
- **Datasets**: 5 imputed datasets
- **Iterations**: 10 per dataset
- **Predictors**: All scale items + demographics

**Assumptions**:
- Data Missing at Random (MAR)
- Predictor variables informationally related to missing data

**Alternatives considered**:
- Listwise deletion: Loses power, introduces bias
- Mean imputation: Underestimates variance
- Single imputation: Doesn't account for uncertainty

### Split-Sample Design

#### Rationale

**Cross-validation** prevents capitalization on chance:

1. **Sample 1 (EFA)**: Exploratory, data-driven factor extraction
2. **Sample 2 (CFA)**: Confirmatory, theory-testing

**Advantages**:
- Stronger evidence for factor structure
- Reduces overfitting
- Demonstrates replicability
- Publication standard for scale validation

**Trade-off**: Requires larger sample, but worth it for rigor

**Alternative**: If sample too small, consider:
- Single-sample CFA only
- Cross-validation with independent dataset
- Bootstrap validation

### Regression Analysis

#### Univariate vs. Multiple Regression

We use **univariate (simple) regression**:

**Rationale**:
- Isolates unique effect of each predictor
- Avoids multicollinearity issues
- Easier interpretation
- Appropriate for exploratory convergent validity

**When multiple regression preferred**:
- Controlling for confounds
- Testing mediation/moderation
- Estimating unique variance
- Theory specifies multiple predictors

#### Assumption Testing

All regression models checked for:

1. **Normality of residuals**: Shapiro-Wilk test
2. **Heteroscedasticity**: Breusch-Pagan test
3. **Linearity**: Visual inspection of scatterplots
4. **Independence**: Assumed from study design

**Violations**: Reported and may require transformations or robust methods

## Interpretation Guidelines

### How to Read Results Tables

> **Note:** All numbers in the examples below (loadings, fit indices, reliability coefficients) are **illustrative placeholders to show how to read the tables**. They are not results from this study.

#### EFA Results

```
Factor Loadings:
               Factor1  Factor2  Communality
Item_1         0.856    0.123    0.749
Item_2         0.742    0.089    0.558
...
```

**Interpretation**:
- **Loadings >0.40**: Substantial (item assigned to factor)
- **Cross-loadings**: Items loading >0.40 on multiple factors
- **Communality**: Proportion of variance explained (h²)

#### CFA Results

```
Model Fit:
χ² = 245.67, df = 142, p < .001
CFI = 0.942
TLI = 0.931
RMSEA = 0.052 [90% CI: 0.041, 0.063]
SRMR = 0.058
```

**Interpretation**:
- **χ²**: Significant is common with large samples (de-emphasize)
- **CFI, TLI**: Close to 1.0 indicates good fit
- **RMSEA**: Lower is better, include confidence interval
- **SRMR**: Average residual correlation

#### Reliability Results

```
Cronbach's α = 0.887
McDonald's ω_t = 0.901
```

**Interpretation**:
- Values >0.80 indicate good internal consistency
- Omega typically slightly higher than alpha
- Large discrepancy suggests tau-equivalence violated

#### Invariance Results

```
Configural:  CFI = 0.945, RMSEA = 0.054
Metric:      CFI = 0.941, RMSEA = 0.056
  ΔCFI = -0.004, ΔRMSEA = 0.002 → Metric invariance supported
Scalar:      CFI = 0.935, RMSEA = 0.059  
  ΔCFI = -0.006, ΔRMSEA = 0.003 → Scalar invariance supported
```

**Interpretation**:
- Small ΔCFI and ΔRMSEA → Invariance supported
- Supported invariance → Valid group comparisons possible

### Common Questions

**Q: My CFI is 0.89. Is my model bad?**
A: Not necessarily. Fit indices are guidelines, not rules. Consider:
- Sample size (affects some indices)
- Model complexity
- Theoretical meaningfulness
- Other fit indices
- Modification indices for improvement

**Q: What if invariance isn't supported?**
A: Options include:
1. **Partial invariance**: Free specific parameters
2. **Post-hoc investigation**: Why are groups different?
3. **Separate analyses**: Analyze groups independently
4. **Reconsider grouping**: Is grouping variable appropriate?

**Q: Should I delete low-loading items?**
A: Consider:
- **Theory**: Does item belong conceptually?
- **Reliability**: Impact on scale reliability
- **Content validity**: Balance coverage vs. parsimony
- **Publication**: Document all decisions

**Q: When should I use robust estimators?**
A: Consider robust estimators when:
- Assumptions violated
- Non-normal data
- Small samples
- Ordinal data with few categories

## References

### Key Papers (Alphabetical)

1. Cheung, G. W., & Rensvold, R. B. (2002). Evaluating goodness-of-fit indexes for testing measurement invariance. *Structural Equation Modeling, 9*(2), 233-255.

2. Fabrigar, L. R., Wegener, D. T., MacCallum, R. C., & Strahan, E. J. (1999). Evaluating the use of exploratory factor analysis in psychological research. *Psychological Methods, 4*(3), 272-299.

3. Horn, J. L. (1965). A rationale and test for the number of factors in factor analysis. *Psychometrika, 30*(2), 179-185.

4. Hu, L. T., & Bentler, P. M. (1999). Cutoff criteria for fit indexes in covariance structure analysis. *Structural Equation Modeling, 6*(1), 1-55.

5. McDonald, R. P. (1999). *Test theory: A unified treatment*. Mahwah, NJ: Erlbaum.

6. Tabachnick, B. G., & Fidell, L. S. (2013). *Using multivariate statistics* (6th ed.). Boston, MA: Pearson.

### Recommended Textbooks

- Brown, T. A. (2015). *Confirmatory factor analysis for applied research* (2nd ed.). New York, NY: Guilford Press.
- Kline, R. B. (2016). *Principles and practice of structural equation modeling* (4th ed.). New York, NY: Guilford Press.
- Revelle, W. (2020). *Psych: Procedures for psychological, psychometric, and personality research*. Evanston, IL: Northwestern University.

## Additional Resources

- [Factor Analysis Decision Tree](https://www.tandfonline.com/doi/full/10.1080/00223891.2018.1449025)
- [SEM Fit Index Interpretations](https://davidakenny.net/cm/fit.htm)
- [McDonald's Omega Calculator](https://www.real-statistics.com/reliability/omega/)

---

**Questions or suggestions for documentation improvements?**  
Please open an issue or submit a pull request.
