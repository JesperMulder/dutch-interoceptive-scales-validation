# Output Directory

This directory contains all analysis outputs including Excel workbooks, figures, and supplementary results.

## ⚠️ Important Note

**Output files are NOT tracked in version control** to keep the repository clean and manageable. This directory is included in `.gitignore`.

## Output Structure

After running the analysis, this directory will contain:

```
output/
├── README.md                               # This file
│
├── IAS_Analysis_Results.xlsx               # IAS psychometric results
├── IATS_Analysis_Results.xlsx              # IATS psychometric results
├── Univariate_Regression_Results.xlsx      # Regression analyses
│
├── IAS_parallel_analysis.png               # IAS scree plot
├── IATS_parallel_analysis.png              # IATS scree plot
│
├── scatter_IAS_Totaal_ICQ_Totaal.png      # Regression scatterplots
├── scatter_IAS_Totaal_BPQ_Totaal.png
├── scatter_IAS_Totaal_BDI_Totaal.png
├── scatter_IAS_Totaal_TAS_Totaal.png
│
├── scatter_IATS_Totaal_ICQ_Totaal.png
├── scatter_IATS_Totaal_BPQ_Totaal.png
├── scatter_IATS_Totaal_BDI_Totaal.png
└── scatter_IATS_Totaal_TAS_Totaal.png
```

## Excel Workbooks

### IAS_Analysis_Results.xlsx

Comprehensive psychometric results for the Interoceptive Accuracy Scale (IAS).

**Worksheets**:

1. **EFA_Results**
   - Factorability tests (KMO, Bartlett's test)
   - Number of factors determined
   - Factor loadings (pattern matrix)
   - Communalities
   - Variance explained by factors
   - Factor correlations (if oblique rotation)

2. **Reliability**
   - Overall scale reliability
     - Cronbach's alpha
     - McDonald's omega (hierarchical and total)
     - Number of items
   - Factor-specific reliability (if multidimensional)

3. **CFA_Results**
   - Model fit indices
     - Chi-square (χ²) and degrees of freedom
     - Comparative Fit Index (CFI)
     - Tucker-Lewis Index (TLI)
     - Root Mean Square Error of Approximation (RMSEA)
     - Standardized Root Mean Square Residual (SRMR)
     - Akaike Information Criterion (AIC)
     - Bayesian Information Criterion (BIC)
   - Standardized factor loadings
   - Model comparison (EFA-derived vs. single-factor)

4. **Measurement_Invariance**
   - Configural invariance (baseline model)
   - Metric invariance (equal loadings)
   - Scalar invariance (equal loadings and intercepts)
   - Model comparison statistics
     - ΔCFI (change in CFI)
     - ΔRMSEA (change in RMSEA)
   - Group-specific fit indices
   - Invariance decision (supported/not supported)

**Usage**:
- Open in Excel, LibreOffice, or Google Sheets
- Tables are formatted with headers
- Ready for inclusion in manuscripts or presentations
- Can be copy-pasted into Word or LaTeX documents

### IATS_Analysis_Results.xlsx

Identical structure to IAS results, but for the Interoceptive Attention Scale (IATS).

**Worksheets**: Same as IAS_Analysis_Results.xlsx

### Univariate_Regression_Results.xlsx

Results from univariate linear regression analyses testing relationships between interoceptive scales and validity measures.

**Worksheets**:

1. **IAS_Summary**
   - Summary table of all IAS regressions
   - Columns:
     - Predictor variable
     - Sample size (N)
     - R² and Adjusted R²
     - Pearson correlation (r)
     - Regression coefficient (B)
     - Standard error (SE)
     - t-statistic
     - p-value
     - Confidence intervals
     - F-statistic
     - Assumption test results

2. **IATS_Summary**
   - Summary table of all IATS regressions
   - Same structure as IAS_Summary

3. **IAS_[Predictor]** (one sheet per predictor)
   - Detailed results for specific regression model
   - Model information (DV, IV, N, R², F-statistic)
   - Coefficient table (B, SE, t, p, CI)
   - Assumption test results
     - Residuals normality (p-value)
     - Heteroscedasticity test (p-value)

4. **IATS_[Predictor]** (one sheet per predictor)
   - Same structure as IAS predictor sheets

**Predictors tested**:
- ICQ_Totaal (Interoceptive Confidence)
- BPQ_Totaal (Body Perception)
- BDI_Totaal (Depression)
- TAS_Totaal (Alexithymia)

## Figure Files

All figures are saved at **300 DPI** in PNG format, suitable for publication.

### Parallel Analysis Scree Plots

**Files**:
- `IAS_parallel_analysis.png`
- `IATS_parallel_analysis.png`

**Content**:
- Eigenvalues from real data (blue line with markers)
- Mean eigenvalues from random data (red dashed line)
- 95th percentile of random eigenvalues (red dotted line)
- Kaiser's criterion reference line (eigenvalue = 1)

**Interpretation**:
- Factors retained where real eigenvalues exceed random eigenvalues
- Intersection point indicates optimal number of factors

**Dimensions**: 10" × 6" (3000 × 1800 pixels at 300 DPI)

### Regression Scatterplots

**Files** (8 total):
- `scatter_IAS_Totaal_[Predictor].png` (4 files)
- `scatter_IATS_Totaal_[Predictor].png` (4 files)

**Content**:
- Bivariate scatterplot (points with alpha transparency)
- Regression line (red solid line)
- Statistics panel (R², Pearson r, p-value)
- Axis labels with variable names
- Title indicating regression model

**Interpretation**:
- Visual assessment of linearity
- Strength and direction of relationship
- Presence of outliers or influential points

**Dimensions**: 8" × 6" (2400 × 1800 pixels at 300 DPI)

## File Naming Conventions

### Excel Files
- Format: `[Scale]_Analysis_Results.xlsx`
- Examples: `IAS_Analysis_Results.xlsx`, `IATS_Analysis_Results.xlsx`

### Figures
- Scree plots: `[Scale]_parallel_analysis.png`
- Scatterplots: `scatter_[DV]_[IV].png`

## Using Output Files

### For Manuscripts

**Tables**:
1. Open Excel file
2. Navigate to relevant worksheet
3. Copy table
4. Paste into Word document
5. Apply journal-specific formatting

**Figures**:
1. Insert figure into document
2. Resize maintaining aspect ratio
3. Add figure caption
4. Reference in text

**Example caption**:
> Figure 1. Parallel analysis scree plot for the Interoceptive Accuracy Scale (IAS). Real data eigenvalues (blue line) are compared against the 95th percentile of random data eigenvalues (red dotted line). The intersection suggests retaining [X] factors.

### For Presentations

**PowerPoint/Keynote**:
- Insert PNG figures directly (high resolution maintained)
- Copy tables from Excel and paste as formatted tables
- Resize and format as needed

**LaTeX**:
```latex
\begin{figure}[h]
  \centering
  \includegraphics[width=0.8\textwidth]{output/IAS_parallel_analysis.png}
  \caption{Parallel analysis scree plot for the IAS.}
  \label{fig:ias_scree}
\end{figure}
```

### For Supplementary Materials

- Combine Excel workbooks into single supplementary file
- Include all figures in supplementary figures document
- Reference in main manuscript

## Archiving

### Recommended Archival Practices

**Before publication**:
1. Create dated archive: `output_YYYYMMDD.zip`
2. Include README documenting:
   - Analysis script version
   - Date of analysis
   - Package versions (from session info)
   - Any notes on analysis decisions

**Long-term storage**:
- Store outputs separate from code repository
- Use institutional data repository if available
- Include persistent identifier (DOI) if publishing supplementary materials

**Organization**:
```
analysis_archive_20241215/
├── script/
│   └── dutch_interoceptive_validation.py
├── output/
│   ├── [all output files]
│   └── analysis_log.txt
└── session_info.txt
```

## Troubleshooting

### Common Issues

**"Output file already exists"**
- Expected behavior: Files are overwritten each run
- To preserve: Rename previous outputs before re-running
- Solution: Add timestamps to output filenames if needed

**"Cannot open Excel file"**
- Ensure file is not currently open in Excel
- Check file permissions
- Try opening with LibreOffice or Google Sheets

**"Figures look pixelated"**
- Check DPI setting (should be 300)
- Don't resize PNG files (regenerate if needed)
- Use vector formats (.pdf, .svg) if available

**"Missing worksheets in Excel"**
- Check console output for errors
- Some analyses may fail if assumptions violated
- Review error messages in terminal

### Quality Checks

Before using outputs, verify:

- [ ] All expected worksheets present
- [ ] No #DIV/0! or #ERROR! in Excel
- [ ] Figures render correctly
- [ ] Statistics are reasonable (e.g., no negative variances)
- [ ] Sample sizes match across analyses
- [ ] Model fit indices within expected ranges

## Regenerating Outputs

To regenerate all outputs:

```bash
# Ensure environment is activated
conda activate interoceptive-validation

# Run analysis
python dutch_interoceptive_validation.py

# Outputs will be created/overwritten in output/
```

**Note**: Outputs depend on:
- Input data (must be available)
- Random seed (set in script for reproducibility)
- Package versions (use provided environment)

## Output Checklist

After running analysis, verify presence of:

### Excel Workbooks (3 files)
- [ ] IAS_Analysis_Results.xlsx
- [ ] IATS_Analysis_Results.xlsx
- [ ] Univariate_Regression_Results.xlsx

### Figures (10 files)
- [ ] IAS_parallel_analysis.png
- [ ] IATS_parallel_analysis.png
- [ ] scatter_IAS_Totaal_ICQ_Totaal.png
- [ ] scatter_IAS_Totaal_BPQ_Totaal.png
- [ ] scatter_IAS_Totaal_BDI_Totaal.png
- [ ] scatter_IAS_Totaal_TAS_Totaal.png
- [ ] scatter_IATS_Totaal_ICQ_Totaal.png
- [ ] scatter_IATS_Totaal_BPQ_Totaal.png
- [ ] scatter_IATS_Totaal_BDI_Totaal.png
- [ ] scatter_IATS_Totaal_TAS_Totaal.png

### Quality Checks
- [ ] All Excel files open without errors
- [ ] All figures display correctly
- [ ] Sample sizes consistent across outputs
- [ ] No missing or suspicious values

## Contact

If you encounter issues with output files:

1. Check error messages in console
2. Review troubleshooting section above
3. Consult documentation in `docs/`
4. Open an issue on GitHub with:
   - Error message
   - Steps to reproduce
   - System information

---

**Last updated**: 2026

**Note**: This directory is automatically created if it doesn't exist. Never commit output files to version control - they can be regenerated from the script and data.
