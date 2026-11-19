# Data Directory

This directory contains the input data for the psychometric validation analyses.

## ⚠️ Important Note

**Data files are NOT tracked in version control** for privacy and ethical reasons. This directory contains only documentation and is included in `.gitignore`.

## Data Requirements

### File Format

- **Required format**: SPSS (.sav) file
- **Encoding**: UTF-8 recommended
- **File size**: No specific limit, but large files (>100MB) may require additional memory

### Required Variables

#### Scale Items

All scale items should be coded as numeric variables with appropriate value labels.

**IAS (Interoceptive Accuracy Scale)**
- Variables: `IAS_1`, `IAS_2`, ..., `IAS_21`
- Items: 21
- Response scale: [Document your scale, e.g., 1-5 Likert]

**IATS (Interoceptive Attention Scale)**
- Variables: `IATS_1`, `IATS_2`, ..., `IATS_21`
- Items: 21
- Response scale: [Document your scale]

**ICQ (Interoceptive Confidence Questionnaire)**
- Variables: `ICQ_1`, `ICQ_2`, ..., `ICQ_20`
- Items: 20
- Response scale: [Document your scale]

**BPQ (Body Perception Questionnaire)**
- Awareness subscale: `BPQ_Awareness_1` to `BPQ_Awareness_26` (26 items)
- ANS subscale: `BPQ_ANS_1` to `BPQ_ANS_20` (20 items)
- Total: 46 items
- Response scale: [Document your scale]

**BDI (Beck Depression Inventory)**
- Variables: `BDI_1`, `BDI_2`, ..., `BDI_21`
- Items: 21
- Response scale: [Document your scale]

**TAS (Toronto Alexithymia Scale)**
- Variables: `TAS_1`, `TAS_2`, ..., `TAS_20`
- Items: 20
- Response scale: [Document your scale]

#### Total Scores

Pre-calculated total scores (optional but recommended for validation):

- `IAS_Totaal`: Sum of IAS items
- `IATS_Totaal`: Sum of IATS items
- `ICQ_Totaal`: Sum of ICQ items
- `BPQ_Awareness_Totaal`: Sum of BPQ Awareness items
- `BPQ_ANS_Totaal`: Sum of BPQ ANS items
- `BPQ_Totaal`: Sum of all BPQ items
- `BDI_Totaal`: Sum of BDI items
- `TAS_Totaal`: Sum of TAS items

#### Demographics and Covariates

**Required**:
- `Geslacht`: Gender (for measurement invariance testing)
  - Coding: [e.g., 1 = Male, 2 = Female]
  
- `Leeftijd`: Age in years
  - Type: Numeric
  - Range: [e.g., 18-99]

**Optional** (for descriptive statistics):
- `Geen_ChronischeAandoening`: Chronic illness status
- `Inkomen_Cat`: Income category
- `SES`: Socioeconomic status

### Data Quality Requirements

#### Missing Data

- **Maximum acceptable**: <20% missing per variable
- **Pattern**: Completely at Random (MCAR) or At Random (MAR) preferred
- **Handling**: Multiple imputation (MICE) is applied automatically

#### Outliers

- Univariate outliers (>3 SD) are identified and reported
- Multivariate outliers should be investigated before analysis
- Consider data entry errors vs. valid extreme responses

### Variable Naming Conventions

If your variable names differ from those expected, update the variable lists in the script:

```python
# Example: If your IAS items are named IAS01, IAS02, ...
IAS_ITEMS = [f'IAS{i:02d}' for i in range(1, 22)]

# Example: If gender is coded as 'sex' instead of 'Geslacht'
STRATIFY_VARIABLE = 'sex'
```

## Data Preparation Checklist

Before running the analysis, check that:

- [ ] Data is in SPSS (.sav) format
- [ ] All required scale items are present
- [ ] Variable names match those specified in script or have been updated
- [ ] Missing values are properly coded (not as 999 or -99)
- [ ] Response scales are documented
- [ ] No duplicate participant IDs
- [ ] Data has been checked for entry errors
- [ ] Ethical approval obtained for data use
- [ ] Appropriate permissions for data sharing (if applicable)

## Data Security and Ethics

### Privacy

- **Personal identifiers**: Remove all direct identifiers before analysis
- **De-identification**: Confirm that no one can trace the data back to individuals
- **Sensitive variables**: Handle with appropriate security measures

### Ethical Considerations

- Obtain institutional review board (IRB) approval before data collection
- Obtain informed consent from all participants
- Follow data protection regulations (GDPR, HIPAA, etc.)
- Maintain participant confidentiality
- Store data securely with appropriate access controls

### Data Sharing

When sharing analysis results:

- Never commit raw data files to version control
- Share only aggregated or de-identified results
- Follow your institution's data sharing policies
- Consider using restricted access repositories if sharing is necessary

## Example Data Structure

```
data/
├── README.md                          # This file
├── Part1_data.sav                     # Main dataset (NOT in git)
├── data_dictionary.csv                # Variable descriptions (optional)
└── codebook.pdf                       # Detailed codebook (optional)
```

## Troubleshooting

### Common Issues

**"Data file not found"**
- Place the data file is in the `data/` directory
- Update `DATA_PATH` in script configuration
- Check file extension (.sav)

**"Variable not found"**
- Verify variable names in SPSS file match script expectations
- Update variable name lists in script if needed
- Check for spelling/capitalization differences

**"Too many missing values"**
- Review data collection procedures
- Consider listwise deletion for variables with >20% missing
- Document missing data patterns

**"SPSS file won't open"**
- Check SPSS file version (older versions preferred)
- Try re-saving in SPSS with compatibility mode
- Use `pyreadstat` to inspect file structure

### Getting Help

If you encounter data-related issues:

1. Check variable names: `df.columns.tolist()`
2. Check data types: `df.dtypes`
3. Check missing patterns: `df.isnull().sum()`
4. Inspect first rows: `df.head()`

## References

- Field, A. (2013). *Discovering statistics using IBM SPSS statistics* (4th ed.). Sage.
- Tabachnick, B. G., & Fidell, L. S. (2013). *Using multivariate statistics* (6th ed.). Pearson.
- van Buuren, S. (2018). *Flexible imputation of missing data* (2nd ed.). CRC Press.
