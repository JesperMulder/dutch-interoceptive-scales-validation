# \# Data Documentation

# 

# This folder contains the dataset used for validating the Dutch translations of the IAS and IATS questionnaires.

# 

# \## Data Privacy

# 

# \*\*Important\*\*: The actual data files are not included in this public repository to protect participant privacy. Only the analysis code is shared publicly.

# 

# \## Expected Data Format

# 

# The analysis script expects a data file (`.sav` or `.csv` format).

# 

# \### Required Variables

# 

# \#### Participant Information

# \- `Persoon\_ID`: Unique participant identifier

# \- `Leeftijd`: Age (numeric)

# \- `Geslacht`: Gender (coded as appropriate for your study)

# 

# \#### IAS Items (Interoceptive Accuracy Scale)

# \- `IAS\_1` through `IAS\_21`: Individual IAS item responses

# \- `IAS\_Totaal`: IAS total score (sum of items)

# 

# \#### IATS Items (Interoceptive Attention Scale)

# \- `IATS\_1` through `IATS\_21`: Individual IATS item responses

# \- `IATS\_Totaal`: IATS total score (sum of items)

# 

# \#### Validation Questionnaires (Total Scores)

# \- `BPQ\_Totaal`: Body Perception Questionnaire total score

# \- `ICQ\_Totaal`: Interoceptive Confusion Questionnaire total score

# \- `BDI\_Totaal`: Beck Depression Inventory total score

# \- `TAS\_Totaal`: Toronto Alexithymia Scale total score

# 

# \### Data Requirements

# 

# \#### Scale Responses

# \- \*\*IAS/IATS Items\*\*: Should be on the original response scale (1-5 Likert scale)

# \- \*\*Total Scores\*\*: Pre-calculated total scores for each questionnaire

# \- \*\*Missing Data\*\*: The script will handle missing data appropriately

# 

# \### File Formats Supported

# 

# \#### SPSS Files (.sav)

# ```python

# \# The script will load using:

# df = pd.read\_spss('your\_file.sav', convert\_categoricals=False)

# ```

# 

# \#### CSV Files (.csv)

# ```python

# \# Alternatively, for CSV files:

# df = pd.read\_csv('your\_file.csv')

# ```

# 

# \### Data Quality Checks

# 

# The analysis script performs the following data quality checks:

# 

# 1\. \*\*Outlier Detection\*\*: Removes participants with Z-scores > 3 on any total score

# 2\. \*\*Missing Data\*\*: Reports missing data patterns

# 3\. \*\*Range Checks\*\*: Ensures responses are within expected ranges

# 

# \### Variable Coding

# 

# \#### Gender (`Geslacht`)

# \- 1 = Male, 2 = Female

# 

# \#### Response Scales

# \- \*\*IAS\*\*: 1 = Strongly disagree, 5 = Strongly agree

# \- \*\*IATS\*\*: 1 = Strongly disagree, 5 = Strongly agree

# \- \*\*Validation questionnaires\*\*: See citation

# 

# \## Data Preparation Steps

# 

# Before running the analysis:

# 

# 1\. \*\*Place your data file\*\* in this `data/` folder

# 2\. \*\*Update the file path\*\* in `analysis.py`:

# &nbsp;  ```python

# &nbsp;  DATA\_PATH = 'data/your\_actual\_filename.sav'

# &nbsp;  ```

# 3\. \*\*Verify variable names\*\* match those expected by the script

# 4\. \*\*Check data types\*\* are appropriate (numeric for scales, categorical for demographics)

# 

# \## Confidentiality

# 

# \- All data should be de-identified before analysis

# \- Participant IDs should not be traceable to individuals

# \- Follow your institution's ethical guidelines for data sharing

# \- Consider data encryption for sensitive information

# 

# \## Data Citation

# 

# When using this analysis approach, please cite both:

# 1\. The original questionnaire developers

# 2\. This validation study

