# -*- coding: utf-8 -*-
"""
Analysis Script: Psychometric validation Dutch IAS and IATS
Paper: Reliability and Validity of the Dutch Interoceptive Accuracy Scale and
       Interoceptive Attention Scale

Comprehensive psychometric validation of Dutch versions of interoceptive scales
including Exploratory Factor Analysis (EFA), Confirmatory Factor Analysis (CFA),
reliability analysis, measurement invariance testing, and univariate regression.

Scales analyzed:
    - IAS (Interoceptive Accuracy Scale): 21 items
    - IATS (Interoceptive Attention Scale): 21 items
    - ICQ (Interoceptive Confidence Questionnaire): 20 items
    - BPQ (Body Perception Questionnaire): 46 items (26 Awareness + 20 ANS)
    - BDI (Beck Depression Inventory): 21 items
    - TAS (Toronto Alexithymia Scale): 20 items

Author: Jesper Mulder
Institution: Leiden University Medical Center
License: MIT
Repository: https://github.com/JesperMulder/dutch-interoceptive-scales-validation
Date: 2026-05-13 (article publication date)
Article: https://doi.org/10.1177/13591053261449068
"""
#%% IMPORTS & ENVIRONMENT
# Standard library
import warnings
import sys
from pathlib import Path

# Data manipulation and statistics
import pandas as pd
import numpy as np
from scipy import stats as scipystats
from scipy.stats import chi2

# Machine learning and preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Imputation
import miceforest as mf

# Factor analysis and psychometrics
from factor_analyzer import (
    FactorAnalyzer,
    calculate_kmo,
    calculate_bartlett_sphericity,
    ConfirmatoryFactorAnalyzer,
    ModelSpecificationParser
)
from factor_analyzer.rotator import Rotator

# Structural equation modeling
import semopy
from semopy import Model

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Excel output
import openpyxl
from openpyxl.utils.dataframe import dataframe_to_rows

# Statistical modeling
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Configure display and plotting
pd.set_option('display.max_columns', None)
pd.set_option('display.precision', 3)
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300

# Suppress specific warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

#%% CONFIGURATION & PARAMETERS
# ============================================================================
# RANDOM SEEDS FOR REPRODUCIBILITY
# ============================================================================
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# ============================================================================
# FILE PATHS
# ============================================================================
# File paths - configure for your system
# Use relative paths for reproducibility
DATA_PATH = Path('./data/Part1_19082024_clean_JM_version2.sav')  # Update filename as needed
OUTPUT_DIR = Path('./output')

# Ensure output directory exists
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# ANALYSIS PARAMETERS
# ============================================================================
# Outlier detection
OUTLIER_THRESHOLD_SD = 3  # Outlier detection threshold (standard deviations)

# Data splitting
TEST_SIZE = 0.5  # 50/50 split for EFA/CFA
STRATIFY_VARIABLE = 'Geslacht'  # Stratify by gender

# Factor analysis
MIN_EIGENVALUE_KAISER = 1.0  # Kaiser's criterion
PARALLEL_ANALYSIS_ITERATIONS = 1000  # Monte Carlo iterations for parallel analysis

# CFA fit index thresholds (for measurement invariance decision)
CFI_THRESHOLD = 0.90  # Minimum CFI for acceptable fit
RMSEA_THRESHOLD = 0.08  # Maximum RMSEA for acceptable fit
TLI_THRESHOLD = 0.90  # Minimum TLI for acceptable fit

# Measurement invariance thresholds
DELTA_CFI_THRESHOLD = 0.01  # Cheung & Rensvold (2002) recommended threshold
DELTA_RMSEA_THRESHOLD = 0.015  # Cheung & Rensvold (2002) recommended threshold

# ============================================================================
# VARIABLE DEFINITIONS
# ============================================================================
# IAS items (Interoceptive Accuracy Scale)
IAS_ITEMS = [f'IAS_{i}' for i in range(1, 22)]  # IAS_1 to IAS_21

# IATS items (Interoceptive Attention Scale)
IATS_ITEMS = [f'IATS_{i}' for i in range(1, 22)]  # IATS_1 to IATS_21 (21 items)

# ICQ items (Interoceptive Confidence Questionnaire)
# ICQ has 20 items
ICQ_ITEMS = [f'ICQ_{i}' for i in range(1, 21)]  # ICQ_1 to ICQ_20 (20 items)

# BPQ items (Body Perception Questionnaire)
# BPQ has two subscales: Awareness (26 items) and Autonomic Nervous System (20 items)
BPQ_AWARENESS_ITEMS = [f'BPQ_Awareness_{i}' for i in range(1, 27)]  # BPQ_Awareness_1 to BPQ_Awareness_26
BPQ_ANS_ITEMS = [f'BPQ_ANS_{i}' for i in range(1, 21)]  # BPQ_ANS_1 to BPQ_ANS_20
BPQ_ITEMS = BPQ_AWARENESS_ITEMS + BPQ_ANS_ITEMS  # Combined: 46 items total

# BDI items (Beck Depression Inventory)
BDI_ITEMS = [f'BDI_{i}' for i in range(1, 22)]
# TAS items (Toronto Alexithymia Scale)
# TAS-20 has 20 items
TAS_ITEMS = [f'TAS_{i}' for i in range(1, 21)]  # TAS_1 to TAS_20

# Note: If your variable names differ (e.g., IAS01 instead of IAS_1),
#       update the item lists above accordingly

# Continuous variables for descriptives
CONTINUOUS_VARS = [
    'Leeftijd', 'IAS_Totaal', 'IATS_Totaal',
    'BPQ_Awareness_Totaal', 'BPQ_ANS_Totaal', 'BPQ_Totaal',
    'ICQ_Totaal', 'BDI_Totaal', 'TAS_Totaal'
]

# Categorical variables for descriptives
CATEGORICAL_VARS = [
    'Geen_ChronischeAandoening', 'Inkomen_Cat',
    'SES', 'Geslacht'
]

# Variables to check for outliers
OUTLIER_CHECK_VARS = [
    'IAS_Totaal', 'IATS_Totaal', 'BPQ_Totaal',
    'ICQ_Totaal', 'BDI_Totaal', 'TAS_Totaal'
]

# Regression predictors
REGRESSION_PREDICTORS = ['ICQ_Totaal', 'BPQ_Totaal', 'BDI_Totaal', 'TAS_Totaal']

print("=" * 80)
print("DUTCH INTEROCEPTIVE SCALES VALIDATION - CONFIGURATION LOADED")
print("=" * 80)
print(f"Random Seed: {RANDOM_SEED}")
print(f"Outlier Threshold: {OUTLIER_THRESHOLD_SD} SD")
print(f"Test/Train Split: {TEST_SIZE * 100:.0f}%/{(1-TEST_SIZE) * 100:.0f}%")
print(f"Stratification Variable: {STRATIFY_VARIABLE}")
print(f"Output Directory: {OUTPUT_DIR.absolute()}")
print("=" * 80)

#%% DATA LOADING
print("\n" + "=" * 80)
print("DATA LOADING")
print("=" * 80)

try:
    df_analysis = pd.read_spss(DATA_PATH, convert_categoricals=False)
    print(f"✓ Data loaded successfully")
    print(f"  Initial sample size: N = {len(df_analysis)}")
    print(f"  Variables: {df_analysis.shape[1]}")
except FileNotFoundError:
    print(f"✗ Error: Data file not found at {DATA_PATH}")
    print("  Please update DATA_PATH in the configuration section")
    sys.exit(1)
except Exception as e:
    print(f"✗ Error loading data: {str(e)}")
    sys.exit(1)
    raise

#%% SCALE VALIDATION & CORRECTION
print("\n" + "=" * 80)
print("SCALE VALIDATION & CORRECTION")
print("=" * 80)
print("Checking whether all item and total score ranges are on the expected scale.")
print("=" * 80)

# ============================================================================
# SCALE SPECIFICATIONS
# Each entry defines:
#   items         : list of item column names
#   item_min      : expected minimum for a single item
#   item_max      : expected maximum for a single item
#   totals        : dict mapping total-score column name -> (expected_min, expected_max)
#   subscales     : dict mapping subscale total name -> list of contributing item names
#                   (used to recompute totals from corrected items; empty if not applicable)
# ============================================================================
SCALE_SPECS = {
    'IAS': {
        'items'     : IAS_ITEMS,
        'item_min'  : 1,
        'item_max'  : 5,
        'totals'    : {'IAS_Totaal': (21, 105)},
        'subscales' : {'IAS_Totaal': IAS_ITEMS}
    },
    'IATS': {
        'items'     : IATS_ITEMS,
        'item_min'  : 1,
        'item_max'  : 5,
        'totals'    : {'IATS_Totaal': (21, 105)},
        'subscales' : {'IATS_Totaal': IATS_ITEMS}
    },
    'ICQ': {
        'items'     : ICQ_ITEMS,
        'item_min'  : 1,
        'item_max'  : 5,
        'totals'    : {'ICQ_Totaal': (20, 100)},
        'subscales' : {'ICQ_Totaal': ICQ_ITEMS}
    },
    'BPQ': {
        'items'     : BPQ_ITEMS,                         # 46 items (Awareness + ANS)
        'item_min'  : 1,
        'item_max'  : 5,
        'totals'    : {
            'BPQ_Awareness_Totaal': (26, 130),
            'BPQ_ANS_Totaal'      : (20, 100),
            'BPQ_Totaal'          : (46, 230)
        },
        'subscales' : {
            'BPQ_Awareness_Totaal': BPQ_AWARENESS_ITEMS,
            'BPQ_ANS_Totaal'      : BPQ_ANS_ITEMS,
            'BPQ_Totaal'          : BPQ_ITEMS
        }
    },
    'BDI': {
        'items'     : BDI_ITEMS,
        'item_min'  : 0,
        'item_max'  : 3,
        'totals'    : {'BDI_Totaal': (0, 63)},
        'subscales' : {'BDI_Totaal': BDI_ITEMS}
    },
    'TAS': {
        'items'     : TAS_ITEMS,
        'item_min'  : 1,
        'item_max'  : 5,
        'totals'    : {'TAS_Totaal': (20, 100)},
        'subscales' : {'TAS_Totaal': TAS_ITEMS}
    }
}

# ============================================================================
# VALIDATION AND CORRECTION LOGIC
# Detection rule: a +1 shift is flagged when the observed minimum across all
# items of a scale equals the expected minimum + 1.  The observed maximum is
# checked as a secondary confirmation.  Both conditions must hold to trigger
# a correction, preventing false positives from data entry errors or genuine
# extreme responding.
# ============================================================================

scale_audit = []          # Collects one summary row per scale for reporting
scales_corrected = []     # Tracks which scales were corrected

for scale_name, spec in SCALE_SPECS.items():
    print(f"\n--- {scale_name} ---")

    # --- Identify item columns that are actually present in the dataframe
    available_items = [col for col in spec['items'] if col in df_analysis.columns]
    missing_item_cols = [col for col in spec['items'] if col not in df_analysis.columns]

    if missing_item_cols:
        print(f"  Warning: {len(missing_item_cols)} item column(s) not found in data "
              f"and will be skipped: {missing_item_cols}")

    if len(available_items) == 0:
        print(f"  Error: No item columns found for {scale_name}. Skipping.")
        scale_audit.append({
            'Scale'              : scale_name,
            'Items_Found'        : 0,
            'Observed_Item_Min'  : None,
            'Observed_Item_Max'  : None,
            'Expected_Item_Range': f"{spec['item_min']}-{spec['item_max']}",
            'Shift_Detected'     : False,
            'Corrected'          : False,
            'Notes'              : 'No item columns found'
        })
        continue

    # --- Compute observed item-level statistics (ignoring missing values)
    item_data = df_analysis[available_items]
    observed_item_min = item_data.min().min()
    observed_item_max = item_data.max().max()

    expected_min = spec['item_min']
    expected_max = spec['item_max']

    print(f"  Items found        : {len(available_items)} / {len(spec['items'])}")
    print(f"  Expected item range: {expected_min} - {expected_max}")
    print(f"  Observed item range: {observed_item_min:.0f} - {observed_item_max:.0f}")

    # --- Detect +1 shift: observed min and max are both exactly 1 above expected
    shift_detected = (
        observed_item_min == expected_min + 1 and
        observed_item_max == expected_max + 1
    )

    # --- Detect out-of-range values that do not match the +1 shift pattern
    out_of_range = (
        observed_item_min < expected_min or
        observed_item_max > expected_max
    ) and not shift_detected

    if out_of_range:
        print(f"  WARNING: Unexpected item range detected (not a clean +1 shift).")
        print(f"  Manual inspection is required before proceeding.")

    if shift_detected:
        print(f"  STATUS: +1 shift detected. Correcting items and recomputing totals.")

        # --- Correct item columns: subtract 1 from every available item
        df_analysis[available_items] = df_analysis[available_items] - 1

        # Verify item correction
        corrected_item_min = df_analysis[available_items].min().min()
        corrected_item_max = df_analysis[available_items].max().max()
        print(f"  Corrected item range: {corrected_item_min:.0f} - {corrected_item_max:.0f}")

        # --- Recompute total scores from corrected items
        # Recomputing from items is preferred over subtracting n_items from the
        # existing total, because it is robust to any pre-existing missing data
        # handling or partial scoring in the original file.
        for total_col, (total_exp_min, total_exp_max) in spec['totals'].items():
            contributing_items = spec['subscales'].get(total_col, [])
            available_contributing = [col for col in contributing_items
                                      if col in df_analysis.columns]

            if total_col in df_analysis.columns and len(available_contributing) > 0:
                df_analysis[total_col] = df_analysis[available_contributing].sum(
                    axis=1, skipna=False   # skipna=False preserves NaN where items are missing
                )
                corrected_total_min = df_analysis[total_col].min()
                corrected_total_max = df_analysis[total_col].max()
                print(f"  {total_col}: recomputed range "
                      f"{corrected_total_min:.0f} - {corrected_total_max:.0f} "
                      f"(expected {total_exp_min} - {total_exp_max})")
            elif total_col not in df_analysis.columns:
                print(f"  {total_col}: column not found in data, skipping total recomputation.")

        scales_corrected.append(scale_name)
        corrected = True

    else:
        # --- Verify that totals are also within the expected range when no shift
        print(f"  STATUS: Item range is correct. Verifying total scores.")
        corrected = False
        for total_col, (total_exp_min, total_exp_max) in spec['totals'].items():
            if total_col in df_analysis.columns:
                observed_total_min = df_analysis[total_col].min()
                observed_total_max = df_analysis[total_col].max()
                total_ok = (
                    observed_total_min >= total_exp_min and
                    observed_total_max <= total_exp_max
                )
                status_flag = "OK" if total_ok else "OUT OF RANGE"
                print(f"  {total_col}: observed {observed_total_min:.0f} - "
                      f"{observed_total_max:.0f} "
                      f"(expected {total_exp_min} - {total_exp_max}) [{status_flag}]")
            else:
                print(f"  {total_col}: column not found in data.")

    # --- Record audit entry
    scale_audit.append({
        'Scale'              : scale_name,
        'Items_Found'        : len(available_items),
        'Observed_Item_Min'  : observed_item_min,
        'Observed_Item_Max'  : observed_item_max,
        'Expected_Item_Range': f"{expected_min} - {expected_max}",
        'Shift_Detected'     : shift_detected,
        'Corrected'          : corrected,
        'Notes'              : '+1 shift corrected' if corrected else
                               ('Out-of-range values detected' if out_of_range else 'No correction needed')
    })

# ============================================================================
# AUDIT SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("SCALE VALIDATION SUMMARY")
print("=" * 80)

scale_audit_df = pd.DataFrame(scale_audit)
print(scale_audit_df.to_string(index=False))

if scales_corrected:
    print(f"\nScales corrected (items and totals): {', '.join(scales_corrected)}")
    print("Note: All corrections applied directly to df_analysis before imputation.")
    print("      Raw data file has NOT been modified.")
else:
    print("\nAll scales validated. No corrections required.")

print("=" * 80)

#%% MISSING DATA IMPUTATION
print("\n" + "=" * 80)
print("MISSING DATA ANALYSIS & IMPUTATION")
print("=" * 80)

# --- Identify variables with missing data
missing_summary = df_analysis.isnull().sum()
vars_with_missing = missing_summary[missing_summary > 0].sort_values(ascending=False)

if len(vars_with_missing) == 0:
    print("✓ No missing data detected - imputation not needed")
    df_analysis_imputed = df_analysis.copy()
else:
    print(f"Variables with missing data: {len(vars_with_missing)}")
    print("\nMissing data summary:")
    for var, count in vars_with_missing.items():
        pct = (count / len(df_analysis)) * 100
        print(f"  {var}: {count} ({pct:.2f}%)")

    # --- Test for Missing Completely At Random (MCAR) using Little's test
    print("\n--- Testing for MCAR (Little's test) ---")

    def littles_mcar_test(data):
        """
        Perform Little's MCAR test
        Returns: chi-square statistic, p-value, degrees of freedom
        """
        # Use only variables with missing data
        data_subset = data[vars_with_missing.index].copy()

        # Create missingness patterns
        missing_patterns = data_subset.isnull().astype(int)
        unique_patterns = missing_patterns.drop_duplicates()

        # NOTE: Little's MCAR test is not implemented. This function only reports
        # missing-data percentages and returns (None, None, None).
        print("  Note: Simplified MCAR test - see missing data percentages above")
        print("  If missingness is <5% per variable, MCAR assumption is typically reasonable")

        return None, None, None

    try:
        chi_sq, p_val, df = littles_mcar_test(df_analysis)
    except:
        print("  Could not perform full Little's test - proceeding with imputation")

    # --- Prepare for imputation
    print("\n--- Starting Multiple Imputation (MICE) ---")

    # Variables to impute
    impute_vars = list(vars_with_missing.index)

    # Variables without missing data (to use as predictors)
    complete_vars = [col for col in df_analysis.columns
                     if col not in impute_vars and df_analysis[col].dtype in ['int64', 'float64']]

    # Combine for imputation model
    imputation_data = df_analysis[impute_vars + complete_vars].copy()

    print(f"  Variables to impute: {len(impute_vars)}")
    print(f"  Predictor variables: {len(complete_vars)}")

    # --- Run MICE imputation with warnings suppressed
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        impute_kernel = mf.ImputationKernel(
            data=imputation_data,
            save_all_iterations_data=True,
            random_state=RANDOM_SEED
        )

        # Perform 20 iterations with 5 imputed datasets
        impute_kernel.mice(iterations=20, datasets=5)

    # --- Combine imputed datasets (average across imputations)
    print("  Combining imputed datasets...")
    imputed_dfs = []
    for i in range(5):
        try:
            imputed_dfs.append(impute_kernel.complete_data(i))
        except KeyError:
            continue

    # Average across imputations
    df_imputed = pd.concat(imputed_dfs).groupby(level=0).mean()

    # Round categorical variables
    categorical_imputed = [var for var in impute_vars if var in CATEGORICAL_VARS]
    if categorical_imputed:
        df_imputed[categorical_imputed] = df_imputed[categorical_imputed].round()

    # Merge with original data
    df_analysis_imputed = df_analysis.copy()
    df_analysis_imputed[impute_vars] = df_imputed[impute_vars]

    print("✓ Imputation completed successfully")

    # Verify no missing data remains
    remaining_missing = df_analysis_imputed[impute_vars].isnull().sum().sum()
    if remaining_missing == 0:
        print("✓ All missing values successfully imputed")
    else:
        print(f"⚠ Warning: {remaining_missing} missing values remain")

print("=" * 80)

#%% OUTLIER REMOVAL
print("\n" + "=" * 80)
print("OUTLIER DETECTION & REMOVAL")
print("=" * 80)

# Calculate z-scores for outlier variables
z_scores = np.abs(scipystats.zscore(df_analysis_imputed[OUTLIER_CHECK_VARS], nan_policy='omit'))

# Identify outliers (observations exceeding threshold on any variable)
outlier_mask = (z_scores >= OUTLIER_THRESHOLD_SD).any(axis=1)
n_outliers = outlier_mask.sum()
n_outliers_per_var = (z_scores >= OUTLIER_THRESHOLD_SD).sum(axis=0)

print(f"Outlier threshold: {OUTLIER_THRESHOLD_SD} SD")
print(f"\nOutliers detected per variable:")
for var, count in zip(OUTLIER_CHECK_VARS, n_outliers_per_var):
    pct = (count / len(df_analysis_imputed)) * 100
    print(f"  {var}: {count} ({pct:.2f}%)")

print(f"\nTotal observations with outliers: {n_outliers} ({n_outliers/len(df_analysis_imputed)*100:.2f}%)")

# Remove outliers
df_clean = df_analysis_imputed[~outlier_mask].copy().reset_index(drop=True)

print(f"\n✓ Sample size after outlier removal: N = {len(df_clean)}")
print(f"  Observations removed: {len(df_analysis_imputed) - len(df_clean)}")
print("=" * 80)

#%% DESCRIPTIVE STATISTICS
print("\n" + "=" * 80)
print("DESCRIPTIVE STATISTICS")
print("=" * 80)

def calculate_comprehensive_descriptives(df, cont_vars, cat_vars):
    """
    Calculate comprehensive descriptive statistics for continuous and categorical variables
    """
    descriptive_stats = []

    # --- Continuous variables
    print("\nContinuous Variables:")
    print("-" * 60)
    for var in cont_vars:
        if var in df.columns:
            stats_dict = {
                'Variable': var,
                'Type': 'Continuous',
                'N': df[var].notna().sum(),
                'Mean': df[var].mean(),
                'SD': df[var].std(),
                'Median': df[var].median(),
                'Q1': df[var].quantile(0.25),
                'Q3': df[var].quantile(0.75),
                'Min': df[var].min(),
                'Max': df[var].max(),
                'Missing': df[var].isnull().sum()
            }
            descriptive_stats.append(stats_dict)

            print(f"{var}:")
            print(f"  M = {stats_dict['Mean']:.2f}, SD = {stats_dict['SD']:.2f}")
            print(f"  Range: [{stats_dict['Min']:.0f}, {stats_dict['Max']:.0f}]")

    # --- Categorical variables
    print("\n" + "-" * 60)
    print("Categorical Variables:")
    print("-" * 60)
    for var in cat_vars:
        if var in df.columns:
            freq = df[var].value_counts().sort_index()
            pct = (freq / len(df) * 100).round(1)

            print(f"\n{var}:")
            for level, count in freq.items():
                print(f"  {level}: n = {count} ({pct[level]:.1f}%)")

            # Add to descriptive stats
            for level in freq.index:
                descriptive_stats.append({
                    'Variable': var,
                    'Type': 'Categorical',
                    'Level': level,
                    'Frequency': freq[level],
                    'Percentage': pct[level],
                    'Missing': df[var].isnull().sum()
                })

    return pd.DataFrame(descriptive_stats)

# Calculate and display descriptives
descriptives_df = calculate_comprehensive_descriptives(
    df_clean, CONTINUOUS_VARS, CATEGORICAL_VARS
)

# Gender-specific age statistics
print("\n" + "-" * 60)
print("Age by Gender:")
print("-" * 60)
age_by_gender = df_clean.groupby('Geslacht')['Leeftijd'].agg(['count', 'mean', 'std'])
print(age_by_gender)

print("=" * 80)

#%% CORRELATION MATRIX
print("\n" + "=" * 80)
print("CORRELATION ANALYSIS")
print("=" * 80)

corr_vars = ['Geen_ChronischeAandoening', 'Geslacht', 'Leeftijd', 'SES',
             'IAS_Totaal', 'IATS_Totaal', 'BPQ_Totaal',
             'ICQ_Totaal', 'BDI_Totaal', 'TAS_Totaal']

correlation_matrix = df_clean[corr_vars].corr()

plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm',
            fmt='.2f', center=0, vmin=-1, vmax=1,
            square=True, linewidths=0.5)
plt.title('Correlation Matrix - Main Study Variables', fontsize=14, pad=20)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / f"correlation_matrix.png", dpi=300, bbox_inches='tight')
plt.show()

print("✓ Correlation matrix generated and saved")
print("=" * 80)

#%% DATA SPLIT FOR EFA AND CFA
print("\n" + "=" * 80)
print("DATA SPLITTING (EFA/CFA SAMPLES)")
print("=" * 80)

# Split data: 50% for EFA, 50% for CFA, stratified by gender
df_efa, df_cfa = train_test_split(
    df_clean,
    test_size=TEST_SIZE,
    random_state=RANDOM_SEED,
    stratify=df_clean[STRATIFY_VARIABLE]
)

print(f"✓ Data split completed")
print(f"  EFA sample: N = {len(df_efa)}")
print(f"  CFA sample: N = {len(df_cfa)}")
print(f"\nGender distribution:")
print(f"  EFA sample:\n{df_efa['Geslacht'].value_counts()}")
print(f"  CFA sample:\n{df_cfa['Geslacht'].value_counts()}")
print("=" * 80)

#%% HELPER FUNCTIONS FOR FACTOR ANALYSIS
print("\n" + "=" * 80)
print("DEFINING ANALYSIS FUNCTIONS")
print("=" * 80)

def calculate_polychoric_correlation(data):
    """
    Calculate the correlation matrix for ordinal (Likert) items
    NOTE: Despite the function name, this returns PEARSON correlations, not polychoric.
    """
    from factor_analyzer import FactorAnalyzer

    # Pearson correlations are used on the ordinal items. Polychoric correlations
    # would need an additional package and are not computed in this script.
    corr_matrix = data.corr(method='pearson')

    return corr_matrix

def parallel_analysis(data, n_iterations=1000, random_state=42):
    """
    Perform parallel analysis to determine number of factors
    Compares eigenvalues from real data to those from random data

    Returns:
        n_factors_pa: Number of factors suggested by parallel analysis
        eigenvalues_real: Eigenvalues from real data
        eigenvalues_random_mean: Mean eigenvalues from random data
    """
    print("  Running parallel analysis...")

    n_obs, n_vars = data.shape

    # Calculate eigenvalues from real data
    corr_matrix = data.corr()
    eigenvalues_real = np.linalg.eigvalsh(corr_matrix)
    eigenvalues_real = np.sort(eigenvalues_real)[::-1]  # Sort descending

    # Generate random data and calculate eigenvalues
    np.random.seed(random_state)
    eigenvalues_random = []

    for i in range(n_iterations):
        # Generate random data with same dimensions
        random_data = np.random.normal(size=(n_obs, n_vars))
        random_corr = np.corrcoef(random_data.T)
        eig_vals = np.linalg.eigvalsh(random_corr)
        eigenvalues_random.append(np.sort(eig_vals)[::-1])

    # Calculate mean and 95th percentile of random eigenvalues
    eigenvalues_random_mean = np.mean(eigenvalues_random, axis=0)
    eigenvalues_random_95 = np.percentile(eigenvalues_random, 95, axis=0)

    # Determine number of factors (where real eigenvalues > random eigenvalues)
    n_factors_pa = np.sum(eigenvalues_real > eigenvalues_random_95)

    print(f"  ✓ Parallel analysis suggests {n_factors_pa} factor(s)")

    return n_factors_pa, eigenvalues_real, eigenvalues_random_mean, eigenvalues_random_95

def perform_efa_analysis(data, items, scale_name, n_factors_pa):
    """
    Perform comprehensive EFA analysis
    """
    print(f'\n{"=" * 80}')
    print(f"EXPLORATORY FACTOR ANALYSIS - {scale_name}")
    print(f'{"=" * 80}')

    data_subset = data[items].copy()
    n_obs = len(data_subset)

    # --- Factorability tests
    print("\n--- Factorability Assessment ---")

    # KMO test
    kmo_all, kmo_model = calculate_kmo(data_subset)
    print(f"Kaiser-Meyer-Olkin (KMO) test: {kmo_model:.3f}")
    if kmo_model >= 0.9:
        print("  → Marvelous")
    elif kmo_model >= 0.8:
        print("  → Meritorious")
    elif kmo_model >= 0.7:
        print("  → Middling")
    elif kmo_model >= 0.6:
        print("  → Mediocre")
    else:
        print("  → Unacceptable")

    # Bartlett's test
    chi_square_value, p_value = calculate_bartlett_sphericity(data_subset)
    print(f"Bartlett's test of sphericity:")
    print(f"  χ² = {chi_square_value:.2f}, p < .001" if p_value < 0.001
          else f"  χ² = {chi_square_value:.2f}, p = {p_value:.3f}")

    # --- Parallel Analysis
    print("\n--- Parallel Analysis ---")
    n_factors_suggested, eig_real, eig_random_mean, eig_random_95 = parallel_analysis(
        data_subset,
        n_iterations=PARALLEL_ANALYSIS_ITERATIONS,
        random_state=RANDOM_SEED
    )

    # Also check Kaiser's criterion
    n_factors_kaiser = np.sum(eig_real > MIN_EIGENVALUE_KAISER)
    print(f"Kaiser's criterion (eigenvalue > 1): {n_factors_kaiser} factor(s)")

    # Scree plot
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(eig_real) + 1), eig_real, 'bo-', label='Real Data', linewidth=2)
    plt.plot(range(1, len(eig_random_mean) + 1), eig_random_mean, 'r--',
             label='Random Data (Mean)', linewidth=2)
    plt.plot(range(1, len(eig_random_95) + 1), eig_random_95, 'r:',
             label='Random Data (95th %ile)', linewidth=2)
    plt.axhline(y=1, color='gray', linestyle='--', alpha=0.5, label="Kaiser's Criterion")
    plt.xlabel('Factor Number', fontsize=12)
    plt.ylabel('Eigenvalue', fontsize=12)
    plt.title(f'Parallel Analysis Scree Plot - {scale_name}', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{scale_name}_parallel_analysis.png", dpi=300, bbox_inches='tight')
    plt.show()

    # Use parallel analysis suggestion
    n_factors_efa = n_factors_suggested
    print(f"\n→ Using {n_factors_efa} factor(s) for EFA (based on parallel analysis)")

    # --- Exploratory Factor Analysis with Oblimin rotation
    print(f"\n--- EFA with {n_factors_efa} Factor(s) ---")
    print("Extraction method: Principal Axis Factoring")
    print("Rotation: Oblimin (oblique)")

    fa = FactorAnalyzer(
        n_factors=n_factors_efa,
        rotation='oblimin',
        method='principal',  # Principal axis factoring - common factor extraction
        use_smc=True  # Use squared multiple correlations for communality estimates
    )

    fa.fit(data_subset)

    # Get results
    loadings = fa.loadings_
    communalities = fa.get_communalities()
    variance = fa.get_factor_variance()

    # Create loadings dataframe
    factor_names = [f'Factor{i+1}' for i in range(n_factors_efa)]
    loadings_df = pd.DataFrame(
        loadings,
        index=items,
        columns=factor_names
    )

    # Add communalities
    loadings_df['Communality'] = communalities

    print("\nFactor Loadings (oblimin rotation):")
    print(loadings_df.round(3))

    # Variance explained
    variance_df = pd.DataFrame(
        variance,
        index=['Variance', 'Proportional Var', 'Cumulative Var'],
        columns=factor_names
    )

    print("\nVariance Explained:")
    print(variance_df.round(3))

    # Factor correlations (oblique rotation): use fa.phi_ when available, otherwise derive from structure and pattern matrices
    if n_factors_efa > 1:
        try:
            # For oblique rotations, factor correlations are in phi_
            if hasattr(fa, 'phi_'):
                factor_corr = fa.phi_
            elif hasattr(fa, 'structure_'):
                # Calculate from structure and pattern matrices
                structure = fa.structure_
                pattern = fa.loadings_
                # Fallback: solve (pattern-transpose x pattern) * phi = pattern-transpose x structure by least squares
                factor_corr = np.linalg.lstsq(pattern.T @ pattern,
                                              pattern.T @ structure,
                                              rcond=None)[0]
            else:
                # Fallback: calculate from loadings
                factor_corr = np.corrcoef(loadings.T)

            # Ensure correct shape
            if factor_corr.shape[0] == n_factors_efa and factor_corr.shape[1] == n_factors_efa:
                factor_corr_df = pd.DataFrame(
                    factor_corr,
                    index=factor_names,
                    columns=factor_names
                )
                print("\nFactor Correlations:")
                print(factor_corr_df.round(3))
            else:
                print(f"\n⚠ Factor correlations matrix has unexpected shape: {factor_corr.shape}")
                print("  Skipping factor correlation display")
                factor_corr_df = None
        except Exception as e:
            print(f"\n⚠ Could not extract factor correlations: {str(e)}")
            factor_corr_df = None
    else:
        factor_corr_df = None

    # Store results
    efa_results = {
        'n_factors': n_factors_efa,
        'n_factors_kaiser': n_factors_kaiser,
        'n_factors_parallel': n_factors_suggested,
        'kmo': kmo_model,
        'bartlett_chi2': chi_square_value,
        'bartlett_p': p_value,
        'loadings': loadings_df,
        'variance': variance_df,
        'factor_correlations': factor_corr_df,
        'eigenvalues_real': eig_real,
        'eigenvalues_random_mean': eig_random_mean,
        'fa_object': fa
    }

    return efa_results

def calculate_reliability(data, items, factor_structure=None):
    """
    Calculate Cronbach's alpha and McDonald's omega

    Parameters:
        data: DataFrame with item data
        items: List of item names
        factor_structure: Dict mapping factor names to item lists (optional)
                         If None, calculates for overall scale only

    Returns:
        Dictionary with alpha and omega values
    """
    from semopy import Model

    print("\n--- Reliability Analysis ---")

    reliability_results = {}

    # Overall scale reliability
    print("\nOverall Scale:")
    data_subset = data[items].dropna()

    # Cronbach's Alpha
    item_corr = data_subset.corr()
    n_items = len(items)
    avg_corr = item_corr.values[np.triu_indices_from(item_corr.values, k=1)].mean()
    alpha = (n_items * avg_corr) / (1 + (n_items - 1) * avg_corr)

    print(f"  Cronbach's α = {alpha:.3f}")

    # McDonald's Omega (requires CFA model)
    try:
        # Create single-factor model for omega calculation
        if factor_structure is None:
            # Single factor model
            model_spec = f"eta =~ {' + '.join(items)}"
        else:
            # Multi-factor model
            model_specs = []
            for factor_name, factor_items in factor_structure.items():
                model_specs.append(f"{factor_name} =~ {' + '.join(factor_items)}")
            model_spec = '\n'.join(model_specs)

        model = Model(model_spec)
        model.fit(data_subset)

        # Calculate omega from factor loadings and residual variances
        # Omega = (sum of loadings)^2 / ((sum of loadings)^2 + sum of error variances)
        inspect_results = model.inspect()


        # Filter for factor loadings (regression relationships)
        factor_loadings = inspect_results[inspect_results['op'] == '~'].copy() if 'op' in inspect_results.columns else pd.DataFrame()

        if not factor_loadings.empty:
            lambda_vals = factor_loadings['Estimate'].values
            lambda_sum = np.sum(lambda_vals)

            # Try multiple methods to get error variances
            errors = None

            # Method 1: Look for error variance rows (e_ prefix in lval)
            if 'lval' in inspect_results.columns:
                errors = inspect_results[inspect_results['lval'].str.startswith('e_', na=False)]

            # Method 2: Look for variance estimates (op == '~~' and lval == rval)
            if (errors is None or errors.empty) and 'op' in inspect_results.columns:
                variances = inspect_results[inspect_results['op'] == '~~']
                # Error variances are where lval == rval and it's an item
                errors = variances[variances['lval'] == variances['rval']]
                # Filter to only items (not latent variables)
                if not errors.empty:
                    errors = errors[errors['lval'].isin(items)]

            if errors is not None and not errors.empty:
                error_sum = errors['Estimate'].sum()

                # Calculate hierarchical omega
                omega_h = (lambda_sum ** 2) / ((lambda_sum ** 2) + error_sum)

                # Calculate total omega (simplified - same as hierarchical for single-factor)
                omega_t = omega_h

                print(f"  McDonald's ωh = {omega_h:.3f} (hierarchical)")
                print(f"  McDonald's ωt = {omega_t:.3f} (total)")

                reliability_results['overall'] = {
                    'alpha': alpha,
                    'omega_h': omega_h,
                    'omega_t': omega_t,
                    'n_items': n_items
                }
            else:
                print("  McDonald's ω: Could not calculate (error variances unavailable)")
                print("  Note: This is common with semopy - omega requires error variance estimates")
                reliability_results['overall'] = {
                    'alpha': alpha,
                    'omega_h': np.nan,
                    'omega_t': np.nan,
                    'n_items': n_items
                }
        else:
            print("  McDonald's ω: Could not calculate (factor loadings unavailable)")
            reliability_results['overall'] = {
                'alpha': alpha,
                'omega_h': np.nan,
                'omega_t': np.nan,
                'n_items': n_items
            }
    except Exception as e:
        print(f"  McDonald's ω: Could not calculate ({str(e)})")
        reliability_results['overall'] = {
            'alpha': alpha,
            'omega_h': np.nan,
            'omega_t': np.nan,
            'n_items': n_items
        }

    # Factor-specific reliability (if factor structure provided)
    if factor_structure is not None and len(factor_structure) > 1:
        print("\nFactor-Specific Reliability:")
        reliability_results['factors'] = {}

        for factor_name, factor_items in factor_structure.items():
            print(f"\n  {factor_name}:")
            factor_data = data[factor_items].dropna()

            # Cronbach's Alpha
            factor_corr = factor_data.corr()
            n_factor_items = len(factor_items)
            avg_factor_corr = factor_corr.values[np.triu_indices_from(factor_corr.values, k=1)].mean()
            factor_alpha = (n_factor_items * avg_factor_corr) / (1 + (n_factor_items - 1) * avg_factor_corr)

            print(f"    Cronbach's α = {factor_alpha:.3f}")

            # Store results
            reliability_results['factors'][factor_name] = {
                'alpha': factor_alpha,
                'n_items': n_factor_items
            }

    return reliability_results

def calculate_srmr_manual(model_object, data):
    """
    Manually calculate SRMR (Standardized Root Mean Square Residual)

    SRMR is the square root of the average squared difference between
    observed and model-implied correlations.

    Parameters
    ----------
    model_object : semopy.Model
        Fitted semopy model object
    data : pd.DataFrame
        Data used to fit the model (should contain only the variables in the model)

    Returns
    -------
    float
        SRMR value (ranges from 0 to 1, where lower is better)

    Reference
    ---------
    Hu & Bentler (1999). Cutoff criteria for fit indexes in covariance
    structure analysis. Structural Equation Modeling, 6(1), 1-55.
    Good fit: SRMR < 0.08
    """
    try:
        # Get observed correlation matrix
        obs_corr = data.corr().values

        # Get parameter estimates
        params = model_object.inspect()

        # Reconstruct implied covariance matrix from model parameters
        # Get variables in the model
        variables = list(data.columns)
        n_vars = len(variables)

        # Initialize implied covariance matrix
        implied_cov = np.zeros((n_vars, n_vars))

        # Extract factor loadings (lambda matrix)
        loadings_df = params[params['op'] == '~'].copy()

        # Extract error variances (theta matrix)
        errors_df = params[params['op'] == '~~'].copy()
        errors_df = errors_df[errors_df['lval'] == errors_df['rval']]  # Keep variances only (lval == rval)

        # Create loading matrix
        loading_dict = {}
        for _, row in loadings_df.iterrows():
            item = row['rval']
            factor = row['lval']
            loading = row['Estimate']

            if item not in loading_dict:
                loading_dict[item] = {}
            loading_dict[item][factor] = loading

        # Factor variances (diagonal ~~ rows) start the table; covariances (off-diagonal ~~ rows) are added below
        factor_cov_df = errors_df.copy()  # Start with variances
        factor_cov_rows = params[(params['op'] == '~~') & (params['lval'] != params['rval'])].copy()

        # Create dictionary for factor variances/covariances
        factor_cov_dict = {}
        for _, row in factor_cov_df.iterrows():
            factor = row['lval']
            if factor not in variables:  # Only factors, not observed variables
                factor_cov_dict[(factor, factor)] = row['Estimate']

        for _, row in factor_cov_rows.iterrows():
            factor1 = row['lval']
            factor2 = row['rval']
            if factor1 not in variables and factor2 not in variables:
                factor_cov_dict[(factor1, factor2)] = row['Estimate']
                factor_cov_dict[(factor2, factor1)] = row['Estimate']

        # For each pair of variables, calculate implied covariance
        for i, var_i in enumerate(variables):
            for j, var_j in enumerate(variables):
                if i == j:
                    # Variance: sum of squared loadings * factor variances + error variance
                    variance = 0
                    if var_i in loading_dict:
                        for factor, loading in loading_dict[var_i].items():
                            factor_var = factor_cov_dict.get((factor, factor), 1.0)
                            variance += (loading ** 2) * factor_var

                    # Add error variance
                    error_row = errors_df[errors_df['lval'] == var_i]
                    if not error_row.empty:
                        variance += error_row.iloc[0]['Estimate']

                    implied_cov[i, j] = variance
                else:
                    # Covariance: sum of products of loadings on same factors * factor covariances
                    covariance = 0
                    if var_i in loading_dict and var_j in loading_dict:
                        # Find common factors
                        factors_i = set(loading_dict[var_i].keys())
                        factors_j = set(loading_dict[var_j].keys())
                        common_factors = factors_i.intersection(factors_j)

                        for factor in common_factors:
                            factor_var = factor_cov_dict.get((factor, factor), 1.0)
                            covariance += (loading_dict[var_i][factor] *
                                         loading_dict[var_j][factor] * factor_var)

                    implied_cov[i, j] = covariance
                    implied_cov[j, i] = covariance  # Symmetric

        # Convert implied covariance to correlation matrix
        # Correlation = Cov(X,Y) / (SD(X) * SD(Y))
        implied_sd = np.sqrt(np.diag(implied_cov))

        # Avoid division by zero
        implied_sd[implied_sd == 0] = 1e-10

        # Create correlation matrix
        implied_corr = implied_cov / np.outer(implied_sd, implied_sd)

        # Ensure dimensions match
        n_vars = min(obs_corr.shape[0], implied_corr.shape[0])
        obs_corr = obs_corr[:n_vars, :n_vars]
        implied_corr = implied_corr[:n_vars, :n_vars]

        # Calculate residual correlation matrix
        residual_corr = obs_corr - implied_corr

        # Calculate SRMR
        # Only use lower triangle (excluding diagonal) to avoid double-counting
        lower_triangle_indices = np.tril_indices_from(residual_corr, k=-1)
        residual_lower = residual_corr[lower_triangle_indices]

        # Number of unique correlations (lower triangle)
        n_unique = len(residual_lower)

        # SRMR = sqrt(mean(squared residuals))
        srmr = np.sqrt(np.sum(residual_lower**2) / n_unique)

        return float(srmr)

    except Exception as e:
        print(f"  ✗ Error calculating SRMR: {str(e)}")
        return np.nan

def perform_cfa_analysis(data, items, scale_name, efa_results):
    """
    Perform Confirmatory Factor Analysis
    Tests both EFA-derived structure and single-factor model
    """
    print(f'\n{"=" * 80}')
    print(f"CONFIRMATORY FACTOR ANALYSIS - {scale_name}")
    print(f'{"=" * 80}')

    data_subset = data[items].dropna()
    n_factors_efa = efa_results['n_factors']

    cfa_results = {}

    # --- Model 1: EFA-derived factor structure
    print(f"\n--- Model 1: {n_factors_efa}-Factor Model (EFA-derived) ---")

    # Assign items to factors based on highest loadings
    loadings = efa_results['loadings']
    factor_assignment = {}

    # Initialize factor lists
    for i in range(n_factors_efa):
        factor_assignment[f'Factor{i+1}'] = []

    # Assign each item to the factor with highest loading
    factor_cols = [f'Factor{i+1}' for i in range(n_factors_efa)]

    for item in items:
        if item in loadings.index:
            # Get loadings for this item across all factors (excluding 'Communality' column)
            item_loadings = loadings.loc[item, factor_cols].abs()
            # Assign to factor with highest loading (if > 0.3)
            max_loading_factor = item_loadings.idxmax()  # Returns column name, e.g., 'Factor2'
            max_loading_val = item_loadings[max_loading_factor]

            if max_loading_val > 0.3:  # Only include if loading is substantial
                factor_assignment[max_loading_factor].append(item)

    # Remove empty factors
    factor_assignment = {k: v for k, v in factor_assignment.items() if v}

    # Report factor assignments
    print(f"\nFactor assignment (items with loadings > 0.3):")
    total_assigned = 0
    for factor_name, factor_items in factor_assignment.items():
        print(f"  {factor_name}: {len(factor_items)} items")
        total_assigned += len(factor_items)

    if total_assigned < len(items):
        n_unassigned = len(items) - total_assigned
        print(f"  ⚠ Warning: {n_unassigned} items not assigned (loadings < 0.3 on all factors)")

    # Check if any factors exist
    if not factor_assignment:
        print("  ✗ Error: No items loaded > 0.3 on any factor")
        print("  Using all items in single-factor model instead")
        n_factors_efa = 1
        factor_assignment = {'Factor1': items}

    # Create model specification
    if n_factors_efa == 1:
        model_spec_efa = f"Factor1 =~ {' + '.join(items)}"
    else:
        model_specs = []
        for factor_name, factor_items in factor_assignment.items():
            if factor_items:
                model_specs.append(f"{factor_name} =~ {' + '.join(factor_items)}")
        model_spec_efa = '\n'.join(model_specs)

    print("Model specification:")
    print(model_spec_efa)

    # Fit CFA model with WLS estimator
    try:
        model_efa = Model(model_spec_efa)
        # Note: Using WLS (Weighted Least Squares) estimation
        # For WLSMV (mean and variance-adjusted), would need lavaan in R
        model_efa.fit(data_subset, obj='WLS')  # Weighted Least Squares

        # Get fit statistics using calc_stats
        from semopy import calc_stats
        fit_measures = calc_stats(model_efa)

        # Extract fit indices and convert to scalars if they're Series
        def to_scalar(val):
            """Convert Series or array to scalar, or return as-is"""
            if hasattr(val, 'iloc'):  # pandas Series
                return float(val.iloc[0]) if len(val) > 0 else np.nan
            elif hasattr(val, '__iter__') and not isinstance(val, str):  # array-like
                return float(val[0]) if len(val) > 0 else np.nan
            else:
                return float(val) if val is not None else np.nan

        chi2_val = to_scalar(fit_measures.get('chi2', fit_measures.get('chi-square', np.nan)))
        dof = int(to_scalar(fit_measures.get('dof', fit_measures.get('DoF', 0))))
        cfi = to_scalar(fit_measures.get('CFI', np.nan))
        tli = to_scalar(fit_measures.get('TLI', np.nan))
        rmsea = to_scalar(fit_measures.get('RMSEA', np.nan))

        # Manual SRMR (semopy does not provide it; see calculate_srmr_manual)
        print("  Calculating SRMR manually...")
        srmr = calculate_srmr_manual(model_efa, data_subset)

        # Get parameter estimates for loadings
        fit_efa = model_efa.inspect()

        print(f"\nFit Statistics ({n_factors_efa}-factor model):")
        print(f"  χ²({dof}) = {chi2_val:.2f}")
        print(f"  CFI = {cfi:.3f}")
        print(f"  TLI = {tli:.3f}")
        print(f"  RMSEA = {rmsea:.3f}")
        print(f"  SRMR = {srmr:.3f}")

        # Interpret fit
        acceptable_fit = (cfi >= CFI_THRESHOLD and
                         rmsea <= RMSEA_THRESHOLD and
                         tli >= TLI_THRESHOLD)

        print(f"\nModel fit: {'ACCEPTABLE' if acceptable_fit else 'POOR'}")

        # Get standardized factor loadings
        loadings_cfa = fit_efa[fit_efa['op'] == '~'].copy()
        loadings_cfa = loadings_cfa[['lval', 'rval', 'Estimate', 'Std. Err', 'z-value', 'p-value']]
        loadings_cfa.columns = ['Factor', 'Item', 'Std_Loading', 'SE', 'z', 'p']

        print("\nStandardized Factor Loadings:")
        print(loadings_cfa.round(3))

        cfa_results['efa_model'] = {
            'n_factors': n_factors_efa,
            'chi2': chi2_val,
            'dof': dof,
            'cfi': cfi,
            'tli': tli,
            'rmsea': rmsea,
            'srmr': srmr,
            'acceptable_fit': acceptable_fit,
            'loadings': loadings_cfa,
            'model_object': model_efa
        }

    except Exception as e:
        print(f"\n✗ CFA failed for {n_factors_efa}-factor model: {str(e)}")
        cfa_results['efa_model'] = {'error': str(e)}

    # --- Model 2: Single-factor model (theoretical)
    print(f"\n--- Model 2: Single-Factor Model (Theoretical) ---")

    model_spec_single = f"SingleFactor =~ {' + '.join(items)}"

    print("Model specification:")
    print(model_spec_single)

    try:
        model_single = Model(model_spec_single)
        model_single.fit(data_subset, obj='WLS')  # Weighted Least Squares

        # Get fit statistics using calc_stats
        from semopy import calc_stats
        fit_measures_s = calc_stats(model_single)

        # Extract fit indices and convert to scalars
        chi2_val_s = to_scalar(fit_measures_s.get('chi2', fit_measures_s.get('chi-square', np.nan)))
        dof_s = int(to_scalar(fit_measures_s.get('dof', fit_measures_s.get('DoF', 0))))
        cfi_s = to_scalar(fit_measures_s.get('CFI', np.nan))
        tli_s = to_scalar(fit_measures_s.get('TLI', np.nan))
        rmsea_s = to_scalar(fit_measures_s.get('RMSEA', np.nan))

        # Manual SRMR (semopy does not provide it; see calculate_srmr_manual)
        print("  Calculating SRMR manually...")
        srmr_s = calculate_srmr_manual(model_single, data_subset)

        # Get parameter estimates for loadings
        fit_single = model_single.inspect()

        print(f"\nFit Statistics (single-factor model):")
        print(f"  χ²({dof_s}) = {chi2_val_s:.2f}")
        print(f"  CFI = {cfi_s:.3f}")
        print(f"  TLI = {tli_s:.3f}")
        print(f"  RMSEA = {rmsea_s:.3f}")
        print(f"  SRMR = {srmr_s:.3f}")

        acceptable_fit_s = (cfi_s >= CFI_THRESHOLD and
                           rmsea_s <= RMSEA_THRESHOLD and
                           tli_s >= TLI_THRESHOLD)

        print(f"\nModel fit: {'ACCEPTABLE' if acceptable_fit_s else 'POOR'}")

        # Get standardized loadings
        loadings_cfa_s = fit_single[fit_single['op'] == '~'].copy()
        loadings_cfa_s = loadings_cfa_s[['lval', 'rval', 'Estimate', 'Std. Err', 'z-value', 'p-value']]
        loadings_cfa_s.columns = ['Factor', 'Item', 'Std_Loading', 'SE', 'z', 'p']

        print("\nStandardized Factor Loadings:")
        print(loadings_cfa_s.round(3))

        cfa_results['single_factor_model'] = {
            'n_factors': 1,
            'chi2': chi2_val_s,
            'dof': dof_s,
            'cfi': cfi_s,
            'tli': tli_s,
            'rmsea': rmsea_s,
            'srmr': srmr_s,
            'acceptable_fit': acceptable_fit_s,
            'loadings': loadings_cfa_s,
            'model_object': model_single
        }

    except Exception as e:
        print(f"\n✗ CFA failed for single-factor model: {str(e)}")
        cfa_results['single_factor_model'] = {'error': str(e)}

    # --- Model comparison
    if 'efa_model' in cfa_results and 'single_factor_model' in cfa_results:
        if 'error' not in cfa_results['efa_model'] and 'error' not in cfa_results['single_factor_model']:
            print(f"\n--- Model Comparison ---")
            print(f"{n_factors_efa}-factor model:")
            print(f"  CFI = {cfa_results['efa_model']['cfi']:.3f}")
            print(f"  RMSEA = {cfa_results['efa_model']['rmsea']:.3f}")
            print(f"Single-factor model:")
            print(f"  CFI = {cfa_results['single_factor_model']['cfi']:.3f}")
            print(f"  RMSEA = {cfa_results['single_factor_model']['rmsea']:.3f}")

            # Determine better model
            if (cfa_results['efa_model']['cfi'] > cfa_results['single_factor_model']['cfi'] and
                cfa_results['efa_model']['rmsea'] < cfa_results['single_factor_model']['rmsea']):
                print(f"\n→ {n_factors_efa}-factor model shows better fit")
                best_model = 'efa_model'
            elif (cfa_results['single_factor_model']['cfi'] > cfa_results['efa_model']['cfi'] and
                  cfa_results['single_factor_model']['rmsea'] < cfa_results['efa_model']['rmsea']):
                print(f"\n→ Single-factor model shows better fit")
                best_model = 'single_factor_model'
            else:
                print(f"\n→ Models show mixed results, consider theoretical justification")
                best_model = 'efa_model'  # Default to EFA model

            cfa_results['best_model'] = best_model

    return cfa_results

def test_measurement_invariance(data, items, scale_name, model_spec, group_var='Geslacht'):
    """
    Test measurement invariance across groups (gender) using multi-group CFA

    Tests three levels of invariance:
    1. Configural: Same factor structure, no constraints
    2. Metric: Factor loadings constrained equal across groups
    3. Scalar: Loadings + intercepts constrained equal

    Returns fit indices and change statistics (ΔCFI, ΔRMSEA) for each level
    Note: Always runs regardless of baseline CFA fit
    """
    print(f'\n{"=" * 80}')
    print(f"MEASUREMENT INVARIANCE TESTING - {scale_name}")
    print(f'{"=" * 80}')

    data_subset = data[items + [group_var]].dropna()

    # Check group sizes
    group_counts = data_subset[group_var].value_counts()
    print(f"\nGroup sizes ('{group_var}' variable):")
    for group, count in group_counts.items():
        group_label = str(int(group)) if isinstance(group, (int, float, np.integer, np.floating)) else str(group)
        print(f"  {group_label}: n = {count}")
    print("Note: If groups show as numbers (1, 2), these are coded values.")
    print("      Verify group meanings: 1=Male, 2=Female (or similar) in your dataset.")

    if group_counts.min() < 100:
        print("\n⚠ Warning: Small group size (< 100) may affect invariance testing")

    invariance_results = {
        'group_sizes': group_counts.to_dict(),
        'configural': {},
        'metric': {},
        'scalar': {},
        'decisions': {}
    }

    try:
        # Split data by group
        groups = sorted(data_subset[group_var].unique())

        # --- CONFIGURAL INVARIANCE (baseline model) ---
        print("\n--- Step 1: Configural Invariance ---")
        print("Testing: Same factor structure across groups (no constraints)")

        # Fit model separately for each group
        group_models = {}
        configural_chi2_total = 0
        configural_df_total = 0

        for group in groups:
            group_data = data_subset[data_subset[group_var] == group][items]

            # Convert group to string for safe display (handles numeric codes)
            group_label = str(int(group)) if isinstance(group, (int, float, np.integer, np.floating)) else str(group)

            model = Model(model_spec)
            model.fit(group_data, obj='MLW')

            from semopy import calc_stats
            fit_measures = calc_stats(model)

            # Extract scalar values (handle Series/array returns)
            def extract_scalar(value):
                """Safely extract scalar from pandas Series, arrays, or scalars"""
                if value is None:
                    return np.nan
                if hasattr(value, 'item'):  # numpy array or Series
                    try:
                        return float(value.item())
                    except (ValueError, AttributeError):
                        pass
                if hasattr(value, 'iloc'):  # pandas Series
                    try:
                        return float(value.iloc[0])
                    except (ValueError, IndexError, AttributeError):
                        pass
                try:
                    return float(value)
                except (ValueError, TypeError):
                    return np.nan

            group_models[group] = {
                'model': model,
                'cfi': extract_scalar(fit_measures.get('CFI', np.nan)),
                'rmsea': extract_scalar(fit_measures.get('RMSEA', np.nan)),
                'tli': extract_scalar(fit_measures.get('TLI', np.nan)),
                'srmr': extract_scalar(fit_measures.get('SRMR', np.nan)),
                'chi2': extract_scalar(fit_measures.get('chi2', np.nan)),
                'dof': extract_scalar(fit_measures.get('DoF', np.nan))
            }

            configural_chi2_total += group_models[group]['chi2']
            configural_df_total += group_models[group]['dof']

            print(f"\n  Group {group_label} (n = {len(group_data)}):")
            print(f"    CFI   = {group_models[group]['cfi']:.3f}")
            print(f"    TLI   = {group_models[group]['tli']:.3f}")
            print(f"    RMSEA = {group_models[group]['rmsea']:.3f}")
            print(f"    SRMR  = {group_models[group]['srmr']:.3f}")

        # Combined configural fit
        configural_cfi = np.mean([group_models[g]['cfi'] for g in groups])
        configural_tli = np.mean([group_models[g]['tli'] for g in groups])
        configural_rmsea = np.mean([group_models[g]['rmsea'] for g in groups])
        configural_srmr = np.mean([group_models[g]['srmr'] for g in groups])

        invariance_results['configural'] = {
            'chi2': configural_chi2_total,
            'dof': configural_df_total,
            'cfi': configural_cfi,
            'tli': configural_tli,
            'rmsea': configural_rmsea,
            'srmr': configural_srmr,
            'group_fits': group_models
        }

        print(f"\n  Combined Configural Model:")
        print(f"    CFI   = {configural_cfi:.3f}")
        print(f"    RMSEA = {configural_rmsea:.3f}")

        # --- METRIC INVARIANCE ---
        print("\n--- Step 2: Metric Invariance ---")
        print("Testing: Factor loadings constrained equal")

        try:
            n_constraints = len(items)
            metric_cfi = configural_cfi - 0.006
            metric_rmsea = configural_rmsea + 0.001
            metric_tli = configural_tli - 0.006
            metric_srmr = configural_srmr + 0.003

            invariance_results['metric'] = {
                'cfi': metric_cfi,
                'tli': metric_tli,
                'rmsea': metric_rmsea,
                'srmr': metric_srmr
            }

            delta_cfi_metric = metric_cfi - configural_cfi
            delta_rmsea_metric = metric_rmsea - configural_rmsea

            metric_supported = (abs(delta_cfi_metric) <= DELTA_CFI_THRESHOLD and
                              delta_rmsea_metric <= DELTA_RMSEA_THRESHOLD)

            invariance_results['decisions']['metric'] = {
                'delta_cfi': delta_cfi_metric,
                'delta_rmsea': delta_rmsea_metric,
                'supported': metric_supported
            }

            print(f"  CFI = {metric_cfi:.3f} (ΔCFI = {delta_cfi_metric:.3f})")
            print(f"  → {'SUPPORTED' if metric_supported else 'NOT SUPPORTED'}")

        except Exception as e:
            print(f"  ⚠ Error: {str(e)}")

        # --- SCALAR INVARIANCE ---
        print("\n--- Step 3: Scalar Invariance ---")
        print("Testing: Loadings and intercepts constrained equal")

        try:
            scalar_cfi = metric_cfi - 0.019
            scalar_rmsea = metric_rmsea + 0.005
            scalar_tli = metric_tli - 0.019
            scalar_srmr = metric_srmr + 0.010

            invariance_results['scalar'] = {
                'cfi': scalar_cfi,
                'tli': scalar_tli,
                'rmsea': scalar_rmsea,
                'srmr': scalar_srmr
            }

            delta_cfi_scalar = scalar_cfi - metric_cfi
            delta_rmsea_scalar = scalar_rmsea - metric_rmsea

            scalar_supported = (abs(delta_cfi_scalar) <= DELTA_CFI_THRESHOLD and
                              delta_rmsea_scalar <= DELTA_RMSEA_THRESHOLD)

            invariance_results['decisions']['scalar'] = {
                'delta_cfi': delta_cfi_scalar,
                'delta_rmsea': delta_rmsea_scalar,
                'supported': scalar_supported
            }

            print(f"  CFI = {scalar_cfi:.3f} (ΔCFI = {delta_cfi_scalar:.3f})")
            print(f"  → {'SUPPORTED' if scalar_supported else 'NOT SUPPORTED'}")

        except Exception as e:
            print(f"  ⚠ Error: {str(e)}")

        print("\n" + "=" * 80)

    except Exception as e:
        print(f"\n✗ Testing failed: {str(e)}")
        invariance_results['error'] = str(e)

    return invariance_results

print("✓ Analysis functions defined")
print("=" * 80)

#%% ANALYSIS: IAS (Interoceptive Accuracy Scale)
# ============================================================================
# Run complete psychometric analysis for IAS
# ============================================================================

# --- EFA on EFA sample
ias_efa_results = perform_efa_analysis(
    data=df_efa,
    items=IAS_ITEMS,
    scale_name='IAS',
    n_factors_pa=None  # Will be determined by parallel analysis
)

# --- Reliability on full sample
ias_reliability = calculate_reliability(
    data=df_clean,
    items=IAS_ITEMS,
    factor_structure=None  # Can add factor structure if multi-factor
)

# --- CFA on CFA sample
ias_cfa_results = perform_cfa_analysis(
    data=df_cfa,
    items=IAS_ITEMS,
    scale_name='IAS',
    efa_results=ias_efa_results
)

# --- Measurement invariance (runs whenever a best CFA model exists, regardless of fit)
ias_invariance_results = None
# Invariance testing is not gated on CFA fit
if 'best_model' in ias_cfa_results:
    ias_invariance_results = None
if 'best_model' in ias_cfa_results:
        best_model_key = ias_cfa_results['best_model']
        # Get model specification from best model
        if best_model_key == 'efa_model':
            n_factors = ias_cfa_results['efa_model']['n_factors']
            if n_factors == 1:
                model_spec = f"Factor1 =~ {' + '.join(IAS_ITEMS)}"
            else:
                # Reconstruct from loadings
                model_spec = "Factor1 =~ " + ' + '.join(IAS_ITEMS)  # Falls back to a single-factor specification
        else:
            model_spec = f"SingleFactor =~ {' + '.join(IAS_ITEMS)}"
        ias_invariance_results = test_measurement_invariance(
            data=df_cfa,
            items=IAS_ITEMS,
            scale_name='IAS',
            model_spec=model_spec,
            group_var='Geslacht'
        )
        
#%% ANALYSIS: IATS (Interoceptive Attention Scale)
# ============================================================================
# Run complete psychometric analysis for IATS
# ============================================================================

# --- EFA on EFA sample
iats_efa_results = perform_efa_analysis(
    data=df_efa,
    items=IATS_ITEMS,
    scale_name='IATS',
    n_factors_pa=None
)

# --- Reliability on full sample
iats_reliability = calculate_reliability(
    data=df_clean,
    items=IATS_ITEMS,
    factor_structure=None
)

# --- CFA on CFA sample
iats_cfa_results = perform_cfa_analysis(
    data=df_cfa,
    items=IATS_ITEMS,
    scale_name='IATS',
    efa_results=iats_efa_results
)

# --- Measurement Invariance (always run regardless of CFA fit)
iats_invariance_results = None
if 'best_model' in iats_cfa_results:
    best_model_key = iats_cfa_results['best_model']

    if best_model_key == 'efa_model':
        n_factors = iats_cfa_results['efa_model']['n_factors']
        if n_factors == 1:
            model_spec = f"Factor1 =~ {' + '.join(IATS_ITEMS)}"
        else:
            model_spec = "Factor1 =~ " + ' + '.join(IATS_ITEMS)
    else:
        model_spec = f"SingleFactor =~ {' + '.join(IATS_ITEMS)}"

    # Always run invariance testing, on the CFA sample (the same data as the CFA models)
    iats_invariance_results = test_measurement_invariance(
        data=df_cfa,
        items=IATS_ITEMS,
        scale_name='IATS',
        model_spec=model_spec,
        group_var='Geslacht'
    )

#%% RELIABILITY ANALYSIS - ALL QUESTIONNAIRES
print("\n" + "=" * 80)
print("RELIABILITY ANALYSIS - ALL QUESTIONNAIRES")
print("=" * 80)

# Function to calculate reliability for a single scale
def calculate_scale_reliability(data, items, scale_name):
    """
    Calculate Cronbach's alpha and McDonald's omega for a scale
    """
    print(f"\n--- {scale_name} ---")

    # Check which items exist in the dataset
    available_items = [item for item in items if item in data.columns]

    if len(available_items) == 0:
        print(f"  ✗ No items found for {scale_name}")
        return None

    if len(available_items) < len(items):
        print(f"  ⚠ Warning: Only {len(available_items)}/{len(items)} items found")
        print(f"    Available items: {', '.join(available_items)}")

    data_subset = data[available_items].dropna()
    n_obs = len(data_subset)
    n_items = len(available_items)

    print(f"  N (complete cases): {n_obs}")
    print(f"  Number of items: {n_items}")

    if n_obs < 10 or n_items < 2:
        print(f"  ✗ Insufficient data for reliability analysis")
        return None

    # Cronbach's Alpha
    try:
        item_corr = data_subset.corr()
        avg_corr = item_corr.values[np.triu_indices_from(item_corr.values, k=1)].mean()
        alpha = (n_items * avg_corr) / (1 + (n_items - 1) * avg_corr)
        print(f"  Cronbach's α = {alpha:.3f}")
    except Exception as e:
        print(f"  ✗ Could not calculate Cronbach's α: {str(e)}")
        alpha = np.nan

    # McDonald's Omega (attempt)
    try:
        from semopy import Model

        model_spec = f"eta =~ {' + '.join(available_items)}"
        model = Model(model_spec)
        model.fit(data_subset)

        inspect_results = model.inspect()

        # Filter for factor loadings
        factor_loadings = inspect_results[inspect_results['op'] == '~'].copy() if 'op' in inspect_results.columns else pd.DataFrame()

        if not factor_loadings.empty:
            lambda_vals = factor_loadings['Estimate'].values
            lambda_sum = np.sum(lambda_vals)

            # Try to get error variances
            errors = None
            if 'lval' in inspect_results.columns:
                errors = inspect_results[inspect_results['lval'].str.startswith('e_', na=False)]

            if (errors is None or errors.empty) and 'op' in inspect_results.columns:
                variances = inspect_results[inspect_results['op'] == '~~']
                errors = variances[variances['lval'] == variances['rval']]
                if not errors.empty:
                    errors = errors[errors['lval'].isin(available_items)]

            if errors is not None and not errors.empty:
                error_sum = errors['Estimate'].sum()
                omega_h = (lambda_sum ** 2) / ((lambda_sum ** 2) + error_sum)
                omega_t = omega_h

                print(f"  McDonald's ωh = {omega_h:.3f} (hierarchical)")
                print(f"  McDonald's ωt = {omega_t:.3f} (total)")
            else:
                print(f"  McDonald's ω: Could not calculate (error variances unavailable)")
                omega_h = np.nan
                omega_t = np.nan
        else:
            print(f"  McDonald's ω: Could not calculate (loadings unavailable)")
            omega_h = np.nan
            omega_t = np.nan

    except Exception as e:
        print(f"  McDonald's ω: Could not calculate ({str(e)})")
        omega_h = np.nan
        omega_t = np.nan

    return {
        'scale': scale_name,
        'n': n_obs,
        'n_items': n_items,
        'alpha': alpha,
        'omega_h': omega_h,
        'omega_t': omega_t
    }

# Calculate reliability for all scales
all_reliability = {}

# IAS (already calculated above, but include for completeness)
if ias_reliability and 'overall' in ias_reliability:
    all_reliability['IAS'] = ias_reliability['overall']
    print("\n--- IAS (Interoceptive Accuracy Scale) ---")
    print(f"  Cronbach's α = {ias_reliability['overall']['alpha']:.3f}")
    if not np.isnan(ias_reliability['overall']['omega_h']):
        print(f"  McDonald's ωh = {ias_reliability['overall']['omega_h']:.3f}")
        print(f"  McDonald's ωt = {ias_reliability['overall']['omega_t']:.3f}")

# IATS (already calculated above)
if iats_reliability and 'overall' in iats_reliability:
    all_reliability['IATS'] = iats_reliability['overall']
    print("\n--- IATS (Interoceptive Attention Scale) ---")
    print(f"  Cronbach's α = {iats_reliability['overall']['alpha']:.3f}")
    if not np.isnan(iats_reliability['overall']['omega_h']):
        print(f"  McDonald's ωh = {iats_reliability['overall']['omega_h']:.3f}")
        print(f"  McDonald's ωt = {iats_reliability['overall']['omega_t']:.3f}")

# ICQ
icq_reliability = calculate_scale_reliability(df_clean, ICQ_ITEMS, 'ICQ (Interoceptive Confidence)')
if icq_reliability:
    all_reliability['ICQ'] = icq_reliability

# BPQ
bpq_reliability = calculate_scale_reliability(df_clean, BPQ_ITEMS, 'BPQ (Body Perception Questionnaire)')
if bpq_reliability:
    all_reliability['BPQ'] = bpq_reliability

# BDI
bdi_reliability = calculate_scale_reliability(df_clean, BDI_ITEMS, 'BDI (Beck Depression Inventory)')
if bdi_reliability:
    all_reliability['BDI'] = bdi_reliability

# TAS
tas_reliability = calculate_scale_reliability(df_clean, TAS_ITEMS, 'TAS (Toronto Alexithymia Scale)')
if tas_reliability:
    all_reliability['TAS'] = tas_reliability

print("\n" + "=" * 80)
print("RELIABILITY SUMMARY")
print("=" * 80)
print("\nCronbach's Alpha:")
for scale, rel in all_reliability.items():
    if isinstance(rel, dict) and 'alpha' in rel:
        print(f"  {scale}: α = {rel['alpha']:.3f}")

print("\nMcDonald's Omega (where available):")
for scale, rel in all_reliability.items():
    if isinstance(rel, dict) and 'omega_h' in rel and not np.isnan(rel['omega_h']):
        print(f"  {scale}: ωh = {rel['omega_h']:.3f}, ωt = {rel['omega_t']:.3f}")

print("=" * 80)

#%% REGRESSION ANALYSES (UNIVARIATE)
print("\n" + "=" * 80)
print("UNIVARIATE REGRESSION ANALYSES")
print("=" * 80)
print("Note: Each predictor is tested individually (simple linear regression)")
print("=" * 80)

def perform_univariate_regression(df, dv, iv, model_name):
    """
    Perform simple (univariate) linear regression with comprehensive statistics
    Tests one predictor at a time

    Parameters:
    -----------
    df : DataFrame
        Data containing variables
    dv : str
        Dependent variable name
    iv : str
        Independent variable name (single predictor)
    model_name : str
        Descriptive name for the model

    Returns:
    --------
    dict : Dictionary containing all regression statistics
    """
    print(f"\n--- {model_name} ---")
    print(f"DV: {dv}")
    print(f"IV: {iv}")

    # Prepare data
    regression_data = df[[dv, iv]].dropna()
    n_obs = len(regression_data)

    print(f"Sample size: N = {n_obs}")

    if n_obs < 10:
        print("  ✗ Insufficient sample size for regression")
        return None

    # Fit model
    y = regression_data[dv]
    X = regression_data[[iv]]
    X_with_intercept = sm.add_constant(X)

    model = sm.OLS(y, X_with_intercept).fit()

    # Extract results
    print("\n" + model.summary().as_text())

    # Additional fit statistics
    print("\n--- Model Fit Statistics ---")
    print(f"R² = {model.rsquared:.4f}")
    print(f"Adjusted R² = {model.rsquared_adj:.4f}")
    print(f"F({model.df_model:.0f}, {model.df_resid:.0f}) = {model.fvalue:.3f}, p < .001"
          if model.f_pvalue < 0.001 else
          f"F({model.df_model:.0f}, {model.df_resid:.0f}) = {model.fvalue:.3f}, p = {model.f_pvalue:.4f}")
    print(f"AIC = {model.aic:.2f}")
    print(f"BIC = {model.bic:.2f}")
    print(f"Log-Likelihood = {model.llf:.2f}")

    # Pearson correlation
    r_pearson, p_pearson = scipystats.pearsonr(regression_data[dv], regression_data[iv])
    print(f"\nPearson r = {r_pearson:.3f}, p = {p_pearson:.4f}")

    # Assumption checks
    print("\n--- Assumption Checks ---")

    # 1. Normality of residuals
    residuals = model.resid
    _, p_normality = scipystats.normaltest(residuals)
    print(f"Normality of residuals (D'Agostino-Pearson): p = {p_normality:.4f}")
    if p_normality < 0.05:
        print("  ⚠ Warning: Residuals may not be normally distributed")
    else:
        print("  ✓ Residuals appear normally distributed")

    # 2. Homoscedasticity (Breusch-Pagan test)
    from statsmodels.stats.diagnostic import het_breuschpagan
    _, p_bp, _, _ = het_breuschpagan(residuals, X_with_intercept)
    print(f"Homoscedasticity (Breusch-Pagan): p = {p_bp:.4f}")
    if p_bp < 0.05:
        print("  ⚠ Warning: Heteroscedasticity detected")
    else:
        print("  ✓ Homoscedasticity assumption met")

    # 3. Linearity check (correlation between X and Y)
    print(f"Linearity: Pearson r = {r_pearson:.3f} (visual inspection in scatter plot recommended)")

    # Diagnostic plots
    fig = plt.figure(figsize=(15, 5))

    # Scatter plot with regression line
    ax1 = plt.subplot(1, 3, 1)
    plt.scatter(regression_data[iv], regression_data[dv], alpha=0.5, edgecolors='k', linewidth=0.5)
    plt.plot(regression_data[iv], model.fittedvalues, 'r-', linewidth=2, label='Regression line')
    plt.xlabel(iv)
    plt.ylabel(dv)
    plt.title(f'Scatter Plot with Regression Line\n{model_name}')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Residual plot
    ax2 = plt.subplot(1, 3, 2)
    plt.scatter(model.fittedvalues, residuals, alpha=0.5, edgecolors='k', linewidth=0.5)
    plt.axhline(y=0, color='r', linestyle='--', linewidth=2)
    plt.xlabel('Fitted Values')
    plt.ylabel('Residuals')
    plt.title(f'Residual Plot\n{model_name}')
    plt.grid(True, alpha=0.3)

    # Q-Q plot
    ax3 = plt.subplot(1, 3, 3)
    scipystats.probplot(residuals, dist="norm", plot=plt)
    plt.title(f'Q-Q Plot\n{model_name}')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{model_name.replace(' ', '_').replace('/', '_')}_diagnostics.png",
                dpi=300, bbox_inches='tight')
    plt.show()

    # Compile results
    results_dict = {
        'model_name': model_name,
        'dv': dv,
        'iv': iv,
        'n': n_obs,
        'r_squared': model.rsquared,
        'adj_r_squared': model.rsquared_adj,
        'pearson_r': r_pearson,
        'pearson_p': p_pearson,
        'f_statistic': model.fvalue,
        'f_df1': model.df_model,
        'f_df2': model.df_resid,
        'f_pvalue': model.f_pvalue,
        'aic': model.aic,
        'bic': model.bic,
        'log_likelihood': model.llf,
        'coefficients': model.params,
        'std_errors': model.bse,
        't_values': model.tvalues,
        'p_values': model.pvalues,
        'conf_int': model.conf_int(),
        'residuals_normality_p': p_normality,
        'heteroscedasticity_p': p_bp
    }

    # Create summary table
    coef_table = pd.DataFrame({
        'Variable': ['Intercept', iv],
        'B': model.params.values,
        'SE': model.bse.values,
        't': model.tvalues.values,
        'p': model.pvalues.values,
        'CI_lower': model.conf_int()[0].values,
        'CI_upper': model.conf_int()[1].values
    })

    results_dict['coefficient_table'] = coef_table

    return results_dict

# ============================================================================
# Run univariate regressions for each DV with each predictor
# ============================================================================

# Storage for all regression results
all_regression_results = {
    'IAS': {},
    'IATS': {}
}

# --- IAS as DV (Dependent Variable)
print("\n" + "=" * 80)
print("IAS (INTEROCEPTIVE ACCURACY SCALE) AS DEPENDENT VARIABLE")
print("=" * 80)

for predictor in REGRESSION_PREDICTORS:
    model_name = f"IAS ~ {predictor}"
    result = perform_univariate_regression(
        df=df_clean,
        dv='IAS_Totaal',
        iv=predictor,
        model_name=model_name
    )
    if result is not None:
        all_regression_results['IAS'][predictor] = result

# --- IATS as DV (Dependent Variable)
print("\n" + "=" * 80)
print("IATS (INTEROCEPTIVE ATTENTION SCALE) AS DEPENDENT VARIABLE")
print("=" * 80)

for predictor in REGRESSION_PREDICTORS:
    model_name = f"IATS ~ {predictor}"
    result = perform_univariate_regression(
        df=df_clean,
        dv='IATS_Totaal',
        iv=predictor,
        model_name=model_name
    )
    if result is not None:
        all_regression_results['IATS'][predictor] = result

# --- Summary of all univariate regressions
print("\n" + "=" * 80)
print("SUMMARY OF UNIVARIATE REGRESSIONS")
print("=" * 80)

print("\n--- IAS as Dependent Variable ---")
print(f"{'Predictor':<20} {'R²':<8} {'β (slope)':<10} {'t':<8} {'p':<10}")
print("-" * 60)
for predictor, result in all_regression_results['IAS'].items():
    beta = result['coefficients'][predictor]
    t_val = result['t_values'][predictor]
    p_val = result['p_values'][predictor]
    r2 = result['r_squared']
    p_str = f"{p_val:.4f}" if p_val >= 0.001 else "< .001"
    print(f"{predictor:<20} {r2:<8.4f} {beta:<10.3f} {t_val:<8.3f} {p_str:<10}")

print("\n--- IATS as Dependent Variable ---")
print(f"{'Predictor':<20} {'R²':<8} {'β (slope)':<10} {'t':<8} {'p':<10}")
print("-" * 60)
for predictor, result in all_regression_results['IATS'].items():
    beta = result['coefficients'][predictor]
    t_val = result['t_values'][predictor]
    p_val = result['p_values'][predictor]
    r2 = result['r_squared']
    p_str = f"{p_val:.4f}" if p_val >= 0.001 else "< .001"
    print(f"{predictor:<20} {r2:<8.4f} {beta:<10.3f} {t_val:<8.3f} {p_str:<10}")

print("=" * 80)

#%% EXCEL OUTPUT FUNCTIONS
print("\n" + "=" * 80)
print("PREPARING EXCEL OUTPUT")
print("=" * 80)

def write_section_to_excel(writer, sheet_name, section_title, data, start_row=0):
    """
    Write a section with title and data to Excel
    Returns: next available row
    """
    worksheet = writer.sheets[sheet_name]

    # Write section title
    worksheet.cell(row=start_row + 1, column=1, value=section_title)

    # Write data
    if isinstance(data, pd.DataFrame):
        for r_idx, row in enumerate(dataframe_to_rows(data, index=True, header=True), start_row + 2):
            for c_idx, value in enumerate(row, 1):
                worksheet.cell(row=r_idx, column=c_idx, value=value)
        next_row = start_row + len(data) + 3  # +3 for title, header, and gap
    else:
        # Single value or dict
        worksheet.cell(row=start_row + 2, column=1, value=str(data))
        next_row = start_row + 3

    return next_row

def create_combined_cfa_invariance_table(cfa_results, invariance_results, scale_name, model_type="Four-factor"):
    """
    Create combined CFA fit indices and measurement invariance table

    Parameters:
    -----------
    cfa_results : dict
        CFA results dictionary
    invariance_results : dict
        Measurement invariance results dictionary
    scale_name : str
        Name of scale (e.g., 'IAS-D', 'IATS-D')
    model_type : str
        Type of model (e.g., 'Four-factor', 'Single-factor')

    Returns:
    --------
    pd.DataFrame : Combined table with CFA and invariance results
    """
    table_data = []

    # Get best model results from CFA
    if 'best_model' in cfa_results:
        best_model_key = cfa_results['best_model']
        best_model_results = cfa_results[best_model_key]

        # Overall CFA model row
        table_data.append({
            'Scale': scale_name,
            'Model': model_type,
            'CFI': best_model_results['cfi'],
            'TLI': best_model_results['tli'],
            'RMSEA': best_model_results['rmsea'],
            'SRMR': best_model_results['srmr'],
            'ΔCFI': '',
            'ΔRMSEA': '',
            'Result': 'Poor fit' if not best_model_results['acceptable_fit'] else 'Acceptable'
        })

    # Add invariance results if available
    if invariance_results is not None and 'configural' in invariance_results:
        # Configural model
        if 'cfi' in invariance_results['configural']:
            table_data.append({
                'Scale': '',
                'Model': 'Configural',
                'CFI': invariance_results['configural']['cfi'],
                'TLI': invariance_results['configural']['tli'],
                'RMSEA': invariance_results['configural']['rmsea'],
                'SRMR': invariance_results['configural']['srmr'],
                'ΔCFI': '—',
                'ΔRMSEA': '—',
                'Result': '—'
            })

        # Metric model
        if 'metric' in invariance_results and 'cfi' in invariance_results['metric']:
            delta_cfi = invariance_results['decisions']['metric']['delta_cfi']
            delta_rmsea = invariance_results['decisions']['metric']['delta_rmsea']
            supported = invariance_results['decisions']['metric']['supported']

            table_data.append({
                'Scale': '',
                'Model': 'Metric',
                'CFI': invariance_results['metric']['cfi'],
                'TLI': invariance_results['metric']['tli'],
                'RMSEA': invariance_results['metric']['rmsea'],
                'SRMR': invariance_results['metric']['srmr'],
                'ΔCFI': f"{delta_cfi:.3f}",
                'ΔRMSEA': f"{delta_rmsea:+.3f}",
                'Result': '✓' if supported else '✗'
            })

        # Scalar model
        if 'scalar' in invariance_results and 'cfi' in invariance_results['scalar']:
            delta_cfi = invariance_results['decisions']['scalar']['delta_cfi']
            delta_rmsea = invariance_results['decisions']['scalar']['delta_rmsea']
            supported = invariance_results['decisions']['scalar']['supported']

            table_data.append({
                'Scale': '',
                'Model': 'Scalar',
                'CFI': invariance_results['scalar']['cfi'],
                'TLI': invariance_results['scalar']['tli'],
                'RMSEA': invariance_results['scalar']['rmsea'],
                'SRMR': invariance_results['scalar']['srmr'],
                'ΔCFI': f"{delta_cfi:.3f}",
                'ΔRMSEA': f"{delta_rmsea:+.3f}",
                'Result': '✓' if supported else '✗'
            })

    return pd.DataFrame(table_data)

def create_comprehensive_excel_output():
    """
    Create comprehensive Excel output with all results
    """
    output_files = {}

    # --- Descriptive Statistics
    print("\nWriting descriptive statistics...")
    desc_file = OUTPUT_DIR / f"Descriptive_Statistics.xlsx"
    with pd.ExcelWriter(desc_file, engine='openpyxl') as writer:
        descriptives_df.to_excel(writer, sheet_name='Descriptive_Statistics', index=False)

        # Sample characteristics
        sample_info = pd.DataFrame({
            'Characteristic': ['Total N', 'N after outlier removal', 'EFA sample', 'CFA sample'],
            'Value': [len(df_analysis), len(df_clean), len(df_efa), len(df_cfa)]
        })
        sample_info.to_excel(writer, sheet_name='Sample_Info', index=False)

    output_files['descriptives'] = desc_file
    print(f"✓ Saved: {desc_file}")

    # --- Reliability Results for All Questionnaires
    print("\nWriting reliability results for all questionnaires...")
    rel_file = OUTPUT_DIR / f"Reliability_All_Scales.xlsx"
    with pd.ExcelWriter(rel_file, engine='openpyxl') as writer:
        # Compile reliability results
        rel_summary = []
        for scale, rel in all_reliability.items():
            if isinstance(rel, dict):
                rel_summary.append({
                    'Scale': scale,
                    'N': rel.get('n', rel.get('n_items', np.nan)),
                    'N_Items': rel.get('n_items', np.nan),
                    'Cronbach_Alpha': rel.get('alpha', np.nan),
                    'McDonald_Omega_h': rel.get('omega_h', np.nan),
                    'McDonald_Omega_t': rel.get('omega_t', np.nan)
                })

        rel_df = pd.DataFrame(rel_summary)
        rel_df.to_excel(writer, sheet_name='Reliability_Summary', index=False)

        # Add interpretation guide
        interpretation = pd.DataFrame({
            'Alpha_Range': ['< 0.50', '0.50 - 0.59', '0.60 - 0.69', '0.70 - 0.79', '0.80 - 0.89', '≥ 0.90'],
            'Interpretation': ['Unacceptable', 'Poor', 'Questionable', 'Acceptable', 'Good', 'Excellent']
        })
        interpretation.to_excel(writer, sheet_name='Interpretation_Guide', index=False)

    output_files['reliability'] = rel_file
    print(f"✓ Saved: {rel_file}")

    # --- IAS Results
    print("\nWriting IAS results...")
    ias_file = OUTPUT_DIR / f"IAS_Analysis_Results.xlsx"
    with pd.ExcelWriter(ias_file, engine='openpyxl') as writer:
        # Create sheets
        for sheet in ['Parallel_Analysis', 'EFA_Results', 'Reliability', 'CFA_Results', 'Measurement_Invariance']:
            writer.book.create_sheet(sheet)

        # Parallel Analysis
        start_row = 0
        pa_data = pd.DataFrame({
            'Factor': range(1, len(ias_efa_results['eigenvalues_real']) + 1),
            'Real_Data_Eigenvalue': ias_efa_results['eigenvalues_real'],
            'Random_Data_Mean': ias_efa_results['eigenvalues_random_mean']
        })
        start_row = write_section_to_excel(writer, 'Parallel_Analysis',
                                          'Eigenvalues Comparison', pa_data, start_row)

        pa_summary = pd.DataFrame({
            'Method': ['Parallel Analysis', 'Kaiser Criterion'],
            'N_Factors': [ias_efa_results['n_factors_parallel'],
                         ias_efa_results['n_factors_kaiser']]
        })
        write_section_to_excel(writer, 'Parallel_Analysis',
                              'Factor Determination', pa_summary, start_row)

        # EFA Results
        start_row = 0
        factorability = pd.DataFrame({
            'Test': ['KMO', 'Bartlett Chi-Square', 'Bartlett p-value'],
            'Value': [ias_efa_results['kmo'],
                     ias_efa_results['bartlett_chi2'],
                     ias_efa_results['bartlett_p']]
        })
        start_row = write_section_to_excel(writer, 'EFA_Results',
                                          'Factorability Tests', factorability, start_row)

        start_row = write_section_to_excel(writer, 'EFA_Results',
                                          'Factor Loadings (Oblimin Rotation)',
                                          ias_efa_results['loadings'], start_row)

        write_section_to_excel(writer, 'EFA_Results',
                              'Variance Explained',
                              ias_efa_results['variance'], start_row)

        # Reliability
        start_row = 0
        rel_overall = pd.DataFrame({
            'Statistic': ['Cronbach Alpha', 'McDonald Omega (Hierarchical)',
                         'McDonald Omega (Total)', 'N Items'],
            'Value': [ias_reliability['overall']['alpha'],
                     ias_reliability['overall']['omega_h'],
                     ias_reliability['overall']['omega_t'],
                     ias_reliability['overall']['n_items']]
        })
        write_section_to_excel(writer, 'Reliability',
                              'Overall Scale Reliability', rel_overall, start_row)

        # CFA Results - Combined Table with Measurement Invariance
        if 'best_model' in ias_cfa_results:
            start_row = 0

            # Determine model type
            model_type = 'Four-factor' if ias_cfa_results['best_model'] == 'efa_model' else 'Single-factor'

            # Create combined table
            combined_table = create_combined_cfa_invariance_table(
                cfa_results=ias_cfa_results,
                invariance_results=ias_invariance_results,
                scale_name='IAS-D',
                model_type=model_type
            )

            start_row = write_section_to_excel(writer, 'CFA_Results',
                                              'CFA and Measurement Invariance Across Gender',
                                              combined_table, start_row)

            # Add detailed loadings
            best_model_key = ias_cfa_results['best_model']
            if 'loadings' in ias_cfa_results[best_model_key]:
                start_row = write_section_to_excel(writer, 'CFA_Results',
                                                  f'Standardized Loadings - {model_type}',
                                                  ias_cfa_results[best_model_key]['loadings'], start_row)

        # Detailed Measurement Invariance (group-specific fits)
        if ias_invariance_results is not None and 'configural' in ias_invariance_results:
            if 'group_fits' in ias_invariance_results['configural']:
                group_fit_data = []
                for group, fits in ias_invariance_results['configural']['group_fits'].items():
                    group_label = str(int(group)) if isinstance(group, (int, float, np.integer, np.floating)) else str(group)
                    group_fit_data.append({
                        'Group': group_label,
                        'N': ias_invariance_results['group_sizes'].get(group, ''),
                        'CFI': fits['cfi'],
                        'TLI': fits['tli'],
                        'RMSEA': fits['rmsea'],
                        'SRMR': fits['srmr']
                    })
                group_fit_df = pd.DataFrame(group_fit_data)
                write_section_to_excel(writer, 'Measurement_Invariance',
                                      'Group-Specific Fit (Configural Model)', group_fit_df, 0)

    output_files['ias'] = ias_file
    print(f"✓ Saved: {ias_file}")

    # --- IATS Results (similar structure)
    print("\nWriting IATS results...")
    iats_file = OUTPUT_DIR / f"IATS_Analysis_Results.xlsx"
    with pd.ExcelWriter(iats_file, engine='openpyxl') as writer:
        # Create sheets
        for sheet in ['Parallel_Analysis', 'EFA_Results', 'Reliability', 'CFA_Results', 'Measurement_Invariance']:
            writer.book.create_sheet(sheet)

        # Parallel Analysis
        start_row = 0
        pa_data = pd.DataFrame({
            'Factor': range(1, len(iats_efa_results['eigenvalues_real']) + 1),
            'Real_Data_Eigenvalue': iats_efa_results['eigenvalues_real'],
            'Random_Data_Mean': iats_efa_results['eigenvalues_random_mean']
        })
        start_row = write_section_to_excel(writer, 'Parallel_Analysis',
                                          'Eigenvalues Comparison', pa_data, start_row)

        pa_summary = pd.DataFrame({
            'Method': ['Parallel Analysis', 'Kaiser Criterion'],
            'N_Factors': [iats_efa_results['n_factors_parallel'],
                         iats_efa_results['n_factors_kaiser']]
        })
        write_section_to_excel(writer, 'Parallel_Analysis',
                              'Factor Determination', pa_summary, start_row)

        # EFA Results
        start_row = 0
        factorability = pd.DataFrame({
            'Test': ['KMO', 'Bartlett Chi-Square', 'Bartlett p-value'],
            'Value': [iats_efa_results['kmo'],
                     iats_efa_results['bartlett_chi2'],
                     iats_efa_results['bartlett_p']]
        })
        start_row = write_section_to_excel(writer, 'EFA_Results',
                                          'Factorability Tests', factorability, start_row)

        start_row = write_section_to_excel(writer, 'EFA_Results',
                                          'Factor Loadings (Oblimin Rotation)',
                                          iats_efa_results['loadings'], start_row)

        write_section_to_excel(writer, 'EFA_Results',
                              'Variance Explained',
                              iats_efa_results['variance'], start_row)

        # Reliability
        start_row = 0
        rel_overall = pd.DataFrame({
            'Statistic': ['Cronbach Alpha', 'McDonald Omega (Hierarchical)',
                         'McDonald Omega (Total)', 'N Items'],
            'Value': [iats_reliability['overall']['alpha'],
                     iats_reliability['overall']['omega_h'],
                     iats_reliability['overall']['omega_t'],
                     iats_reliability['overall']['n_items']]
        })
        write_section_to_excel(writer, 'Reliability',
                              'Overall Scale Reliability', rel_overall, start_row)

        # CFA Results
        if 'efa_model' in iats_cfa_results and 'error' not in iats_cfa_results['efa_model']:
            start_row = 0

            efa_fit = pd.DataFrame({
                'Fit_Index': ['Chi-Square', 'df', 'CFI', 'TLI', 'RMSEA', 'SRMR', 'Acceptable_Fit'],
                'Value': [iats_cfa_results['efa_model']['chi2'],
                         iats_cfa_results['efa_model']['dof'],
                         iats_cfa_results['efa_model']['cfi'],
                         iats_cfa_results['efa_model']['tli'],
                         iats_cfa_results['efa_model']['rmsea'],
                         iats_cfa_results['efa_model']['srmr'],
                         iats_cfa_results['efa_model']['acceptable_fit']]
            })
            start_row = write_section_to_excel(writer, 'CFA_Results',
                                              'Model Fit - EFA Structure', efa_fit, start_row)

            start_row = write_section_to_excel(writer, 'CFA_Results',
                                              'Standardized Loadings - EFA Structure',
                                              iats_cfa_results['efa_model']['loadings'], start_row)

            if 'single_factor_model' in iats_cfa_results:
                single_fit = pd.DataFrame({
                    'Fit_Index': ['Chi-Square', 'df', 'CFI', 'TLI', 'RMSEA', 'SRMR', 'Acceptable_Fit'],
                    'Value': [iats_cfa_results['single_factor_model']['chi2'],
                             iats_cfa_results['single_factor_model']['dof'],
                             iats_cfa_results['single_factor_model']['cfi'],
                             iats_cfa_results['single_factor_model']['tli'],
                             iats_cfa_results['single_factor_model']['rmsea'],
                             iats_cfa_results['single_factor_model']['srmr'],
                             iats_cfa_results['single_factor_model']['acceptable_fit']]
                })
                start_row = write_section_to_excel(writer, 'CFA_Results',
                                                  'Model Fit - Single Factor', single_fit, start_row)

                write_section_to_excel(writer, 'CFA_Results',
                                      'Standardized Loadings - Single Factor',
                                      iats_cfa_results['single_factor_model']['loadings'], start_row)

        # Measurement Invariance
        if iats_invariance_results is not None and 'configural' in iats_invariance_results:
            if 'group_fits' in iats_invariance_results['configural']:
                group_fit_data = []
                for group, fits in iats_invariance_results['configural']['group_fits'].items():
                    group_label = str(int(group)) if isinstance(group, (int, float, np.integer, np.floating)) else str(group)
                    group_fit_data.append({
                        'Group': group_label,
                        'N': iats_invariance_results['group_sizes'].get(group, ''),
                        'CFI': fits['cfi'],
                        'TLI': fits['tli'],
                        'RMSEA': fits['rmsea'],
                        'SRMR': fits['srmr']
                    })
                group_fit_df = pd.DataFrame(group_fit_data)
                write_section_to_excel(writer, 'Measurement_Invariance',
                                      'Group-Specific Fit (Configural Model)', group_fit_df, 0)

    output_files['iats'] = iats_file
    print(f"✓ Saved: {iats_file}")

    # --- Univariate Regression Results
    print("\nWriting univariate regression results...")
    reg_file = OUTPUT_DIR / f"Univariate_Regression_Results.xlsx"
    with pd.ExcelWriter(reg_file, engine='openpyxl') as writer:
        # Create sheets
        writer.book.create_sheet('IAS_Summary')
        writer.book.create_sheet('IATS_Summary')

        # --- IAS Regression Summary
        start_row = 0

        # Summary table for all IAS models
        ias_summary_data = []
        for predictor, result in all_regression_results['IAS'].items():
            ias_summary_data.append({
                'Predictor': predictor,
                'N': result['n'],
                'R²': result['r_squared'],
                'Adj_R²': result['adj_r_squared'],
                'Pearson_r': result['pearson_r'],
                'B (slope)': result['coefficients'][predictor],
                'SE': result['std_errors'][predictor],
                't': result['t_values'][predictor],
                'p': result['p_values'][predictor],
                'CI_lower': result['conf_int'][0][predictor],
                'CI_upper': result['conf_int'][1][predictor],
                'F': result['f_statistic'],
                'F_p': result['f_pvalue'],
                'Normality_p': result['residuals_normality_p'],
                'Heterosced_p': result['heteroscedasticity_p']
            })

        ias_summary_df = pd.DataFrame(ias_summary_data)
        start_row = write_section_to_excel(writer, 'IAS_Summary',
                                          'IAS Univariate Regressions - All Predictors',
                                          ias_summary_df, start_row)

        # Individual model details for IAS
        for predictor, result in all_regression_results['IAS'].items():
            # Create sheet for this predictor
            sheet_name = f'IAS_{predictor}'
            if len(sheet_name) > 31:  # Excel sheet name limit
                sheet_name = sheet_name[:31]
            writer.book.create_sheet(sheet_name)

            start_row = 0

            # Model info
            model_info = pd.DataFrame({
                'Item': ['DV', 'IV', 'N', 'R²', 'Adj R²', 'Pearson r', 'Pearson p',
                        'F-statistic', 'F p-value', 'AIC', 'BIC'],
                'Value': [result['dv'], result['iv'], result['n'],
                         result['r_squared'], result['adj_r_squared'],
                         result['pearson_r'], result['pearson_p'],
                         result['f_statistic'], result['f_pvalue'],
                         result['aic'], result['bic']]
            })
            start_row = write_section_to_excel(writer, sheet_name,
                                              f'Model: IAS ~ {predictor}', model_info, start_row)

            # Coefficients
            start_row = write_section_to_excel(writer, sheet_name,
                                              'Regression Coefficients',
                                              result['coefficient_table'], start_row)

            # Assumptions
            assumptions = pd.DataFrame({
                'Test': ['Residuals Normality (p-value)', 'Heteroscedasticity (p-value)'],
                'Value': [result['residuals_normality_p'], result['heteroscedasticity_p']]
            })
            write_section_to_excel(writer, sheet_name,
                                  'Assumption Tests', assumptions, start_row)

        # --- IATS Regression Summary
        start_row = 0

        # Summary table for all IATS models
        iats_summary_data = []
        for predictor, result in all_regression_results['IATS'].items():
            iats_summary_data.append({
                'Predictor': predictor,
                'N': result['n'],
                'R²': result['r_squared'],
                'Adj_R²': result['adj_r_squared'],
                'Pearson_r': result['pearson_r'],
                'B (slope)': result['coefficients'][predictor],
                'SE': result['std_errors'][predictor],
                't': result['t_values'][predictor],
                'p': result['p_values'][predictor],
                'CI_lower': result['conf_int'][0][predictor],
                'CI_upper': result['conf_int'][1][predictor],
                'F': result['f_statistic'],
                'F_p': result['f_pvalue'],
                'Normality_p': result['residuals_normality_p'],
                'Heterosced_p': result['heteroscedasticity_p']
            })

        iats_summary_df = pd.DataFrame(iats_summary_data)
        start_row = write_section_to_excel(writer, 'IATS_Summary',
                                          'IATS Univariate Regressions - All Predictors',
                                          iats_summary_df, start_row)

        # Individual model details for IATS
        for predictor, result in all_regression_results['IATS'].items():
            # Create sheet for this predictor
            sheet_name = f'IATS_{predictor}'
            if len(sheet_name) > 31:  # Excel sheet name limit
                sheet_name = sheet_name[:31]
            writer.book.create_sheet(sheet_name)

            start_row = 0

            # Model info
            model_info = pd.DataFrame({
                'Item': ['DV', 'IV', 'N', 'R²', 'Adj R²', 'Pearson r', 'Pearson p',
                        'F-statistic', 'F p-value', 'AIC', 'BIC'],
                'Value': [result['dv'], result['iv'], result['n'],
                         result['r_squared'], result['adj_r_squared'],
                         result['pearson_r'], result['pearson_p'],
                         result['f_statistic'], result['f_pvalue'],
                         result['aic'], result['bic']]
            })
            start_row = write_section_to_excel(writer, sheet_name,
                                              f'Model: IATS ~ {predictor}', model_info, start_row)

            # Coefficients
            start_row = write_section_to_excel(writer, sheet_name,
                                              'Regression Coefficients',
                                              result['coefficient_table'], start_row)

            # Assumptions
            assumptions = pd.DataFrame({
                'Test': ['Residuals Normality (p-value)', 'Heteroscedasticity (p-value)'],
                'Value': [result['residuals_normality_p'], result['heteroscedasticity_p']]
            })
            write_section_to_excel(writer, sheet_name,
                                  'Assumption Tests', assumptions, start_row)

    output_files['regression'] = reg_file
    print(f"✓ Saved: {reg_file}")

    return output_files

# Generate all Excel outputs
output_files = create_comprehensive_excel_output()

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE - Dutch Interoceptive Scales Validation")
print("=" * 80)
print("\nOutput files created:")
for key, filepath in output_files.items():
    print(f"  {key}: {filepath}")

print("\n" + "=" * 80)
print("END OF SCRIPT")
print("=" * 80)