"""
Dutch Interoceptive Scales Validation Analysis
==============================================

This script performs validation analyses on Dutch translations of:
- Interoceptive Accuracy Scale (IAS) 
- Interoceptive Attention Scale (IATS)

Key variables retained: Persoon_ID, Leeftijd, Geslacht, IAS, IATS, BPQ, ICQ, BDI, TAS
Analyses: PCA, Internal Consistency, CFA (1-4 factors), Regression

Author: J. Mulder
Date: 15-07-2025
Reference: https://www.medrxiv.org/content/10.1101/2025.05.06.25326009v1.full-text
"""

#%% Import packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy import stats as scipystats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from factor_analyzer import calculate_kmo, calculate_bartlett_sphericity, Rotator
from semopy import Model, stats
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Set display options
pd.set_option('display.max_columns', None)
sns.set_style("whitegrid")

#%% Configuration
# File paths - UPDATE THESE FOR YOUR SETUP
DATA_PATH = 'data/your_data.sav'  # Place your data file in the data/ folder
OUTPUT_PATH = 'output/'

# Analysis parameters
OUTLIER_THRESHOLD = 3  # Z-score threshold for outlier removal
RANDOM_STATE = 42
PCA_VARIANCE_THRESHOLD = 0.70

#%% Data Loading and Preparation
def load_and_prepare_data(file_path):
    """Load data and prepare essential variables"""
    print("Loading data...")
    
    try:
        # Load data (adjust based on your file format)
        if file_path.endswith('.sav'):
            df = pd.read_spss(file_path, convert_categoricals=False)
        elif file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        else:
            raise ValueError("Unsupported file format. Use .sav or .csv")
            
        print(f"Data loaded successfully. Shape: {df.shape}")
        return df
        
    except Exception as e:
        print(f"Error loading data: {e}")
        raise

def remove_outliers(df, variables, threshold=3):
    """Remove outliers based on z-score threshold"""
    print(f"Removing outliers (threshold: {threshold} SD)...")
    
    z_scores = np.abs(scipystats.zscore(df[variables]))
    df_clean = df[(z_scores < threshold).all(axis=1)]
    
    n_removed = len(df) - len(df_clean)
    print(f"Removed {n_removed} outliers ({n_removed/len(df)*100:.1f}%)")
    print(f"Final sample size: {len(df_clean)}")
    
    return df_clean

#%% Descriptive Statistics
def calculate_descriptives(df):
    """Calculate and display descriptive statistics"""
    print("\n" + "="*50)
    print("DESCRIPTIVE STATISTICS")
    print("="*50)
    
    # Continuous variables
    cont_vars = ['Leeftijd', 'IAS_Totaal', 'IATS_Totaal', 'BPQ_Totaal', 
                 'ICQ_Totaal', 'BDI_Totaal', 'TAS_Totaal']
    
    print("\nContinuous Variables:")
    desc_stats = df[cont_vars].describe()
    print(desc_stats.round(2))
    
    # Age and gender breakdown
    print(f"\nMean age: {df['Leeftijd'].mean():.1f} ± {df['Leeftijd'].std():.1f}")
    
    if 'Geslacht' in df.columns:
        gender_counts = df['Geslacht'].value_counts()
        gender_pct = (gender_counts / len(df) * 100).round(1)
        print("Gender distribution:")
        for gender, count in gender_counts.items():
            print(f"  {gender}: {count} ({gender_pct[gender]}%)")
    
    return desc_stats

#%% Principal Component Analysis
def perform_pca(df, columns, scale_name, n_components_range=[1,2,3,4]):
    """Perform PCA with rotation and return results for multiple factor solutions"""
    print(f"\n" + "="*50)
    print(f"PCA ANALYSIS - {scale_name}")
    print("="*50)
    
    data = df[columns].dropna()
    
    # Standardize data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    
    # Test assumptions
    kmo_all, kmo_model = calculate_kmo(data_scaled)
    bartlett_chi2, bartlett_p = calculate_bartlett_sphericity(data_scaled)
    
    print(f"Sample adequacy:")
    print(f"  KMO: {kmo_model:.3f}")
    print(f"  Bartlett's test: χ² = {bartlett_chi2:.2f}, p < {bartlett_p:.3f}")
    
    # Fit PCA
    pca = PCA()
    pca.fit(data_scaled)
    
    eigenvalues = pca.explained_variance_
    explained_var = pca.explained_variance_ratio_
    cumulative_var = np.cumsum(explained_var)
    
    # Kaiser criterion
    n_kaiser = np.sum(eigenvalues > 1)
    print(f"\nEigenvalues > 1 (Kaiser criterion): {n_kaiser}")
    print(f"Variance explained by {n_kaiser} components: {cumulative_var[n_kaiser-1]:.3f}")
    
    # Create scree plot
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(eigenvalues) + 1), eigenvalues, 'o-', linewidth=2)
    plt.axhline(y=1, color='r', linestyle='--', alpha=0.7)
    plt.xlabel('Component Number')
    plt.ylabel('Eigenvalue')
    plt.title(f'Scree Plot - {scale_name}')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Create output directory if it doesn't exist
    import os
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    plt.savefig(f'{OUTPUT_PATH}{scale_name.lower()}_scree_plot.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Store results for different factor solutions
    pca_results = {}
    
    for n_factors in n_components_range:
        if n_factors <= len(columns):
            # Get loadings
            loadings = pca.components_[:n_factors].T * np.sqrt(eigenvalues[:n_factors])
            
            # Apply varimax rotation (with error handling)
            try:
                rotator = Rotator(method='varimax')
                loadings_rotated = rotator.fit_transform(loadings)
                rotation_applied = True
            except Exception as e:
                print(f"    Warning: Rotation failed ({e}), using unrotated loadings")
                loadings_rotated = loadings
                rotation_applied = False
            
            # Create loadings DataFrame
            factor_names = [f'Factor_{i+1}' for i in range(n_factors)]
            loadings_df = pd.DataFrame(
                loadings_rotated, 
                index=columns, 
                columns=factor_names
            )
            
            pca_results[f'{n_factors}_factor'] = {
                'loadings': loadings_df,
                'components': pca.components_[:n_factors],  # Store raw components for CFA
                'explained_variance': explained_var[:n_factors].sum(),
                'eigenvalues': eigenvalues[:n_factors]
            }
            
            print(f"\n{n_factors}-Factor Solution:")
            print(f"  Variance explained: {explained_var[:n_factors].sum():.3f}")
            if rotation_applied:
                print("  Rotated factor loadings (|loading| > 0.3):")
            else:
                print("  Unrotated factor loadings (|loading| > 0.3):")
            for factor in factor_names:
                items = loadings_df[np.abs(loadings_df[factor]) > 0.3].index.tolist()
                if items:
                    print(f"    {factor}: {', '.join(items)}")
                else:
                    print(f"    {factor}: No items with |loading| > 0.3")
    
    return pca_results

#%% Internal Consistency Analysis
def calculate_cronbach_alpha(df):
    """Calculate Cronbach's alpha"""
    n_items = df.shape[1]
    total_variance = df.sum(axis=1).var(ddof=1)
    item_variances = df.var(axis=0, ddof=1).sum()
    alpha = (n_items / (n_items - 1)) * (1 - (item_variances / total_variance))
    return alpha

def analyze_internal_consistency(df, columns, scale_name, pca_results):
    """Analyze internal consistency for different factor structures"""
    print(f"\n" + "="*50)
    print(f"INTERNAL CONSISTENCY - {scale_name}")
    print("="*50)
    
    data = df[columns].dropna()
    
    # Overall alpha
    overall_alpha = calculate_cronbach_alpha(data)
    print(f"Overall Cronbach's α: {overall_alpha:.3f}")
    
    # Alpha for each factor structure
    consistency_results = {'overall': overall_alpha}
    
    for structure_name, structure_data in pca_results.items():
        loadings_df = structure_data['loadings']
        n_factors = loadings_df.shape[1]
        
        print(f"\n{structure_name.replace('_', '-').title()} Structure:")
        factor_alphas = {}
        
        for factor_col in loadings_df.columns:
            # Assign items to factors based on highest absolute loading
            factor_items = []
            for item in loadings_df.index:
                max_loading_factor = loadings_df.loc[item].abs().idxmax()
                if max_loading_factor == factor_col and abs(loadings_df.loc[item, factor_col]) > 0.3:
                    factor_items.append(item)
            
            if len(factor_items) > 1:  # Need at least 2 items for alpha
                factor_alpha = calculate_cronbach_alpha(data[factor_items])
                factor_alphas[factor_col] = factor_alpha
                print(f"  {factor_col}: α = {factor_alpha:.3f} ({len(factor_items)} items)")
            else:
                print(f"  {factor_col}: insufficient items ({len(factor_items)})")
        
        consistency_results[structure_name] = factor_alphas
    
    return consistency_results

#%% Confirmatory Factor Analysis
def create_model_spec_from_pca(pca_components, num_factors):
    """Generate model specification dictionary based on PCA components and number of factors."""
    model_spec = {f'F{i+1}': [] for i in range(num_factors)}
    
    for idx, column in enumerate(range(pca_components.shape[1])):
        # Get the absolute values of the loadings for this item across all factors
        loadings = [np.abs(pca_components[factor, idx]) for factor in range(num_factors)]
        
        # Assign to the factor with the highest absolute loading
        best_factor = np.argmax(loadings)
        model_spec[f'F{best_factor+1}'].append(idx + 1)
    
    return model_spec

def create_model_spec(model_spec, prefix=""):
    model_desc = ""
    for factor, items in model_spec.items():
        model_desc += f"{prefix}{factor} =~ " + " + ".join([f"{prefix}{item}" for item in items]) + "\n"
    return model_desc

def calculate_fit_statistics(model, data):
    model.fit(data)
    dof = stats.calc_dof(model)
    chi_square = stats.calc_chi2(model)
    cfi = stats.calc_cfi(model)
    tli = stats.calc_tli(model)
    rmsea = stats.calc_rmsea(model)
    
    return chi_square, cfi, tli, rmsea, dof

def extract_factor_loadings(model):
    """Extract and return the factor loadings from a fitted model."""
    params = model.inspect()
    loadings = params[params['op'] == '~']
    return loadings[['lval', 'rval', 'Estimate']]

def run_cfa_analysis(df, columns, scale_name, pca_results):
    """Run CFA for different factor structures"""
    print(f"\n" + "="*50)
    print(f"CONFIRMATORY FACTOR ANALYSIS - {scale_name}")
    print("="*50)
    
    data = df[columns].dropna()
    
    # Split data for cross-validation
    train_data, test_data = train_test_split(data, test_size=0.5, random_state=RANDOM_STATE)
    print(f"Training sample: n = {len(train_data)}")
    print(f"Test sample: n = {len(test_data)}")
    
    cfa_results = {}
    
    for structure_name, structure_data in pca_results.items():
        print(f"\n{structure_name.replace('_', '-').title()} Model:")
        
        try:
            # Extract number of factors from structure name
            num_factors = int(structure_name.split('_')[0])
            
            # Create model specification using raw PCA components
            model_spec = create_model_spec_from_pca(structure_data['components'], num_factors)
            model_desc = create_model_spec(model_spec, scale_name + "_")
            print("Model specification:")
            print(model_desc)
            
            # Fit model
            model = Model(model_desc)
            
            # Calculate fit indices
            fit_stats = calculate_fit_statistics(model, train_data)
            loadings = extract_factor_loadings(model)
            
            fit_stats_dict = {
                'Chi-square': fit_stats[0],
                'df': fit_stats[4],
                'p-value': 1 - scipystats.chi2.cdf(fit_stats[0], fit_stats[4]) if fit_stats[4] > 0 else np.nan,
                'CFI': fit_stats[1],
                'TLI': fit_stats[2],
                'RMSEA': fit_stats[3]
            }
            
            print("Fit indices:")
            for stat, value in fit_stats_dict.items():
                if isinstance(value, float):
                    print(f"  {stat}: {value:.3f}")
                else:
                    print(f"  {stat}: {value}")
            
            # Evaluate fit
            good_fit = (fit_stats[1] > 0.95 and fit_stats[2] > 0.95 and fit_stats[3] < 0.06)
            acceptable_fit = (fit_stats[1] > 0.90 and fit_stats[2] > 0.90 and fit_stats[3] < 0.08)
            
            if good_fit:
                print("  → Good model fit")
            elif acceptable_fit:
                print("  → Acceptable model fit")
            else:
                print("  → Poor model fit")
            
            cfa_results[structure_name] = {
                'model': model,
                'fit_stats': fit_stats,
                'fit_stats_dict': fit_stats_dict,
                'model_spec': model_desc,
                'loadings': loadings
            }
            
        except Exception as e:
            print(f"  Error fitting model: {e}")
            cfa_results[structure_name] = None
    
    return cfa_results

#%% Regression Analysis
def run_regression_analyses(df):
    """Run univariate regression analyses"""
    print(f"\n" + "="*50)
    print("REGRESSION ANALYSES")
    print("="*50)
    
    dependent_vars = ['IAS_Totaal', 'IATS_Totaal']
    independent_vars = ['ICQ_Totaal', 'BPQ_Totaal', 'BDI_Totaal', 'TAS_Totaal']
    
    regression_results = []
    
    for dep_var in dependent_vars:
        print(f"\nDependent Variable: {dep_var}")
        print("-" * 40)
        
        for ind_var in independent_vars:
            # Prepare data
            data_subset = df[[dep_var, ind_var]].dropna()
            X = sm.add_constant(data_subset[ind_var])
            y = data_subset[dep_var]
            
            # Fit model
            model = sm.OLS(y, X).fit()
            
            # Extract results
            coef = model.params[ind_var]
            se = model.bse[ind_var]
            t_stat = model.tvalues[ind_var]
            p_value = model.pvalues[ind_var]
            r_squared = model.rsquared
            
            print(f"{ind_var}: β = {coef:.3f}, SE = {se:.3f}, t = {t_stat:.3f}, "
                  f"p = {p_value:.3f}, R² = {r_squared:.3f}")
            
            regression_results.append({
                'Dependent': dep_var,
                'Independent': ind_var,
                'Beta': coef,
                'SE': se,
                't': t_stat,
                'p': p_value,
                'R_squared': r_squared
            })
    
    return pd.DataFrame(regression_results)

#%% Main Analysis Pipeline
def main():
    """Main analysis pipeline"""
    print("DUTCH INTEROCEPTIVE SCALES VALIDATION ANALYSIS")
    print("=" * 60)
    
    # Load and prepare data
    df = load_and_prepare_data(DATA_PATH)
    
    # Define key variables
    outlier_vars = ['IAS_Totaal', 'IATS_Totaal', 'BPQ_Totaal', 
                    'ICQ_Totaal', 'BDI_Totaal', 'TAS_Totaal']
    
    ias_items = [f'IAS_{i}' for i in range(1, 22)]
    iats_items = [f'IATS_{i}' for i in range(1, 22)]
    
    # Remove outliers
    df_clean = remove_outliers(df, outlier_vars, OUTLIER_THRESHOLD)
    
    # Descriptive statistics
    descriptives = calculate_descriptives(df_clean)
    
    # PCA Analysis
    ias_pca_results = perform_pca(df_clean, ias_items, "IAS")
    iats_pca_results = perform_pca(df_clean, iats_items, "IATS")
    
    # Internal Consistency
    ias_consistency = analyze_internal_consistency(df_clean, ias_items, "IAS", ias_pca_results)
    iats_consistency = analyze_internal_consistency(df_clean, iats_items, "IATS", iats_pca_results)
    
    # Confirmatory Factor Analysis
    ias_cfa_results = run_cfa_analysis(df_clean, ias_items, "IAS", ias_pca_results)
    iats_cfa_results = run_cfa_analysis(df_clean, iats_items, "IATS", iats_pca_results)
    
    # Regression Analysis
    regression_results = run_regression_analyses(df_clean)
    
    print(f"\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print(f"Results saved to: {OUTPUT_PATH}")
    print("Check the output folder for generated plots and results.")
    
    return {
        'descriptives': descriptives,
        'ias_pca': ias_pca_results,
        'iats_pca': iats_pca_results,
        'ias_consistency': ias_consistency,
        'iats_consistency': iats_consistency,
        'ias_cfa': ias_cfa_results,
        'iats_cfa': iats_cfa_results,
        'regression': regression_results
    }

# Run analysis if script is executed directly
if __name__ == "__main__":
    results = main()