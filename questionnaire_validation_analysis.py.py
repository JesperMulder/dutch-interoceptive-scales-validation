"""
Validation of Dutch Interoceptive Accuracy Scale (IAS) and Interoceptive Attention Scale (IATS)

This script performs psychometric validation analyses for the Dutch translations of the 
Interoceptive Accuracy Scale and Interoceptive Attention Scale.

Reference: [Your paper citation will go here]

Authors: [Your name]
Date: [Date]
"""

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
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

def load_data(file_path):
    """
    Load the dataset from the specified file path.
    
    Parameters:
    file_path (str): Path to the data file
    
    Returns:
    pd.DataFrame: Loaded dataset
    """
    try:
        df = pd.read_spss(file_path, convert_categoricals=False)
        print(f"Data loaded successfully. Shape: {df.shape}")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        raise

def select_variables(df):
    """
    Select only the variables needed for the final analyses.
    
    Parameters:
    df (pd.DataFrame): Full dataset
    
    Returns:
    pd.DataFrame: Dataset with selected variables only
    """
    # Core variables to keep
    keep_vars = ['Persoon_ID', 'Leeftijd', 'Geslacht']
    
    # Add questionnaire variables
    keep_vars.extend(['IAS_Totaal', 'IATS_Totaal', 'BPQ_Totaal', 
                     'ICQ_Totaal', 'BDI_Totaal', 'TAS_Totaal'])
    
    # Add individual IAS and IATS items
    ias_items = [f'IAS_{i}' for i in range(1, 22)]
    iats_items = [f'IATS_{i}' for i in range(1, 22)]
    keep_vars.extend(ias_items + iats_items)
    
    # Select only existing columns
    available_vars = [var for var in keep_vars if var in df.columns]
    df_selected = df[available_vars].copy()
    
    print(f"Selected {len(available_vars)} variables for analysis")
    return df_selected

def remove_outliers(df, threshold=3):
    """
    Remove outliers based on z-score threshold.
    
    Parameters:
    df (pd.DataFrame): Input dataset
    threshold (float): Z-score threshold for outlier removal
    
    Returns:
    pd.DataFrame: Dataset without outliers
    """
    # Variables to check for outliers
    outlier_vars = ['IAS_Totaal', 'IATS_Totaal', 'BPQ_Totaal',
                    'ICQ_Totaal', 'BDI_Totaal', 'TAS_Totaal']
    
    # Calculate z-scores
    z_scores = np.abs(scipystats.zscore(df[outlier_vars]))
    
    # Remove outliers
    df_clean = df[(z_scores < threshold).all(axis=1)].copy()
    
    n_removed = len(df) - len(df_clean)
    print(f"Removed {n_removed} outliers (z-score > {threshold})")
    print(f"Final sample size: {len(df_clean)}")
    
    return df_clean

def describe_sample(df):
    """
    Generate descriptive statistics for the sample.
    
    Parameters:
    df (pd.DataFrame): Clean dataset
    
    Returns:
    dict: Dictionary containing descriptive statistics
    """
    descriptives = {}
    
    # Sample size
    descriptives['n_total'] = len(df)
    
    # Age statistics
    descriptives['age_mean'] = df['Leeftijd'].mean()
    descriptives['age_std'] = df['Leeftijd'].std()
    descriptives['age_range'] = (df['Leeftijd'].min(), df['Leeftijd'].max())
    
    # Sex distribution
    sex_counts = df['Geslacht'].value_counts()
    descriptives['sex_distribution'] = sex_counts
    descriptives['percent_female'] = (sex_counts.get(1, 0) / len(df)) * 100
    
    # Questionnaire scores
    questionnaire_vars = ['IAS_Totaal', 'IATS_Totaal', 'BPQ_Totaal',
                         'ICQ_Totaal', 'BDI_Totaal', 'TAS_Totaal']
    descriptives['questionnaire_stats'] = df[questionnaire_vars].describe()
    
    print("=== SAMPLE CHARACTERISTICS ===")
    print(f"Total N: {descriptives['n_total']}")
    print(f"Age: M = {descriptives['age_mean']:.1f}, SD = {descriptives['age_std']:.1f}")
    print(f"Age range: {descriptives['age_range'][0]:.0f}-{descriptives['age_range'][1]:.0f}")
    print(f"Female: {descriptives['percent_female']:.1f}%")
    print("\nQuestionnaire Descriptives:")
    print(descriptives['questionnaire_stats'])
    
    return descriptives

def perform_pca(df, item_columns, scale_name):
    """
    Perform Principal Component Analysis with varimax rotation.
    
    Parameters:
    df (pd.DataFrame): Dataset
    item_columns (list): List of item column names
    scale_name (str): Name of the scale (for reporting)
    
    Returns:
    dict: PCA results including components, eigenvalues, and fit indices
    """
    print(f"\n=== PCA FOR {scale_name.upper()} ===")
    
    # Prepare data
    data = df[item_columns].dropna()
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    
    # KMO and Bartlett's test
    kmo_all, kmo_model = calculate_kmo(data_scaled)
    bartlett_chi2, bartlett_p = calculate_bartlett_sphericity(data_scaled)
    
    print(f"KMO: {kmo_model:.3f}")
    print(f"Bartlett's test: χ² = {bartlett_chi2:.2f}, p < .001" if bartlett_p < 0.001 
          else f"Bartlett's test: χ² = {bartlett_chi2:.2f}, p = {bartlett_p:.3f}")
    
    # Perform PCA
    pca = PCA()
    pca.fit(data_scaled)
    
    # Extract eigenvalues and variance explained
    eigenvalues = pca.explained_variance_
    variance_explained = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(variance_explained)
    
    # Kaiser criterion
    n_components_kaiser = np.sum(eigenvalues > 1)
    print(f"Components with eigenvalue > 1: {n_components_kaiser}")
    
    # Get component loadings with varimax rotation
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
    
    # Apply varimax rotation
    rotator = Rotator(method='varimax')
    loadings_rotated = rotator.fit_transform(loadings)
    
    # Create results dataframe
    component_df = pd.DataFrame(
        loadings_rotated,
        index=item_columns,
        columns=[f'PC{i+1}' for i in range(len(item_columns))]
    )
    
    # Calculate communalities
    communalities = np.sum(loadings_rotated**2, axis=1)
    
    results = {
        'eigenvalues': eigenvalues,
        'variance_explained': variance_explained,
        'cumulative_variance': cumulative_variance,
        'n_components_kaiser': n_components_kaiser,
        'components': component_df,
        'communalities': communalities,
        'kmo': kmo_model,
        'bartlett_chi2': bartlett_chi2,
        'bartlett_p': bartlett_p
    }
    
    # Print key results
    print("Eigenvalues (first 5):", eigenvalues[:5].round(3))
    print("Variance explained (first 5):", variance_explained[:5].round(3))
    
    return results

def calculate_reliability(df, item_columns, factor_structure=None):
    """
    Calculate Cronbach's alpha reliability coefficients.
    
    Parameters:
    df (pd.DataFrame): Dataset
    item_columns (list): List of item column names
    factor_structure (dict, optional): Factor structure for subscale analysis
    
    Returns:
    dict: Reliability results
    """
    def cronbach_alpha(data):
        """Calculate Cronbach's alpha"""
        n_items = data.shape[1]
        total_var = data.sum(axis=1).var(ddof=1)
        item_vars = data.var(axis=0, ddof=1).sum()
        alpha = (n_items / (n_items - 1)) * (1 - (item_vars / total_var))
        return alpha
    
    results = {}
    
    # Overall reliability
    data = df[item_columns].dropna()
    results['overall_alpha'] = cronbach_alpha(data)
    
    # Item-total correlations and alpha if item deleted
    item_analysis = {}
    for item in item_columns:
        # Alpha if item deleted
        data_without_item = data.drop(columns=[item])
        alpha_without = cronbach_alpha(data_without_item)
        
        # Item-total correlation
        item_total_corr = data[item].corr(data.drop(columns=[item]).sum(axis=1))
        
        item_analysis[item] = {
            'alpha_if_deleted': alpha_without,
            'item_total_correlation': item_total_corr
        }
    
    results['item_analysis'] = item_analysis
    
    # Factor-based reliabilities if structure provided
    if factor_structure:
        factor_alphas = {}
        for factor_name, factor_items in factor_structure.items():
            if len(factor_items) > 1:  # Need at least 2 items for reliability
                factor_data = data[[col for col in item_columns if any(str(i) in col for i in factor_items)]]
                if not factor_data.empty:
                    factor_alphas[factor_name] = cronbach_alpha(factor_data)
        results['factor_alphas'] = factor_alphas
    
    return results

def create_cfa_model_specification(pca_results, n_factors, prefix):
    """
    Create CFA model specification based on PCA results.
    
    Parameters:
    pca_results (dict): Results from PCA
    n_factors (int): Number of factors
    prefix (str): Variable prefix (e.g., 'IAS_', 'IATS_')
    
    Returns:
    str: Model specification for semopy
    """
    components = pca_results['components']
    n_items = len(components)
    
    # Get the loadings for the specified number of factors
    loadings = components.iloc[:, :n_factors].abs()
    
    # Create factor assignments
    factor_assignments = {}
    for factor in range(n_factors):
        factor_assignments[f'F{factor + 1}'] = []
    
    # Assign each item to the factor with the highest loading
    for item_idx in range(n_items):
        # Get loadings for this item across all factors
        item_loadings = loadings.iloc[item_idx, :].values
        
        # Find the factor with the highest loading
        best_factor_idx = np.argmax(item_loadings)
        
        # Assign item to that factor
        factor_name = f'F{best_factor_idx + 1}'
        item_name = f"{prefix}{item_idx + 1}"
        factor_assignments[factor_name].append(item_name)
    
    # Build model specification string
    model_spec_parts = []
    for factor_name, items in factor_assignments.items():
        if len(items) > 0:  # Only include factors with items
            model_spec_parts.append(f"{factor_name} =~ " + " + ".join(items))
    
    model_spec = "\n".join(model_spec_parts)
    return model_spec

def run_cfa_models(df, item_columns, pca_results, prefix, scale_name):
    """
    Run Confirmatory Factor Analysis for 1-4 factor models.
    
    Parameters:
    df (pd.DataFrame): Dataset
    item_columns (list): List of item column names
    pca_results (dict): Results from PCA
    prefix (str): Variable prefix
    scale_name (str): Scale name for reporting
    
    Returns:
    dict: CFA results for different factor models
    """
    print(f"\n=== CFA FOR {scale_name.upper()} ===")
    
    # Split data for cross-validation
    df_train, df_test = train_test_split(df, test_size=0.5, random_state=42)
    
    results = {}
    
    for n_factors in [1, 2, 3, 4]:
        try:
            # Create model specification
            model_spec = create_cfa_model_specification(pca_results, n_factors, prefix)
            
            if not model_spec:  # Skip if no valid model
                continue
                
            # Fit model
            model = Model(model_spec)
            model.fit(df_train[item_columns])
            
            # Calculate fit statistics
            chi2 = stats.calc_chi2(model)
            dof = stats.calc_dof(model)
            cfi = stats.calc_cfi(model)
            tli = stats.calc_tli(model)
            rmsea = stats.calc_rmsea(model)
            
            # Get factor loadings
            params = model.inspect()
            loadings = params[params['op'] == '=~'][['lval', 'rval', 'Estimate']]
            
            # Calculate p-value safely
            if dof > 0 and not np.isnan(chi2):
                p_value = 1 - scipystats.chi2.cdf(chi2, dof)
            else:
                p_value = np.nan
            
            results[f'{n_factors}_factor'] = {
                'model_spec': model_spec,
                'fit_indices': {
                    'chi2': chi2,
                    'dof': dof,
                    'p_value': p_value,
                    'cfi': cfi,
                    'tli': tli,
                    'rmsea': rmsea
                },
                'loadings': loadings
            }
            
            print(f"\n{n_factors}-Factor Model:")
            if not np.isnan(p_value):
                print(f"  χ²({dof}) = {chi2:.2f}, p = {p_value:.3f}")
            else:
                print(f"  χ²({dof}) = {chi2:.2f}, p = NaN")
            print(f"  CFI = {cfi:.3f}, TLI = {tli:.3f}, RMSEA = {rmsea:.3f}")
            
        except Exception as e:
            print(f"Error fitting {n_factors}-factor model: {e}")
            continue
    
    return results

def run_regression_analyses(df):
    """
    Run regression analyses examining relationships between scales.
    
    Parameters:
    df (pd.DataFrame): Dataset
    
    Returns:
    dict: Regression results
    """
    print("\n=== REGRESSION ANALYSES ===")
    
    dependent_vars = ['IAS_Totaal', 'IATS_Totaal']
    independent_vars = ['ICQ_Totaal', 'BPQ_Totaal', 'BDI_Totaal', 'TAS_Totaal']
    
    results = {}
    
    for dep_var in dependent_vars:
        results[dep_var] = {}
        
        print(f"\nDependent Variable: {dep_var}")
        print("-" * 40)
        
        for ind_var in independent_vars:
            # Simple regression
            X = sm.add_constant(df[ind_var])
            y = df[dep_var]
            model = sm.OLS(y, X).fit()
            
            results[dep_var][ind_var] = {
                'coefficient': model.params[ind_var],
                'std_error': model.bse[ind_var],
                't_value': model.tvalues[ind_var],
                'p_value': model.pvalues[ind_var],
                'r_squared': model.rsquared,
                'confidence_interval': model.conf_int().loc[ind_var].tolist()
            }
            
            print(f"{ind_var}: β = {model.params[ind_var]:.3f}, "
                  f"t = {model.tvalues[ind_var]:.3f}, "
                  f"p = {model.pvalues[ind_var]:.3f}, "
                  f"R² = {model.rsquared:.3f}")
    
    return results

def save_results(results_dict, output_path):
    """
    Save all analysis results to Excel file.
    
    Parameters:
    results_dict (dict): Dictionary containing all analysis results
    output_path (str): Path for output Excel file
    """
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        
        # 1. Descriptive statistics
        if 'descriptives' in results_dict:
            desc_data = []
            desc = results_dict['descriptives']
            
            desc_data.append({'Measure': 'Sample Size', 'Value': desc['n_total']})
            desc_data.append({'Measure': 'Mean Age', 'Value': f"{desc['age_mean']:.1f}"})
            desc_data.append({'Measure': 'Age SD', 'Value': f"{desc['age_std']:.1f}"})
            desc_data.append({'Measure': 'Age Range', 'Value': f"{desc['age_range'][0]:.0f}-{desc['age_range'][1]:.0f}"})
            desc_data.append({'Measure': 'Percent Female', 'Value': f"{desc['percent_female']:.1f}%"})
            
            desc_df = pd.DataFrame(desc_data)
            desc_df.to_excel(writer, sheet_name='Descriptives', index=False)
            
            # Questionnaire descriptives
            if 'questionnaire_stats' in desc:
                desc['questionnaire_stats'].to_excel(writer, sheet_name='Questionnaire_Stats', index=True)
        
        # 2. IAS PCA Results
        if 'ias_pca' in results_dict:
            ias_pca = results_dict['ias_pca']
            
            # Eigenvalues and variance explained
            pca_summary = pd.DataFrame({
                'Component': [f'PC{i+1}' for i in range(len(ias_pca['eigenvalues']))],
                'Eigenvalue': ias_pca['eigenvalues'],
                'Variance_Explained': ias_pca['variance_explained'],
                'Cumulative_Variance': ias_pca['cumulative_variance']
            })
            pca_summary.to_excel(writer, sheet_name='IAS_PCA_Summary', index=False)
            
            # Component loadings
            ias_pca['components'].to_excel(writer, sheet_name='IAS_PCA_Loadings', index=True)
            
            # Communalities
            comm_df = pd.DataFrame({
                'Item': ias_pca['components'].index,
                'Communality': ias_pca['communalities']
            })
            comm_df.to_excel(writer, sheet_name='IAS_Communalities', index=False)
            
            # Fit indices
            fit_df = pd.DataFrame([{
                'KMO': ias_pca['kmo'],
                'Bartlett_Chi2': ias_pca['bartlett_chi2'],
                'Bartlett_p': ias_pca['bartlett_p'],
                'Kaiser_Components': ias_pca['n_components_kaiser']
            }])
            fit_df.to_excel(writer, sheet_name='IAS_PCA_Fit', index=False)
        
        # 3. IATS PCA Results
        if 'iats_pca' in results_dict:
            iats_pca = results_dict['iats_pca']
            
            # Eigenvalues and variance explained
            pca_summary = pd.DataFrame({
                'Component': [f'PC{i+1}' for i in range(len(iats_pca['eigenvalues']))],
                'Eigenvalue': iats_pca['eigenvalues'],
                'Variance_Explained': iats_pca['variance_explained'],
                'Cumulative_Variance': iats_pca['cumulative_variance']
            })
            pca_summary.to_excel(writer, sheet_name='IATS_PCA_Summary', index=False)
            
            # Component loadings
            iats_pca['components'].to_excel(writer, sheet_name='IATS_PCA_Loadings', index=True)
            
            # Communalities
            comm_df = pd.DataFrame({
                'Item': iats_pca['components'].index,
                'Communality': iats_pca['communalities']
            })
            comm_df.to_excel(writer, sheet_name='IATS_Communalities', index=False)
            
            # Fit indices
            fit_df = pd.DataFrame([{
                'KMO': iats_pca['kmo'],
                'Bartlett_Chi2': iats_pca['bartlett_chi2'],
                'Bartlett_p': iats_pca['bartlett_p'],
                'Kaiser_Components': iats_pca['n_components_kaiser']
            }])
            fit_df.to_excel(writer, sheet_name='IATS_PCA_Fit', index=False)
        
        # 4. IAS Reliability Results
        if 'ias_reliability' in results_dict:
            ias_rel = results_dict['ias_reliability']
            
            # Overall alpha
            alpha_df = pd.DataFrame([{
                'Scale': 'IAS',
                'Overall_Alpha': ias_rel['overall_alpha']
            }])
            alpha_df.to_excel(writer, sheet_name='IAS_Reliability', index=False, startrow=0)
            
            # Item analysis
            item_data = []
            for item, stats in ias_rel['item_analysis'].items():
                item_data.append({
                    'Item': item,
                    'Alpha_if_Deleted': stats['alpha_if_deleted'],
                    'Item_Total_Correlation': stats['item_total_correlation']
                })
            item_df = pd.DataFrame(item_data)
            item_df.to_excel(writer, sheet_name='IAS_Item_Analysis', index=False)
            
            # Factor alphas if available
            if 'factor_alphas' in ias_rel:
                factor_data = []
                for factor, alpha in ias_rel['factor_alphas'].items():
                    factor_data.append({'Factor': factor, 'Alpha': alpha})
                factor_df = pd.DataFrame(factor_data)
                factor_df.to_excel(writer, sheet_name='IAS_Factor_Alphas', index=False)
        
        # 5. IATS Reliability Results
        if 'iats_reliability' in results_dict:
            iats_rel = results_dict['iats_reliability']
            
            # Overall alpha
            alpha_df = pd.DataFrame([{
                'Scale': 'IATS',
                'Overall_Alpha': iats_rel['overall_alpha']
            }])
            alpha_df.to_excel(writer, sheet_name='IATS_Reliability', index=False, startrow=0)
            
            # Item analysis
            item_data = []
            for item, stats in iats_rel['item_analysis'].items():
                item_data.append({
                    'Item': item,
                    'Alpha_if_Deleted': stats['alpha_if_deleted'],
                    'Item_Total_Correlation': stats['item_total_correlation']
                })
            item_df = pd.DataFrame(item_data)
            item_df.to_excel(writer, sheet_name='IATS_Item_Analysis', index=False)
            
            # Factor alphas if available
            if 'factor_alphas' in iats_rel:
                factor_data = []
                for factor, alpha in iats_rel['factor_alphas'].items():
                    factor_data.append({'Factor': factor, 'Alpha': alpha})
                factor_df = pd.DataFrame(factor_data)
                factor_df.to_excel(writer, sheet_name='IATS_Factor_Alphas', index=False)
        
        # 6. IAS CFA Results
        if 'ias_cfa' in results_dict:
            ias_cfa = results_dict['ias_cfa']
            
            # Fit indices summary
            fit_data = []
            for model_name, results in ias_cfa.items():
                fit_indices = results['fit_indices']
                fit_data.append({
                    'Model': model_name,
                    'Chi2': fit_indices['chi2'],
                    'df': fit_indices['dof'],
                    'p_value': fit_indices['p_value'],
                    'CFI': fit_indices['cfi'],
                    'TLI': fit_indices['tli'],
                    'RMSEA': fit_indices['rmsea']
                })
            fit_summary_df = pd.DataFrame(fit_data)
            fit_summary_df.to_excel(writer, sheet_name='IAS_CFA_Fit_Indices', index=False)
            
            # Individual model loadings
            for model_name, results in ias_cfa.items():
                if 'loadings' in results and not results['loadings'].empty:
                    sheet_name = f'IAS_CFA_{model_name}_Loadings'
                    # Truncate sheet name if too long
                    if len(sheet_name) > 31:
                        sheet_name = f'IAS_{model_name}_Load'
                    results['loadings'].to_excel(writer, sheet_name=sheet_name, index=False)
        
        # 7. IATS CFA Results
        if 'iats_cfa' in results_dict:
            iats_cfa = results_dict['iats_cfa']
            
            # Fit indices summary
            fit_data = []
            for model_name, results in iats_cfa.items():
                fit_indices = results['fit_indices']
                fit_data.append({
                    'Model': model_name,
                    'Chi2': fit_indices['chi2'],
                    'df': fit_indices['dof'],
                    'p_value': fit_indices['p_value'],
                    'CFI': fit_indices['cfi'],
                    'TLI': fit_indices['tli'],
                    'RMSEA': fit_indices['rmsea']
                })
            fit_summary_df = pd.DataFrame(fit_data)
            fit_summary_df.to_excel(writer, sheet_name='IATS_CFA_Fit_Indices', index=False)
            
            # Individual model loadings
            for model_name, results in iats_cfa.items():
                if 'loadings' in results and not results['loadings'].empty:
                    sheet_name = f'IATS_CFA_{model_name}_Loadings'
                    # Truncate sheet name if too long
                    if len(sheet_name) > 31:
                        sheet_name = f'IATS_{model_name}_Load'
                    results['loadings'].to_excel(writer, sheet_name=sheet_name, index=False)
        
        # 8. Regression Results
        if 'regression' in results_dict:
            regression = results_dict['regression']
            
            # Compile all regression results into one table
            reg_data = []
            for dep_var, predictors in regression.items():
                for ind_var, stats in predictors.items():
                    reg_data.append({
                        'Dependent_Variable': dep_var,
                        'Independent_Variable': ind_var,
                        'Coefficient': stats['coefficient'],
                        'Std_Error': stats['std_error'],
                        't_value': stats['t_value'],
                        'p_value': stats['p_value'],
                        'R_squared': stats['r_squared'],
                        'CI_Lower': stats['confidence_interval'][0],
                        'CI_Upper': stats['confidence_interval'][1]
                    })
            
            reg_df = pd.DataFrame(reg_data)
            reg_df.to_excel(writer, sheet_name='Regression_Results', index=False)
            
            # Separate sheets for each dependent variable
            for dep_var in regression.keys():
                dep_data = [row for row in reg_data if row['Dependent_Variable'] == dep_var]
                dep_df = pd.DataFrame(dep_data)
                sheet_name = f'Regression_{dep_var}'
                if len(sheet_name) > 31:
                    sheet_name = f'Reg_{dep_var}'
                dep_df.to_excel(writer, sheet_name=sheet_name, index=False)
    
    print(f"Results saved to: {output_path}")
    print("\nExcel sheets created:")
    print("- Descriptives: Sample characteristics")
    print("- Questionnaire_Stats: Descriptive statistics for scales")
    print("- IAS_PCA_*: Principal component analysis results for IAS")
    print("- IATS_PCA_*: Principal component analysis results for IATS")
    print("- IAS_Reliability, IAS_Item_Analysis: Internal consistency for IAS")
    print("- IATS_Reliability, IATS_Item_Analysis: Internal consistency for IATS")
    print("- IAS_CFA_*, IATS_CFA_*: Confirmatory factor analysis results")
    print("- Regression_*: Regression analysis results")

def main():
    """
    Main analysis function.
    """
    print("=== DUTCH INTEROCEPTIVE SCALES VALIDATION ===")
    print("Starting psychometric validation analyses...\n")
    
    # Configuration
    DATA_PATH = "data/Part1_19082024_clean_JM_version2.sav"  # Update this path
    OUTPUT_PATH = "output/validation_results.xlsx"
    
    try:
        # 1. Load and prepare data
        df = load_data(DATA_PATH)
        df = select_variables(df)
        df = remove_outliers(df)
        
        # 2. Descriptive statistics
        descriptives = describe_sample(df)
        
        # Define item columns
        ias_items = [f'IAS_{i}' for i in range(1, 22)]
        iats_items = [f'IATS_{i}' for i in range(1, 22)]
        
        # 3. Principal Component Analysis
        ias_pca = perform_pca(df, ias_items, "IAS")
        iats_pca = perform_pca(df, iats_items, "IATS")
        
        # 4. Internal consistency
        print("\n=== RELIABILITY ANALYSES ===")
        ias_reliability = calculate_reliability(df, ias_items)
        iats_reliability = calculate_reliability(df, iats_items)
        
        print(f"IAS Overall α = {ias_reliability['overall_alpha']:.3f}")
        print(f"IATS Overall α = {iats_reliability['overall_alpha']:.3f}")
        
        # 5. Confirmatory Factor Analysis
        ias_cfa = run_cfa_models(df, ias_items, ias_pca, "IAS_", "IAS")
        iats_cfa = run_cfa_models(df, iats_items, iats_pca, "IATS_", "IATS")
        
        # 6. Regression analyses
        regression_results = run_regression_analyses(df)
        
        # 7. Compile and save results
        all_results = {
            'descriptives': descriptives,
            'ias_pca': ias_pca,
            'iats_pca': iats_pca,
            'ias_reliability': ias_reliability,
            'iats_reliability': iats_reliability,
            'ias_cfa': ias_cfa,
            'iats_cfa': iats_cfa,
            'regression': regression_results
        }
        
        save_results(all_results, OUTPUT_PATH)
        
        print("\n=== ANALYSIS COMPLETE ===")
        print("All analyses completed successfully!")
        
    except Exception as e:
        print(f"Analysis failed: {e}")
        raise

if __name__ == "__main__":
    main()