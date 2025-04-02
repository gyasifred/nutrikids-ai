import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import scipy.stats as stats
import shap
import lime
import lime.lime_tabular
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from utils import load_and_filter_data
"""
Malnutrition Prediction Analysis Tool
====================================

This script provides a comprehensive framework for analyzing Large Language Model (LLM) 
predictions on malnutrition cases. It combines exploratory data analysis with 
explainable AI techniques (SHAP and LIME) to understand prediction patterns.

Key functionalities:
1. Data loading and preparation
2. Factor extraction from explanations
3. Statistical analysis of factors in correct vs. incorrect predictions
4. Visualization of factor distributions and correlations
5. Surrogate model training for interpretability
6. SHAP and LIME analysis for feature importance
7. Analysis of factor combinations and their relationship with malnutrition

The script helps identify patterns in prediction errors, determine which factors 
are most strongly associated with malnutrition, and provides both statistical and 
visual insights into model behavior.

Usage:
    python malnutrition_analysis.py --file path/to/prediction.csv

Input format:
    The expected input is a CSV file with columns:
    - patient_id: Unique identifier for each patient
    - true_label: Ground truth malnutrition status ('yes' or 'no')
    - predicted_label: LLM predicted malnutrition status ('yes' or 'no')
    - explanation: Text explanation provided by the LLM for its prediction

Output:
    - Statistical summaries printed to console
    - Visualizations saved to 'figures/' directory
    - CSV files with analysis results

"""


def extract_prompt_factors(explanations):
    """
    Extract mentions of specific clinical factors from explanation texts.
    
    This function scans each explanation for keywords related to predefined clinical factors
    and creates a binary matrix indicating which factors were mentioned.
    
    Args:
        explanations (pandas.Series): Series of explanation texts
        
    Returns:
        pandas.DataFrame: Binary DataFrame where each row corresponds to an explanation
                         and each column represents a factor (1=mentioned, 0=not mentioned)
    """
    # Define factors from the prompt based on the provided classification criteria
    prompt_factors = {
        'recent_illness': [
            'illness', 'infection', 'sick', 'disease', 'surgery', 'trauma', 'hospitalization',
            'acute', 'chronic', 'recovery', 'post-operative', 'procedure', 'injury', 'wound',
            'fever', 'immune', 'inflammation', 'cancer', 'tumor', 'sepsis', 'respiratory',
            'pneumonia', 'icu', 'intensive care', 'bed rest', 'immobility', 'immobilization',
            'complications', 'comorbidity', 'organ failure', 'pathology', 'dysfunction'
        ],
        
        'socioeconomic': [
            'socioeconomic', 'poverty', 'food insecurity', 'income', 'access', 'resources',
            'financial', 'economic', 'unemployed', 'unemployment', 'homeless', 'housing',
            'instability', 'education', 'literacy', 'transportation', 'safety', 'vulnerable',
            'marginalized', 'disadvantaged', 'welfare', 'assistance', 'snap', 'wic', 'food stamps',
            'food desert', 'food bank', 'meal program', 'social support', 'cooking facilities',
            'refrigeration', 'utilities', 'social determinants', 'barriers'
        ],
        
        'symptoms': [
            'fatigue', 'weakness', 'appetite', 'weight loss', 'muscle wasting', 'edema',
            'lethargy', 'malaise', 'exhaustion', 'tiredness', 'cachexia', 'sarcopenia',
            'frailty', 'thinness', 'emaciation', 'wasting', 'dehydration', 'swelling',
            'lethargic', 'weak', 'tired', 'anorexia', 'lack of energy', 'hair loss',
            'poor wound healing', 'skin changes', 'pallor', 'brittle nails', 'bruising',
            'anasarca', 'kwashiorkor', 'marasmus', 'poor growth', 'failure to thrive'
        ],
        
        'family_history': [
            'family', 'genetic', 'hereditary', 'familial', 'parent',
            'sibling', 'relative', 'mother', 'father', 'inherited', 'predisposition',
            'risk factor', 'generational', 'ancestry', 'lineage', 'pedigree',
            'household', 'domestic', 'family environment', 'caregiver', 'guardian',
            'family dynamic', 'family structure', 'family function', 'parenting',
            'family medical history', 'relative with condition'
        ],
        
        'lab_results': [
            'albumin', 'hemoglobin', 'lab', 'laboratory', 'vitamin', 'mineral', 'deficiency',
            'protein', 'prealbumin', 'transferrin', 'ferritin', 'iron', 'folate', 'b12',
            'vitamin d', 'vitamin a', 'zinc', 'calcium', 'magnesium', 'phosphorus',
            'electrolytes', 'sodium', 'potassium', 'chloride', 'anemia', 'lymphocyte',
            'creatinine', 'nitrogen balance', 'cholesterol', 'triglycerides', 'glucose',
            'a1c', 'chemistry panel', 'cbc', 'complete blood count', 'micronutrient'
        ],
        
        'medications': [
            'medication', 'drug', 'side effect', 'treatment', 'therapy', 'chemotherapy', 'diuretic',
            'antidepressant', 'antibiotic', 'corticosteroid', 'steroid', 'immunosuppressant',
            'anticonvulsant', 'laxative', 'sedative', 'antipsychotic', 'nsaid', 'opioid',
            'antacid', 'ppi', 'proton pump inhibitor', 'metformin', 'insulin', 'polypharmacy',
            'appetite suppressant', 'stimulant', 'antiemetic', 'prescription', 'over-the-counter',
            'supplement', 'herbal', 'medication adherence', 'regimen', 'dose', 'interaction'
        ],
        
        'mental_health': [
            'depression', 'anxiety', 'mental', 'psychological', 'eating disorder', 'cognitive',
            'psychiatric', 'mood', 'stress', 'trauma', 'ptsd', 'social isolation', 'loneliness',
            'grief', 'bereavement', 'schizophrenia', 'bipolar', 'dementia', 'alzheimer',
            'memory', 'confusion', 'disorientation', 'anorexia nervosa', 'bulimia',
            'binge eating', 'avoidant food intake', 'pica', 'rumination', 'suicidal',
            'self-neglect', 'substance abuse', 'alcohol', 'addiction', 'emotional wellbeing'
        ],
        
        'malabsorption': [
            'diarrhea', 'malabsorption', 'absorption', 'digestive', 'gastrointestinal',
            'celiac', 'crohn', 'ulcerative colitis', 'ibd', 'inflammatory bowel disease',
            'short bowel', 'intestinal resection', 'pancreatic insufficiency', 'cystic fibrosis',
            'bile acid deficiency', 'steatorrhea', 'constipation', 'bloating', 'gas',
            'abdominal pain', 'nausea', 'vomiting', 'dysphagia', 'odynophagia',
            'gerd', 'reflux', 'gastritis', 'enteritis', 'colitis', 'gastroparesis',
            'bariatric surgery', 'ostomy', 'tube feeding', 'parenteral nutrition'
        ],
        
        'weight_height': [
            'weight-for-height', 'weight for height', 'weight/height', 'bmi', 'bmi-for-age', 'bmi for age',
            'anthropometric', 'anthropometry', 'growth chart', 'weight', 'height', 'length',
            'stature', 'percentile', 'growth curve', 'weight trajectory', 'weight velocity',
            'stunting', 'wasting', 'underweight', 'low weight', 'height-for-age', 'weight-for-age',
            'body mass', 'weight status', 'growth faltering', 'growth failure', 'poor growth',
            'weight fluctuation', 'weight change', 'birth weight', 'head circumference'
        ],
        
        'arm_circumference': [
            'mid-upper arm', 'muac', 'arm circumference',
            'mid-arm muscle circumference', 'mamc', 'arm muscle area', 'triceps skinfold',
            'arm anthropometry', 'limb anthropometry', 'upper limb measurement',
            'arm diameter', 'arm girth', 'midupper arm measurement', 'upper arm',
            'arm muscle', 'arm fat', 'anthropometric assessment', 'arm measurement',
            'limb measurement', 'body composition assessment'
        ],
        
        'z_score': [
            'z score', 'z-score', '-1', '-2', '-3',
            'standard deviation', 'percentile', 'sd', 'standard score', 'normalized score',
            'statistical measure', 'growth reference', 'who standards', 'cdc charts',
            'distribution', 'normal curve', 'bell curve', 'reference population',
            'cutoff', 'threshold', 'below average', 'significantly below', 'severely below',
            'moderate deficit', 'severe deficit', 'mild deficit', '+1', '+2', '+3'
        ],
        
        'dietary_intake': [
            'diet', 'nutrition', 'intake', 'consumption', 'eating', 'meal', 'feeding',
            'calorie', 'caloric', 'protein', 'carbohydrate', 'fat', 'nutrient', 'macronutrient',
            'micronutrient', 'diet quality', 'diet diversity', 'meal frequency', 'portion size',
            'restrictive diet', 'elimination diet', 'food allergy', 'food intolerance',
            'food selectivity', 'picky eating', 'food refusal', 'feeding difficulty',
            'appetite regulation', 'hunger', 'satiety', 'meal skipping', 'fasting'
        ],
        
        'hydration': [
            'hydration', 'fluid', 'water', 'dehydration', 'hyperhydration', 'thirst',
            'dry mouth', 'poor skin turgor', 'urine output', 'concentrated urine',
            'fluid balance', 'oral intake', 'liquid consumption', 'drinking', 'beverage',
            'iv fluid', 'intravenous', 'rehydration', 'fluid restriction', 'fluid overload'
        ],
        
        'functional_status': [
            'functional', 'function', 'mobility', 'activity', 'exercise', 'physical activity',
            'adl', 'activities of daily living', 'iadl', 'instrumental activities',
            'independence', 'dependence', 'limitation', 'disability', 'impairment',
            'weakness', 'strength', 'endurance', 'fatigue', 'energy', 'vitality',
            'performance', 'capacity', 'rehabilitation', 'physical therapy', 'occupational therapy'
        ]
    }

    # Initialize factors DataFrame
    factors_df = pd.DataFrame(0, index=range(len(explanations)),
                             columns=list(prompt_factors.keys()))

    # Extract factor mentions
    for i, exp in enumerate(explanations):
        if pd.isna(exp):
            continue

        exp_lower = exp.lower()
        for factor, keywords in prompt_factors.items():
            # Check if any keyword is mentioned in the explanation
            if any(keyword.lower() in exp_lower for keyword in keywords):
                factors_df.loc[i, factor] = 1

    return factors_df

def train_surrogate_model(factors_df, labels):
    """
    Train a RandomForest model as a surrogate for SHAP and LIME analysis.
    
    A surrogate model helps us interpret what factors the LLM may be using
    in its decision-making process.
    
    Args:
        factors_df (pandas.DataFrame): DataFrame of binary factor indicators
        labels (pandas.Series): Binary labels (1 for malnutrition, 0 for none)
        
    Returns:
        tuple: (trained model, test data) for use in explainability techniques
    """
    X_train, X_test, y_train, y_test = train_test_split(factors_df, labels, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    print("Surrogate Model Performance:")
    print(classification_report(y_test, model.predict(X_test)))
    
    # Create confusion matrix for surrogate model
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, model.predict(X_test))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
               xticklabels=['No Malnutrition', 'Malnutrition'],
               yticklabels=['No Malnutrition', 'Malnutrition'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Surrogate Model Confusion Matrix')
    plt.tight_layout()
    plt.savefig('figures/surrogate_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return model, X_test

def shap_analysis(model, X_test):
    """
    Perform SHAP (SHapley Additive exPlanations) analysis to determine feature importance.
    
    SHAP values help understand how each feature contributes to predictions,
    both globally and for individual instances.
    
    Args:
        model: Trained surrogate model
        X_test (pandas.DataFrame): Test data containing factor indicators
        
    Returns:
        None (saves visualizations to files)
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    
    # Summary plot showing overall feature importance
    plt.figure(figsize=(12, 6))
    shap.summary_plot(shap_values[1], X_test, show=False)
    plt.savefig('figures/shap_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Dependence plots for top features
    for feature in X_test.columns[:5]:  # Plot top 5 features
        plt.figure(figsize=(8, 5))
        shap.dependence_plot(feature, shap_values[1], X_test, show=False)
        plt.savefig(f'figures/shap_dependence_{feature}.png', dpi=300, bbox_inches='tight')
        plt.close()

def lime_analysis(model, X_test, sample_indices=None):
    """
    Perform LIME (Local Interpretable Model-agnostic Explanations) analysis on sample instances.
    
    LIME helps understand model decisions for specific instances by creating
    local approximations of the model.
    
    Args:
        model: Trained surrogate model
        X_test (pandas.DataFrame): Test data containing factor indicators
        sample_indices (list, optional): Indices of samples to explain. 
                                        Defaults to [0] if None.
        
    Returns:
        None (saves HTML explanations to files)
    """
    # Create explainer for the surrogate model
    explainer = lime.lime_tabular.LimeTabularExplainer(
        X_test.values, 
        feature_names=X_test.columns,
        class_names=['No', 'Yes'], 
        discretize_continuous=True
    )
    
    # If no indices provided, use first instance
    if sample_indices is None:
        sample_indices = [0]
    
    # Generate explanations for each sample
    for idx in sample_indices:
        if idx < len(X_test):
            exp = explainer.explain_instance(
                X_test.iloc[idx].values, 
                model.predict_proba
            )
            exp.save_to_file(f'figures/lime_explanation_sample_{idx}.html')

def analyze_factor_combinations(factors_df, malnutrition_status):
    """
    Analyze combinations of factors and their association with malnutrition.
    
    This function identifies which combinations of factors are most strongly
    associated with malnutrition status.
    
    Args:
        factors_df (pandas.DataFrame): DataFrame with binary indicators for factors
        malnutrition_status (pandas.Series): Series with malnutrition decisions ('yes'/'no')
        
    Returns:
        pandas.DataFrame: Factor combinations ranked by their association with malnutrition
    """
    # Convert malnutrition status to binary (1 for "yes", 0 for "no")
    y = (malnutrition_status == 'yes').astype(int)

    # Find all combinations of factors (up to 3 factors)
    max_factors = min(3, len(factors_df.columns))
    combinations = []

    # For each possible combination size
    for k in range(1, max_factors + 1):
        # Generate all combinations of k factors
        for combo in itertools.combinations(factors_df.columns, k):
            # Get rows where all factors in this combination are present
            mask = factors_df[list(combo)].all(axis=1)

            # Skip if too few examples
            if mask.sum() < 5:
                continue

            # Calculate association with malnutrition
            malnutrition_rate = y[mask].mean()
            baseline_rate = y.mean()
            lift = malnutrition_rate / baseline_rate if baseline_rate > 0 else 0

            combinations.append({
                'combination': ' + '.join(combo),
                'num_factors': k,
                'count': mask.sum(),
                'malnutrition_rate': malnutrition_rate,
                'baseline_rate': baseline_rate,
                'lift': lift
            })

    # Convert to DataFrame and sort
    combinations_df = pd.DataFrame(combinations)
    if len(combinations_df) > 0:
        combinations_df = combinations_df.sort_values('lift', ascending=False)

    return combinations_df

def statistical_testing(correct_factors, incorrect_factors):
    """
    Perform statistical tests to compare factor differences between correct and incorrect predictions.
    
    Uses chi-square tests for categorical differences and t-tests for mean differences.
    
    Args:
        correct_factors (pandas.DataFrame): Factor matrix for correct predictions
        incorrect_factors (pandas.DataFrame): Factor matrix for incorrect predictions
        
    Returns:
        pandas.DataFrame: Results of statistical tests for each factor
    """
    factor_names = correct_factors.columns
    results = []
    
    for factor in factor_names:
        correct_vals = correct_factors[factor]
        incorrect_vals = incorrect_factors[factor]
        
        # Chi-square test for categorical differences
        chi2, p_chi2 = stats.chisquare([correct_vals.sum(), incorrect_vals.sum()])
        
        # T-test for mean differences
        t_stat, p_ttest = stats.ttest_ind(correct_vals, incorrect_vals, equal_var=False)
        
        results.append({
            'factor': factor,
            'correct_mean': correct_vals.mean(),
            'incorrect_mean': incorrect_vals.mean(),
            'difference': correct_vals.mean() - incorrect_vals.mean(),
            'chi2_p_value': p_chi2,
            'ttest_p_value': p_ttest
        })
    
    results_df = pd.DataFrame(results).sort_values('ttest_p_value')
    results_df.to_csv('figures/statistical_tests.csv', index=False)
    print("Statistical tests saved to figures/statistical_tests.csv")
    
    return results_df

def plot_factor_analysis(factors_df, malnutrition_status, factor_combinations=None):
    """
    Create visualizations of factor analysis results.
    
    Args:
        factors_df (pandas.DataFrame): DataFrame with binary indicators for factors
        malnutrition_status (pandas.Series): Series with malnutrition decisions
        factor_combinations (pandas.DataFrame, optional): DataFrame with factor combinations
        
    Returns:
        pandas.DataFrame: Factor statistics
    """
    # Convert malnutrition status to binary
    y = (malnutrition_status == 'yes').astype(int)

    # Calculate factor prevalence and association with malnutrition
    factor_stats = pd.DataFrame(index=factors_df.columns)
    factor_stats['prevalence'] = factors_df.mean()
    factor_stats['malnutrition_correlation'] = [factors_df[col].corr(y) for col in factors_df.columns]
    factor_stats['present_in_malnutrition'] = [factors_df[col][y == 1].mean() for col in factors_df.columns]
    factor_stats['present_in_normal'] = [factors_df[col][y == 0].mean() for col in factors_df.columns]
    factor_stats = factor_stats.sort_values('malnutrition_correlation', ascending=False)

    # Create figures directory if it doesn't exist
    import os
    if not os.path.exists('figures'):
        os.makedirs('figures')

    # Plot factor correlation with malnutrition
    plt.figure(figsize=(12, 8))
    sns.barplot(x='malnutrition_correlation', y=factor_stats.index, data=factor_stats.reset_index(),
               palette='viridis')
    plt.title('Correlation of Prompt Factors with Malnutrition Status', fontsize=16)
    plt.xlabel('Correlation Coefficient', fontsize=14)
    plt.ylabel('Factor', fontsize=14)
    plt.tight_layout()
    plt.savefig('figures/factor_correlation.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Plot factor prevalence by malnutrition status
    plt.figure(figsize=(14, 8))
    comparison_data = factor_stats[['present_in_malnutrition', 'present_in_normal']].copy()
    comparison_data = pd.melt(comparison_data.reset_index(), id_vars=['index'],
                             value_vars=['present_in_malnutrition', 'present_in_normal'],
                             var_name='status', value_name='prevalence')

    sns.barplot(x='index', y='prevalence', hue='status', data=comparison_data, palette='Set2')
    plt.title('Prevalence of Factors by Malnutrition Status', fontsize=16)
    plt.xlabel('Factor', fontsize=14)
    plt.ylabel('Prevalence', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Status')
    plt.tight_layout()
    plt.savefig('figures/factor_prevalence_by_status.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Plot top factor combinations if available
    if factor_combinations is not None and len(factor_combinations) > 0:
        plt.figure(figsize=(14, 10))
        top_combos = factor_combinations.head(15) if len(factor_combinations) >= 15 else factor_combinations
        sns.barplot(x='lift', y='combination', hue='num_factors', data=top_combos, palette='viridis')
        plt.title('Top Factor Combinations Associated with Malnutrition', fontsize=16)
        plt.xlabel('Lift (Relative Risk)', fontsize=14)
        plt.ylabel('Factor Combination', fontsize=14)
        plt.legend(title='Number of Factors')
        plt.tight_layout()
        plt.savefig('figures/factor_combinations.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Plot distribution of combination sizes
        plt.figure(figsize=(10, 6))
        sns.countplot(x='num_factors', data=factor_combinations, palette='viridis')
        plt.title('Distribution of Factor Combination Sizes', fontsize=16)
        plt.xlabel('Number of Factors in Combination', fontsize=14)
        plt.ylabel('Count', fontsize=14)
        plt.tight_layout()
        plt.savefig('figures/factor_combination_sizes.png', dpi=300, bbox_inches='tight')
        plt.close()

    return factor_stats

def analyze_prediction_types(data_dict):
    """
    Analyze factors by prediction type (TP, TN, FP, FN).
    
    Args:
        data_dict (dict): Dictionary with different subsets of predictions
        
    Returns:
        pandas.DataFrame: Factor prevalence by prediction type
    """
    # Analysis by prediction type (TP, TN, FP, FN)
    prediction_types = {
        'True Positive': data_dict['correct_yes'],
        'True Negative': data_dict['correct_no'],
        'False Positive': data_dict['incorrect_yes'],
        'False Negative': data_dict['incorrect_no']
    }

    print("\nAnalyzing factors by prediction type (TP, TN, FP, FN)...")

    # Extract factors for each prediction type
    type_factors = {}
    type_prevalence = {}

    for pred_type, df in prediction_types.items():
        if len(df) > 0:
            type_factors[pred_type] = extract_prompt_factors(df['explanation'])
            type_prevalence[pred_type] = type_factors[pred_type].mean()

    # Create combined DataFrame for all prediction types
    all_prevalence = pd.DataFrame(type_prevalence)

    # Plot factor prevalence by prediction type
    plt.figure(figsize=(14, 10))
    all_prevalence_melted = pd.melt(all_prevalence.reset_index(),
                                   id_vars=['index'],
                                   var_name='prediction_type',
                                   value_name='prevalence')

    sns.barplot(x='prevalence', y='index', hue='prediction_type',
               data=all_prevalence_melted, palette='Set2')
    plt.title('Factor Prevalence by Prediction Type', fontsize=16)
    plt.xlabel('Prevalence', fontsize=14)
    plt.ylabel('Factor', fontsize=14)
    plt.legend(title='Prediction Type')
    plt.tight_layout()
    plt.savefig('figures/factor_by_prediction_type.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return all_prevalence

def analyze_llm_predictions(file_path):
    """
    Main function to analyze LLM predictions on malnutrition cases.
    
    Performs a comprehensive analysis of model performance, factor importance,
    and error patterns.
    
    Args:
        file_path (str): Path to the CSV file with LLM predictions
        
    Returns:
        dict: Dictionary containing analysis results
    """
    # Create figures directory if it doesn't exist
    import os
    if not os.path.exists('figures'):
        os.makedirs('figures')
    
    # Load and split the data
    data_dict = load_and_filter_data(file_path)
    full_df = data_dict['full_df']

    print(f"Dataset size: {len(full_df)} records")
    print(f"Correct predictions: {len(data_dict['correct_predictions'])} ({len(data_dict['correct_predictions'])/len(full_df):.2%})")
    print(f"Incorrect predictions: {len(data_dict['incorrect_predictions'])} ({len(data_dict['incorrect_predictions'])/len(full_df):.2%})")

    # Classification metrics
    y_true = (full_df['true_label'] == 'yes').astype(int)
    y_pred = (full_df['predicted_label'] == 'yes').astype(int)

    print("\nClassification Metrics:")
    print(classification_report(y_true, y_pred, target_names=['No Malnutrition', 'Malnutrition']))

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No Malnutrition', 'Malnutrition'],
                yticklabels=['No Malnutrition', 'Malnutrition'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('figures/confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Extract prompt factors from explanations
    print("\nExtracting prompt factors from explanations...")
    factors_df = extract_prompt_factors(full_df['explanation'])

    # Add factors to the original dataframe
    full_df_with_factors = pd.concat([full_df, factors_df], axis=1)
    full_df_with_factors.to_csv('figures/full_data_with_factors.csv', index=False)

    # Calculate factor statistics for the full dataset
    print("\nAnalyzing factors across all predictions...")
    factor_stats = plot_factor_analysis(factors_df, full_df['true_label'])

    # Print top factors associated with malnutrition
    print("\nTop factors associated with malnutrition:")
    print(factor_stats.sort_values('malnutrition_correlation', ascending=False).head(5))

    # Train surrogate model for explainability
    print("\nTraining surrogate model for explainability...")
    surrogate_model, X_test = train_surrogate_model(factors_df, y_true)
    
    # SHAP Analysis
    print("\nPerforming SHAP analysis...")
    shap_analysis(surrogate_model, X_test)
    
    # LIME Analysis (on sample instances)
    print("\nPerforming LIME analysis...")
    sample_indices = [0, 1, 2]  # Analyze first three samples
    lime_analysis(surrogate_model, X_test, sample_indices)

    # Analyze factor combinations
    print("\nAnalyzing factor combinations...")
    factor_combinations = analyze_factor_combinations(factors_df, full_df['true_label'])

    if len(factor_combinations) > 0:
        # Print top factor combinations
        print("\nTop factor combinations associated with malnutrition:")
        print(factor_combinations.head(5))
        factor_combinations.to_csv('figures/factor_combinations.csv', index=False)

        # Generate plots with combinations
        plot_factor_analysis(factors_df, full_df['true_label'], factor_combinations)
    else:
        print("Not enough data to analyze factor combinations.")

    # Separate analysis for correct vs. incorrect predictions
    print("\nAnalyzing factors in correct vs. incorrect predictions...")

    # Extract factors for correct and incorrect predictions
    correct_factors = extract_prompt_factors(data_dict['correct_predictions']['explanation'])
    incorrect_factors = extract_prompt_factors(data_dict['incorrect_predictions']['explanation'])

    # Statistical testing of differences
    stat_results = statistical_testing(correct_factors, incorrect_factors)

    # Calculate factor prevalence in correct vs. incorrect predictions
    correct_prevalence = correct_factors.mean()
    incorrect_prevalence = incorrect_factors.mean()

    # Calculate difference in factor usage
    factor_diff = pd.DataFrame({
        'correct_prevalence': correct_prevalence,
        'incorrect_prevalence': incorrect_prevalence,
        'difference': correct_prevalence - incorrect_prevalence
    })
    factor_diff = factor_diff.sort_values('difference', ascending=False)
    factor_diff.to_csv('figures/factor_difference.csv', index=True)

    # Plot difference in factor usage
    plt.figure(figsize=(12, 8))
    sns.barplot(x='difference', y=factor_diff.index, data=factor_diff.reset_index(), palette='RdBu_r')
    plt.axvline(x=0, color='black', linestyle='-', alpha=0.5)
    plt.title('Difference in Factor Usage: Correct vs. Incorrect Predictions', fontsize=16)
    plt.xlabel('Difference in Prevalence (Correct - Incorrect)', fontsize=14)
    plt.ylabel('Factor', fontsize=14)
    plt.tight_layout()
    plt.savefig('figures/factor_difference.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Print factors more common in correct predictions
    print("\nFactors more common in correct predictions:")
    print(factor_diff[factor_diff['difference'] > 0].sort_values('difference', ascending=False).head(5))

    # Print factors more common in incorrect predictions
    print("\nFactors more common in incorrect predictions:")
    print(factor_diff[factor_diff['difference'] < 0].sort_values('difference').head(5))

    # Analysis by prediction type (TP, TN, FP, FN)
    prediction_type_prevalence = analyze_prediction_types(data_dict)
    prediction_type_prevalence.to_csv('figures/prediction_type_factor_prevalence.csv')

    # Return analysis results
    results = {
        'full_df_with_factors': full_df_with_factors,
        'factor_stats': factor_stats,
        'factor_combinations': factor_combinations,
        'statistical_tests': stat_results,
        'factor_difference': factor_diff,
        'prediction_type_prevalence': prediction_type_prevalence,
        'surrogate_model': surrogate_model
    }

    print("\nAnalysis complete. Results saved to figures/ directory.")
    return results

def main(file_path="prediction.csv"):
    """
    Entry point for the script.
    
    Args:
        file_path (str): Path to the CSV file with LLM predictions
        
    Returns:
        None
    """
    print(f"Analyzing LLM malnutrition predictions from: {file_path}")
    analysis_results = analyze_llm_predictions(file_path)
    
    # Generate summary report
    with open('figures/analysis_summary.txt', 'w') as f:
        f.write("Malnutrition Prediction Analysis Summary\n")
        f.write("=======================================\n\n")
        f.write(f"Dataset: {file_path}\n")
        f.write(f"Total records: {len(analysis_results['full_df_with_factors'])}\n\n")
        
        f.write("Top factors associated with malnutrition:\n")
        for factor in analysis_results['factor_stats'].sort_values('malnutrition_correlation', ascending=False).head(5).index:
            f.write(f"- {factor}\n")
        
        f.write("\nFactors more common in correct predictions:\n")
        for factor in analysis_results['factor_difference'][analysis_results['factor_difference']['difference'] > 0].sort_values('difference', ascending=False).head(3).index:
            f.write(f"- {factor}\n")
            
        f.write("\nFactors more common in incorrect predictions:\n")
        for factor in analysis_results['factor_difference'][analysis_results['factor_difference']['difference'] < 0].sort_values('difference').head(3).index:
            f.write(f"- {factor}\n")
    
    print("Analysis summary saved to figures/analysis_summary.txt")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze LLM malnutrition predictions')
    parser.add_argument('--file', type=str, default="./llama_zero_shot/predictions.csv",
                        help='Path to the CSV file with predictions')
    
    args = parser.parse_args()
    main(args.file)
