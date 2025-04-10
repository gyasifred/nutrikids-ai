#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from scipy import stats
from statsmodels.stats.proportion import proportion_confint
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score, confusion_matrix
import warnings
from utils import (
    load_and_filter_data,
    extract_clinical_measurements,
    extract_criteria_mentions,
    analyze_measurement_thresholds,
    analyze_criteria_frequency,
    analyze_criteria_correlation,
    visualize_criteria_frequency,
    plot_measurement_distributions,
    analyze_severity_classifications,
    generate_measurement_summary,
    analyze_measurement_criteria_alignment,
    analyze_clinical_symptoms,
    analyze_dietary_factors,
    analyze_risk_factors,
    extract_patient_demographics
)

# Set plotting style
plt.style.use('seaborn')
sns.set_palette("Set2")
plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['font.size'] = 12

def main():
    # ------------------------------------------------
    # 1. Configuration and Data Loading
    # ------------------------------------------------
    
    # Create output directories
    os.makedirs('figures', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    os.makedirs('tables', exist_ok=True)
    
    # Load data
    file_path = "./llama_zero_shot/prediction.csv"
    data = load_and_filter_data(file_path)
    
    # Define comprehensive criteria dictionary
    criteria_dict = {
        # Anthropometric measurements
        'BMI': ['bmi', 'body mass index', 'kg/m2', 'quetelet index'],
        'weight_for_height': ['weight for height', 'weight-for-height', 'wfh', 'whz'],
        'BMI_for_age': ['bmi for age', 'bmi-for-age', 'baz'],
        'MUAC': ['muac', 'mid upper arm circumference', 'mid-upper arm circumference', 'arm circumference'],
        'weight_loss': ['weight loss', 'lost weight', 'decrease in weight', 'declining weight', 'unintentional weight loss'],
        
        # Clinical symptoms
        'muscle_wasting': ['muscle wasting', 'muscle loss', 'decreased muscle mass', 'muscle atrophy', 'sarcopenia'],
        'fatigue': ['fatigue', 'weakness', 'tired', 'low energy', 'lethargy'],
        'skin_changes': ['skin changes', 'dry skin', 'thin skin', 'skin breakdown', 'poor skin turgor'],
        'hair_changes': ['hair changes', 'hair loss', 'thin hair', 'brittle hair'],
        'edema': ['edema', 'swelling', 'fluid retention', 'pitting edema'],
        
        # Dietary intake
        'inadequate_intake': ['inadequate intake', 'poor intake', 'decreased intake', 'reduced appetite'],
        'caloric_deficit': ['caloric deficit', 'insufficient calories', 'low calorie', 'calorie restriction'],
        'protein_deficit': ['protein deficit', 'low protein', 'insufficient protein', 'protein-energy malnutrition'],
        'food_insecurity': ['food insecurity', 'limited access to food', 'cannot afford food', 'food scarcity'],
        
        # Medical conditions
        'chronic_illness': ['chronic illness', 'chronic disease', 'comorbidity', 'long-term condition'],
        'gi_disorders': ['gi disorder', 'gastrointestinal disorder', 'malabsorption', 'diarrhea', 'vomiting'],
        'infection': ['infection', 'sepsis', 'inflammatory', 'infectious process'],
        'cancer': ['cancer', 'malignancy', 'oncology', 'tumor', 'chemotherapy'],
        
        # Risk factors
        'medications': ['medication', 'drug induced', 'steroid', 'chemotherapy', 'immunosuppressants'],
        'mental_health': ['depression', 'dementia', 'cognitive impairment', 'psychiatric', 'anxiety'],
        'socioeconomic': ['socioeconomic', 'homeless', 'poverty', 'financial', 'low income'],
        'functional_status': ['functional decline', 'immobility', 'bed bound', 'decreased activity'],
        
        # Lab markers
        'lab_markers': ['albumin', 'prealbumin', 'transferrin', 'hemoglobin', 'lymphocyte', 'protein'],
        
        # Special categories
        'pediatric': ['child', 'infant', 'pediatric', 'growth chart', 'percentile'],
        'geriatric': ['elderly', 'geriatric', 'frail', 'aging', 'older adult']
    }

    # ------------------------------------------------
    # 2. Comprehensive Analysis Pipeline
    # ------------------------------------------------
    
    # Initialize results dictionary
    results = {
        'data': data,
        'criteria_analysis': {},
        'measurement_analysis': {},
        'symptom_analysis': {},
        'dietary_analysis': {},
        'risk_factor_analysis': {},
        'demographics': {}
    }
    
    # A. Criteria Mention Analysis
    print("Running criteria mention analysis...")
    for key in ['full_df', 'correct_predictions', 'incorrect_predictions', 
                'correct_yes', 'correct_no', 'incorrect_yes', 'incorrect_no']:
        results['criteria_analysis'][key] = extract_criteria_mentions(
            data[key]['explanation'], 
            criteria_dict
        )
    
    # B. Clinical Measurement Extraction
    print("Extracting clinical measurements...")
    for key in ['full_df', 'correct_predictions', 'incorrect_predictions', 
                'correct_yes', 'correct_no', 'incorrect_yes', 'incorrect_no']:
        results['measurement_analysis'][key] = extract_clinical_measurements(
            data[key]['explanation']
        )
    
    # C. Patient Demographics
    print("Extracting patient demographics...")
    results['demographics'] = extract_patient_demographics(data['full_df']['explanation'])
    
    # D. Clinical Symptom Analysis
    print("Analyzing clinical symptoms...")
    results['symptom_analysis'] = analyze_clinical_symptoms(
        data['full_df']['explanation'], 
        data['full_df']['true_label']
    )
    
    # E. Dietary Factor Analysis
    print("Analyzing dietary factors...")
    results['dietary_analysis'] = analyze_dietary_factors(
        data['full_df']['explanation'], 
        data['full_df']['true_label']
    )
    
    # F. Risk Factor Analysis
    print("Analyzing risk factors...")
    results['risk_factor_analysis'] = analyze_risk_factors(
        data['full_df']['explanation'], 
        data['full_df']['true_label']
    )
    
    # ------------------------------------------------
    # 3. Comparative Analysis by Prediction Accuracy
    # ------------------------------------------------
    
    # Initialize comparative results
    comparative_results = {
        'correct_vs_incorrect': {},
        'true_positives': {},
        'true_negatives': {},
        'false_positives': {},
        'false_negatives': {}
    }
    
    # A. Criteria Frequency Comparison
    print("Comparing criteria frequency...")
    for group in ['correct_predictions', 'incorrect_predictions']:
        comparative_results['correct_vs_incorrect'][group] = analyze_criteria_frequency(
            results['criteria_analysis'][group],
            data[group]['true_label']
        )
    
    # B. Threshold Analysis
    print("Analyzing measurement thresholds...")
    for group in ['correct_predictions', 'incorrect_predictions']:
        comparative_results['correct_vs_incorrect'][f'{group}_thresholds'] = analyze_measurement_thresholds(
            results['measurement_analysis'][group],
            data[group]['true_label']
        )
    
    # C. Four-group Analysis (TP, TN, FP, FN)
    print("Analyzing four prediction groups...")
    for group, label in [('correct_yes', 'true_positives'), 
                         ('correct_no', 'true_negatives'),
                         ('incorrect_yes', 'false_positives'),
                         ('incorrect_no', 'false_negatives')]:
        # Criteria correlation
        comparative_results[label]['criteria_correlation'] = analyze_criteria_correlation(
            results['criteria_analysis'][group],
            data[group]['true_label']
        )
        
        # Measurement distributions
        if len(results['measurement_analysis'][group]) > 0:
            comparative_results[label]['measurement_distributions'] = plot_measurement_distributions(
                results['measurement_analysis'][group],
                data[group]['true_label']
            )
        
        # Severity classifications
        comparative_results[label]['severity_classifications'] = analyze_severity_classifications(
            results['measurement_analysis'][group],
            data[group]['true_label']
        )
    
    # ------------------------------------------------
    # 4. Visualization and Output Generation
    # ------------------------------------------------
    
    print("Generating visualizations and reports...")
    
    # A. Criteria Frequency Visualization
    freq_fig = visualize_criteria_frequency(
        comparative_results['correct_vs_incorrect']['correct_predictions'],
        top_n=20
    )
    freq_fig.savefig('figures/criteria_frequency_comparison.png', dpi=300, bbox_inches='tight')
    plt.close(freq_fig)
    
    # B. Measurement Distribution Plots
    for group in ['correct_predictions', 'incorrect_predictions']:
        if len(results['measurement_analysis'][group]) > 0:
            figs = plot_measurement_distributions(
                results['measurement_analysis'][group],
                data[group]['true_label']
            )
            
            for measure, fig in figs.items():
                fig.savefig(f'figures/{group}_{measure}_distribution.png', dpi=300, bbox_inches='tight')
                plt.close(fig)
    
    # C. Generate Comprehensive Report
    report_sections = []
    
    # 1. Executive Summary
    report_sections.append("""
MALNUTRITION PREDICTION ANALYSIS REPORT
======================================

This report provides a comprehensive analysis of malnutrition prediction performance, 
including criteria utilization patterns, measurement distributions, and clinical 
factor associations across correctly and incorrectly classified cases.
""")
    
    # 2. Dataset Overview
    report_sections.append(f"""
DATASET OVERVIEW
----------------
Total cases: {len(data['full_df'])}
Correct predictions: {len(data['correct_predictions'])} ({len(data['correct_predictions'])/len(data['full_df']):.1%})
Incorrect predictions: {len(data['incorrect_predictions'])} ({len(data['incorrect_predictions'])/len(data['full_df']):.1%})

Breakdown:
- True Positives: {len(data['correct_yes'])}
- True Negatives: {len(data['correct_no'])}
- False Positives: {len(data['incorrect_yes'])}
- False Negatives: {len(data['incorrect_no'])}
""")
    
    # 3. Criteria Analysis Summary
    report_sections.append("""
CRITERIA ANALYSIS SUMMARY
------------------------
""")
    
    # Top criteria in correct predictions
    top_correct = comparative_results['correct_vs_incorrect']['correct_predictions'].nlargest(5, 'risk_ratio')
    report_sections.append("Most predictive criteria in correct predictions:")
    for _, row in top_correct.iterrows():
        report_sections.append(f"- {row['criteria']}: RR={row['risk_ratio']:.1f}, p={row['p_value']:.4f}")
    
    # Top criteria in incorrect predictions
    top_incorrect = comparative_results['correct_vs_incorrect']['incorrect_predictions'].nlargest(5, 'risk_ratio')
    report_sections.append("
Most problematic criteria in incorrect predictions:")
    for _, row in top_incorrect.iterrows():
        report_sections.append(f"- {row['criteria']}: RR={row['risk_ratio']:.1f}, p={row['p_value']:.4f}")
    
    # 4. Measurement Analysis Summary
    report_sections.append("""

MEASUREMENT ANALYSIS SUMMARY
---------------------------
""")
    
    # Threshold performance
    for group in ['correct_predictions', 'incorrect_predictions']:
        thresholds = comparative_results['correct_vs_incorrect'][f'{group}_thresholds']
        report_sections.append(f"
Measurement performance in {group.replace('_', ' ')}:")
        
        for measure, stats in thresholds.items():
            if 'error' not in stats:
                report_sections.append(
                    f"- {measure}: AUC={stats['auc']:.2f}, "
                    f"Threshold={stats['optimal_threshold']:.1f} "
                    f"(Sens={stats['sensitivity']:.1%}, Spec={stats['specificity']:.1%})"
                )
    
    # 5. Clinical Factor Analysis
    report_sections.append("""

CLINICAL FACTOR ANALYSIS
------------------------
""")
    
    # Symptoms
    report_sections.append("
Top clinical symptoms associated with malnutrition:")
    top_symptoms = results['symptom_analysis'].nlargest(5, 'prevalence_ratio')
    for _, row in top_symptoms.iterrows():
        report_sections.append(
            f"- {row['symptom']}: PR={row['prevalence_ratio']:.1f}, "
            f"AR={row['attributable_risk']:.2f}"
        )
    
    # Dietary factors
    report_sections.append("
Top dietary factors associated with malnutrition:")
    top_dietary = results['dietary_analysis'].nlargest(5, 'odds_ratio')
    for _, row in top_dietary.iterrows():
        report_sections.append(
            f"- {row['factor']}: OR={row['odds_ratio']:.1f}"
        )
    
    # Risk factors
    report_sections.append("
Top risk factors associated with malnutrition:")
    top_risk = results['risk_factor_analysis'].nlargest(5, 'relative_risk')
    for _, row in top_risk.iterrows():
        report_sections.append(
            f"- {row['factor']}: RR={row['relative_risk']:.1f}, "
            f"AR={row['attributable_risk']:.2f}"
        )
    
    # 6. Recommendations
    report_sections.append("""

RECOMMENDATIONS
---------------
1. Focus on improving recognition of: """ + ", ".join(top_incorrect['criteria'].tolist()) + """
2. Validate measurement thresholds for: """ + ", ".join(list(comparative_results['correct_vs_incorrect']['correct_predictions_thresholds'].keys())[:3]) + """
3. Consider additional training on: """ + ", ".join(top_risk['factor'].tolist()[:3]) + """
""")
    
    # Save full report
    with open('results/full_analysis_report.txt', 'w') as f:
        f.write("\n".join(report_sections))
    
    # ------------------------------------------------
    # 5. Data Export
    # ------------------------------------------------
    
    print("Exporting results to files...")
    
    # A. Save DataFrames to CSV
    for category in ['criteria_analysis', 'measurement_analysis']:
        for key, df in results[category].items():
            df.to_csv(f'tables/{category}_{key}.csv', index=False)
    
    # B. Save analysis results
    pd.DataFrame(results['symptom_analysis']).to_csv('tables/symptom_analysis.csv', index=False)
    pd.DataFrame(results['dietary_analysis']).to_csv('tables/dietary_analysis.csv', index=False)
    pd.DataFrame(results['risk_factor_analysis']).to_csv('tables/risk_factor_analysis.csv', index=False)
    results['demographics'].to_csv('tables/patient_demographics.csv', index=False)
    
    # C. Save comparative results
    for group in ['correct_predictions', 'incorrect_predictions']:
        comparative_results['correct_vs_incorrect'][group].to_csv(
            f'tables/{group}_criteria_frequency.csv', 
            index=False
        )
    
    # ------------------------------------------------
    # 6. Final Output
    # ------------------------------------------------
    
    print("""
Analysis complete!
------------------
Results saved to:
- /figures/      : Visualizations
- /results/      : Summary reports
- /tables/       : Detailed data tables

Key findings have been compiled in results/full_analysis_report.txt
""")
    
    return {
        'results': results,
        'comparative_results': comparative_results
    }

if __name__ == "__main__":
    analysis_results = main()
