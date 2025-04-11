#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# Removed seaborn import as it's not necessary for the plotting style anymore
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
plt.style.use('ggplot')  # Use a valid style like 'ggplot'
# sns.set_palette("Set2")  # Removed seaborn's set_palette call since sns is no longer needed
plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['font.size'] = 12

def analyze_reasoning_consistency(explanation, original_note):
    """
    Analyze the consistency between the explanation and original note.
    
    Args:
        explanation (str): The model-generated explanation
        original_note (str): The original clinical input note
        
    Returns:
        dict: Results of consistency analysis
    """
    # Convert to lowercase for case-insensitive matching
    exp_lower = explanation.lower()
    note_lower = original_note.lower()
    
    # Find potential hallucinations (statements in explanation not in original note)
    # This is a simple approach - a more sophisticated NLP approach would be better
    key_findings = []
    hallucinations = []
    unsupported_claims = []
    
    # Extract sentences from explanation
    explanation_sentences = re.split(r'(?<=[.!?])\s+', explanation)
    
    for sentence in explanation_sentences:
        # Skip very short sentences
        if len(sentence.strip()) < 15:
            continue
            
        # Look for key clinical terms that might indicate reasoning
        clinical_terms = ['bmi', 'weight', 'malnutrition', 'diagnosis', 
                         'evidence', 'symptom', 'nutrition', 'intake']
                         
        has_clinical_term = any(term in sentence.lower() for term in clinical_terms)
        
        # Check if key sentence content appears in original note
        sentence_tokens = set(re.findall(r'\b\w+\b', sentence.lower()))
        matching_words = sum(1 for word in sentence_tokens if word in note_lower)
        matching_ratio = matching_words / len(sentence_tokens) if sentence_tokens else 0
        
        if has_clinical_term:
            if matching_ratio < 0.3:  # Less than 30% of words match
                hallucinations.append(sentence)
            elif matching_ratio < 0.5:  # Between 30-50% match
                unsupported_claims.append(sentence)
            elif matching_ratio > 0.7:  # Strong match
                key_findings.append(sentence)
    
    # Check for overconfident language not supported by evidence
    overconfident_terms = ['clearly', 'definitely', 'absolutely', 'certainly', 
                          'undoubtedly', 'obvious', 'evident', 'conclusive']
    
    overconfident_statements = []
    for term in overconfident_terms:
        pattern = rf'\b{term}\b'
        matches = re.finditer(pattern, exp_lower)
        for match in matches:
            # Extract the surrounding context (sentence containing the term)
            start = max(0, exp_lower.rfind('.', 0, match.start()) + 1)
            end = exp_lower.find('.', match.end())
            if end == -1:
                end = len(exp_lower)
            context = explanation[start:end+1].strip()
            overconfident_statements.append(context)
    
    return {
        'key_findings': key_findings,
        'hallucinations': hallucinations,
        'unsupported_claims': unsupported_claims,
        'overconfident_statements': overconfident_statements,
        'hallucination_count': len(hallucinations),
        'unsupported_claim_count': len(unsupported_claims),
        'overconfidence_count': len(overconfident_statements)
    }

def analyze_reasoning_patterns(data_dict):
    """
    Analyze reasoning patterns across different prediction categories.
    
    Args:
        data_dict (dict): Dictionary containing the categorized DataFrames
        
    Returns:
        dict: Analysis results
    """
    results = {}
    
    # Analyze each prediction category
    for category in ['TP_data', 'TN_data', 'FP_data', 'FN_data']:
        df = data_dict[category]
        if len(df) == 0:
            results[category] = {'error': f'No data in {category}'}
            continue
            
        # Initialize counters
        reasoning_analysis = {
            'total_records': len(df),
            'hallucination_counts': [],
            'unsupported_claim_counts': [],
            'overconfidence_counts': [],
            'examples': {
                'hallucinations': [],
                'unsupported_claims': [],
                'overconfident_statements': []
            }
        }
        
        # Analyze each record in the category
        for _, row in df.iterrows():
            analysis = analyze_reasoning_consistency(row['explanation'], row['original_note'])
            
            # Accumulate counts
            reasoning_analysis['hallucination_counts'].append(analysis['hallucination_count'])
            reasoning_analysis['unsupported_claim_counts'].append(analysis['unsupported_claim_count'])
            reasoning_analysis['overconfidence_counts'].append(analysis['overconfidence_count'])
            
            # Save examples (limit to a few per category)
            for key in ['hallucinations', 'unsupported_claims', 'overconfident_statements']:
                if analysis[key] and len(reasoning_analysis['examples'][key]) < 5:
                    for item in analysis[key]:
                        example = {
                            'patient_id': row['patient_id'],
                            'text': item
                        }
                        reasoning_analysis['examples'][key].append(example)
        
        # Calculate summary statistics
        reasoning_analysis['avg_hallucinations'] = np.mean(reasoning_analysis['hallucination_counts'])
        reasoning_analysis['avg_unsupported'] = np.mean(reasoning_analysis['unsupported_claim_counts'])
        reasoning_analysis['avg_overconfidence'] = np.mean(reasoning_analysis['overconfidence_counts'])
        
        results[category] = reasoning_analysis
    
    return results

def visualize_reasoning_comparison(reasoning_results):
    """
    Create visualizations comparing reasoning patterns across prediction categories.
    
    Args:
        reasoning_results (dict): Results from analyze_reasoning_patterns
        
    Returns:
        dict: Dictionary of matplotlib figures
    """
    figures = {}
    
    # Prepare data for plotting
    categories = ['TP', 'TN', 'FP', 'FN']
    available_categories = [cat for cat in categories if f'{cat}_data' in reasoning_results 
                           and 'error' not in reasoning_results[f'{cat}_data']]
    
    if len(available_categories) < 2:
        return {'error': 'Not enough data categories for comparison visualization'}
    
    # 1. Bar chart of average reasoning issues
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    
    metrics = ['avg_hallucinations', 'avg_unsupported', 'avg_overconfidence']
    metric_labels = ['Hallucinations', 'Unsupported Claims', 'Overconfident Statements']
    x = np.arange(len(available_categories))
    width = 0.25
    
    for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
        values = [reasoning_results[f'{cat}_data'][metric] for cat in available_categories]
        ax1.bar(x + (i-1)*width, values, width, label=label)
    
    ax1.set_xticks(x)
    ax1.set_xticklabels(available_categories)
    ax1.set_ylabel('Average Count per Record')
    ax1.set_title('Reasoning Issues by Prediction Category')
    ax1.legend()
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    
    figures['reasoning_issues_comparison'] = fig1
    
    # 2. Stacked bar chart showing proportion of records with issues
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    
    # Calculate proportions
    proportions = []
    for cat in available_categories:
        cat_data = reasoning_results[f'{cat}_data']
        total = cat_data['total_records']
        
        with_hallucinations = sum(1 for count in cat_data['hallucination_counts'] if count > 0) / total
        with_unsupported = sum(1 for count in cat_data['unsupported_claim_counts'] if count > 0) / total
        with_overconfidence = sum(1 for count in cat_data['overconfidence_counts'] if count > 0) / total
        
        proportions.append({
            'category': cat,
            'with_hallucinations': with_hallucinations,
            'with_unsupported': with_unsupported,
            'with_overconfidence': with_overconfidence
        })
    
    # Plot stacked bars
    bottoms = np.zeros(len(available_categories))
    for metric, label in zip(['with_hallucinations', 'with_unsupported', 'with_overconfidence'], metric_labels):
        values = [prop[metric] for prop in proportions]
        ax2.bar(available_categories, values, bottom=bottoms, label=label)
        bottoms += values
    
    ax2.set_ylim(0, 1)
    ax2.set_ylabel('Proportion of Records')
    ax2.set_title('Proportion of Records with Reasoning Issues')
    ax2.legend()
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    
    figures['issue_proportions'] = fig2
    
    # 3. Comparison of error categories between FP and FN (focus on failures)
    if 'FP_data' in reasoning_results and 'FN_data' in reasoning_results and \
       'error' not in reasoning_results['FP_data'] and 'error' not in reasoning_results['FN_data']:
        
        fig3, ax3 = plt.subplots(figsize=(10, 6))
        
        fp_data = reasoning_results['FP_data']
        fn_data = reasoning_results['FN_data']
        
        metrics = ['avg_hallucinations', 'avg_unsupported', 'avg_overconfidence']
        x = np.arange(len(metrics))
        width = 0.35
        
        fp_values = [fp_data[metric] for metric in metrics]
        fn_values = [fn_data[metric] for metric in metrics]
        
        ax3.bar(x - width/2, fp_values, width, label='False Positives')
        ax3.bar(x + width/2, fn_values, width, label='False Negatives')
        
        ax3.set_xticks(x)
        ax3.set_xticklabels(metric_labels)
        ax3.set_ylabel('Average Count per Record')
        ax3.set_title('Comparison of Reasoning Issues in False Predictions')
        ax3.legend()
        ax3.grid(axis='y', linestyle='--', alpha=0.7)
        
        figures['fp_fn_comparison'] = fig3
    
    return figures

def generate_reasoning_report(reasoning_results):
    """
    Generate a comprehensive report on reasoning analysis.
    
    Args:
        reasoning_results (dict): Results from analyze_reasoning_patterns
        
    Returns:
        str: Report text
    """
    report_sections = []
    
    # Introduction
    report_sections.append("""
REASONING CONSISTENCY ANALYSIS REPORT
=====================================

This report analyzes reasoning patterns, inconsistencies, and potential hallucinations
in the model's explanations compared to the original clinical notes.
""")
    
    # Overall summary
    total_records = sum(
        reasoning_results[f'{cat}_data']['total_records'] 
        for cat in ['TP', 'TN', 'FP', 'FN'] 
        if f'{cat}_data' in reasoning_results and 'error' not in reasoning_results[f'{cat}_data']
    )
    
    report_sections.append(f"""
OVERALL SUMMARY
--------------
Total records analyzed: {total_records}

Reasoning issues detected:
""")
    
    # Calculate overall averages
    all_hallucinations = []
    all_unsupported = []
    all_overconfident = []
    
    for cat in ['TP', 'TN', 'FP', 'FN']:
        key = f'{cat}_data'
        if key in reasoning_results and 'error' not in reasoning_results[key]:
            all_hallucinations.extend(reasoning_results[key]['hallucination_counts'])
            all_unsupported.extend(reasoning_results[key]['unsupported_claim_counts'])
            all_overconfident.extend(reasoning_results[key]['overconfidence_counts'])
    
    avg_hallucinations = np.mean(all_hallucinations) if all_hallucinations else 0
    avg_unsupported = np.mean(all_unsupported) if all_unsupported else 0
    avg_overconfident = np.mean(all_overconfident) if all_overconfident else 0
    
    report_sections.append(f"""
- Average hallucinations per record: {avg_hallucinations:.2f}
- Average unsupported claims per record: {avg_unsupported:.2f}
- Average overconfident statements per record: {avg_overconfident:.2f}
""")
    
    # Detailed analysis by category
    report_sections.append("""
ANALYSIS BY PREDICTION CATEGORY
------------------------------
""")
    
    for cat in ['TP', 'TN', 'FP', 'FN']:
        key = f'{cat}_data'
        if key in reasoning_results and 'error' not in reasoning_results[key]:
            data = reasoning_results[key]
            
            report_sections.append(f"""
{cat} ({"True Positive" if cat == "TP" else "True Negative" if cat == "TN" else "False Positive" if cat == "FP" else "False Negative"})
{'-' * len(cat)}

Records: {data['total_records']}
Average issues:
- Hallucinations: {data['avg_hallucinations']:.2f}
- Unsupported claims: {data['avg_unsupported']:.2f}
- Overconfident statements: {data['avg_overconfidence']:.2f}
""")
            
            # Add examples for each category
            for issue_type in ['hallucinations', 'unsupported_claims', 'overconfident_statements']:
                examples = data['examples'][issue_type]
                if examples:
                    report_sections.append(f"""
Example {issue_type.replace('_', ' ')}:
""")
                    for i, example in enumerate(examples[:3], 1):
                        report_sections.append(f"{i}. Patient {example['patient_id']}: \"{example['text']}\"")
    
    # Comparison between correct and incorrect predictions
    report_sections.append("""

COMPARISON: CORRECT vs. INCORRECT PREDICTIONS
-------------------------------------------
""")
    
    correct_cats = ['TP_data', 'TN_data']
    incorrect_cats = ['FP_data', 'FN_data']
    
    correct_hall = []
    correct_unsupp = []
    correct_overconf = []
    
    incorrect_hall = []
    incorrect_unsupp = []
    incorrect_overconf = []
    
    for cat in correct_cats:
        if cat in reasoning_results and 'error' not in reasoning_results[cat]:
            correct_hall.extend(reasoning_results[cat]['hallucination_counts'])
            correct_unsupp.extend(reasoning_results[cat]['unsupported_claim_counts'])
            correct_overconf.extend(reasoning_results[cat]['overconfidence_counts'])
    
    for cat in incorrect_cats:
        if cat in reasoning_results and 'error' not in reasoning_results[cat]:
            incorrect_hall.extend(reasoning_results[cat]['hallucination_counts'])
            incorrect_unsupp.extend(reasoning_results[cat]['unsupported_claim_counts'])
            incorrect_overconf.extend(reasoning_results[cat]['overconfidence_counts'])
    
    if correct_hall and incorrect_hall:
        avg_correct_hall = np.mean(correct_hall)
        avg_incorrect_hall = np.mean(incorrect_hall)
        hall_ratio = avg_incorrect_hall / avg_correct_hall if avg_correct_hall > 0 else float('inf')
        
        avg_correct_unsupp = np.mean(correct_unsupp)
        avg_incorrect_unsupp = np.mean(incorrect_unsupp)
        unsupp_ratio = avg_incorrect_unsupp / avg_correct_unsupp if avg_correct_unsupp > 0 else float('inf')
        
        avg_correct_overconf = np.mean(correct_overconf)
        avg_incorrect_overconf = np.mean(incorrect_overconf)
        overconf_ratio = avg_incorrect_overconf / avg_correct_overconf if avg_correct_overconf > 0 else float('inf')
        
        report_sections.append(f"""
Hallucinations:
- Correct predictions: {avg_correct_hall:.2f} per record
- Incorrect predictions: {avg_incorrect_hall:.2f} per record
- Ratio (incorrect/correct): {hall_ratio:.2f}x

Unsupported claims:
- Correct predictions: {avg_correct_unsupp:.2f} per record
- Incorrect predictions: {avg_incorrect_unsupp:.2f} per record
- Ratio (incorrect/correct): {unsupp_ratio:.2f}x

Overconfident statements:
- Correct predictions: {avg_correct_overconf:.2f} per record
- Incorrect predictions: {avg_incorrect_overconf:.2f} per record
- Ratio (incorrect/correct): {overconf_ratio:.2f}x
""")
    
    # Key insights
    report_sections.append("""
KEY INSIGHTS
-----------
""")
    
    # Generate insights based on the data
    if correct_hall and incorrect_hall:
        if hall_ratio > 1.5:
            report_sections.append(f"• Hallucinations are {hall_ratio:.1f}x more common in incorrect predictions.")
        
        if unsupp_ratio > 1.5:
            report_sections.append(f"• Unsupported claims are {unsupp_ratio:.1f}x more common in incorrect predictions.")
        
        if overconf_ratio > 1.5:
            report_sections.append(f"• Overconfident statements are {overconf_ratio:.1f}x more common in incorrect predictions.")
    
    # Check which error category has the highest incidence in FP vs FN
    if 'FP_data' in reasoning_results and 'FN_data' in reasoning_results and \
       'error' not in reasoning_results['FP_data'] and 'error' not in reasoning_results['FN_data']:
        
        fp_data = reasoning_results['FP_data']
        fn_data = reasoning_results['FN_data']
        
        # Compare hallucinations
        if fp_data['avg_hallucinations'] > fn_data['avg_hallucinations'] * 1.2:
            report_sections.append("• False Positives contain more hallucinations than False Negatives.")
        elif fn_data['avg_hallucinations'] > fp_data['avg_hallucinations'] * 1.2:
            report_sections.append("• False Negatives contain more hallucinations than False Positives.")
        
        # Compare unsupported claims
        if fp_data['avg_unsupported'] > fn_data['avg_unsupported'] * 1.2:
            report_sections.append("• False Positives contain more unsupported claims than False Negatives.")
        elif fn_data['avg_unsupported'] > fp_data['avg_unsupported'] * 1.2:
            report_sections.append("• False Negatives contain more unsupported claims than False Positives.")
        
        # Compare overconfidence
        if fp_data['avg_overconfidence'] > fn_data['avg_overconfidence'] * 1.2:
            report_sections.append("• False Positives show more overconfidence than False Negatives.")
        elif fn_data['avg_overconfidence'] > fp_data['avg_overconfidence'] * 1.2:
            report_sections.append("• False Negatives show more overconfidence than False Positives.")
    
    # Recommendations
    report_sections.append("""

RECOMMENDATIONS
-------------
""")
    
    # Create recommendations based on findings
    if correct_hall and incorrect_hall:
        max_ratio = max(hall_ratio, unsupp_ratio, overconf_ratio)
        max_issue = "hallucinations" if max_ratio == hall_ratio else "unsupported claims" if max_ratio == unsupp_ratio else "overconfident statements"
        
        report_sections.append(f"1. Focus on reducing {max_issue} in model explanations.")
        
    if 'FP_data' in reasoning_results and 'FN_data' in reasoning_results and \
       'error' not in reasoning_results['FP_data'] and 'error' not in reasoning_results['FN_data']:
        
        fp_data = reasoning_results['FP_data']
        fn_data = reasoning_results['FN_data']
        
        fp_highest = max(fp_data['avg_hallucinations'], fp_data['avg_unsupported'], fp_data['avg_overconfidence'])
        fn_highest = max(fn_data['avg_hallucinations'], fn_data['avg_unsupported'], fn_data['avg_overconfidence'])
        
        fp_issue = "hallucinations" if fp_highest == fp_data['avg_hallucinations'] else \
                  "unsupported claims" if fp_highest == fp_data['avg_unsupported'] else \
                  "overconfident statements"
                  
        fn_issue = "hallucinations" if fn_highest == fn_data['avg_hallucinations'] else \
                  "unsupported claims" if fn_highest == fn_data['avg_unsupported'] else \
                  "overconfident statements"
        
        report_sections.append(f"2. To reduce False Positives: Address {fp_issue} in explanations.")
        report_sections.append(f"3. To reduce False Negatives: Address {fn_issue} in explanations.")
    
    report_sections.append("""
4. Improve annotation guidelines to promote:
   - Evidence-based reasoning
   - Appropriate confidence calibration
   - Clear citation of specific observations from original notes
""")
    
    return "\n".join(report_sections)
    
def main():
    # ------------------------------------------------
    # 1. Configuration and Data Loading
    # ------------------------------------------------
    
    # Create output directories
    os.makedirs('figures', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    os.makedirs('tables', exist_ok=True)
    
    # Load data
    file_path = "./LLM_pre_eval/gemma/prediction.csv"
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
        'demographics': {},
        'reasoning_analysis': {}  # New addition for reasoning analysis
    }
    
    # A. Criteria Mention Analysis
    print("Running criteria mention analysis...")
    for key in ['full_df', 'TP_data', 'TN_data', 'FP_data', 'FN_data']:
        results['criteria_analysis'][key] = extract_criteria_mentions(
            data[key]['explanation'], 
            criteria_dict
        )
    
    # B. Clinical Measurement Extraction
    print("Extracting clinical measurements...")
    for key in ['full_df', 'TP_data', 'TN_data', 'FP_data', 'FN_data']:
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
    
    # G. NEW: Reasoning Consistency Analysis
    print("Analyzing reasoning consistency and hallucinations...")
    results['reasoning_analysis'] = analyze_reasoning_patterns(data)
    
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
    for group in ['TP_data', 'TN_data', 'FP_data', 'FN_data']:
        comparative_results['correct_vs_incorrect'][group] = analyze_criteria_frequency(
            results['criteria_analysis'][group],
            data[group]['true_label']
        )
    
    # B. Threshold Analysis
    print("Analyzing measurement thresholds...")
    for group in ['TP_data', 'TN_data', 'FP_data', 'FN_data']:
        comparative_results['correct_vs_incorrect'][f'{group}_thresholds'] = analyze_measurement_thresholds(
            results['measurement_analysis'][group],
            data[group]['true_label']
        )
    
    # C. Four-group Analysis (TP, TN, FP, FN)
    print("Analyzing four prediction groups...")
    for group, label in [('TP_data', 'true_positives'), 
                         ('TN_data', 'true_negatives'),
                         ('FP_data', 'false_positives'),
                         ('FN_data', 'false_negatives')]:
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
        comparative_results['correct_vs_incorrect']['TP_data'],
        top_n=20
    )
    freq_fig.savefig('figures/criteria_frequency_comparison.png', dpi=300, bbox_inches='tight')
    plt.close(freq_fig)
    
    # B. Measurement Distribution Plots
    for group in ['TP_data', 'TN_data', 'FP_data', 'FN_data']:
        if len(results['measurement_analysis'][group]) > 0:
            figs = plot_measurement_distributions(
                results['measurement_analysis'][group],
                data[group]['true_label']
            )
            
            for measure, fig in figs.items():
                fig.savefig(f'figures/{group}_{measure}_distribution.png', dpi=300, bbox_inches='tight')
                plt.close(fig)
    
    # C. NEW: Reasoning Analysis Visualizations
    reasoning_figs = visualize_reasoning_comparison(results['reasoning_analysis'])
    for name, fig in reasoning_figs.items():
        if name != 'error':
            fig.savefig(f'figures/reasoning_{name}.png', dpi=300, bbox_inches='tight')
            plt.close(fig)
    
    # D. Generate Comprehensive Report
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
Correct predictions: {len(data['TP_data']) + len(data['TN_data'])} ({(len(data['TP_data']) + len(data['TN_data']))/len(data['full_df']):.1%})
Incorrect predictions: {len(data['FP_data']) + len(data['FN_data'])} ({(len(data['FP_data']) + len(data['FN_data']))/len(data['full_df']):.1%})

Breakdown:
- True Positives: {len(data['TP_data'])}
- True Negatives: {len(data['TN_data'])}
- False Positives: {len(data['FP_data'])}
- False Negatives: {len(data['FN_data'])}
""")
    
    # 3. Criteria Analysis Summary
    report_sections.append("""
CRITERIA ANALYSIS SUMMARY
------------------------
""")
    
    # Top criteria in correct predictions
    top_correct_tp = comparative_results['correct_vs_incorrect']['TP_data'].nlargest(5, 'risk_ratio')
    report_sections.append("Most predictive criteria in true positive predictions:")
    for _, row in top_correct_tp.iterrows():
        report_sections.append(f"- {row['criteria']}: RR={row['risk_ratio']:.1f}, p={row['p_value']:.4f}")
    
    # Top criteria in incorrect predictions
    top_incorrect_fp = comparative_results['correct_vs_incorrect']['FP_data'].nlargest(5, 'risk_ratio')
    report_sections.append("Most problematic criteria in false positive predictions:")
    for _, row in top_incorrect_fp.iterrows():
        report_sections.append(f"- {row['criteria']}: RR={row['risk_ratio']:.1f}, p={row['p_value']:.4f}")
    
    # 4. Measurement Analysis Summary
    report_sections.append("""

MEASUREMENT ANALYSIS SUMMARY
---------------------------
""")
    
    # Threshold performance
    for group in ['TP_data', 'TN_data', 'FP_data', 'FN_data']:
        thresholds = comparative_results['correct_vs_incorrect'][f'{group}_thresholds']
        report_sections.append(f"Measurement performance in {group.replace('_', ' ')}:")
        
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
   # Clinical Symptoms
    report_sections.append("Top clinical symptoms associated with malnutrition:")
    top_symptoms = results['symptom_analysis'].nlargest(5, 'prevalence_ratio')
    for _, row in top_symptoms.iterrows():
        report_sections.append(
            f"- {row['symptom']}: PR={row['prevalence_ratio']:.1f}, "
            f"AR={row['attributable_risk']:.2f}"
        )

    # Dietary factors
    report_sections.append("Top dietary factors associated with malnutrition:")
    top_dietary = results['dietary_analysis'].nlargest(5, 'odds_ratio')
    for _, row in top_dietary.iterrows():
        report_sections.append(
            f"- {row['factor']}: OR={row['odds_ratio']:.1f}"
        )

    # Risk factors
    report_sections.append("Top risk factors associated with malnutrition:")
    top_risk = results['risk_factor_analysis'].nlargest(5, 'relative_risk')
    for _, row in top_risk.iterrows():
        report_sections.append(
            f"- {row['factor']}: RR={row['relative_risk']:.1f}, "
            f"AR={row['attributable_risk']:.2f}"
        )

    
    # 6. NEW: Reasoning Analysis Summary
    report_sections.append("""

REASONING ANALYSIS SUMMARY
------------------------
""")
    
    # Compare correct vs incorrect predictions for reasoning issues
    correct_hall = []
    correct_unsupp = []
    correct_overconf = []
    
    incorrect_hall = []
    incorrect_unsupp = []
    incorrect_overconf = []
    
    for cat in ['TP_data', 'TN_data']:
        if cat in results['reasoning_analysis'] and 'error' not in results['reasoning_analysis'][cat]:
            correct_hall.extend(results['reasoning_analysis'][cat]['hallucination_counts'])
            correct_unsupp.extend(results['reasoning_analysis'][cat]['unsupported_claim_counts'])
            correct_overconf.extend(results['reasoning_analysis'][cat]['overconfidence_counts'])
    
    for cat in ['FP_data', 'FN_data']:
        if cat in results['reasoning_analysis'] and 'error' not in results['reasoning_analysis'][cat]:
            incorrect_hall.extend(results['reasoning_analysis'][cat]['hallucination_counts'])
            incorrect_unsupp.extend(results['reasoning_analysis'][cat]['unsupported_claim_counts'])
            incorrect_overconf.extend(results['reasoning_analysis'][cat]['overconfidence_counts'])
    
    if correct_hall and incorrect_hall:
        avg_correct_hall = np.mean(correct_hall)
        avg_incorrect_hall = np.mean(incorrect_hall)
        hall_ratio = avg_incorrect_hall / avg_correct_hall if avg_correct_hall > 0 else float('inf')
        
        avg_correct_unsupp = np.mean(correct_unsupp)
        avg_incorrect_unsupp = np.mean(incorrect_unsupp)
        unsupp_ratio = avg_incorrect_unsupp / avg_correct_unsupp if avg_correct_unsupp > 0 else float('inf')
        
        avg_correct_overconf = np.mean(correct_overconf)
        avg_incorrect_overconf = np.mean(incorrect_overconf)
        overconf_ratio = avg_incorrect_overconf / avg_correct_overconf if avg_correct_overconf > 0 else float('inf')
        
        report_sections.append(f"""
    Hallucinations comparison:
    - Correct predictions: {avg_correct_hall:.2f} per record
    - Incorrect predictions: {avg_incorrect_hall:.2f} per record
    - Ratio (incorrect/correct): {hall_ratio:.2f}x

    Unsupported claims comparison:
    - Correct predictions: {avg_correct_unsupp:.2f} per record
    - Incorrect predictions: {avg_incorrect_unsupp:.2f} per record
    - Ratio (incorrect/correct): {unsupp_ratio:.2f}x

    Overconfident statements comparison:
    - Correct predictions: {avg_correct_overconf:.2f} per record
    - Incorrect predictions: {avg_incorrect_overconf:.2f} per record
    - Ratio (incorrect/correct): {overconf_ratio:.2f}x
    """)
    
    # 7. Recommendations
    report_sections.append("""

    RECOMMENDATIONS
    ---------------
    """)
            
    # Clinical criteria recommendations
    top_incorrect = comparative_results['correct_vs_incorrect']['FP_data'].nlargest(5, 'risk_ratio')
    report_sections.append("1. Focus on improving recognition of: " + ", ".join(top_incorrect['criteria'].tolist()[:3]))
    
    # Measurement threshold recommendations
    report_sections.append("2. Validate measurement thresholds for: " + ", ".join(list(comparative_results['correct_vs_incorrect']['TP_data_thresholds'].keys())[:3]))
    
    # Risk factor recommendations
    report_sections.append("3. Consider additional training on: " + ", ".join(top_risk['factor'].tolist()[:3]))
    
    # NEW: Reasoning recommendations
    if correct_hall and incorrect_hall:
        max_ratio = max(hall_ratio, unsupp_ratio, overconf_ratio)
        max_issue = "hallucinations" if max_ratio == hall_ratio else "unsupported claims" if max_ratio == unsupp_ratio else "overconfident statements"
        
        report_sections.append(f"4. Focus on reducing {max_issue} in model explanations.")
    
    report_sections.append("""
    5. Improve annotation guidelines to promote:
    - Evidence-based reasoning
    - Appropriate confidence calibration
    - Clear citation of specific observations from original notes
    """)
        
    # Save full report
    with open('results/full_analysis_report.txt', 'w') as f:
        f.write("\n".join(report_sections))
    
    # Save reasoning analysis report
    reasoning_report = generate_reasoning_report(results['reasoning_analysis'])
    with open('results/reasoning_analysis_report.txt', 'w') as f:
        f.write(reasoning_report)
    
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
    for group in ['TP_data', 'TN_data', 'FP_data', 'FN_data']:
        comparative_results['correct_vs_incorrect'][group].to_csv(
            f'tables/{group}_criteria_frequency.csv', 
            index=False
        )
    
    # D. Save reasoning analysis summary
    # Create a summary DataFrame from reasoning analysis
    reasoning_summary = {
        'category': [],
        'total_records': [],
        'avg_hallucinations': [],
        'avg_unsupported': [],
        'avg_overconfidence': []
    }
    
    for cat in ['TP_data', 'TN_data', 'FP_data', 'FN_data']:
        if cat in results['reasoning_analysis'] and 'error' not in results['reasoning_analysis'][cat]:
            reasoning_summary['category'].append(cat)
            reasoning_summary['total_records'].append(results['reasoning_analysis'][cat]['total_records'])
            reasoning_summary['avg_hallucinations'].append(results['reasoning_analysis'][cat]['avg_hallucinations'])
            reasoning_summary['avg_unsupported'].append(results['reasoning_analysis'][cat]['avg_unsupported'])
            reasoning_summary['avg_overconfidence'].append(results['reasoning_analysis'][cat]['avg_overconfidence'])
    
    pd.DataFrame(reasoning_summary).to_csv('tables/reasoning_analysis_summary.csv', index=False)
    
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

Key findings have been compiled in:
- results/full_analysis_report.txt
- results/reasoning_analysis_report.txt
""")
    
    return {
        'results': results,
        'comparative_results': comparative_results
    }

if __name__ == "__main__":
    analysis_results = main()
