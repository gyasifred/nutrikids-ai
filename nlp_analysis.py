#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from pathlib import Path
from collections import Counter
from scipy import stats
from utils import (
    load_and_filter_data,
    preprocess_text,
    analyze_topics,
    plot_topic_analysis,
    create_keyword_network,
    plot_keyword_network,
    analyze_explanation_sentiment,
    plot_sentiment_analysis,
    get_named_entities,
    plot_entity_analysis
)

# Set plotting style
plt.style.use('seaborn')
sns.set_palette("Set2")
plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['font.size'] = 12
warnings.filterwarnings('ignore', category=UserWarning)

def compare_explanation_with_note(explanations, notes):
    """
    Compare explanations with their corresponding original notes to assess alignment.
    
    Args:
        explanations (pd.Series): Series containing explanation texts
        notes (pd.Series): Series containing original clinical notes
        
    Returns:
        dict: Analysis results including alignment metrics and overlapping content
    """
    from sklearn.feature_extraction.text import CountVectorizer
    import numpy as np
    
    results = {
        'alignment_scores': [],
        'key_term_overlap': [],
        'novel_terms_in_explanation': [],
        'missing_key_terms': []
    }
    
    # Preprocess both explanations and notes
    processed_explanations = [preprocess_text(text) for text in explanations]
    processed_notes = [preprocess_text(text) for text in notes]
    
    # For each explanation-note pair
    for exp, note in zip(processed_explanations, processed_notes):
        # Calculate term overlap using Jaccard similarity
        exp_terms = set(exp.split())
        note_terms = set(note.split())
        
        # Calculate Jaccard similarity
        if len(exp_terms.union(note_terms)) > 0:
            jaccard = len(exp_terms.intersection(note_terms)) / len(exp_terms.union(note_terms))
        else:
            jaccard = 0
            
        results['alignment_scores'].append(jaccard)
        
        # Find overlapping key terms
        overlap = exp_terms.intersection(note_terms)
        results['key_term_overlap'].append(list(overlap))
        
        # Find terms in explanation not in note (potential hallucinations)
        novel_terms = exp_terms - note_terms
        results['novel_terms_in_explanation'].append(list(novel_terms))
        
        # Find key terms from note missing in explanation
        missing_terms = note_terms - exp_terms
        results['missing_key_terms'].append(list(missing_terms))
    
    # Calculate average alignment score
    results['mean_alignment_score'] = np.mean(results['alignment_scores'])
    
    # Calculate percentage of explanations with low alignment (<0.3)
    results['low_alignment_rate'] = sum(1 for score in results['alignment_scores'] if score < 0.3) / len(results['alignment_scores'])
    
    return results

def analyze_evidence_support(explanations, notes):
    """
    Analyze the strength of evidence support in explanations based on original notes.
    
    Args:
        explanations (pd.Series): Series containing explanation texts
        notes (pd.Series): Series containing original clinical notes
        
    Returns:
        dict: Analysis results including evidence support metrics
    """
    import re
    import numpy as np
    
    # Expanded clinical indicators organized by category
    clinical_indicators = {
        # Anthropometric measurements
        'anthropometric': [
            'bmi', 'weight', 'height', 'weight-for-height', 'muac', 'mid-upper arm circumference',
            'weight loss', 'weight gain', 'underweight', 'overweight', 'obese', 'thin', 'emaciated',
            'percentile', 'z-score', 'triceps skinfold', 'muscle mass', 'body composition'
        ],
        
        # Clinical symptoms
        'clinical': [
            'muscle wasting', 'fatigue', 'weakness', 'lethargy', 'skin changes', 'hair changes',
            'edema', 'dermatitis', 'glossitis', 'stomatitis', 'poor wound healing', 'bruising',
            'pallor', 'dry skin', 'brittle nails', 'hair loss', 'muscle atrophy', 'sarcopenia'
        ],
        
        # Dietary factors
        'dietary': [
            'caloric intake', 'protein intake', 'diet', 'supplement', 'feeding', 'appetite',
            'meal', 'nutrition', 'nutrient', 'malnutrition', 'deficiency', 'vitamin', 'mineral',
            'food insecurity', 'limited access', 'poor diet', 'inadequate intake', 'fasting',
            'anorexia', 'tube feeding', 'tpn', 'parenteral nutrition', 'enteral nutrition'
        ],
        
        # Medical conditions
        'medical': [
            'chronic illness', 'gastrointestinal', 'infection', 'malabsorption', 'diarrhea',
            'vomiting', 'nausea', 'constipation', 'dysphagia', 'gastroparesis', 'celiac',
            'crohn', 'ulcerative colitis', 'pancreatic insufficiency', 'liver disease',
            'cancer', 'diabetes', 'respiratory disease', 'renal disease', 'hiv', 'tuberculosis'
        ],
        
        # Lab values
        'labs': [
            'albumin', 'prealbumin', 'transferrin', 'total protein', 'lymphocyte count',
            'cholesterol', 'hemoglobin', 'hematocrit', 'ferritin', 'folate', 'b12',
            'vitamin d', 'zinc', 'magnesium', 'calcium', 'nitrogen balance'
        ],
        
        # Risk factors
        'risk_factors': [
            'medications', 'polypharmacy', 'depression', 'anxiety', 'cognitive impairment',
            'dementia', 'socioeconomic', 'poverty', 'homelessness', 'social isolation',
            'elderly', 'pediatric', 'pregnancy', 'alcohol', 'substance abuse', 'surgery',
            'hospitalization', 'immobility', 'disability'
        ]
    }
    
    # Flatten the dictionary for easier searching
    all_indicators = []
    for category in clinical_indicators.values():
        all_indicators.extend(category)
    
    results = {
        'score': [],
        'evidence_count': [],
        'unsupported_claims': [],
        'category_coverage': []  # New metric to track which categories are covered
    }
    
    for exp, note in zip(explanations, notes):
        exp_lower = exp.lower()
        note_lower = note.lower()
        
        # 1. Count clinical indicators mentioned in both explanation and note
        supported_indicators = 0
        unsupported_indicators = 0
        unsupported_claims = []
        
        # Track which categories are covered in the explanation
        covered_categories = set()
        
        for indicator in all_indicators:
            if indicator in exp_lower:
                if indicator in note_lower:
                    supported_indicators += 1
                    # Find which category this indicator belongs to
                    for category, indicators in clinical_indicators.items():
                        if indicator in indicators:
                            covered_categories.add(category)
                            break
                else:
                    unsupported_indicators += 1
                    unsupported_claims.append(indicator)
        
        # 2. Calculate evidence support score (higher is better)
        if supported_indicators + unsupported_indicators > 0:
            support_score = supported_indicators / (supported_indicators + unsupported_indicators)
        else:
            support_score = 0
        
        # 3. Calculate category coverage (percentage of categories mentioned)
        category_coverage = len(covered_categories) / len(clinical_indicators)
        
        results['score'].append(support_score)
        results['evidence_count'].append(supported_indicators)
        results['unsupported_claims'].append(unsupported_claims)
        results['category_coverage'].append(category_coverage)
    
    # Calculate average evidence support score
    results['mean_support_score'] = np.mean(results['score'])
    
    # Calculate percentage of explanations with low evidence support (<0.5)
    results['low_support_rate'] = sum(1 for score in results['score'] if score < 0.5) / len(results['score'])
    
    # Calculate average category coverage
    results['mean_category_coverage'] = np.mean(results['category_coverage'])
    
    return results

def detect_hallucinations(explanations, notes):
    """
    Detect potential hallucinations (information in explanations not found in notes).
    
    Args:
        explanations (pd.Series): Series containing explanation texts
        notes (pd.Series): Series containing original clinical notes
        
    Returns:
        dict: Analysis results including hallucination instances and metrics
    """
    import re
    import numpy as np
    from nltk.tokenize import sent_tokenize
    import spacy
    
    # Load spaCy model for NER (used to identify clinical entities)
    try:
        nlp = spacy.load("en_core_web_sm")
    except:
        # If model not available, download it
        import subprocess
        subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
        nlp = spacy.load("en_core_web_sm")
    
    results = {
        'hallucination_scores': [],
        'hallucinated_instances': [],
        'hallucinated_entities': [],
        'confidence_phrases': []
    }
    
    # Confidence phrases that might indicate hallucinations when not supported
    confidence_phrases = [
        'clearly', 'obviously', 'definitely', 'certainly', 'undoubtedly',
        'strongly', 'severely', 'significantly', 'markedly', 'notably',
        'evidently', 'absolutely', 'extremely', 'very', 'highly'
    ]
    
    for i, (exp, note) in enumerate(zip(explanations, notes)):
        # Process with spaCy to get entities
        exp_doc = nlp(exp)
        note_doc = nlp(note)
        
        # Extract entities from explanation and note
        exp_entities = [(ent.text.lower(), ent.label_) for ent in exp_doc.ents]
        note_entities = [ent.text.lower() for ent in note_doc.ents]
        
        # Check for entities in explanation not found in note
        hallucinated_ents = []
        for ent_text, ent_label in exp_entities:
            if ent_text not in note.lower() and len(ent_text) > 1:  # Skip single character entities
                hallucinated_ents.append((ent_text, ent_label))
        
        # Check for confident statements containing potential hallucinations
        confidence_matches = []
        for phrase in confidence_phrases:
            pattern = r'\b' + phrase + r'\b'
            if re.search(pattern, exp.lower()):
                # If confident phrase found, check surrounding text
                for match in re.finditer(pattern, exp.lower()):
                    # Get sentence containing the match
                    sentences = sent_tokenize(exp)
                    for sent in sentences:
                        if phrase in sent.lower():
                            # Check if sentence contains hallucinated entity
                            if any(ent[0] in sent.lower() for ent in hallucinated_ents):
                                confidence_matches.append(sent)
        
        # Calculate hallucination score based on ratio of hallucinated entities
        if exp_entities:
            hall_score = len(hallucinated_ents) / len(exp_entities)
        else:
            hall_score = 0
            
        # Record results
        results['hallucination_scores'].append(hall_score)
        results['hallucinated_entities'].append(hallucinated_ents)
        results['confidence_phrases'].append(confidence_matches)
        
        # Flag as potential hallucination if score above threshold or confidence phrases with hallucinations
        if hall_score > 0.2 or confidence_matches:
            results['hallucinated_instances'].append(i)
    
    # Calculate metrics
    results['mean_hallucination_score'] = np.mean(results['hallucination_scores'])
    results['hallucination_rate'] = len(results['hallucinated_instances']) / len(explanations)
    
    return results

def plot_comparative_analysis(comparison_results, hallucination_results, output_dir='figures/nlp'):
    """
    Generate visualizations for the comparison between explanations and notes.
    
    Args:
        comparison_results (dict): Results from compare_explanation_with_note
        hallucination_results (dict): Results from detect_hallucinations
        output_dir (str): Directory to save figures
        
    Returns:
        dict: Dictionary of generated figures
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    import os
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    figures = {}
    
    # 1. Plot alignment score distribution
    fig1 = plt.figure(figsize=(10, 6))
    sns.histplot(comparison_results['alignment_scores'], bins=20, kde=True)
    plt.title('Distribution of Explanation-Note Alignment Scores')
    plt.xlabel('Alignment Score (Jaccard Similarity)')
    plt.ylabel('Count')
    plt.axvline(x=0.3, color='red', linestyle='--', label='Low Alignment Threshold')
    plt.legend()
    plt.savefig(f'{output_dir}/alignment_scores.png', bbox_inches='tight')
    figures['alignment_dist'] = fig1
    
    # 2. Plot evidence support scores
    if 'evidence_support' in comparison_results:
        fig2 = plt.figure(figsize=(10, 6))
        sns.histplot(comparison_results['evidence_support']['score'], bins=20, kde=True)
        plt.title('Distribution of Evidence Support Scores')
        plt.xlabel('Evidence Support Score')
        plt.ylabel('Count')
        plt.axvline(x=0.5, color='red', linestyle='--', label='Low Support Threshold')
        plt.legend()
        plt.savefig(f'{output_dir}/evidence_support.png', bbox_inches='tight')
        figures['evidence_support'] = fig2
    
    # 3. Plot hallucination score distribution
    fig3 = plt.figure(figsize=(10, 6))
    sns.histplot(hallucination_results['hallucination_scores'], bins=20, kde=True)
    plt.title('Distribution of Hallucination Scores')
    plt.xlabel('Hallucination Score')
    plt.ylabel('Count')
    plt.axvline(x=0.2, color='red', linestyle='--', label='Hallucination Threshold')
    plt.legend()
    plt.savefig(f'{output_dir}/hallucination_scores.png', bbox_inches='tight')
    figures['hallucination_dist'] = fig3
    
    # 4. Plot relationship between alignment and hallucination
    fig4 = plt.figure(figsize=(10, 6))
    plt.scatter(comparison_results['alignment_scores'], 
                hallucination_results['hallucination_scores'],
                alpha=0.6)
    plt.title('Relationship Between Alignment and Hallucination')
    plt.xlabel('Alignment Score')
    plt.ylabel('Hallucination Score')
    plt.axhline(y=0.2, color='red', linestyle='--', label='Hallucination Threshold')
    plt.axvline(x=0.3, color='blue', linestyle='--', label='Low Alignment Threshold')
    plt.legend()
    plt.savefig(f'{output_dir}/alignment_vs_hallucination.png', bbox_inches='tight')
    figures['alignment_vs_hallucination'] = fig4
    
    return figures

def main():
    # ------------------------------------------------
    # 1. Configuration and Data Loading
    # ------------------------------------------------
    
    # Create output directories
    os.makedirs('figures/nlp', exist_ok=True)
    os.makedirs('results/nlp', exist_ok=True)
    os.makedirs('tables/nlp', exist_ok=True)
    
    # Additional directories for comparison analysis
    os.makedirs('figures/nlp/comparison', exist_ok=True)
    os.makedirs('tables/nlp/comparison', exist_ok=True)
    
    # Load data
    file_path = "./llama_zero_shot/prediction.csv"
    data = load_and_filter_data(file_path)
    
    # ------------------------------------------------
    # 2. NLP Analysis Pipeline
    # ------------------------------------------------
    
    # Initialize results dictionary
    results = {
        'data': data,
        'topic_analysis': {},
        'keyword_networks': {},
        'sentiment_analysis': {},
        'entity_analysis': {},
        'explanation_note_comparison': {}  # New section for comparison analysis
    }
    
    # A. Topic Modeling Analysis
    print("Running topic modeling analysis...")
    for key in ['full_df', 'correct_predictions', 'incorrect_predictions', 
                'TP_data', 'TN_data', 'FP_data', 'FN_data']:
        results['topic_analysis'][key] = analyze_topics(
            data[key]['explanation'],
            data[key]['true_label'],
            n_topics=5
        )
    
    # B. Keyword Network Analysis
    print("Creating keyword networks...")
    for key in ['full_df', 'correct_predictions', 'incorrect_predictions', 
                'TP_data', 'TN_data', 'FP_data', 'FN_data']:  # Added individual prediction groups
        network = create_keyword_network(
            data[key]['explanation'],
            data[key]['true_label']
        )
        results['keyword_networks'][key] = network
    
    # C. Sentiment Analysis
    print("Analyzing sentiment patterns...")
    for key in ['full_df', 'correct_predictions', 'incorrect_predictions', 
                'TP_data', 'TN_data', 'FP_data', 'FN_data']:
        results['sentiment_analysis'][key] = analyze_explanation_sentiment(
            data[key]['explanation'],
            data[key]['true_label']
        )
    
    # D. Named Entity Recognition
    print("Extracting named entities...")
    for key in ['full_df', 'correct_predictions', 'incorrect_predictions', 
                'TP_data', 'TN_data', 'FP_data', 'FN_data']:  # Added individual prediction groups
        results['entity_analysis'][key] = get_named_entities(
            data[key]['explanation']
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
        'false_negatives': {},
        'explanation_fidelity': {}  # New section for explanation-note comparison
    }
    
    # A. Topic Comparison
    print("Comparing topics between groups...")
    for group in ['correct_predictions', 'incorrect_predictions']:
        comparative_results['correct_vs_incorrect'][f'{group}_topics'] = {
            'topics': results['topic_analysis'][group]['topics'],
            'prevalence': results['topic_analysis'][group]['prevalence_by_status']
        }
    
    # B. Sentiment Comparison
    print("Comparing sentiment between groups...")
    for group in ['correct_predictions', 'incorrect_predictions']:
        comparative_results['correct_vs_incorrect'][f'{group}_sentiment'] = {
            'stats': results['sentiment_analysis'][group]['sentiment_stats'],
            'effect_size': {
                'cohen_d': results['sentiment_analysis'][group]['sentiment_stats'].iloc[0]['cohen_d'],
                'p_value': results['sentiment_analysis'][group]['p_value']
            }
        }
    
    # C. Four-group Analysis (TP, TN, FP, FN)
    print("Analyzing four prediction groups...")
    for group, label in [('TP_data', 'true_positives'), 
                         ('TN_data', 'true_negatives'),
                         ('FP_data', 'false_positives'),
                         ('FN_data', 'false_negatives')]:
        # Topic analysis
        comparative_results[label]['topics'] = {
            'main_topics': list(results['topic_analysis'][group]['topics'].values())
        }
        
        # Sentiment analysis
        comparative_results[label]['sentiment'] = {
            'mean': results['sentiment_analysis'][group]['sentiment_stats']['mean'].to_dict()
        }
    
    # ------------------------------------------------
    # 3a. Comparative Analysis between Explanations and Original Notes
    # ------------------------------------------------

    print("Analyzing explanation alignment with original notes...")
    
    # A. Detect potential hallucinations (content in explanation not found in notes)
    for key in ['full_df', 'correct_predictions', 'incorrect_predictions', 
                'TP_data', 'TN_data', 'FP_data', 'FN_data']:
        results['explanation_note_comparison'][key] = compare_explanation_with_note(
            data[key]['explanation'],
            data[key]['original_note']
        )
        
    # B. Measure evidence support strength
    print("Quantifying evidence support in explanations...")
    for key in ['full_df', 'correct_predictions', 'incorrect_predictions', 
                'TP_data', 'TN_data', 'FP_data', 'FN_data']:
        results['explanation_note_comparison'][key]['evidence_support'] = analyze_evidence_support(
            data[key]['explanation'],
            data[key]['original_note']
        )

    # C. Identify hallucination rate by prediction group
    print("Detecting potential hallucinations...")
    hallucination_results = {}
    for key in ['TP_data', 'TN_data', 'FP_data', 'FN_data', 
                'full_df', 'correct_predictions', 'incorrect_predictions']:
        hallucination_results[key] = detect_hallucinations(
            data[key]['explanation'],
            data[key]['original_note']
        )

    # D. Comparative metrics between groups
    comparative_results['explanation_fidelity'] = {
        'hallucination_rates': {
            'TP': len(hallucination_results['TP_data']['hallucinated_instances']) / len(data['TP_data']),
            'TN': len(hallucination_results['TN_data']['hallucinated_instances']) / len(data['TN_data']),
            'FP': len(hallucination_results['FP_data']['hallucinated_instances']) / len(data['FP_data']),
            'FN': len(hallucination_results['FN_data']['hallucinated_instances']) / len(data['FN_data'])
        },
        'evidence_strength': {
            'TP': np.mean(results['explanation_note_comparison']['TP_data']['evidence_support']['score']),
            'TN': np.mean(results['explanation_note_comparison']['TN_data']['evidence_support']['score']),
            'FP': np.mean(results['explanation_note_comparison']['FP_data']['evidence_support']['score']),
            'FN': np.mean(results['explanation_note_comparison']['FN_data']['evidence_support']['score'])
        },
        'alignment_scores': {
            'TP': np.mean(results['explanation_note_comparison']['TP_data']['alignment_scores']),
            'TN': np.mean(results['explanation_note_comparison']['TN_data']['alignment_scores']),
            'FP': np.mean(results['explanation_note_comparison']['FP_data']['alignment_scores']),
            'FN': np.mean(results['explanation_note_comparison']['FN_data']['alignment_scores'])
        }
    }
    
    # E. Correlation between explanation fidelity and prediction accuracy
    # Calculate correlation between alignment scores and prediction correctness
    all_alignment_scores = []
    all_correctness = []
    
    for i, row in data['full_df'].iterrows():
        idx = i  # Use index in full dataframe
        group = row['prediction_result']
        if group in ['TP', 'TN']:
            correctness = 1  # Correct prediction
        else:
            correctness = 0  # Incorrect prediction
            
        # Find alignment score for this instance
        if idx < len(results['explanation_note_comparison']['full_df']['alignment_scores']):
            alignment = results['explanation_note_comparison']['full_df']['alignment_scores'][idx]
            all_alignment_scores.append(alignment)
            all_correctness.append(correctness)
    
    # Calculate correlation if we have enough data
    if len(all_alignment_scores) > 1:
        corr, p_value = stats.pearsonr(all_alignment_scores, all_correctness)
        comparative_results['explanation_fidelity']['correlation'] = {
            'alignment_correctness_corr': corr,
            'alignment_correctness_p': p_value
        }
    
    # ------------------------------------------------
    # 4. Visualization and Output Generation
    # ------------------------------------------------
    
    print("Generating visualizations and reports...")
    
    # A. Topic Modeling Visualizations
    for key in ['full_df', 'correct_predictions', 'incorrect_predictions']:
        figs = plot_topic_analysis(
            results['topic_analysis'][key],
            output_dir=f'figures/nlp/{key}',
            figsize_multiplier=1.2
        )
        for fig_name, fig in figs.items():
            plt.close(fig)
    
    # B. Keyword Network Visualizations
    for key in ['full_df', 'correct_predictions', 'incorrect_predictions']:
        fig = plot_keyword_network(
            results['keyword_networks'][key],
            output_dir=f'figures/nlp/{key}',
            figsize=(16, 12)
        )
        if fig:
            plt.close(fig)
    
    # C. Sentiment Analysis Visualizations
    for key in ['full_df', 'correct_predictions', 'incorrect_predictions']:
        figs = plot_sentiment_analysis(
            results['sentiment_analysis'][key],
            output_dir=f'figures/nlp/{key}'
        )
        for fig_name, fig in figs.items():
            plt.close(fig)
    
    # D. Entity Analysis Visualizations
    for key in ['full_df', 'correct_predictions', 'incorrect_predictions']:
        figs = plot_entity_analysis(
            results['entity_analysis'][key],
            output_dir=f'figures/nlp/{key}'
        )
        for fig_name, fig in figs.items():
            plt.close(fig)
    
    # E. Explanation-Note Comparison Visualizations
    print("Generating explanation-note comparison visualizations...")
    
    # Plot hallucination rates by prediction group
    fig_hallucination = plt.figure(figsize=(10, 6))
    rates = comparative_results['explanation_fidelity']['hallucination_rates']
    plt.bar(rates.keys(), rates.values(), color=['green', 'blue', 'orange', 'red'])
    plt.title('Hallucination Rate by Prediction Group')
    plt.ylabel('Rate')
    plt.ylim(0, 1)  # Ensure scale from 0-1 for rate visualization
    plt.savefig('figures/nlp/comparison/hallucination_rates.png', bbox_inches='tight')
    plt.close(fig_hallucination)
    
    # Plot evidence strength by prediction group
    fig_evidence = plt.figure(figsize=(10, 6))
    evidence = comparative_results['explanation_fidelity']['evidence_strength']
    plt.bar(evidence.keys(), evidence.values(), color=['green', 'blue', 'orange', 'red'])
    plt.title('Evidence Support Strength by Prediction Group')
    plt.ylabel('Average Score')
    plt.ylim(0, 1)  # Ensure scale from 0-1 for score visualization
    plt.savefig('figures/nlp/comparison/evidence_strength.png', bbox_inches='tight')
    plt.close(fig_evidence)
    
    # Plot alignment scores by prediction group
    fig_alignment = plt.figure(figsize=(10, 6))
    alignment = comparative_results['explanation_fidelity']['alignment_scores']
    plt.bar(alignment.keys(), alignment.values(), color=['green', 'blue', 'orange', 'red'])
    plt.title('Explanation-Note Alignment by Prediction Group')
    plt.ylabel('Average Alignment Score')
    plt.ylim(0, 1)  # Ensure scale from 0-1 for score visualization
    plt.savefig('figures/nlp/comparison/alignment_scores.png', bbox_inches='tight')
    plt.close(fig_alignment)
    
    # Plot correlation between alignment and correctness if available
    if 'correlation' in comparative_results['explanation_fidelity']:
        fig_corr = plt.figure(figsize=(10, 6))
        plt.scatter(all_alignment_scores, all_correctness, alpha=0.5)
        plt.title(f'Correlation between Alignment and Correctness (r={comparative_results["explanation_fidelity"]["correlation"]["alignment_correctness_corr"]:.3f})')
        plt.xlabel('Explanation-Note Alignment Score')
        plt.ylabel('Prediction Correctness (1=Correct, 0=Incorrect)')
        plt.grid(True, alpha=0.3)
        plt.savefig('figures/nlp/comparison/alignment_correctness_correlation.png', bbox_inches='tight')
        plt.close(fig_corr)
    
    # Generate detailed visualizations for each prediction group
    for key in ['TP_data', 'TN_data', 'FP_data', 'FN_data']:
        figs = plot_comparative_analysis(
            results['explanation_note_comparison'][key],
            hallucination_results[key],
            output_dir=f'figures/nlp/comparison/{key}'
        )
        for fig_name, fig in figs.items():
            plt.close(fig)
            
    # F. Generate Comparative Plots Between Correct and Incorrect Predictions
    
    # Compare hallucination scores between correct and incorrect
    fig_hall_compare = plt.figure(figsize=(12, 6))
    
    # Create data for plotting
    correct_hall = hallucination_results['correct_predictions']['hallucination_scores']
    incorrect_hall = hallucination_results['incorrect_predictions']['hallucination_scores']
    
    # Plot distributions
    plt.hist([correct_hall, incorrect_hall], bins=20, alpha=0.6, label=['Correct Predictions', 'Incorrect Predictions'], color=['green', 'red'])
    plt.axvline(np.mean(correct_hall), color='green', linestyle='dashed', linewidth=2, label=f'Correct Mean: {np.mean(correct_hall):.3f}')
    plt.axvline(np.mean(incorrect_hall), color='red', linestyle='dashed', linewidth=2, label=f'Incorrect Mean: {np.mean(incorrect_hall):.3f}')
    plt.title('Comparison of Hallucination Scores: Correct vs. Incorrect Predictions')
    plt.xlabel('Hallucination Score')
    plt.ylabel('Frequency')
    plt.legend()
    plt.savefig('figures/nlp/comparison/correct_vs_incorrect_hallucination.png', bbox_inches='tight')
    plt.close(fig_hall_compare)
    
    # E. Generate Comprehensive Report
    report_sections = []
    
    # 1. Executive Summary
    report_sections.append("""
NLP ANALYSIS REPORT
==================

This report provides a comprehensive natural language processing analysis of malnutrition 
prediction explanations, including topic modeling, sentiment analysis, keyword networks, 
named entity recognition, and explanation-note comparison across different prediction groups.
""")
    
    # 2. Dataset Overview
    report_sections.append(f"""
DATASET OVERVIEW
----------------
Total cases: {len(data['full_df'])}
Correct predictions: {len(data['correct_predictions'])} ({len(data['correct_predictions'])/len(data['full_df']):.1%})
Incorrect predictions: {len(data['incorrect_predictions'])} ({len(data['incorrect_predictions'])/len(data['full_df']):.1%})

Breakdown:
- True Positives: {len(data['TP_data'])}
- True Negatives: {len(data['TN_data'])}
- False Positives: {len(data['FP_data'])}
- False Negatives: {len(data['FN_data'])}
""")
    
    # 3. Topic Analysis Summary
    report_sections.append("""
TOPIC ANALYSIS SUMMARY
----------------------
""")
    
    # Main topics in correct predictions
    correct_topics = results['topic_analysis']['correct_predictions']['topics']
    report_sections.append("Main topics in correct predictions:")
    for topic_id, words in correct_topics.items():
        report_sections.append(f"- Topic {topic_id}: {', '.join(words[:5])}...")
    
    # Main topics in incorrect predictions
    incorrect_topics = results['topic_analysis']['incorrect_predictions']['topics']
    report_sections.append("\nMain topics in incorrect predictions:")
    for topic_id, words in incorrect_topics.items():
        report_sections.append(f"- Topic {topic_id}: {', '.join(words[:5])}...")
    
    # 4. Sentiment Analysis Summary
    report_sections.append("""

SENTIMENT ANALYSIS SUMMARY
-------------------------
""")
    
    # Sentiment comparison
    correct_sentiment = results['sentiment_analysis']['correct_predictions']['sentiment_stats']
    incorrect_sentiment = results['sentiment_analysis']['incorrect_predictions']['sentiment_stats']
    
    report_sections.append("Sentiment in correct predictions:")
    report_sections.append(f"- Malnourished: mean={correct_sentiment.loc['yes']['mean']:.2f}")
    report_sections.append(f"- Not malnourished: mean={correct_sentiment.loc['no']['mean']:.2f}")
    
    report_sections.append("\nSentiment in incorrect predictions:")
    report_sections.append(f"- Malnourished: mean={incorrect_sentiment.loc['yes']['mean']:.2f}")
    report_sections.append(f"- Not malnourished: mean={incorrect_sentiment.loc['no']['mean']:.2f}")
    
    # 5. Keyword Network Summary
    report_sections.append("""

KEYWORD NETWORK SUMMARY
-----------------------
""")
    
    full_network = results['keyword_networks']['full_df']
    if full_network:
        top_nodes = sorted(full_network.nodes(data=True), 
                          key=lambda x: x[1]['count'], 
                          reverse=True)[:5]
        
        report_sections.append("Most frequent keywords in explanations:")
        for node in top_nodes:
            report_sections.append(
                f"- {node[0]} (count={node[1]['count']}, "
                f"malnutrition ratio={node[1]['malnutrition_ratio']:.2f})"
            )
    
    # 6. Named Entity Summary
    report_sections.append("""

NAMED ENTITY SUMMARY
--------------------
""")
    
    full_entities = results['entity_analysis']['full_df']
    if full_entities:
        top_categories = sorted(
            [(k, sum(v.values())) for k, v in full_entities['category_counts'].items()],
            key=lambda x: x[1],
            reverse=True
        )[:5]
        
        report_sections.append("Most frequent entity categories:")
        for category, count in top_categories:
            report_sections.append(f"- {category}: {count} entities")
    
    # 7. Explanation-Note Comparison Summary
    report_sections.append("""

EXPLANATION-NOTE COMPARISON
--------------------------
""")

    # Hallucination stats
    report_sections.append("Hallucination rates by prediction group:")
    for group, rate in comparative_results['explanation_fidelity']['hallucination_rates'].items():
        report_sections.append(f"- {group}: {rate:.2%}")

    # Evidence support
    report_sections.append("\nEvidence support strength by prediction group:")
    for group, score in comparative_results['explanation_fidelity']['evidence_strength'].items():
        report_sections.append(f"- {group}: {score:.2f}")
        
    # Alignment scores
    report_sections.append("\nExplanation-note alignment by prediction group:")
    for group, score in comparative_results['explanation_fidelity']['alignment_scores'].items():
        report_sections.append(f"- {group}: {score:.2f}")

    # Key findings from comparison
    report_sections.append("\nKey findings from explanation-note comparison:")

    # Find the group with highest hallucination rate
    worst_group = max(comparative_results['explanation_fidelity']['hallucination_rates'].items(), 
                      key=lambda x: x[1])[0]
    report_sections.append(f"- Highest hallucination rate observed in {worst_group} predictions")

    # Find the group with lowest evidence support
    lowest_evidence = min(comparative_results['explanation_fidelity']['evidence_strength'].items(),
                          key=lambda x: x[1])[0]
    report_sections.append(f"- Lowest evidence support observed in {lowest_evidence} predictions")

    # Check if there's correlation between hallucination and prediction error
    hall_correct = (comparative_results['explanation_fidelity']['hallucination_rates']['TP'] + 
                    comparative_results['explanation_fidelity']['hallucination_rates']['TN']) / 2
    hall_incorrect = (comparative_results['explanation_fidelity']['hallucination_rates']['FP'] + 
                      comparative_results['explanation_fidelity']['hallucination_rates']['FN']) / 2
    report_sections.append(f"- {'Higher' if hall_incorrect > hall_correct else 'Similar or lower'} hallucination rates in incorrect predictions compared to correct ones")
    
    # Add correlation between alignment and correctness if available
    if 'correlation' in comparative_results['explanation_fidelity']:
        corr = comparative_results['explanation_fidelity']['correlation']['alignment_correctness_corr']
        p_val = comparative_results['explanation_fidelity']['correlation']['alignment_correctness_p']
        significance = "statistically significant" if p_val < 0.05 else "not statistically significant"
        report_sections.append(f"- Correlation between explanation-note alignment and prediction correctness: r={corr:.3f} ({significance}, p={p_val:.3f})")
    
    # 8. Recommendations
    report_sections.append("""

RECOMMENDATIONS
---------------
1. Focus on improving explanations related to: """ + ", ".join(list(incorrect_topics.values())[0][:3]) + """
2. Monitor sentiment patterns in: """ + ("FP/FN" if abs(incorrect_sentiment.loc['yes']['mean'] - incorrect_sentiment.loc['no']['mean']) > 0.2 else "all cases") + """
3. Validate terminology usage for: """ + ", ".join([cat[0] for cat in top_categories[:3]]) + """
4. Improve evidence support in: """ + lowest_evidence + """ predictions
5. Reduce hallucination rate in: """ + worst_group + """ predictions
6. """ + ("Strengthen alignment between explanations and source notes" if 'correlation' in comparative_results['explanation_fidelity'] and comparative_results['explanation_fidelity']['correlation']['alignment_correctness_corr'] > 0.3 else "Address inconsistencies between explanations and source notes") + """
""")
    
    # Save full report
    with open('results/nlp/full_analysis_report.txt', 'w') as f:
        f.write("\n".join(report_sections))
    
    # ------------------------------------------------
    # 5. Data Export
    # ------------------------------------------------
    
    print("Exporting results to files...")
    
    # A. Save topic data
    for key, analysis in results['topic_analysis'].items():
        if 'topics' in analysis:
            pd.DataFrame({
                'topic_id': list(analysis['topics'].keys()),
                'top_words': [", ".join(words) for words in analysis['topics'].values()]
            }).to_csv(f'tables/nlp/{key}_topics.csv', index=False)
    
    # B. Save sentiment data
    for key, analysis in results['sentiment_analysis'].items():
        if 'sentiment_stats' in analysis:
            analysis['sentiment_stats'].to_csv(f'tables/nlp/{key}_sentiment.csv')
    
    # C. Save entity data
    for key, analysis in results['entity_analysis'].items():
        if 'category_counts' in analysis:
            entity_counts = []
            for category, counts in analysis['category_counts'].items():
                for entity, count in counts.items():
                    entity_counts.append({'category': category, 'entity': entity, 'count': count})
            pd.DataFrame(entity_counts).to_csv(f'tables/nlp/{key}_entities.csv', index=False)
    
    # D. Save explanation-note comparison data
    for key in ['full_df', 'correct_predictions', 'incorrect_predictions', 
                'TP_data', 'TN_data', 'FP_data', 'FN_data']:
        
        # Save hallucination data
        if key in hallucination_results:
            hall_data = pd.DataFrame({
                'hallucination_score': hallucination_results[key]['hallucination_scores'],
                'hallucinated_entities_count': [len(ents) for ents in hallucination_results[key]['hallucinated_entities']],
                'confidence_phrases_count': [len(phrases) for phrases in hallucination_results[key]['confidence_phrases']]
            })
            hall_data.to_csv(f'tables/nlp/comparison/{key}_hallucinations.csv', index=False)
        
        # Save alignment data
        if key in results['explanation_note_comparison']:
            align_data = pd.DataFrame({
                'alignment_score': results['explanation_note_comparison'][key]['alignment_scores'],
                'evidence_support_score': results['explanation_note_comparison'][key]['evidence_support']['score'],
                'evidence_count': results['explanation_note_comparison'][key]['evidence_support']['evidence_count'],
                'unsupported_claims_count': [len(claims) for claims in results['explanation_note_comparison'][key]['evidence_support']['unsupported_claims']]
            })
            align_data.to_csv(f'tables/nlp/comparison/{key}_alignment.csv', index=False)
    
    # E. Save comparative analysis data
    pd.DataFrame({
        'group': list(comparative_results['explanation_fidelity']['hallucination_rates'].keys()),
        'hallucination_rate': list(comparative_results['explanation_fidelity']['hallucination_rates'].values()),
        'evidence_strength': list(comparative_results['explanation_fidelity']['evidence_strength'].values()),
        'alignment_score': list(comparative_results['explanation_fidelity']['alignment_scores'].values()),
    }).to_csv('tables/nlp/comparison/explanation_fidelity_summary.csv', index=False)
    
    # ------------------------------------------------
    # 6. Final Output
    # ------------------------------------------------
    
    print("""
NLP analysis complete!
---------------------
Results saved to:
- /figures/nlp/  : Visualizations
- /results/nlp/   : Summary reports
- /tables/nlp/    : Detailed data tables
- /figures/nlp/comparison/ : Explanation-note comparison visualizations
- /tables/nlp/comparison/  : Explanation-note comparison data

Key findings have been compiled in results/nlp/full_analysis_report.txt
""")
    
    return {
        'results': results,
        'comparative_results': comparative_results,
        'hallucination_results': hallucination_results
    }

if __name__ == "__main__":
    nlp_analysis_results = main()
