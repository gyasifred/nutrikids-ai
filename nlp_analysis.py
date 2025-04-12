#!/usr/bin/env python3
import os
import re
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from pathlib import Path
from collections import Counter
from scipy import stats
from scipy.stats import ttest_ind, pearsonr
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

try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
except ImportError as e:
    raise ImportError("spaCy is required for named entity recognition. Please install spaCy (pip install spacy) and a language model.") from e
except Exception as e:
    raise RuntimeError("spaCy language model 'en_core_web_sm' not found. Please install it via: python -m spacy download en_core_web_sm") from e

# Set plotting style
sns.set_palette("Set2")
plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['font.size'] = 12
warnings.filterwarnings('ignore', category=UserWarning)


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
        'vitamin d', 'zinc', 'magnesium', 'calcium', 'nitrogen balance', 
        'blood pressure', 'bp', 'wbc', 'white blood cell', 'glucose', 'sodium', 'potassium',
        'creatinine', 'bun', 'alt', 'ast', 'bilirubin', 'heart rate', 'hr', 'temperature',
        'temp', 'oxygen', 'o2', 'sats', 'blood sugar'
    ],
    
    # Risk factors
    'risk_factors': [
        'medications', 'polypharmacy', 'depression', 'anxiety', 'cognitive impairment',
        'dementia', 'socioeconomic', 'poverty', 'homelessness', 'social isolation',
        'elderly', 'pediatric', 'pregnancy', 'alcohol', 'substance abuse', 'surgery',
        'hospitalization', 'immobility', 'disability',
        'history of', 'hx of', 'smoking', 'smoker', 'hypertension', 'hypertensive', 
        'diabetic', 'family history'
    ],
    
    # Imaging
    'imaging': [
        'x-ray', 'xray', 'ct scan', 'ctscan', 'mri', 'ultrasound', 'chest x-ray', 
        'cxr', 'ekg', 'ecg', 'angiogram', 'imaging', 'scan'
    ],
    
    # Medications
    'medications': [
        'mg', 'tablet', 'aspirin', 'metformin', 'insulin', 'penicillin', 'ibuprofen',
        'acetaminophen', 'lisinopril', 'metoprolol', 'atorvastatin', 'drug', 'medication',
        'dose', 'pill', 'therapy'
    ]
}

def jaccard_similarity(set1, set2):
    """Compute Jaccard similarity between two sets."""
    if not set1 and not set2:
        return 1.0  # if both are empty, define similarity as 1 (no difference)
    if not set1 or not set2:
        return 0.0
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union

def cohen_d(x, y):
    """Compute Cohen's d effect size for two independent samples x and y."""
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    nx, ny = len(x), len(y)
    # Pooled standard deviation (using unbiased variance ddof=1)
    vx, vy = x.var(ddof=1), y.var(ddof=1)
    pooled_std = np.sqrt(((nx - 1) * vx + (ny - 1) * vy) / (nx + ny - 2)) if (nx + ny) > 2 else 0.0
    if pooled_std == 0:
        return 0.0
    return (x.mean() - y.mean()) / pooled_std

def analyze_evidence_support(df):
    """
    Compute alignment, hallucination, and evidence support metrics for each explanation-note pair in the DataFrame.
    Adds columns to df: 'alignment_score', 'hallucination_score', 'evidence_support_score', 'support_fraction', 'coverage_fraction'.
    """
    # Define confidence/hedging phrases to look for (case-insensitive)
    confidence_phrases = [
        "likely", "possibly", "maybe", "suggests", "suggest", "could", "might",
        "suspect", "perhaps", "appears", "appear", "seems", "I think", "cannot rule out"
    ]
    # Compile regex pattern for these phrases as whole words (case insensitive)
    conf_pattern = re.compile(r'\b(' + '|'.join([re.escape(p) for p in confidence_phrases]) + r')\b', flags=re.IGNORECASE)

    # Prepare lists for metric values
    alignment_scores = []
    hallucination_scores = []
    support_fracs = []
    coverage_fracs = []
    evidence_scores = []

    for _, row in df.iterrows():
        note_text = str(row.get('original_note') or row.get('note') or row.get('note_text') or row.get('clinical_note') or "")
        expl_text = str(row.get('explanation') or row.get('expl') or row.get('model_explanation') or "")

        # Process texts with spaCy (tokenization + NER)
        doc_note = nlp(note_text)
        doc_expl = nlp(expl_text)

        # Compute Jaccard alignment on content words (ignore stopwords and punctuation)
        note_tokens = {token.lemma_.lower() for token in doc_note if token.is_alpha and not token.is_stop}
        expl_tokens = {token.lemma_.lower() for token in doc_expl if token.is_alpha and not token.is_stop}
        alignment = jaccard_similarity(note_tokens, expl_tokens)
        alignment_scores.append(alignment)

        # Hallucination detection:
        # Named entities in note and explanation
        note_ents = {ent.text.lower() for ent in doc_note if ent.ent_type_ != ""}
        expl_ents = {ent.text.lower() for ent in doc_expl if ent.ent_type_ != ""}
        # Compute fraction of explanation's entities that are not present in note
        unsupported_ents = expl_ents - note_ents
        if len(expl_ents) > 0:
            ent_hall_score = len(unsupported_ents) / len(expl_ents)
        else:
            ent_hall_score = 0.0
        # Confidence phrase presence flag
        conf_flag = 1 if conf_pattern.search(expl_text) else 0
        # Adjust hallucination score if confidence phrase present
        if len(expl_ents) > 0:
            if conf_flag:
                ent_hall_score = (len(unsupported_ents) + 1) / (len(expl_ents) + 1)
        else:
            # If no entities in explanation, use confidence flag as indicator
            ent_hall_score = 1.0 if conf_flag else 0.0
        hallucination_scores.append(ent_hall_score)

        # Evidence support analysis:
        # Treat all named entities as clinical indicators for evidence
        common_ents = note_ents.intersection(expl_ents)
        # Support fraction: fraction of explanation's entities that appear in note
        if len(expl_ents) > 0:
            support_fraction = len(common_ents) / len(expl_ents)
        else:
            support_fraction = 1.0 if len(note_ents) == 0 else 0.0
        # Coverage fraction: fraction of note's entities that are mentioned in explanation
        if len(note_ents) > 0:
            coverage_fraction = len(common_ents) / len(note_ents)
        else:
            coverage_fraction = 1.0 if len(expl_ents) == 0 else 0.0

        support_fracs.append(support_fraction)
        coverage_fracs.append(coverage_fraction)

        # Overall evidence support score (harmonic mean of support and coverage, or special-case handling)
        if support_fraction > 0 and coverage_fraction > 0:
            evidence_score = 2 * support_fraction * coverage_fraction / (support_fraction + coverage_fraction)
        else:
            if len(note_ents) == 0 and len(expl_ents) == 0:
                evidence_score = 1.0  # no evidence needed and none given
            elif len(note_ents) > 0 and len(expl_ents) == 0:
                evidence_score = 0.0  # evidence was needed but none provided
            elif len(note_ents) == 0 and len(expl_ents) > 0:
                evidence_score = 0.0  # no evidence needed but explanation provided some (hallucinated evidence)
            else:
                evidence_score = 0.0
        evidence_scores.append(evidence_score)

    # Add metrics to DataFrame
    df['alignment_score'] = alignment_scores
    df['hallucination_score'] = hallucination_scores
    df['support_fraction'] = support_fracs
    df['coverage_fraction'] = coverage_fracs
    df['evidence_support_score'] = evidence_scores
    return df

def category_analysis(df):
    """
    Perform category-specific evidence support analysis.
    Returns dictionaries for category coverage and support rates across the entire dataset,
    and raw counts of category presence in notes and explanations.
    """
    # Define categories based on integrated clinical indicators
    categories = clinical_indicators

    notes_lower = df['original_note'].fillna('').astype(str).str.lower()
    expls_lower = df['explanation'].fillna('').astype(str).str.lower()

    cat_note_count = {cat: 0 for cat in categories}
    cat_expl_count = {cat: 0 for cat in categories}
    cat_coverage_count = {cat: 0 for cat in categories}

    for i in range(len(df)):
        note_text = notes_lower.iloc[i]
        expl_text = expls_lower.iloc[i]
        for cat, keywords in categories.items():
            has_in_note = any(kw in note_text for kw in keywords)
            has_in_expl = any(kw in expl_text for kw in keywords)
            if has_in_note:
                cat_note_count[cat] += 1
            if has_in_expl:
                cat_expl_count[cat] += 1
            if has_in_note and has_in_expl:
                cat_coverage_count[cat] += 1

    category_coverage_rate = {}
    category_support_rate = {}
    for cat in categories:
        if cat_note_count[cat] > 0:
            category_coverage_rate[cat] = cat_coverage_count[cat] / cat_note_count[cat]
        else:
            category_coverage_rate[cat] = 0.0
        if cat_expl_count[cat] > 0:
            category_support_rate[cat] = cat_coverage_count[cat] / cat_expl_count[cat]
        else:
            category_support_rate[cat] = 0.0

    return category_coverage_rate, category_support_rate, cat_note_count, cat_expl_count, cat_coverage_count

def statistical_analysis(df):
    """
    Perform t-tests and compute Cohen's d for alignment and evidence support scores across prediction groups (TP, FP, TN, FN).
    Returns a dictionary with group means and comparison results.
    """
    results = {}
    # Determine prediction groups for binary classification
    if 'true_label' in df.columns and 'pred_label' in df.columns:
        true = df['true_label'].astype(int)
        pred = df['pred_label'].astype(int)
        groups = {
            'TP': df[(true == 1) & (pred == 1)],
            'FP': df[(true == 0) & (pred == 1)],
            'TN': df[(true == 0) & (pred == 0)],
            'FN': df[(true == 1) & (pred == 0)]
        }
        # Compute mean scores per group
        results['alignment_means'] = {g: grp['alignment_score'].mean() for g, grp in groups.items()}
        results['evidence_means'] = {g: grp['evidence_support_score'].mean() for g, grp in groups.items()}

        comparisons = {}
        # Compare predicted positive cases: TP vs FP
        if len(groups['TP']) > 1 and len(groups['FP']) > 1:
            t_stat, p_val = ttest_ind(groups['TP']['alignment_score'], groups['FP']['alignment_score'], equal_var=False)
            comparisons['Align_TP_vs_FP'] = {'t_stat': t_stat, 'p_value': p_val, 'cohen_d': cohen_d(groups['TP']['alignment_score'], groups['FP']['alignment_score'])}
            t_stat2, p_val2 = ttest_ind(groups['TP']['evidence_support_score'], groups['FP']['evidence_support_score'], equal_var=False)
            comparisons['Evidence_TP_vs_FP'] = {'t_stat': t_stat2, 'p_value': p_val2, 'cohen_d': cohen_d(groups['TP']['evidence_support_score'], groups['FP']['evidence_support_score'])}
        # Compare predicted negative cases: TN vs FN
        if len(groups['TN']) > 1 and len(groups['FN']) > 1:
            t_stat, p_val = ttest_ind(groups['TN']['alignment_score'], groups['FN']['alignment_score'], equal_var=False)
            comparisons['Align_TN_vs_FN'] = {'t_stat': t_stat, 'p_value': p_val, 'cohen_d': cohen_d(groups['TN']['alignment_score'], groups['FN']['alignment_score'])}
            t_stat2, p_val2 = ttest_ind(groups['TN']['evidence_support_score'], groups['FN']['evidence_support_score'], equal_var=False)
            comparisons['Evidence_TN_vs_FN'] = {'t_stat': t_stat2, 'p_value': p_val2, 'cohen_d': cohen_d(groups['TN']['evidence_support_score'], groups['FN']['evidence_support_score'])}

        results['comparisons'] = comparisons
    return results

def correlation_analysis(df):
    """
    Compute Pearson correlation between evidence support, hallucination score, and alignment.
    Returns a dict with correlation coefficients and p-values for each pair of metrics.
    """
    corr_results = {}
    align = df['alignment_score']
    hall = df['hallucination_score']
    evid = df['evidence_support_score']
    r1, p1 = pearsonr(align, hall)
    corr_results['align_vs_hallucination'] = {'pearson_r': r1, 'p_value': p1}
    r2, p2 = pearsonr(align, evid)
    corr_results['align_vs_evidence'] = {'pearson_r': r2, 'p_value': p2}
    r3, p3 = pearsonr(evid, hall)
    corr_results['evidence_vs_hallucination'] = {'pearson_r': r3, 'p_value': p3}
    return corr_results

def generate_evidence_visualizations(df, category_coverage_rate, category_support_rate, stats_results, cat_note_count, cat_expl_count, cat_coverage_count, output_dir='figures/nlp/evidence_enhanced'):
    """
    Generate and save visualizations for evidence support analysis:
      - Histograms for alignment, hallucination, and evidence support scores.
      - Category-level bar plots for support and coverage rates.
      - Scatterplots for pairwise correlations between metrics.
      - Bar plot for effect sizes (Cohen's d) of group comparisons.
      - Heatmap for category coverage by prediction group.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Histograms for alignment, hallucination, and evidence support
    metrics = [
        ('alignment_score', 'Alignment Score'),
        ('hallucination_score', 'Hallucination Score'),
        ('evidence_support_score', 'Evidence Support Score')
    ]
    for col, label in metrics:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(df[col], bins=20, kde=False, ax=ax, color='steelblue')
        ax.set_title(f"Distribution of {label}s")
        ax.set_xlabel(label)
        ax.set_ylabel("Count")
        plt.tight_layout()
        fig.savefig(f"{output_dir}/{col}_histogram.png")
        plt.close(fig)

    # Scatterplots for correlation analysis (each pair of metrics)
    scatter_pairs = [
        ('alignment_score', 'hallucination_score'),
        ('alignment_score', 'evidence_support_score'),
        ('evidence_support_score', 'hallucination_score')
    ]
    for x_col, y_col in scatter_pairs:
        fig, ax = plt.subplots(figsize=(5, 5))
        sns.scatterplot(x=df[x_col], y=df[y_col], ax=ax, color='purple', alpha=0.7)
        ax.set_title(f"{x_col.replace('_score','').capitalize()} vs {y_col.replace('_score','').capitalize()}")
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        # Annotate Pearson r on plot
        if df[x_col].count() > 1 and df[y_col].count() > 1:
            r_val = df[x_col].corr(df[y_col])
            ax.text(0.05, 0.95, f"r = {r_val:.2f}", transform=ax.transAxes,
                    fontsize=9, verticalalignment='top')
        plt.tight_layout()
        fig.savefig(f"{output_dir}/scatter_{x_col}_vs_{y_col}.png")
        plt.close(fig)

    # Category-level bar plots for coverage and support rates
    cats = list(category_coverage_rate.keys())
    # Preserve a consistent category order (the original insertion order of categories in category_analysis)
    cov_values = [category_coverage_rate.get(cat, 0) for cat in cats]
    sup_values = [category_support_rate.get(cat, 0) for cat in cats]

    # Coverage rate bar plot
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=cats, y=cov_values, ax=ax, color='skyblue', order=cats)
    ax.set_title("Category Coverage Rate (Note -> Explanation)")
    ax.set_ylabel("Coverage (fraction of evidence mentioned)")
    ax.set_xlabel("Evidence Category")
    ax.set_ylim(0, 1)
    for p in ax.patches:
        height = p.get_height()
        ax.annotate(f"{height*100:.0f}%", (p.get_x() + p.get_width()/2, height),
                    ha='center', va='bottom', fontsize=8)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    fig.savefig(f"{output_dir}/category_coverage_rates.png")
    plt.close(fig)

    # Support rate bar plot
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=cats, y=sup_values, ax=ax, color='lightgreen', order=cats)
    ax.set_title("Category Support Rate (Explanation -> Note)")
    ax.set_ylabel("Support (fraction of mentions supported)")
    ax.set_xlabel("Evidence Category")
    ax.set_ylim(0, 1)
    for p in ax.patches:
        height = p.get_height()
        ax.annotate(f"{height*100:.0f}%", (p.get_x() + p.get_width()/2, height),
                    ha='center', va='bottom', fontsize=8)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    fig.savefig(f"{output_dir}/category_support_rates.png")
    plt.close(fig)

    # Effect size bar plot for group comparisons (Cohen's d)
    comp = stats_results.get('comparisons', {})
    labels = []
    values = []
    for comp_name, stats in comp.items():
        if 'cohen_d' in stats:
            labels.append(comp_name)
            values.append(stats['cohen_d'])
    if labels:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(x=labels, y=values, ax=ax, palette='husl')
        ax.set_title("Effect Sizes (Cohen's d) for Group Comparisons")
        ax.set_ylabel("Cohen's d")
        ax.set_xlabel("Comparison")
        for p in ax.patches:
            height = p.get_height()
            ax.annotate(f"{height:.2f}", (p.get_x() + p.get_width()/2, height),
                        ha='center', va='bottom', fontsize=8)
        plt.xticks(rotation=15, ha='right')
        plt.tight_layout()
        fig.savefig(f"{output_dir}/effect_sizes.png")
        plt.close(fig)

    # Heatmap for category coverage by prediction group
    if 'true_label' in df.columns and 'pred_label' in df.columns:
        group_masks = {
            'TP': (df['true_label'] == 1) & (df['pred_label'] == 1),
            'FP': (df['true_label'] == 0) & (df['pred_label'] == 1),
            'TN': (df['true_label'] == 0) & (df['pred_label'] == 0),
            'FN': (df['true_label'] == 1) & (df['pred_label'] == 0)
        }
        group_cov_matrix = pd.DataFrame(index=clinical_indicators.keys(), columns=['TP', 'FP', 'TN', 'FN'], dtype=float)
        for group, mask in group_masks.items():
            df_group = df[mask]
            notes_g = df_group['original_note'].fillna('').astype(str).str.lower()
            expls_g = df_group['explanation'].fillna('').astype(str).str.lower()
            for cat, keywords in clinical_indicators.items():
                cov_count = 0
                note_count = 0
                for i in range(len(df_group)):
                    note_text = notes_g.iloc[i]
                    expl_text = expls_g.iloc[i]
                    has_note = any(kw in note_text for kw in keywords)
                    has_expl = any(kw in expl_text for kw in keywords)
                    if has_note:
                        note_count += 1
                    if has_note and has_expl:
                        cov_count += 1
                if note_count > 0:
                    group_cov_matrix.loc[cat, group] = cov_count / note_count
                else:
                    group_cov_matrix.loc[cat, group] = np.nan

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(group_cov_matrix * 100, annot=True, fmt=".0f", cmap="YlGnBu",
                    mask=group_cov_matrix.isna(), cbar_kws={'label': 'Coverage (%)'}, ax=ax)
        ax.set_title("Evidence Coverage by Category and Prediction Group")
        ax.set_ylabel("Evidence Category")
        ax.set_xlabel("Prediction Group")
        plt.tight_layout()
        fig.savefig(f"{output_dir}/category_coverage_heatmap.png")
        plt.close(fig)

def main():
    # ------------------------------------------------
    # 1. Configuration and Data Loading
    # ------------------------------------------------
    parser = argparse.ArgumentParser(description='Enhanced NLP Analysis Pipeline for Clinical Explanations')
    parser.add_argument('--input', type=str, default="./llama_zero_shot/prediction.csv", 
                        help='Path to input data CSV file')
    parser.add_argument('--output_dir', type=str, default="./results", 
                        help='Directory for output files')
    args = parser.parse_args()
    
    os.makedirs(f'{args.output_dir}/figures/nlp', exist_ok=True)
    os.makedirs(f'{args.output_dir}/tables/nlp', exist_ok=True)
    os.makedirs(f'{args.output_dir}/figures/nlp/evidence_enhanced', exist_ok=True)
    os.makedirs(f'{args.output_dir}/tables/nlp/evidence_enhanced', exist_ok=True)

    data = load_and_filter_data(args.input)

    # ------------------------------------------------
    # 2. Enhanced NLP Analysis Pipeline
    # ------------------------------------------------
    results = {}

    keys = ['full_df', 'correct_predictions', 'incorrect_predictions', 'TP_data', 'TN_data', 'FP_data', 'FN_data']

    for key in keys:
        print(f"Processing {key} dataset...")
        results[key] = {}
        
        # Standard NLP Analysis
        df = data[key]
        explanations = df['explanation']
        notes = df['original_note']
        if 'true_label' in df.columns:
            labels = df['true_label']
        else:
            # Create dummy labels if not available
            labels = pd.Series([1] * len(df))

        # A. Topic Modeling
        print(f"Running topic modeling for {key}...")
        results[key]['topic_analysis'] = analyze_topics(explanations, labels, n_topics=5)
        plot_topic_analysis(results[key]['topic_analysis'], output_dir=f'{args.output_dir}/figures/nlp/{key}')
        
        # Save topics to CSV
        topic_df = pd.DataFrame({
            'topic_id': list(results[key]['topic_analysis']['topics'].keys()),
            'top_words': [", ".join(words) for words in results[key]['topic_analysis']['topics'].values()]
        })
        topic_df.to_csv(f'{args.output_dir}/tables/nlp/{key}_topics.csv', index=False)

        # B. Keyword Network Analysis
        print(f"Creating keyword network for {key}...")
        results[key]['keyword_network'] = create_keyword_network(explanations, labels)
        plot_keyword_network(results[key]['keyword_network'], output_dir=f'{args.output_dir}/figures/nlp/{key}')

        # C. Sentiment Analysis
        print(f"Analyzing sentiment for {key}...")
        results[key]['sentiment_analysis'] = analyze_explanation_sentiment(explanations, labels)
        plot_sentiment_analysis(results[key]['sentiment_analysis'], output_dir=f'{args.output_dir}/figures/nlp/{key}')
        
        # Save sentiment stats
        sentiment_stats = results[key]['sentiment_analysis']['sentiment_stats']
        sentiment_stats.to_csv(f'{args.output_dir}/tables/nlp/{key}_sentiment.csv')

        # D. Named Entity Recognition
        print(f"Extracting named entities for {key}...")
        results[key]['entity_analysis'] = get_named_entities(explanations)
        plot_entity_analysis(results[key]['entity_analysis'], output_dir=f'{args.output_dir}/figures/nlp/{key}')
        
        # Save entity counts
        entity_counts = []
        for cat, entities in results[key]['entity_analysis']['category_counts'].items():
            for entity, count in entities.items():
                entity_counts.append({'category': cat, 'entity': entity, 'count': count})
        pd.DataFrame(entity_counts).to_csv(f'{args.output_dir}/tables/nlp/{key}_entities.csv', index=False)

        # E. Evidence Support Analysis
        print(f"Analyzing evidence support for {key}...")
        df_enhanced = analyze_evidence_support(df.copy())
        
        # Category-level evidence analysis
        category_coverage_rate, category_support_rate, cat_note_count, cat_expl_count, cat_coverage_count = category_analysis(df_enhanced)
        
        # Statistical tests for evidence metrics by prediction group
        stats_results = statistical_analysis(df_enhanced)
        
        # Correlation analysis between metrics
        corr_results = correlation_analysis(df_enhanced)
        
        # Generate and save visualizations
        os.makedirs(f'{args.output_dir}/figures/nlp/evidence_enhanced/{key}', exist_ok=True)
        generate_evidence_visualizations(
            df_enhanced, category_coverage_rate, category_support_rate, stats_results,
            cat_note_count, cat_expl_count, cat_coverage_count,
            output_dir=f'{args.output_dir}/figures/nlp/evidence_enhanced/{key}'
        )
        
        # Save enhanced dataframe with evidence metrics
        metrics_cols = ['alignment_score', 'hallucination_score', 'evidence_support_score', 
                        'support_fraction', 'coverage_fraction']
        df_enhanced[metrics_cols].to_csv(f'{args.output_dir}/tables/nlp/evidence_enhanced/{key}_metrics.csv', index=False)
        
        # Save category analysis results
        cat_df = pd.DataFrame({
            'category': list(category_coverage_rate.keys()),
            'coverage_rate': list(category_coverage_rate.values()),
            'support_rate': list(category_support_rate.values()),
            'note_count': [cat_note_count[cat] for cat in category_coverage_rate.keys()],
            'explanation_count': [cat_expl_count[cat] for cat in category_coverage_rate.keys()],
            'overlap_count': [cat_coverage_count[cat] for cat in category_coverage_rate.keys()]
        })
        cat_df.to_csv(f'{args.output_dir}/tables/nlp/evidence_enhanced/{key}_categories.csv', index=False)
        
        # Save correlation analysis
        corr_df = pd.DataFrame({
            'metric_pair': list(corr_results.keys()),
            'pearson_r': [stats['pearson_r'] for stats in corr_results.values()],
            'p_value': [stats['p_value'] for stats in corr_results.values()]
        })
        corr_df.to_csv(f'{args.output_dir}/tables/nlp/evidence_enhanced/{key}_correlations.csv', index=False)

    # ------------------------------------------------
    # 3. Generate Comprehensive Summary Report
    # ------------------------------------------------
    report_sections = ["ENHANCED NLP ANALYSIS REPORT\n===========================\n\n"]

    # Dataset summary
    total_cases = len(data['full_df'])
    if 'correct_predictions' in data and 'incorrect_predictions' in data:
        correct_cases = len(data['correct_predictions'])
        incorrect_cases = len(data['incorrect_predictions'])
        report_sections.append(f"Total cases: {total_cases}\nCorrect predictions: {correct_cases}\nIncorrect predictions: {incorrect_cases}\n\n")
    else:
        report_sections.append(f"Total cases: {total_cases}\n\n")

    # Topic modeling summary
    report_sections.append("Topic Modeling Summary:\n")
    for key in keys[:3]:  # Focus on main data splits
        if key in results:
            topics = results[key]['topic_analysis']['topics']
            report_sections.append(f"\nKey topics in {key.replace('_', ' ')}:")
            for tid, words in topics.items():
                report_sections.append(f" - Topic {tid}: {', '.join(words[:5])}...\n")

    # Sentiment analysis summary
    report_sections.append("\nSentiment Analysis Summary:\n")
    for key in keys[:3]:  # Focus on main data splits
        if key in results and 'sentiment_analysis' in results[key]:
            sentiment = results[key]['sentiment_analysis']['sentiment_stats']
            report_sections.append(f"\nAverage Sentiment ({key}):\n")
            for label, row in sentiment.iterrows():
                report_sections.append(f" - {label}: Mean sentiment = {row['mean']:.2f}\n")

    # Named Entity Recognition summary
    report_sections.append("\nNamed Entity Summary (Full Dataset):\n")
    if 'full_df' in results and 'entity_analysis' in results['full_df']:
        full_entities = results['full_df']['entity_analysis']['category_counts']
        for category, entities in full_entities.items():
            top_entities = sorted(entities.items(), key=lambda x: x[1], reverse=True)[:3]
            report_sections.append(f" - {category}: {', '.join([f'{ent}({cnt})' for ent, cnt in top_entities])}\n")

    # Evidence support metrics summary
    report_sections.append("\nEvidence Support Analysis:\n")
    for key in keys[:3]:  # Focus on main data splits
        if key in data:
            df_metrics = data[key].copy()
            if 'evidence_support_score' in df_metrics.columns:
                avg_align = df_metrics['alignment_score'].mean()
                avg_hall = df_metrics['hallucination_score'].mean()
                avg_support = df_metrics['evidence_support_score'].mean()
                
                report_sections.append(f"\n{key.replace('_', ' ')}:")
                report_sections.append(f" - Average Alignment Score: {avg_align:.2f}")
                report_sections.append(f" - Average Hallucination Score: {avg_hall:.2f}")
                report_sections.append(f" - Average Evidence Support Score: {avg_support:.2f}\n")
                
                # Check if we have prediction groups
                if 'true_label' in df_metrics.columns and 'pred_label' in df_metrics.columns:
                    # Calculate scores by prediction group
                    groups = {'TP': (1, 1), 'FP': (0, 1), 'TN': (0, 0), 'FN': (1, 0)}
                    report_sections.append("\nEvidence Support by Prediction Group:\n")
                    for group, (true, pred) in groups.items():
                        mask = (df_metrics['true_label'] == true) & (df_metrics['pred_label'] == pred)
                        group_df = df_metrics[mask]
                        if len(group_df) > 0:
                            avg_support_group = group_df['evidence_support_score'].mean()
                            report_sections.append(f" - {group}: {avg_support_group:.2f} (n={len(group_df)})\n")

    # Write full report
    with open(f'{args.output_dir}/tables/nlp/full_analysis_report.txt', 'w') as f:
        f.writelines(report_sections)

    print("Enhanced NLP analysis complete. Results saved to:", args.output_dir)

if __name__ == "__main__":
    main()
