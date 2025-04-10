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

def main():
    # ------------------------------------------------
    # 1. Configuration and Data Loading
    # ------------------------------------------------
    
    # Create output directories
    os.makedirs('figures/nlp', exist_ok=True)
    os.makedirs('results/nlp', exist_ok=True)
    os.makedirs('tables/nlp', exist_ok=True)
    
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
        'entity_analysis': {}
    }
    
    # A. Topic Modeling Analysis
    print("Running topic modeling analysis...")
    for key in ['full_df', 'correct_predictions', 'incorrect_predictions', 
                'correct_yes', 'correct_no', 'incorrect_yes', 'incorrect_no']:
        results['topic_analysis'][key] = analyze_topics(
            data[key]['explanation'],
            data[key]['true_label'],
            n_topics=5
        )
    
    # B. Keyword Network Analysis
    print("Creating keyword networks...")
    for key in ['full_df', 'correct_predictions', 'incorrect_predictions']:
        network = create_keyword_network(
            data[key]['explanation'],
            data[key]['true_label']
        )
        results['keyword_networks'][key] = network
    
    # C. Sentiment Analysis
    print("Analyzing sentiment patterns...")
    for key in ['full_df', 'correct_predictions', 'incorrect_predictions', 
                'correct_yes', 'correct_no', 'incorrect_yes', 'incorrect_no']:
        results['sentiment_analysis'][key] = analyze_explanation_sentiment(
            data[key]['explanation'],
            data[key]['true_label']
        )
    
    # D. Named Entity Recognition
    print("Extracting named entities...")
    for key in ['full_df', 'correct_predictions', 'incorrect_predictions']:
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
        'false_negatives': {}
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
    for group, label in [('correct_yes', 'true_positives'), 
                         ('correct_no', 'true_negatives'),
                         ('incorrect_yes', 'false_positives'),
                         ('incorrect_no', 'false_negatives')]:
        # Topic analysis
        comparative_results[label]['topics'] = {
            'main_topics': list(results['topic_analysis'][group]['topics'].values())
        }
        
        # Sentiment analysis
        comparative_results[label]['sentiment'] = {
            'mean': results['sentiment_analysis'][group]['sentiment_stats']['mean'].to_dict()
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
    
    # E. Generate Comprehensive Report
    report_sections = []
    
    # 1. Executive Summary
    report_sections.append("""
NLP ANALYSIS REPORT
==================

This report provides a comprehensive natural language processing analysis of malnutrition 
prediction explanations, including topic modeling, sentiment analysis, keyword networks, 
and named entity recognition across different prediction groups.
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
    
    # 7. Recommendations
    report_sections.append("""

RECOMMENDATIONS
---------------
1. Focus on improving explanations related to: """ + ", ".join(list(incorrect_topics.values())[0][:3]) + """
2. Monitor sentiment patterns in: """ + ("FP/FN" if abs(incorrect_sentiment.loc['yes']['mean'] - incorrect_sentiment.loc['no']['mean']) > 0.2 else "all cases") + """
3. Validate terminology usage for: """ + ", ".join([cat[0] for cat in top_categories[:3]]) + """
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

Key findings have been compiled in results/nlp/full_analysis_report.txt
""")
    
    return {
        'results': results,
        'comparative_results': comparative_results
    }

if __name__ == "__main__":
    nlp_analysis_results = main()
