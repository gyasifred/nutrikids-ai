import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier, plot_tree
import re
import nltk
from collections import Counter
import networkx as nx

# Install necessary nltk packages if not already installed
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

def load_and_filter_data(file_path):
    """
    Load the dataset and create filtered DataFrames based on prediction correctness.
    
    Args:
        file_path (str): Path to the CSV file containing prediction data.
        
    Returns:
        dict: A dictionary containing:
        - 'full_df': Original dataset with prediction_result column
        - 'TP_data': True Positives
        - 'TN_data': True Negatives
        - 'FP_data': False Positives
        - 'FN_data': False Negatives
        - 'correct_predictions': All correct predictions
        - 'incorrect_predictions': All incorrect predictions
    """
    # Load dataset
    df = pd.read_csv(file_path)

    # Ensure required columns exist
    required_columns = {'patient_id', 'true_label', 'predicted_label', 'explanation', 'original_note'}
    if not required_columns.issubset(df.columns):
        missing_cols = required_columns - set(df.columns)
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Add prediction_result column
    def classify_prediction(row):
        if row['predicted_label'] == 1 and row['true_label'] == 1:
            return 'TP'
        elif row['predicted_label'] == 0 and row['true_label'] == 0:
            return 'TN'
        elif row['predicted_label'] == 1 and row['true_label'] == 0:
            return 'FP'
        elif row['predicted_label'] == 0 and row['true_label'] == 1:
            return 'FN'
        else:
            return 'Unknown'
            
    df['prediction_result'] = df.apply(classify_prediction, axis=1)
    
    # Create filtered dataframes based on prediction result
    TP_data = df[df['prediction_result'] == 'TP'].reset_index(drop=True)
    TN_data = df[df['prediction_result'] == 'TN'].reset_index(drop=True)
    FP_data = df[df['prediction_result'] == 'FP'].reset_index(drop=True)
    FN_data = df[df['prediction_result'] == 'FN'].reset_index(drop=True)
    
    # Group by correct/incorrect for backward compatibility
    correct_predictions = pd.concat([TP_data, TN_data]).reset_index(drop=True)
    incorrect_predictions = pd.concat([FP_data, FN_data]).reset_index(drop=True)

    return {
        'full_df': df,
        'TP_data': TP_data,
        'TN_data': TN_data,
        'FP_data': FP_data,
        'FN_data': FN_data,
        'correct_predictions': correct_predictions,
        'incorrect_predictions': incorrect_predictions
    }

def extract_decision_features(explanation_text):
    """
    Extract key features and criteria mentioned in the explanation text.
    Returns a dictionary of features and their values/mentions.
    
    Args:
        explanation_text (str): The LLM-generated explanation text
        
    Returns:
        dict: Dictionary of features and their mentions
    """
    # Ensure explanation_text is a string
    if not isinstance(explanation_text, str):
        explanation_text = str(explanation_text) if explanation_text is not None else ''
    
    # Define patterns for common criteria
    patterns = {
        'bmi': r'(?:bmi|body[\s-]*mass[\s-]*index)[\s-]*(for[\s-]*age)?[\s:]*((?:-?\d+\.?\d*)|(?:<?\s*\d+\.?\d*)|(?:\b(?:low|normal|high|severe|moderate|mild|underweight|normal weight|obese|overweight)\b))',
        
        'weight_height': r'(?:weight[\s-]*(?:for|to|\/|-)?[\s-]*(?:height|stature)|wfh|whz)[\s:]*((?:-?\d+\.?\d*)|(?:<?\s*\d+\.?\d*)|(?:-?\d+\.?\d*\s*(?:z-score|z score|zscore|sd|standard deviation))|(?:\b(?:low|normal|high|severe|moderate|mild|deficit|adequate|excess)\b))',
        
        'muac': r'(?:muac|mid[\s-]*(?:upper|)[\s-]*arm[\s-]*circumference|mac)[\s:]*((?:-?\d+\.?\d*\s*(?:cm|mm)?)|(?:<?\s*\d+\.?\d*)|(?:\b(?:low|normal|high|severe|moderate|mild|deficit|reduced|adequate)\b))',
        
        'z_score': r'(?:(?:-?\d+\.?\d*)[\s-]*z[\s-]*(?:score|)|z[\s-]*score[\s:]*((?:-?\d+\.?\d*)|(?:<?\s*-?\d+\.?\d*)))',
        
        # Growth parameters
        'percentile': r'(?:(?:\d+(?:\.?\d*)?)(?:th|st|nd|rd)?[\s-]*percentile|percentile[\s:]*((?:\d+(?:\.?\d*)?)(?:th|st|nd|rd)?))',
        
        'growth': r'(?:growth[\s-]*(?:chart|curve|velocity|rate|failure|faltering)|failure[\s-]*to[\s-]*thrive|ftt|stunting|stunted|linear[\s-]*growth)[\s:]*((?:-?\d+\.?\d*)|(?:\b(?:delayed|normal|accelerated|poor|good|improved|worsening|decline|deceleration)\b))',
        
        'weight_loss': r'(?:weight[\s-]*(?:loss|decrease|reduction|decline)|lost[\s-]*weight)[\s:]*((?:\d+\.?\d*\s*(?:%|percent|kg|lb|pounds)?)|(?:\b(?:significant|severe|moderate|mild|minimal|substantial|rapid|gradual|progressive|unintentional)\b))',
        
        'albumin': r'(?:albumin|serum[\s-]*albumin)[\s:]*((?:\d+\.?\d*\s*(?:g/dl|g/l|g/dL)?)|(?:<?\s*\d+\.?\d*)|(?:\b(?:low|normal|high|decreased|elevated|hypoalbuminemia)\b))',
        
        'prealbumin': r'(?:prealbumin|transthyretin)[\s:]*((?:\d+\.?\d*\s*(?:mg/dl|mg/l|mg/dL)?)|(?:<?\s*\d+\.?\d*)|(?:\b(?:low|normal|high|decreased|elevated)\b))',
        
        'hemoglobin': r'(?:hemoglobin|hgb|hb|haemoglobin)[\s:]*((?:\d+\.?\d*\s*(?:g/dl|g/l|g/dL)?)|(?:<?\s*\d+\.?\d*)|(?:\b(?:low|normal|high|decreased|elevated|anemia|anemic|anaemia|anaemic)\b))',
        
        'illness': r'\b(?:illness|infection|disease|sick|surgery|trauma|hospitalization|hospitalized|admitted|icu|intensive[\s-]*care|complication|recovery)\b',
        
        'symptoms': r'\b(?:fatigue|weakness|appetite|tired|weight[\s-]*loss|muscle[\s-]*wasting|lethargy|malaise|exhaustion|tiredness)\b',
        
        'intake': r'\b(?:intake|consumption|diet|oral[\s-]*intake|eating|meal|feeding|caloric|calorie|protein|carbohydrate|fat|nutrient)\b',
        
        'malnutrition_class': r'\b(?:malnutrition|malnourished|undernourishment|undernutrition|protein[\s-]*energy[\s-]*malnutrition|protein[\s-]*calorie[\s-]*malnutrition)\b',
    }

    features = {}

    # Check for malnutrition severity mentions
    severity_match = re.search(r'\b(severe|moderate|mild)[\s-]*malnutrition\b', explanation_text, re.IGNORECASE)
    if severity_match:
        features['severity'] = severity_match.group(1).lower()

    # Extract values for patterns
    for feature, pattern in patterns.items():
        match = re.search(pattern, explanation_text, re.IGNORECASE)
        if match and match.groups():  # Check if match exists AND has groups
            try:
                # Try to access group 1, which is typically the value we want
                features[feature] = match.group(1).lower() if match.group(1) else 'mentioned'
            except IndexError:
                # If group 1 doesn't exist but we have a match, mark as 'mentioned'
                features[feature] = 'mentioned'
        elif match:
            # Match exists but no capture groups
            features[feature] = 'mentioned'

    # Check for key words indicating strong decisions
    if re.search(r'\b(?:clear|clearly|definite|definitely|obvious|strong|evidence|confirms|confirmed|diagnostic)\b', explanation_text, re.IGNORECASE):
        features['confidence'] = 'high'
    elif re.search(r'\b(?:suggests|suggest|indicative|may|might|could|possible|possibly|probable|probably|likely)\b', explanation_text, re.IGNORECASE):
        features['confidence'] = 'medium'
    elif re.search(r'\b(?:unclear|not clear|uncertain|unsure|insufficient|limited|data|more information|cannot determine)\b', explanation_text, re.IGNORECASE):
        features['confidence'] = 'low'

    # Extract yes/no decision indicators
    if re.search(r'\b(?:is malnourished|has malnutrition|suffers from malnutrition|diagnosis of malnutrition|malnutrition is present)\b', explanation_text, re.IGNORECASE):
        features['decision_indicator'] = 'yes'
    elif re.search(r'\b(?:no malnutrition|not malnourished|does not have malnutrition|absence of malnutrition|no evidence of malnutrition)\b', explanation_text, re.IGNORECASE):
        features['decision_indicator'] = 'no'

    return features


def analyze_decision_patterns(data_dict):
    """
    Analyze decision patterns across different prediction outcomes.
    
    Args:
        data_dict (dict): Dictionary containing DataFrames with prediction results
        
    Returns:
        dict: Analysis results
    """
    # Extract features from explanations for each category
    categories = ['TP_data', 'TN_data', 'FP_data', 'FN_data']
    all_features = {}
    
    for category in categories:
        if category in data_dict and not data_dict[category].empty:
            features_list = []
            for _, row in data_dict[category].iterrows():
                features = extract_decision_features(row['explanation'])
                features['category'] = category
                features['patient_id'] = row['patient_id']
                features_list.append(features)
            
            all_features[category] = pd.DataFrame(features_list)
    
    # Count feature mentions by category
    feature_counts = {}
    all_feature_names = set()
    
    for category, df in all_features.items():
        if not df.empty:
            # Get all columns except 'category' and 'patient_id'
            feature_cols = [col for col in df.columns if col not in ['category', 'patient_id']]
            all_feature_names.update(feature_cols)
            
            # Count non-null values for each feature
            feature_counts[category] = {
                feature: sum(df[feature].notna()) for feature in feature_cols if feature in df
            }
    
    # Create a comparison DataFrame
    comparison_data = []
    for feature in all_feature_names:
        row = {'feature': feature}
        for category in categories:
            if category in feature_counts:
                # Calculate percentage of records in the category that mention this feature
                n_records = len(data_dict[category])
                count = feature_counts[category].get(feature, 0)
                percentage = (count / n_records * 100) if n_records > 0 else 0
                row[f'{category}_count'] = count
                row[f'{category}_pct'] = percentage
        comparison_data.append(row)
    
    feature_comparison_df = pd.DataFrame(comparison_data)
    
    # Create visualizations
    if not feature_comparison_df.empty:
        # Select top features for visualization
        top_features = feature_comparison_df.copy()
        
        # Calculate sum of percentages across categories for sorting
        top_features['total_pct'] = sum(
            top_features.get(f'{cat}_pct', 0) for cat in categories if f'{cat}_pct' in top_features
        )
        
        # Sort and get top 15 features
        top_features = top_features.sort_values('total_pct', ascending=False).head(15)
        
        # Plot feature comparison
        plt.figure(figsize=(12, 10))
        
        # Reshape for seaborn
        plot_data = []
        for _, row in top_features.iterrows():
            for category in categories:
                if f'{category}_pct' in row:
                    plot_data.append({
                        'Feature': row['feature'],
                        'Category': category,
                        'Percentage': row[f'{category}_pct']
                    })
        
        plot_df = pd.DataFrame(plot_data)
        
        # Create grouped bar chart
        sns.barplot(x='Percentage', y='Feature', hue='Category', data=plot_df)
        plt.title('Decision Features by Prediction Category')
        plt.xlabel('Percentage of Mentions')
        plt.tight_layout()
        plt.savefig('decision_features_comparison.png')
        plt.close()
        
        # Create heatmap for feature comparison
        plt.figure(figsize=(14, 12))
        
        # Prepare data for heatmap
        heatmap_data = top_features.set_index('feature')
        heatmap_cols = [col for col in heatmap_data.columns if col.endswith('_pct')]
        heatmap_data = heatmap_data[heatmap_cols]
        
        # Rename columns for better display
        heatmap_data.columns = [col.replace('_pct', '') for col in heatmap_data.columns]
        
        # Create the heatmap
        sns.heatmap(heatmap_data, annot=True, cmap='YlGnBu', fmt='.1f')
        plt.title('Feature Importance Heatmap by Category')
        plt.tight_layout()
        plt.savefig('feature_importance_heatmap.png')
        plt.close()
    
    # Analyze differences between correct and incorrect predictions
    correct_features = pd.concat([all_features.get('TP_data', pd.DataFrame()), 
                                  all_features.get('TN_data', pd.DataFrame())])
    
    incorrect_features = pd.concat([all_features.get('FP_data', pd.DataFrame()), 
                                   all_features.get('FN_data', pd.DataFrame())])
    
    # Compare confidence levels
    confidence_comparison = {}
    
    if not correct_features.empty and 'confidence' in correct_features.columns:
        confidence_comparison['correct'] = correct_features['confidence'].value_counts(normalize=True)
    
    if not incorrect_features.empty and 'confidence' in incorrect_features.columns:
        confidence_comparison['incorrect'] = incorrect_features['confidence'].value_counts(normalize=True)
    
    if confidence_comparison:
        confidence_df = pd.DataFrame(confidence_comparison).fillna(0)
        
        # Visualize confidence distribution
        plt.figure(figsize=(10, 6))
        confidence_df.plot(kind='bar')
        plt.title('Confidence Levels: Correct vs Incorrect Predictions')
        plt.ylabel('Proportion')
        plt.xlabel('Confidence Level')
        plt.tight_layout()
        plt.savefig('confidence_comparison.png')
        plt.close()
    
    # Create confusion patterns analysis
    # How often do decision features align with the final prediction?
    decision_alignment = {category: {} for category in categories}
    
    for category, df in all_features.items():
        if not df.empty and 'decision_indicator' in df.columns:
            expected_indicator = 'yes' if category in ['TP_data', 'FP_data'] else 'no'
            alignment_count = sum(df['decision_indicator'] == expected_indicator)
            total = sum(df['decision_indicator'].notna())
            if total > 0:
                decision_alignment[category]['aligned'] = alignment_count / total * 100
                decision_alignment[category]['misaligned'] = 100 - decision_alignment[category]['aligned']
    
    # Create analysis summary
    return {
        'feature_comparison': feature_comparison_df,
        'confidence_comparison': confidence_comparison,
        'decision_alignment': decision_alignment,
        'all_features': all_features
    }

def create_decision_flow_diagram(features_df, category, output_filename):
    """
    Create a decision flow diagram showing the reasoning path for a prediction category
    
    Args:
        features_df (DataFrame): DataFrame with extracted features
        category (str): The prediction category (TP, TN, FP, FN)
        output_filename (str): Filename for the output image
        
    Returns:
        tuple: (NetworkX graph, position dictionary)
    """
    import networkx as nx
    import matplotlib.pyplot as plt
    
    # Create a graph
    G = nx.DiGraph()
    
    # Add start node
    G.add_node("START")
    
    # Add target node based on category
    if category in ['TP_data', 'FP_data']:
        target_node = "YES (Malnutrition)"
    else:
        target_node = "NO (No Malnutrition)"
    
    # Count feature occurrences
    feature_counts = {}
    for col in features_df.columns:
        if col not in ['category', 'patient_id']:
            # Count non-null values
            feature_counts[col] = sum(features_df[col].notna())
    
    # Sort features by frequency
    sorted_features = sorted(feature_counts.items(), key=lambda x: x[1], reverse=True)
    
    # Take top 10 features
    top_features = [f for f, c in sorted_features[:10] if c > 0]
    
    # Build the graph structure
    if top_features:
        G.add_edge("START", top_features[0], weight=feature_counts[top_features[0]])
        
        for i in range(len(top_features) - 1):
            G.add_edge(top_features[i], top_features[i+1], 
                      weight=min(feature_counts[top_features[i]], feature_counts[top_features[i+1]]))
        
        G.add_edge(top_features[-1], target_node, weight=feature_counts[top_features[-1]])
    
    # Add decision indicator if available
    if 'decision_indicator' in features_df.columns:
        indicator_counts = features_df['decision_indicator'].value_counts()
        
        if 'yes' in indicator_counts and indicator_counts['yes'] > 0:
            if 'yes' not in G.nodes():
                G.add_node('Explicit YES')
            G.add_edge("START", 'Explicit YES', weight=indicator_counts['yes'])
            G.add_edge('Explicit YES', target_node, weight=indicator_counts['yes'])
        
        if 'no' in indicator_counts and indicator_counts['no'] > 0:
            if 'no' not in G.nodes():
                G.add_node('Explicit NO')
            G.add_edge("START", 'Explicit NO', weight=indicator_counts['no'])
            G.add_edge('Explicit NO', target_node, weight=indicator_counts['no'])
    
    # Add confidence level if available
    if 'confidence' in features_df.columns:
        confidence_counts = features_df['confidence'].value_counts()
        
        for level, count in confidence_counts.items():
            if count > 0:
                node_name = f'Confidence: {level}'
                G.add_node(node_name)
                G.add_edge("START", node_name, weight=count)
                G.add_edge(node_name, target_node, weight=count)
    
    # Visualize the graph
    plt.figure(figsize=(14, 10))
    
    # Use a hierarchical layout
    try:
        pos = nx.nx_pydot.graphviz_layout(G, prog='dot')
    except:
        # Fallback to spring layout
        pos = nx.spring_layout(G, seed=42)
    
    # Node colors
    node_colors = []
    for node in G.nodes():
        if node == "START":
            node_colors.append("#3498db")  # Blue
        elif node == target_node:
            node_colors.append("#2ecc71" if "YES" in node else "#e74c3c")  # Green or Red
        elif "Confidence" in node:
            node_colors.append("#f39c12")  # Orange
        elif node in ["Explicit YES", "Explicit NO"]:
            node_colors.append("#9b59b6")  # Purple
        else:
            node_colors.append("#a3e4d7")  # Light green
    
    # Node sizes based on importance
    node_sizes = []
    for node in G.nodes():
        if node in ["START", target_node, "Explicit YES", "Explicit NO"]:
            node_sizes.append(3000)
        elif "Confidence" in node:
            node_sizes.append(2500)
        else:
            # Size based on frequency
            count = feature_counts.get(node, 0)
            max_count = max(feature_counts.values()) if feature_counts else 1
            node_sizes.append(1500 + (count / max_count) * 1000)
    
    # Get edge weights for line thickness
    edges = G.edges()
    weights = [G[u][v]['weight'] for u, v in edges]
    max_weight = max(weights) if weights else 1
    
    # Draw the network
    nx.draw(G, pos, 
            with_labels=True, 
            node_color=node_colors,
            node_size=node_sizes, 
            font_size=10, 
            font_weight='bold',
            width=[1.0 + 3.0 * (w / max_weight) for w in weights], 
            edge_color='gray',
            arrows=True, 
            arrowsize=15)
    
    # Add edge labels
    edge_labels = {(u, v): f"{G[u][v]['weight']}" for u, v in G.edges()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
    
    # Add title
    plt.title(f"Decision Flow Diagram for {category}", fontsize=16)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    return G, pos

def analyze_feature_differences(data_dict, analysis_results):
    """
    Analyze feature differences between correct and incorrect predictions
    
    Args:
        data_dict (dict): Dictionary containing DataFrames with prediction results
        analysis_results (dict): Previous analysis results
        
    Returns:
        DataFrame: Feature difference analysis
    """
    # Create structure for feature analysis
    all_features = analysis_results['all_features']
    
    # Combine TP and TN features
    correct_features = pd.concat([
        all_features.get('TP_data', pd.DataFrame()),
        all_features.get('TN_data', pd.DataFrame())
    ])
    
    # Combine FP and FN features
    incorrect_features = pd.concat([
        all_features.get('FP_data', pd.DataFrame()),
        all_features.get('FN_data', pd.DataFrame())
    ])
    
    # Handle empty DataFrames
    if correct_features.empty or incorrect_features.empty:
        return pd.DataFrame()
    
    # Count feature mentions
    correct_mentions = {}
    for col in correct_features.columns:
        if col not in ['category', 'patient_id']:
            correct_mentions[col] = sum(correct_features[col].notna())
    
    incorrect_mentions = {}
    for col in incorrect_features.columns:
        if col not in ['category', 'patient_id']:
            incorrect_mentions[col] = sum(incorrect_features[col].notna())
    
    # Create comparison DataFrame
    all_features = set(list(correct_mentions.keys()) + list(incorrect_mentions.keys()))
    
    feature_diff = []
    for feature in all_features:
        correct_count = correct_mentions.get(feature, 0)
        incorrect_count = incorrect_mentions.get(feature, 0)
        
        correct_pct = correct_count / len(correct_features) * 100 if len(correct_features) > 0 else 0
        incorrect_pct = incorrect_count / len(incorrect_features) * 100 if len(incorrect_features) > 0 else 0
        
        diff = correct_pct - incorrect_pct
        
        feature_diff.append({
            'feature': feature,
            'correct_count': correct_count,
            'incorrect_count': incorrect_count,
            'correct_pct': correct_pct,
            'incorrect_pct': incorrect_pct,
            'diff': diff
        })
    
    # Create DataFrame and sort by absolute difference
    diff_df = pd.DataFrame(feature_diff)
    diff_df = diff_df.sort_values('diff', key=abs, ascending=False)
    
    # Visualize the differences
    if not diff_df.empty:
        plt.figure(figsize=(12, 8))
        top_diff = diff_df.head(10)
        
        sns.barplot(x='diff', y='feature', data=top_diff)
        plt.axvline(x=0, color='black', linestyle='--')
        plt.title('Feature Importance Difference: Correct vs Incorrect Predictions')
        plt.xlabel('Difference in Percentage Points (Correct - Incorrect)')
        plt.tight_layout()
        plt.savefig('feature_difference_analysis.png')
        plt.close()
    
    return diff_df

def analyze_llm_explanations(file_path):
    """
    Main function to analyze LLM explanations
    
    Args:
        file_path (str): Path to the CSV file with LLM predictions
        
    Returns:
        dict: Analysis results
    """
    # Load and prepare data
    data_dict = load_and_filter_data(file_path)
    
    # Extract decision patterns
    decision_patterns = analyze_decision_patterns(data_dict)
    
    # Create decision flow diagrams for each category
    for category in ['TP_data', 'TN_data', 'FP_data', 'FN_data']:
        if category in data_dict and not data_dict[category].empty:
            if category in decision_patterns['all_features']:
                create_decision_flow_diagram(
                    decision_patterns['all_features'][category],
                    category,
                    f'decision_flow_{category}.png'
                )
    
    # Analyze feature differences
    feature_differences = analyze_feature_differences(data_dict, decision_patterns)
    
    # Print summary statistics
    print("\n===== LLM Explanation Analysis =====")
    print(f"Total samples: {len(data_dict['full_df'])}")
    
    for category in ['TP_data', 'TN_data', 'FP_data', 'FN_data']:
        if category in data_dict:
            print(f"{category}: {len(data_dict[category])} cases " + 
                  f"({len(data_dict[category])/len(data_dict['full_df'])*100:.1f}%)")
    
    print("\n===== Decision Pattern Summary =====")
    if feature_differences is not None and not feature_differences.empty:
        print("\nTop features more common in CORRECT predictions:")
        top_correct = feature_differences[feature_differences['diff'] > 0].head(5)
        for _, row in top_correct.iterrows():
            print(f"- {row['feature']}: +{row['diff']:.1f}% difference")
        
        print("\nTop features more common in INCORRECT predictions:")
        top_incorrect = feature_differences[feature_differences['diff'] < 0].head(5)
        for _, row in top_incorrect.iterrows():
            print(f"- {row['feature']}: {row['diff']:.1f}% difference")
    
    # Output analysis results
    results = {
        'data': data_dict,
        'decision_patterns': decision_patterns,
        'feature_differences': feature_differences
    }
    
    return results

# Main execution
if __name__ == "__main__":
    file_path = "./LLM_pre_eval/gemma/predictions.csv" 
    analysis_results = analyze_llm_explanations(file_path)
