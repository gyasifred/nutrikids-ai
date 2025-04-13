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
    
    # Add full dataset features extraction
    all_features['full_df'] = []
    for _, row in data_dict['full_df'].iterrows():
        features = extract_decision_features(row['explanation'])
        features['category'] = row['prediction_result']  # Store the prediction result as category
        features['patient_id'] = row['patient_id']
        features['true_label'] = row['true_label']  # Store true label for additional analysis
        features['predicted_label'] = row['predicted_label']  # Store predicted label
        all_features['full_df'].append(features)
    
    all_features['full_df'] = pd.DataFrame(all_features['full_df'])
    
    # Process individual categories as before
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
            # Get all columns except 'category', 'patient_id', 'true_label', and 'predicted_label'
            feature_cols = [col for col in df.columns if col not in ['category', 'patient_id', 'true_label', 'predicted_label']]
            all_feature_names.update(feature_cols)
            
            # Count non-null values for each feature
            feature_counts[category] = {
                feature: sum(df[feature].notna()) for feature in feature_cols if feature in df
            }
    
    # Create a comparison DataFrame
    comparison_data = []
    for feature in all_feature_names:
        row = {'feature': feature}
        for category in categories + ['full_df']:
            if category in feature_counts:
                # Calculate percentage of records in the category that mention this feature
                n_records = len(data_dict[category]) if category != 'full_df' else len(data_dict['full_df'])
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
        if category != 'full_df' and not df.empty and 'decision_indicator' in df.columns:
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
    Create a more structured top-down decision flow diagram showing the reasoning path
    
    Args:
        features_df (DataFrame): DataFrame with extracted features
        category (str): The prediction category (TP, TN, FP, FN, or 'full_df' for full dataset)
        output_filename (str): Filename for the output image
        
    Returns:
        tuple: (NetworkX graph, position dictionary)
    """
    import networkx as nx
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Create a graph
    G = nx.DiGraph()
    
    # Add start node
    G.add_node("START")
    
    # Handle the full dataset case differently
    if category == 'full_df':
        # Create both target nodes for full dataset
        yes_node = "YES (Malnutrition)"
        no_node = "NO (No Malnutrition)"
        G.add_node(yes_node)
        G.add_node(no_node)
    else:
        # Add target node based on category
        if category in ['TP_data', 'FP_data']:
            target_node = "YES (Malnutrition)"
        else:
            target_node = "NO (No Malnutrition)"
        G.add_node(target_node)
    
    # Count feature occurrences
    feature_counts = {}
    for col in features_df.columns:
        if col not in ['category', 'patient_id', 'true_label', 'predicted_label']:
            # Count non-null values
            feature_counts[col] = sum(features_df[col].notna())
    
    # Sort features by frequency
    sorted_features = sorted(feature_counts.items(), key=lambda x: x[1], reverse=True)
    
    # Take top features - limited to improve readability
    top_features = [f for f, c in sorted_features[:8] if c > 0]
    
    # Create node hierarchy with explicit levels
    level_nodes = {
        0: ["START"],
        1: [],  # Decision indicators and confidence nodes
        2: [],  # Top features
        3: []   # Outcome nodes
    }
    
    # Add decision indicator if available
    if 'decision_indicator' in features_df.columns:
        indicator_counts = features_df['decision_indicator'].value_counts()
        
        if 'yes' in indicator_counts and indicator_counts['yes'] > 0:
            G.add_node('Explicit YES')
            level_nodes[1].append('Explicit YES')
            G.add_edge("START", 'Explicit YES', weight=indicator_counts['yes'])
            
            if category == 'full_df':
                G.add_edge('Explicit YES', yes_node, weight=indicator_counts['yes'])
            else:
                target = "YES (Malnutrition)" if category in ['TP_data', 'FP_data'] else "NO (No Malnutrition)"
                G.add_edge('Explicit YES', target, weight=indicator_counts['yes'])
        
        if 'no' in indicator_counts and indicator_counts['no'] > 0:
            G.add_node('Explicit NO')
            level_nodes[1].append('Explicit NO')
            G.add_edge("START", 'Explicit NO', weight=indicator_counts['no'])
            
            if category == 'full_df':
                G.add_edge('Explicit NO', no_node, weight=indicator_counts['no'])
            else:
                target = "YES (Malnutrition)" if category in ['TP_data', 'FP_data'] else "NO (No Malnutrition)"
                G.add_edge('Explicit NO', target, weight=indicator_counts['no'])
    
    # Add confidence levels if available
    if 'confidence' in features_df.columns:
        confidence_counts = features_df['confidence'].value_counts()
        
        for level, count in confidence_counts.items():
            if count > 0:
                node_name = f'Confidence: {level}'
                G.add_node(node_name)
                level_nodes[1].append(node_name)
                G.add_edge("START", node_name, weight=count)
                
                if category == 'full_df':
                    high_confidence_yes = sum((features_df['confidence'] == level) & (features_df['predicted_label'] == 1))
                    high_confidence_no = sum((features_df['confidence'] == level) & (features_df['predicted_label'] == 0))
                    
                    if high_confidence_yes > 0:
                        G.add_edge(node_name, yes_node, weight=high_confidence_yes)
                    if high_confidence_no > 0:
                        G.add_edge(node_name, no_node, weight=high_confidence_no)
                else:
                    target = "YES (Malnutrition)" if category in ['TP_data', 'FP_data'] else "NO (No Malnutrition)"
                    G.add_edge(node_name, target, weight=count)
    
    # Add top features to the graph
    for feature in top_features:
        G.add_node(feature)
        level_nodes[2].append(feature)
        G.add_edge("START", feature, weight=feature_counts[feature])
        
        if category == 'full_df':
            # Split the flow proportionally based on predicted labels for features
            feature_yes_count = sum((features_df[feature].notna()) & (features_df['predicted_label'] == 1))
            feature_no_count = sum((features_df[feature].notna()) & (features_df['predicted_label'] == 0))
            
            if feature_yes_count > 0:
                G.add_edge(feature, yes_node, weight=feature_yes_count)
            if feature_no_count > 0:
                G.add_edge(feature, no_node, weight=feature_no_count)
        else:
            target = "YES (Malnutrition)" if category in ['TP_data', 'FP_data'] else "NO (No Malnutrition)"
            G.add_edge(feature, target, weight=feature_counts[feature])
    
    # Add outcome nodes to the last level
    if category == 'full_df':
        level_nodes[3].extend([yes_node, no_node])
    else:
        target = "YES (Malnutrition)" if category in ['TP_data', 'FP_data'] else "NO (No Malnutrition)"
        level_nodes[3].append(target)
    
    # Create positions for a top-down tree layout
    pos = {}
    
    # Position nodes by level
    for level, nodes in level_nodes.items():
        n_nodes = len(nodes)
        if n_nodes > 0:
            # Space nodes horizontally
            for i, node in enumerate(nodes):
                # Set x position to distribute nodes evenly
                x_pos = (i - (n_nodes - 1) / 2) * (10.0 / max(n_nodes, 1))
                y_pos = -level * 5  # Set y position based on level
                pos[node] = np.array([x_pos, y_pos])
    
    # Visualize the graph
    plt.figure(figsize=(16, 12))
    
    # Node colors
    node_colors = []
    for node in G.nodes():
        if node == "START":
            node_colors.append("#3498db")  # Blue
        elif node in ["YES (Malnutrition)", "NO (No Malnutrition)"]:
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
        if node in ["START", "YES (Malnutrition)", "NO (No Malnutrition)"]:
            node_sizes.append(3000)
        elif "Confidence" in node or node in ["Explicit YES", "Explicit NO"]:
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
            font_size=11, 
            font_weight='bold',
            width=[1.0 + 3.0 * (w / max_weight) for w in weights], 
            edge_color='gray',
            arrows=True, 
            arrowsize=15,
            connectionstyle='arc3,rad=0.1')  # Curved edges for better visibility
    
    # Add edge labels
    edge_labels = {(u, v): f"{G[u][v]['weight']}" for u, v in G.edges()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=9)
    
    # Add title
    category_name = "Full Dataset" if category == "full_df" else category.replace('_data', '')
    plt.title(f"Decision Flow Diagram for {category_name}", fontsize=18)
    plt.axis('off')
    
    # Save with better resolution
    plt.savefig(output_filename, dpi=300, bbox_inches='tight', pad_inches=0.5)
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
        if col not in ['category', 'patient_id', 'true_label', 'predicted_label']:
            correct_mentions[col] = sum(correct_features[col].notna())
    
    incorrect_mentions = {}
    for col in incorrect_features.columns:
        if col not in ['category', 'patient_id', 'true_label', 'predicted_label']:
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
        plt.savefig('feature_differences.png')
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
    
    # Also create a flow diagram for the full dataset
    if 'full_df' in decision_patterns['all_features']:
        create_decision_flow_diagram(
            decision_patterns['all_features']['full_df'],
            'full_df',
            'decision_flow_full_dataset.png'
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

# Function to create word clouds from explanations
def create_explanation_wordclouds(data_dict):
    """
    Create word clouds from explanations for each category
    
    Args:
        data_dict (dict): Dictionary containing DataFrames with prediction results
    """
    from wordcloud import WordCloud
    import matplotlib.pyplot as plt
    from nltk.corpus import stopwords
    
    # Get stopwords
    stop_words = set(stopwords.words('english'))
    
    # Add domain specific stopwords
    domain_stopwords = {'patient', 'malnutrition', 'based', 'indicates', 'shows', 'evidence',
                       'since', 'therefore', 'however', 'thus', 'also', 'due', 'because', 'given'}
    stop_words.update(domain_stopwords)
    
    # Create a word cloud for each category
    categories = ['TP_data', 'TN_data', 'FP_data', 'FN_data']
    
    for category in categories:
        if category in data_dict and not data_dict[category].empty:
            # Combine all explanations
            text = ' '.join(data_dict[category]['explanation'].astype(str))
            
            # Create wordcloud
            wordcloud = WordCloud(width=800, height=400, 
                                background_color='white',
                                stopwords=stop_words,
                                max_words=100,
                                collocations=False).generate(text)
            
            # Display
            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.title(f'Word Cloud for {category}')
            plt.tight_layout()
            plt.savefig(f'wordcloud_{category}.png')
            plt.close()
    
    # Also create a combined wordcloud for correct vs incorrect
    correct_text = ' '.join(data_dict['correct_predictions']['explanation'].astype(str))
    incorrect_text = ' '.join(data_dict['incorrect_predictions']['explanation'].astype(str))
    
    # Create wordclouds
    correct_cloud = WordCloud(width=800, height=400, 
                           background_color='white',
                           stopwords=stop_words,
                           max_words=100,
                           collocations=False).generate(correct_text)
    
    incorrect_cloud = WordCloud(width=800, height=400, 
                             background_color='white',
                             stopwords=stop_words,
                             max_words=100,
                             collocations=False).generate(incorrect_text)
    
    # Display side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    ax1.imshow(correct_cloud, interpolation='bilinear')
    ax1.set_title('Correct Predictions')
    ax1.axis('off')
    
    ax2.imshow(incorrect_cloud, interpolation='bilinear')
    ax2.set_title('Incorrect Predictions')
    ax2.axis('off')
    
    plt.tight_layout()
    plt.savefig('wordcloud_comparison.png')
    plt.close()

# Function to analyze feature co-occurrence
def analyze_feature_cooccurrence(decision_patterns):
    """
    Analyze which features tend to co-occur in explanations.
    
    Args:
        decision_patterns (dict): Decision pattern analysis results
        
    Returns:
        DataFrame: Co-occurrence matrix
    """
    # Get the full dataframe with all features
    all_features_df = decision_patterns['all_features']['full_df']
    
    # Get feature columns (exclude metadata columns)
    feature_cols = [col for col in all_features_df.columns 
                   if col not in ['category', 'patient_id', 'true_label', 'predicted_label']]
    
    # Create co-occurrence matrix
    cooccurrence = pd.DataFrame(index=feature_cols, columns=feature_cols)
    
    # Fill matrix with co-occurrence counts
    for i, feature1 in enumerate(feature_cols):
        for feature2 in feature_cols:
            # Count how many times both features appear in the same explanation
            cooccur_count = sum((all_features_df[feature1].notna()) & (all_features_df[feature2].notna()))
            cooccurrence.loc[feature1, feature2] = cooccur_count
    
    # Ensure numeric values for the heatmap (convert to float and handle NaNs)
    cooccurrence = cooccurrence.apply(pd.to_numeric, errors='coerce').fillna(0)
    
    # Visualize top co-occurrences
    # Focus on features that appear a minimum number of times
    min_occurrences = 5
    frequent_features = [col for col in feature_cols 
                        if sum(all_features_df[col].notna()) >= min_occurrences]
    
    # Create a heatmap for frequent features
    if frequent_features:
        plt.figure(figsize=(12, 10))
        sns.heatmap(cooccurrence.loc[frequent_features, frequent_features], 
                   annot=True, cmap='YlGnBu', fmt='g')
        plt.title('Feature Co-occurrence Matrix')
        plt.tight_layout()
        plt.savefig('feature_cooccurrence.png')
        plt.close()
    
    return cooccurrence

# Main execution
if __name__ == "__main__":
    # Set the file path to your predictions CSV
    file_path = "./PHI/test/predictions.csv" 
    
    # Run the analysis
    analysis_results = analyze_llm_explanations(file_path)
    
    # Create additional visualizations
    create_explanation_wordclouds(analysis_results['data'])
    
    # Analyze feature co-occurrence
    cooccurrence = analyze_feature_cooccurrence(analysis_results['decision_patterns'])
    
    print("\nAnalysis complete! Check the generated images for visualizations.")
