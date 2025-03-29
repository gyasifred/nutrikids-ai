import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import re
from wordcloud import WordCloud, STOPWORDS
import spacy
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from collections import Counter
import networkx as nx
from transformers import pipeline
import shap
import warnings
warnings.filterwarnings('ignore')

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Function to load data
def load_data(file_path):
    """Load the malnutrition dataset with explanations"""
    df = pd.read_csv(file_path)
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"Malnutrition distribution: \n{df['malnutrition_status'].value_counts(normalize=True)}")
    return df

# ------------------------------------------------
# 1. Criteria-Based Analysis - Extract mentions of specific criteria from the prompt
# ------------------------------------------------

def extract_criteria_mentions(explanations, criteria_dict):
    """
    Extract mentions of specific criteria categories from explanations
    
    Args:
        explanations: Series of explanation texts
        criteria_dict: Dictionary mapping categories to keywords
        
    Returns:
        DataFrame with binary indicators for each criteria category
    """
    # Initialize DataFrame for criteria mentions
    criteria_df = pd.DataFrame(0, index=range(len(explanations)), 
                              columns=list(criteria_dict.keys()))
    
    # Extract mentions
    for i, exp in enumerate(explanations):
        if pd.isna(exp):
            continue
            
        exp_lower = exp.lower()
        for category, keywords in criteria_dict.items():
            # Check if any keyword is mentioned in the explanation
            if any(keyword.lower() in exp_lower for keyword in keywords):
                criteria_df.loc[i, category] = 1
    
    return criteria_df

def analyze_criteria_importance(criteria_df, malnutrition_status):
    """
    Analyze the importance of each criteria for malnutrition assessment
    
    Args:
        criteria_df: DataFrame with binary indicators for criteria categories
        malnutrition_status: Series with malnutrition decisions
        
    Returns:
        DataFrame with importance metrics for each criteria
    """
    # Convert malnutrition status to binary
    y = (malnutrition_status == 'yes').astype(int)
    
    # Initialize results DataFrame
    results = pd.DataFrame(columns=['criteria', 'frequency', 'malnutrition_correlation', 
                                   'present_in_malnutrition', 'present_in_non_malnutrition'])
    
    # Calculate metrics for each criteria
    for criteria in criteria_df.columns:
        # Frequency of mention
        frequency = criteria_df[criteria].mean()
        
        # Correlation with malnutrition status
        correlation = criteria_df[criteria].corr(y)
        
        # Presence in malnutrition vs non-malnutrition cases
        present_in_malnutrition = criteria_df[criteria][y == 1].mean()
        present_in_non_malnutrition = criteria_df[criteria][y == 0].mean()
        
        # Add to results
        results.loc[len(results)] = [criteria, frequency, correlation, 
                                    present_in_malnutrition, present_in_non_malnutrition]
    
    # Sort by correlation
    results = results.sort_values('malnutrition_correlation', ascending=False)
    
    return results

def plot_criteria_importance(criteria_importance):
    """Plot the importance of each criteria for malnutrition assessment"""
    plt.figure(figsize=(12, 8))
    
    # Create horizontal bar plot
    sns.barplot(x='malnutrition_correlation', y='criteria', data=criteria_importance, 
               palette='viridis')
    
    plt.title('Correlation of Assessment Criteria with Malnutrition Status', fontsize=16)
    plt.xlabel('Correlation Coefficient', fontsize=14)
    plt.ylabel('Assessment Criteria', fontsize=14)
    plt.tight_layout()
    plt.savefig('criteria_importance.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create frequency comparison plot
    plt.figure(figsize=(14, 8))
    
    # Melt the DataFrame for easier plotting
    plot_data = criteria_importance[['criteria', 'present_in_malnutrition', 'present_in_non_malnutrition']]
    plot_data = pd.melt(plot_data, id_vars=['criteria'], 
                        value_vars=['present_in_malnutrition', 'present_in_non_malnutrition'],
                        var_name='status', value_name='frequency')
    
    # Plot
    sns.barplot(x='criteria', y='frequency', hue='status', data=plot_data, palette='Set2')
    
    plt.title('Frequency of Criteria Mentions by Malnutrition Status', fontsize=16)
    plt.xlabel('Assessment Criteria', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Status')
    plt.tight_layout()
    plt.savefig('criteria_frequency_by_status.png', dpi=300, bbox_inches='tight')
    plt.show()

# ------------------------------------------------
# 2. Clinical Measurement Analysis - Extract clinical measurements and thresholds 
# ------------------------------------------------

def extract_clinical_measurements(explanations):
    """
    Extract clinical measurements and their values from explanations
    
    Args:
        explanations: Series of explanation texts
        
    Returns:
        DataFrame with extracted measurements and values
    """
    # Define patterns for common clinical measurements
    patterns = {
        'BMI': r'BMI[\s:]*([\d\.]+)',
        'weight_for_height': r'weight[- ]for[- ]height[:\s]*([-\d\.]+)[\s]*z[-\s]*score',
        'BMI_for_age': r'BMI[- ]for[- ]age[:\s]*([-\d\.]+)[\s]*z[-\s]*score',
        'MUAC': r'mid[- ]upper arm circumference[:\s]*([\d\.]+)',
        'albumin': r'(?:serum\s+)?albumin[:\s]*([\d\.]+)',
        'hemoglobin': r'(?:serum\s+)?h(?:a)?emoglobin[:\s]*([\d\.]+)',
        'weight_loss': r'(?:weight loss|lost)[:\s]*([\d\.]+)(?:\s*%)?'
    }
    
    # Initialize results
    measurements = []
    
    for i, exp in enumerate(explanations):
        if pd.isna(exp):
            continue
            
        row = {'explanation_id': i}
        
        # Extract measurements
        for measure, pattern in patterns.items():
            match = re.search(pattern, exp, re.IGNORECASE)
            if match:
                try:
                    value = float(match.group(1))
                    row[measure] = value
                except ValueError:
                    pass  # Skip if not a valid number
        
        if len(row) > 1:  # Add only if at least one measurement was found
            measurements.append(row)
    
    # Convert to DataFrame
    measurements_df = pd.DataFrame(measurements)
    
    return measurements_df

def analyze_measurement_thresholds(measurements_df, malnutrition_status):
    """
    Analyze the thresholds of clinical measurements for malnutrition classification
    
    Args:
        measurements_df: DataFrame with clinical measurements
        malnutrition_status: Series with malnutrition decisions
        
    Returns:
        Dictionary with threshold analysis for each measurement
    """
    # Join measurements with malnutrition status
    joined_df = measurements_df.copy()
    joined_df['malnutrition'] = malnutrition_status.iloc[joined_df['explanation_id']].reset_index(drop=True)
    
    # Initialize results
    thresholds = {}
    
    # Analyze each measurement
    for measure in measurements_df.columns:
        if measure == 'explanation_id':
            continue
            
        if joined_df[measure].count() < 10:  # Skip if too few data points
            continue
            
        # Calculate statistics
        stats = {
            'count': joined_df[measure].count(),
            'mean': joined_df[measure].mean(),
            'median': joined_df[measure].median(),
            'mean_malnutrition': joined_df[joined_df['malnutrition'] == 'yes'][measure].mean(),
            'mean_non_malnutrition': joined_df[joined_df['malnutrition'] == 'no'][measure].mean(),
            'threshold_candidates': []
        }
        
        # Find potential thresholds
        sorted_values = sorted(joined_df[measure].dropna().unique())
        for i in range(len(sorted_values) - 1):
            threshold = (sorted_values[i] + sorted_values[i+1]) / 2
            below_threshold = joined_df[measure] < threshold
            
            # Calculate precision and recall for this threshold
            true_positives = ((below_threshold) & (joined_df['malnutrition'] == 'yes')).sum()
            false_positives = ((below_threshold) & (joined_df['malnutrition'] == 'no')).sum()
            false_negatives = ((~below_threshold) & (joined_df['malnutrition'] == 'yes')).sum()
            
            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            stats['threshold_candidates'].append({
                'threshold': threshold,
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            })
        
        # Sort threshold candidates by F1 score
        stats['threshold_candidates'] = sorted(stats['threshold_candidates'], 
                                             key=lambda x: x['f1_score'], reverse=True)
        
        thresholds[measure] = stats
    
    return thresholds

def plot_measurement_distributions(measurements_df, malnutrition_status):
    """Plot distributions of clinical measurements by malnutrition status"""
    # Join measurements with malnutrition status
    joined_df = measurements_df.copy()
    joined_df['malnutrition'] = malnutrition_status.iloc[joined_df['explanation_id']].reset_index(drop=True)
    
    # Create plots for each measurement
    for measure in measurements_df.columns:
        if measure == 'explanation_id' or joined_df[measure].count() < 10:
            continue
            
        plt.figure(figsize=(12, 6))
        
        # Create KDE plot
        sns.kdeplot(data=joined_df, x=measure, hue='malnutrition', fill=True, common_norm=False)
        
        # Add vertical lines for group means
        plt.axvline(joined_df[joined_df['malnutrition'] == 'yes'][measure].mean(), 
                   color='blue', linestyle='--', label='Mean (Malnourished)')
        plt.axvline(joined_df[joined_df['malnutrition'] == 'no'][measure].mean(), 
                   color='orange', linestyle='--', label='Mean (Not Malnourished)')
        
        plt.title(f'Distribution of {measure} by Malnutrition Status', fontsize=16)
        plt.xlabel(measure, fontsize=14)
        plt.ylabel('Density', fontsize=14)
        plt.legend(title='Status')
        plt.tight_layout()
        plt.savefig(f'{measure}_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()

# ------------------------------------------------
# 3. Advanced NLP Analysis - Topic Modeling and Semantic Analysis
# ------------------------------------------------

def preprocess_text(text):
    """Preprocess text for NLP analysis"""
    if pd.isna(text):
        return ""
        
    # Convert to lowercase
    text = text.lower()
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words and len(token) > 2]
    
    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    return " ".join(tokens)

def analyze_topics(explanations, malnutrition_status, n_topics=5, n_top_words=10):
    """
    Perform topic modeling on explanations
    
    Args:
        explanations: Series of explanation texts
        malnutrition_status: Series with malnutrition decisions
        n_topics: Number of topics to extract
        n_top_words: Number of top words to display per topic
        
    Returns:
        Dictionary with topic modeling results
    """
    # Preprocess texts
    preprocessed_texts = explanations.apply(preprocess_text)
    
    # Vectorize
    vectorizer = CountVectorizer(max_features=500, min_df=2)
    X = vectorizer.fit_transform(preprocessed_texts)
    feature_names = vectorizer.get_feature_names_out()
    
    # Run LDA
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda.fit(X)
    
    # Get topic-term matrix
    topic_term_matrix = lda.components_
    
    # Get document-topic matrix
    doc_topic_matrix = lda.transform(X)
    
    # Extract top words for each topic
    topics = {}
    for topic_idx, topic in enumerate(topic_term_matrix):
        top_indices = topic.argsort()[:-n_top_words-1:-1]
        top_words = [feature_names[i] for i in top_indices]
        topics[topic_idx] = top_words
    
    # Get dominant topic for each document
    dominant_topics = doc_topic_matrix.argmax(axis=1)
    
    # Calculate topic prevalence by malnutrition status
    topic_prevalence = pd.DataFrame({
        'document_id': range(len(dominant_topics)),
        'dominant_topic': dominant_topics,
        'malnutrition': malnutrition_status.reset_index(drop=True)
    })
    
    prevalence_by_status = topic_prevalence.groupby(['dominant_topic', 'malnutrition']).size().unstack().fillna(0)
    prevalence_by_status = prevalence_by_status.div(prevalence_by_status.sum(axis=0), axis=1)
    
    return {
        'topics': topics,
        'doc_topic_matrix': doc_topic_matrix,
        'dominant_topics': dominant_topics,
        'prevalence_by_status': prevalence_by_status
    }

def plot_topic_analysis(topic_results):
    """Plot topic modeling results"""
    # Plot top words for each topic
    topics = topic_results['topics']
    n_topics = len(topics)
    
    fig, axes = plt.subplots(nrows=n_topics, figsize=(12, 4*n_topics))
    
    for i, (topic_idx, top_words) in enumerate(topics.items()):
        ax = axes[i] if n_topics > 1 else axes
        y_pos = range(len(top_words))
        
        # Create horizontal word count plots using placeholder values (since we don't have counts)
        # We'll use index position as a proxy for importance
        importance = [len(top_words) - j for j in range(len(top_words))]
        ax.barh(y_pos, importance, align='center')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_words)
        ax.invert_yaxis()
        ax.set_title(f'Topic {topic_idx}')
        ax.set_xlabel('Relative Importance')
    
    plt.tight_layout()
    plt.savefig('topic_top_words.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Plot topic prevalence by malnutrition status
    prevalence = topic_results['prevalence_by_status']
    
    plt.figure(figsize=(10, 6))
    prevalence.plot(kind='bar', stacked=False)
    plt.title('Topic Prevalence by Malnutrition Status', fontsize=16)
    plt.xlabel('Topic', fontsize=14)
    plt.ylabel('Proportion of Documents', fontsize=14)
    plt.xticks(rotation=0)
    plt.legend(title='Malnutrition Status')
    plt.tight_layout()
    plt.savefig('topic_prevalence_by_status.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_keyword_network(explanations, malnutrition_status, min_freq=5, max_keywords=30):
    """
    Create a network visualization of co-occurring keywords in explanations
    
    Args:
        explanations: Series of explanation texts
        malnutrition_status: Series with malnutrition decisions
        min_freq: Minimum frequency for a keyword to be included
        max_keywords: Maximum number of keywords to include
        
    Returns:
        NetworkX graph of keyword co-occurrences
    """
    # Preprocess texts
    preprocessed_texts = explanations.apply(preprocess_text)
    
    # Tokenize
    all_tokens = []
    for text in preprocessed_texts:
        all_tokens.extend(text.split())
    
    # Count token frequencies
    token_counts = Counter(all_tokens)
    
    # Get top keywords
    clinical_stopwords = {'patient', 'diagnosis', 'medical', 'clinical', 'hospital', 
                         'assessment', 'history', 'present', 'condition', 'status',
                         'note', 'report', 'examination', 'chart'}
    
    top_keywords = [w for w, c in token_counts.most_common(100) 
                   if c >= min_freq and w not in clinical_stopwords][:max_keywords]
    
    # Create co-occurrence matrix
    co_occurrence = np.zeros((len(top_keywords), len(top_keywords)))
    keyword_to_idx = {word: i for i, word in enumerate(top_keywords)}
    
    # Count co-occurrences
    for text in preprocessed_texts:
        tokens = set(text.split())
        for word1 in tokens:
            if word1 in keyword_to_idx:
                for word2 in tokens:
                    if word2 in keyword_to_idx and word1 != word2:
                        i, j = keyword_to_idx[word1], keyword_to_idx[word2]
                        co_occurrence[i, j] += 1
                        co_occurrence[j, i] += 1  # Symmetric
    
    # Create network
    G = nx.Graph()
    
    # Add nodes
    for i, word in enumerate(top_keywords):
        # Count occurrences in malnourished vs. non-malnourished
        malnourished_count = sum(1 for j, text in enumerate(preprocessed_texts) 
                                if word in text.split() and malnutrition_status.iloc[j] == 'yes')
        total_count = sum(1 for text in preprocessed_texts if word in text.split())
        
        G.add_node(word, count=token_counts[word], 
                  malnutrition_ratio=malnourished_count / total_count if total_count > 0 else 0)
    
    # Add edges
    for i in range(len(top_keywords)):
        for j in range(i+1, len(top_keywords)):
            if co_occurrence[i, j] >= min_freq:
                G.add_edge(top_keywords[i], top_keywords[j], weight=co_occurrence[i, j])
    
    return G

def plot_keyword_network(G):
    """Plot keyword co-occurrence network"""
    plt.figure(figsize=(14, 10))
    
    # Set node colors based on malnutrition ratio
    node_colors = [plt.cm.RdYlBu(1 - G.nodes[node]['malnutrition_ratio']) for node in G.nodes()]
    
    # Set node sizes based on frequency
    node_sizes = [30 * G.nodes[node]['count'] for node in G.nodes()]
    
    # Set edge weights based on co-occurrence
    edge_weights = [0.5 * G[u][v]['weight'] for u, v in G.edges()]
    
    # Position nodes using force-directed layout
    pos = nx.spring_layout(G, k=0.3, iterations=50)
    
    # Draw the network
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, alpha=0.8)
    nx.draw_networkx_edges(G, pos, width=edge_weights, alpha=0.3, edge_color='gray')
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlBu, norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = plt.colorbar(sm)
    cbar.set_label('Association with Malnutrition', rotation=270, labelpad=20)
    
    plt.axis('off')
    plt.title('Keyword Co-occurrence Network in Malnutrition Explanations', fontsize=16)
    plt.tight_layout()
    plt.savefig('keyword_network.png', dpi=300, bbox_inches='tight')
    plt.show()

# ------------------------------------------------
# 4. Classification and Feature Importance Analysis
# ------------------------------------------------

def classify_explanations(explanations, malnutrition_status):
    """
    Train a classifier to predict malnutrition status from explanations
    and analyze feature importance
    
    Args:
        explanations: Series of explanation texts
        malnutrition_status: Series with malnutrition decisions
        
    Returns:
        Dictionary with classification results and feature importance
    """
    # Preprocess texts
    preprocessed_texts = explanations.apply(preprocess_text)
    
    # Vectorize
    vectorizer = TfidfVectorizer(max_features=500, min_df=2)
    X = vectorizer.fit_transform(preprocessed_texts)
    feature_names = vectorizer.get_feature_names_out()
    
    # Convert target to binary
    y = (malnutrition_status == 'yes').astype(int)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Train classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    
    # Evaluate
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    # Get feature importance
    importance = clf.feature_importances_
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    return {
        'classifier': clf,
        'vectorizer': vectorizer,
        'accuracy': accuracy,
        'classification_report': report,
        'confusion_matrix': conf_matrix,
        'feature_importance': feature_importance
    }

def plot_classification_results(classification_results):
    """Plot classification results and feature importance"""
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(classification_results['confusion_matrix'], annot=True, fmt='d', cmap='Blues',
               xticklabels=['Not Malnourished', 'Malnourished'],
               yticklabels=['Not Malnourished', 'Malnourished'])
    plt.xlabel('Predicted', fontsize=14)
    plt.ylabel('Actual', fontsize=14)
    plt.title('Confusion Matrix', fontsize=16)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Plot feature importance
    importance_df = classification_results['feature_importance'].head(20)
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x='importance', y='feature', data=importance_df)
    plt.title('Top 20 Features for Predicting Malnutrition Status', fontsize=16)
    plt.xlabel('Importance', fontsize=14)
    plt.ylabel('Feature', fontsize=14)
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Plot classification report
    report = classification_results['classification_report']
    report_df = pd.DataFrame({
        'precision': [report['0']['precision'], report['1']['precision']],
        'recall': [report['0']['recall'], report['1']['recall']],
        'f1-score': [report['0']['f1-score'], report['1']['f1-score']]
    }, index=['Not Malnourished', 'Malnourished'])
    
    plt.figure(figsize=(10, 6))
    report_df.plot(kind='bar')
    plt.title('Classification Metrics', fontsize=16)
    plt.xlabel('Class', fontsize=14)
    plt.ylabel('Score', fontsize=14)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig('classification_metrics.png', dpi=300, bbox_inches='tight')
    plt.show()

# ------------------------------------------------
# 5. Structured Analysis of Prompt-Specific Factors 
# ------------------------------------------------

def extract_prompt_factors(explanations):
    """
    Extract mentions of specific factors from the original prompt
    
    Args:
        explanations: Series of explanation texts
        
    Returns:
        DataFrame with counts of each factor
    """
    # Define factors from the prompt
    prompt_factors = {
        'recent_illness': ['illness', 'infection', 'sick', 'disease', 'surgery', 'trauma', 'hospitalization'],
        'socioeconomic': ['socioeconomic', 'poverty', 'food insecurity', 'income', 'access', 'resources'],
        'symptoms': ['fatigue', 'weakness', 'appetite', 'weight loss', 'muscle wasting', 'edema'],
        'family_history': ['family', 'genetic', 'hereditary', 'familial', 'parent'],
        'lab_results': ['albumin', 'hemoglobin', 'lab', 'laboratory', 'vitamin', 'mineral', 'deficiency'],
        'medications': ['medication', 'drug', 'side effect', 'treatment', 'therapy', 'chemotherapy', 'diuretic'],
        'mental_health': ['depression', 'anxiety', 'mental', 'psychological', 'eating disorder', 'cognitive'],
        'malabsorption': ['diarrhea', 'malabsorption', 'absorption', 'digestive', 'gastrointestinal']
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

def analyze_factor_combinations(factors_df, malnutrition_status):
    """
    Analyze combinations of factors and their association with malnutrition
    
    Args:
        factors_df: DataFrame with binary indicators for factors
        malnutrition_status: Series with malnutrition decisions
        
    Returns:
        DataFrame with factor combinations and their association with malnutrition
    """
    # Convert malnutrition status to binary
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
    combinations_df = combinations_df.sort_values('lift', ascending=False)
    
    return combinations_df

def plot_factor_analysis(factors_df, malnutrition_status, factor_combinations=None):
    """Plot factor analysis results"""
    # Convert malnutrition status to binary
    y = (malnutrition_status == 'yes').astype(int)
    
    # Calculate factor prevalence and association with malnutrition
    factor_stats = pd.DataFrame(index=factors_df.columns)
    factor_stats['prevalence'] = factors_df.mean()
    factor_stats['malnutrition_correlation'] = [factors_df[col].corr(y) for col in factors_df.columns]
    factor_stats['present_in_malnutrition'] = [factors_df[col][y == 1].mean() for col in factors_df.columns]
    factor_stats['present_in_normal'] = [factors_df[col][y == 0].mean() for col in factors_df.columns]
    factor_stats = factor_stats.sort_values('malnutrition_correlation', ascending=False)
    
    # Plot factor correlation with malnutrition
    plt.figure(figsize=(12, 8))
    sns.barplot(x='malnutrition_correlation', y=factor_stats.index, data=factor_stats.reset_index(), 
               palette='viridis')
    plt.title('Correlation of Prompt Factors with Malnutrition Status', fontsize=16)
    plt.xlabel('Correlation Coefficient', fontsize=14)
    plt.ylabel('Factor', fontsize=14)
    plt.tight_layout()
    plt.savefig('factor_correlation.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Plot factor prevalence by malnutrition status
    plt.figure(figsize=(14, 8))
    comparison_data = factor_stats[['present_in_malnutrition', 'present_in_normal']]
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
    plt.savefig('factor_prevalence_by_status.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Plot top factor combinations if available
    if factor_combinations is not None and len(factor_combinations) > 0:
        plt.figure(figsize=(14, 10))
        top_combos = factor_combinations.head(15)
        sns.barplot(x='lift', y='combination', hue='num_factors', data=top_combos, palette='viridis')
        plt.title('Top Factor Combinations Associated with Malnutrition', fontsize=16)
        plt.xlabel('Lift (Relative Risk)', fontsize=14)
        plt.ylabel('Factor Combination', fontsize=14)
        plt.legend(title='Number of Factors')
        plt.tight_layout()
        plt.savefig('factor_combinations.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Plot distribution of combination sizes
        plt.figure(figsize=(10, 6))
        sns.countplot(x='num_factors', data=factor_combinations, palette='viridis')
        plt.title('Distribution of Factor Combination Sizes', fontsize=16)
        plt.xlabel('Number of Factors in Combination', fontsize=14)
        plt.ylabel('Count', fontsize=14)
        plt.tight_layout()
        plt.savefig('factor_combination_sizes.png', dpi=300, bbox_inches='tight')
        plt.show()

# ------------------------------------------------
# Main execution function
# ------------------------------------------------

def analyze_malnutrition_explanations(file_path):
    """Run the complete analysis pipeline on malnutrition explanations"""
    # Load data
    df = load_data(file_path)
    
    # Define criteria dictionary
    criteria_dict = {
        'BMI': ['BMI', 'body mass index'],
        'weight_for_height': ['weight for height', 'weight-for-height', 'WHZ', 'WFH'],
        'BMI_for_age': ['BMI for age', 'BMI-for-age', 'BAZ'],
        'MUAC': ['MUAC', 'mid-upper arm circumference', 'mid upper arm circumference'],
        'weight_loss': ['weight loss', 'lost weight', 'losing weight'],
        'inadequate_intake': ['inadequate intake', 'poor intake', 'reduced intake', 'insufficient intake'],
        'reduced_appetite': ['reduced appetite', 'poor appetite', 'loss of appetite', 'anorexia'],
        'cachexia': ['cachexia', 'muscle wasting', 'muscle loss', 'wasting'],
        'sarcopenia': ['sarcopenia', 'muscle weakness', 'reduced strength'],
        'edema': ['edema', 'oedema', 'fluid retention', 'swelling'],
        'lab_markers': ['albumin', 'prealbumin', 'transferrin', 'protein', 'hemoglobin', 'lymphocyte']
    }
    
    # 1. Criteria-Based Analysis
    print("\n1. Analyzing assessment criteria...")
    criteria_df = extract_criteria_mentions(df['explanation'], criteria_dict)
    criteria_importance = analyze_criteria_importance(criteria_df, df['malnutrition_status'])
    plot_criteria_importance(criteria_importance)
    
    # 2. Clinical Measurement Analysis
    print("\n2. Analyzing clinical measurements...")
    measurements_df = extract_clinical_measurements(df['explanation'])
    thresholds = analyze_measurement_thresholds(measurements_df, df['malnutrition_status'])
    plot_measurement_distributions(measurements_df, df['malnutrition_status'])
    
    # 3. Advanced NLP Analysis
    print("\n3. Performing NLP analysis...")
    # Topic modeling
    topic_results = analyze_topics(df['explanation'], df['malnutrition_status'])
    plot_topic_analysis(topic_results)
    
    # Keyword network
    print("Creating keyword network...")
    G = create_keyword_network(df['explanation'], df['malnutrition_status'])
    plot_keyword_network(G)
    
    # 4. Classification Analysis
    print("\n4. Training classification model...")
    classification_results = classify_explanations(df['explanation'], df['malnutrition_status'])
    plot_classification_results(classification_results)
    
    # 5. Structured Analysis of Prompt-Specific Factors
    print("\n5. Analyzing prompt-specific factors...")
    factors_df = extract_prompt_factors(df['explanation'])
    import itertools
    factor_combinations = analyze_factor_combinations(factors_df, df['malnutrition_status'])
    plot_factor_analysis(factors_df, df['malnutrition_status'], factor_combinations)
    
    print("\nAnalysis complete. All visualizations saved.")
    
    return {
        'criteria_importance': criteria_importance,
        'clinical_thresholds': thresholds,
        'topic_results': topic_results,
        'classification_results': classification_results,
        'factor_combinations': factor_combinations
    }

# Run the analysis if script is executed directly
if __name__ == "__main__":
    # Set the file path
    file_path = "malnutrition_explanations.csv"
    
    # Run the analysis
    results = analyze_malnutrition_explanations(file_path)
