import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import spacy
from collections import Counter
import networkx as  nx
from utils import load_and_filter_data

# Install necessary nltk packages
nltk.download('punkt')
nltk.download('stopwords')

def extract_decision_features(explanation_text):
    """
    Extract key features and criteria mentioned in the explanation text.
    Returns a dictionary of features and their values/mentions.
    """
   # Define patterns for common criteria
    patterns = {
        # Anthropometric measurements
        'bmi': r'(?:bmi|body[\s-]*mass[\s-]*index)[\s-]*(for[\s-]*age)?[\s:]*((?:-?\d+\.?\d*)|(?:<?\s*\d+\.?\d*)|(?:\b(?:low|normal|high|severe|moderate|mild|underweight|normal weight|obese|overweight)\b))',
        
        'weight_height': r'(?:weight[\s-]*(?:for|to|\/|-)?[\s-]*(?:height|stature)|wfh|whz)[\s:]*((?:-?\d+\.?\d*)|(?:<?\s*\d+\.?\d*)|(?:\b(?:low|normal|high|severe|moderate|mild|deficit|adequate|excess)\b))',
        
        'muac': r'(?:muac|mid[\s-]*(?:upper|)[\s-]*arm[\s-]*circumference|mac)[\s:]*((?:-?\d+\.?\d*\s*(?:cm|mm)?)|(?:<?\s*\d+\.?\d*)|(?:\b(?:low|normal|high|severe|moderate|mild|deficit|reduced|adequate)\b))',
        
        'z_score': r'(?:(?:-?\d+\.?\d*)[\s-]*z[\s-]*(?:score|)|z[\s-]*score[\s:]*((?:-?\d+\.?\d*)|(?:<?\s*-?\d+\.?\d*)))',
        
        # Growth parameters
        'percentile': r'(?:(?:\d+(?:\.?\d*)?)(?:th|st|nd|rd)?[\s-]*percentile|percentile[\s:]*((?:\d+(?:\.?\d*)?)(?:th|st|nd|rd)?))',
        
        'growth': r'(?:growth[\s-]*(?:chart|curve|velocity|rate|failure|faltering)|failure[\s-]*to[\s-]*thrive|ftt|stunting|stunted|linear[\s-]*growth)[\s:]*((?:-?\d+\.?\d*)|(?:\b(?:delayed|normal|accelerated|poor|good|improved|worsening|decline|deceleration)\b))',
        
        'weight_loss': r'(?:weight[\s-]*(?:loss|decrease|reduction|decline)|lost[\s-]*weight)[\s:]*((?:\d+\.?\d*\s*(?:%|percent|kg|lb|pounds)?)|(?:\b(?:significant|severe|moderate|mild|minimal|substantial|rapid|gradual|progressive|unintentional)\b))',
        
        # Lab values
        'albumin': r'(?:albumin|serum[\s-]*albumin)[\s:]*((?:\d+\.?\d*\s*(?:g/dl|g/l)?)|(?:<?\s*\d+\.?\d*)|(?:\b(?:low|normal|high|decreased|elevated|hypoalbuminemia)\b))',
        
        'prealbumin': r'(?:prealbumin|transthyretin)[\s:]*((?:\d+\.?\d*\s*(?:mg/dl|mg/l)?)|(?:<?\s*\d+\.?\d*)|(?:\b(?:low|normal|high|decreased|elevated)\b))',
        
        'hemoglobin': r'(?:hemoglobin|hgb|hb|haemoglobin)[\s:]*((?:\d+\.?\d*\s*(?:g/dl|g/l)?)|(?:<?\s*\d+\.?\d*)|(?:\b(?:low|normal|high|decreased|elevated|anemia|anemic|anaemia|anaemic)\b))',
        
        'lymphocyte': r'(?:lymphocyte|lymphocyte[\s-]*count|total[\s-]*lymphocyte[\s-]*count|tlc|wbc)[\s:]*((?:\d+\.?\d*\s*(?:k/μl|×10\^9/l)?)|(?:<?\s*\d+\.?\d*)|(?:\b(?:low|normal|high|decreased|elevated|lymphopenia|lymphocytopenia)\b))',
        
        'transferrin': r'(?:transferrin|iron[\s-]*binding[\s-]*capacity|tibc)[\s:]*((?:\d+\.?\d*\s*(?:mg/dl|μg/dl)?)|(?:<?\s*\d+\.?\d*)|(?:\b(?:low|normal|high|decreased|elevated)\b))',
        
        'crp': r'(?:crp|c[\s-]*reactive[\s-]*protein)[\s:]*((?:\d+\.?\d*\s*(?:mg/l|mg/dl)?)|(?:<?\s*\d+\.?\d*)|(?:\b(?:low|normal|high|elevated|positive|negative|inflammation)\b))',
        
        # Vitamin and mineral markers
        'vitamin': r'(?:vitamin[\s-]*(?:a|b[1-9]|b12|c|d|e|k)|folate|folic[\s-]*acid|thiamine|riboflavin|niacin|cobalamin)[\s:]*((?:\d+\.?\d*\s*(?:ng/ml|μg/dl|nmol/l)?)|(?:<?\s*\d+\.?\d*)|(?:\b(?:low|normal|high|deficient|deficiency|adequate|excess|toxicity)\b))',
        
        'mineral': r'(?:(?:iron|zinc|calcium|magnesium|phosphorus|selenium|iodine|copper)[\s-]*(?:level|)|ferritin)[\s:]*((?:\d+\.?\d*\s*(?:ng/ml|μg/dl|mg/dl|mmol/l)?)|(?:<?\s*\d+\.?\d*)|(?:\b(?:low|normal|high|deficient|deficiency|adequate|excess|depleted)\b))',
        
        # Clinical factors
        'illness': r'\b(?:illness|infection|disease|sick|surgery|trauma|hospitalization|hospitalized|admitted|icu|intensive[\s-]*care|complication|recovery|post[\s-]*op|procedure|comorbid|comorbidity|chronic|acute|cancer|tumor|sepsis|wound|injury|pneumonia|respiratory|immunocompromised|inflammation|inflammatory|organ[\s-]*failure|critical)\b',
        
        'socioeconomic': r'\b(?:socioeconomic|poverty|food[\s-]*(?:insecurity|desert|access)|economic|financial|unemployed|homeless|housing|instability|education|literacy|transportation|welfare|assistance|snap|wic|food[\s-]*stamps|food[\s-]*bank|meal[\s-]*program|social[\s-]*support|vulnerable|marginalized|disadvantaged|barrier)\b',
        
        'symptoms': r'\b(?:fatigue|weakness|appetite|tired|weight[\s-]*loss|muscle[\s-]*wasting|lethargy|malaise|exhaustion|tiredness|cachexia|sarcopenia|frailty|thinness|emaciation|wasting|dehydration|swelling|lethargic|weak|tired|anorexia|lack[\s-]*of[\s-]*energy|hair[\s-]*loss|poor[\s-]*wound[\s-]*healing|skin[\s-]*changes|pallor|brittle[\s-]*nails|bruising|anasarca|kwashiorkor|marasmus|failure[\s-]*to[\s-]*thrive)\b',
        
        'medication': r'\b(?:medication|drug|treatment|therapy|side[\s-]*effect|chemotherapy|diuretic|antidepressant|antibiotic|corticosteroid|steroid|immunosuppressant|anticonvulsant|laxative|sedative|antipsychotic|nsaid|opioid|antacid|ppi|proton[\s-]*pump[\s-]*inhibitor|metformin|insulin|polypharmacy|appetite[\s-]*suppressant|stimulant|antiemetic|prescription|over[\s-]*the[\s-]*counter|supplement|herbal|regimen|dose|interaction)\b',
        
        # Functional status
        'functional': r'\b(?:functional|function|mobility|activity|exercise|physical[\s-]*activity|adl|activities[\s-]*of[\s-]*daily[\s-]*living|iadl|instrumental[\s-]*activities|independence|dependence|limitation|disability|impairment|weakness|strength|endurance|fatigue|energy|vitality|performance|capacity|rehabilitation|physical[\s-]*therapy|occupational[\s-]*therapy|walking|ambulation|bedridden|chair[\s-]*bound|transfer|gait|handgrip|grip[\s-]*strength)\b',
        
        # Nutritional intake
        'intake': r'\b(?:intake|consumption|diet|oral[\s-]*intake|eating|meal|feeding|caloric|calorie|protein|carbohydrate|fat|nutrient|macronutrient|micronutrient|diet[\s-]*quality|diet[\s-]*diversity|meal[\s-]*frequency|portion[\s-]*size|restrictive[\s-]*diet|elimination[\s-]*diet|food[\s-]*allergy|food[\s-]*intolerance|food[\s-]*selectivity|picky[\s-]*eating|food[\s-]*refusal|feeding[\s-]*difficulty|meal[\s-]*skipping|fasting|reduced[\s-]*intake|poor[\s-]*intake|decreased[\s-]*intake|inadequate[\s-]*intake|insufficient[\s-]*intake|not[\s-]*eating|poor[\s-]*appetite)\b',
        
        # Mental health
        'mental_health': r'\b(?:mental|psychological|psychiatric|mood|depression|anxiety|stress|trauma|ptsd|social[\s-]*isolation|loneliness|grief|bereavement|schizophrenia|bipolar|dementia|alzheimer|memory|confusion|disorientation|anorexia[\s-]*nervosa|bulimia|binge[\s-]*eating|avoidant[\s-]*food[\s-]*intake|pica|rumination|suicidal|self[\s-]*neglect|substance[\s-]*abuse|alcohol|addiction|emotional[\s-]*wellbeing|cognitive)\b',
        
        # Malabsorption
        'malabsorption': r'\b(?:malabsorption|absorption|digestive|gastrointestinal|celiac|crohn|ulcerative[\s-]*colitis|ibd|inflammatory[\s-]*bowel[\s-]*disease|short[\s-]*bowel|intestinal[\s-]*resection|pancreatic[\s-]*insufficiency|cystic[\s-]*fibrosis|bile[\s-]*acid[\s-]*deficiency|steatorrhea|constipation|bloating|gas|abdominal[\s-]*pain|nausea|vomiting|dysphagia|odynophagia|gerd|reflux|gastritis|enteritis|colitis|gastroparesis|bariatric[\s-]*surgery|ostomy|tube[\s-]*feeding|parenteral[\s-]*nutrition|diarrhea|malabsorptive|nutrient[\s-]*absorption)\b',
        
        # Hydration
        'hydration': r'\b(?:hydration|fluid|water|dehydration|hyperhydration|thirst|dry[\s-]*mouth|poor[\s-]*skin[\s-]*turgor|urine[\s-]*output|concentrated[\s-]*urine|fluid[\s-]*balance|oral[\s-]*intake|liquid[\s-]*consumption|drinking|beverage|iv[\s-]*fluid|intravenous|rehydration|fluid[\s-]*restriction|fluid[\s-]*overload|dehydrated|hypovolemic|hypervolemic)\b',
        
        # Edema
        'edema': r'\b(?:edema|oedema|fluid[\s-]*retention|swelling|peripheral[\s-]*edema|dependent[\s-]*edema|pitting[\s-]*edema|non[\s-]*pitting[\s-]*edema|bilateral[\s-]*edema|pedal[\s-]*edema|ankle[\s-]*edema|leg[\s-]*swelling|sacral[\s-]*edema|ascites|anasarca|generalized[\s-]*edema|nutritional[\s-]*edema|hypoalbuminemic[\s-]*edema|protein[\s-]*deficiency[\s-]*edema|excess[\s-]*fluid|fluid[\s-]*accumulation|interstitial[\s-]*fluid|tissue[\s-]*swelling|puffy|waterlogging)\b',
        
        # Physical assessment
        'physical_assessment': r'\b(?:physical[\s-]*assessment|clinical[\s-]*assessment|physical[\s-]*examination|clinical[\s-]*signs|visible[\s-]*ribs|protruding[\s-]*bones|temporal[\s-]*wasting|sunken[\s-]*eyes|sunken[\s-]*cheeks|thin[\s-]*limbs|loss[\s-]*of[\s-]*subcutaneous[\s-]*fat|hollow[\s-]*temples|prominent[\s-]*clavicle|prominent[\s-]*scapula|visible[\s-]*spine|visible[\s-]*pelvis|reduced[\s-]*fat[\s-]*pads|skin[\s-]*tenting|poor[\s-]*skin[\s-]*turgor|dry[\s-]*skin|hair[\s-]*changes|brittle[\s-]*nails|pressure[\s-]*ulcers|pressure[\s-]*sores|delayed[\s-]*wound[\s-]*healing|poor[\s-]*wound[\s-]*healing|muscle[\s-]*tone|muscle[\s-]*mass|anthropometric|bioimpedance|skinfold)\b',
        
        # Classification terms
        'malnutrition_class': r'\b(?:malnutrition|malnourished|undernourishment|undernutrition|protein[\s-]*energy[\s-]*malnutrition|protein[\s-]*calorie[\s-]*malnutrition|pcm|pem|marasmus|kwashiorkor|severe[\s-]*acute[\s-]*malnutrition|sam|moderate[\s-]*acute[\s-]*malnutrition|mam|mild[\s-]*malnutrition|moderate[\s-]*malnutrition|severe[\s-]*malnutrition|at[\s-]*risk[\s-]*for[\s-]*malnutrition|nutritional[\s-]*risk|nutritionally[\s-]*compromised|cachexia|wasting|starvation|failure[\s-]*to[\s-]*thrive|underweight|nutritionally[\s-]*deficient)\b'
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
    if re.search(r'\b(?:clear|clearly|definite|definitely|obvious|strong|evidence|confirms|confirmed)\b', explanation_text, re.IGNORECASE):
        features['confidence'] = 'high'
    elif re.search(r'\b(?:suggests|suggest|indicative|may|might|could|possible|possibly|probable|probably)\b', explanation_text, re.IGNORECASE):
        features['confidence'] = 'medium'
    elif re.search(r'\b(?:unclear|not clear|uncertain|unsure|insufficient|limited|data|more information)\b', explanation_text, re.IGNORECASE):
        features['confidence'] = 'low'

    return features

def build_decision_tree_from_explanations(df):
    """
    Process explanations to extract decision features and build a decision tree
    """
    # Extract features from explanations
    features_list = []
    for index, row in df.iterrows():
        features = extract_decision_features(row['explanation'])
        features['prediction'] = row['predicted_label']
        features['actual'] = row['true_label']
        features['correct'] = row['predicted_label'] == row['true_label']
        features_list.append(features)

    # Convert to DataFrame
    features_df = pd.DataFrame(features_list)

    # Fill NaN values
    features_df = features_df.fillna('not_mentioned')

    # Convert categorical features to one-hot encoding
    categorical_cols = [col for col in features_df.columns if col not in ['prediction', 'actual', 'correct']]
    features_encoded = pd.get_dummies(features_df[categorical_cols])

    # Train a decision tree to model the LLM's decision process
    X = features_encoded
    y = features_df['prediction']  # Predict the LLM's prediction

    # Train with a small depth to make visualization meaningful
    model = DecisionTreeClassifier(max_depth=4, random_state=42)
    model.fit(X, y)

    return model, features_encoded.columns, features_df

def visualize_explanation_elements(df):
    """
    Visualize the frequency of different elements appearing in explanations
    """
    # Separate correct and incorrect predictions
    correct_explanations = df[df['true_label'] == df['predicted_label']]['explanation']
    incorrect_explanations = df[df['true_label'] != df['predicted_label']]['explanation']

    # Create a combined text for word frequency analysis
    all_text = ' '.join(df['explanation'])
    correct_text = ' '.join(correct_explanations)
    incorrect_text = ' '.join(incorrect_explanations)

    # Process the text
    vectorizer = CountVectorizer(stop_words='english', max_features=30)

    # Get word frequencies
    all_word_freq = vectorizer.fit_transform([all_text]).toarray()[0]
    all_words = vectorizer.get_feature_names_out()

    # Fit on all text to use same vocabulary
    vectorizer.fit([all_text])

    correct_word_freq = vectorizer.transform([correct_text]).toarray()[0]
    incorrect_word_freq = vectorizer.transform([incorrect_text]).toarray()[0]

    # Create DataFrame for easier plotting
    word_freq_df = pd.DataFrame({
        'word': all_words,
        'all_freq': all_word_freq,
        'correct_freq': correct_word_freq,
        'incorrect_freq': incorrect_word_freq
    })

    # Sort by total frequency
    word_freq_df = word_freq_df.sort_values('all_freq', ascending=False)

    # Plot the frequencies
    plt.figure(figsize=(15, 8))

    plt.subplot(1, 2, 1)
    sns.barplot(x='all_freq', y='word', data=word_freq_df.head(15))
    plt.title('Most Common Words in All Explanations')
    plt.xlabel('Frequency')

    plt.subplot(1, 2, 2)
    comparison_df = word_freq_df.head(15).melt(id_vars='word',
                                              value_vars=['correct_freq', 'incorrect_freq'],
                                              var_name='prediction_type',
                                              value_name='frequency')
    sns.barplot(x='frequency', y='word', hue='prediction_type', data=comparison_df)
    plt.title('Word Frequency: Correct vs Incorrect Predictions')
    plt.xlabel('Frequency')
    plt.savefig('explanation_word_frequencies.png')
    plt.close()

    return word_freq_df

def create_decision_flow_diagram(features_df):
    """
    Create a decision flow diagram based on frequently co-occurring features
    extracted from malnutrition assessment explanations
    
    Parameters:
    -----------
    features_df : pandas.DataFrame
        DataFrame containing features extracted from malnutrition assessments
        
    Returns:
    --------
    G : networkx.DiGraph
        The directed graph representing the decision flow
    pos : dict
        Node positions for visualization
    """
    import networkx as nx
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    import numpy as np
    
    # Extract features only for correct predictions
    correct_features = features_df[features_df['correct'] == True]

    # Create a graph
    G = nx.DiGraph()

    # Add start node
    G.add_node("START")

    # Identify key features from the extract_decision_features function
    key_features = [
        # Anthropometric measurements
        'bmi', 'weight_height', 'muac', 'z_score', 
        # Growth parameters
        'percentile', 'growth', 'weight_loss',
        # Lab values
        'albumin', 'prealbumin', 'hemoglobin', 'lymphocyte', 'transferrin', 'crp',
        # Vitamin and mineral markers
        'vitamin', 'mineral',
        # Clinical factors
        'illness', 'socioeconomic', 'symptoms', 'medication',
        # Functional status
        'functional',
        # Nutritional intake
        'intake',
        # Mental health
        'mental_health',
        # Malabsorption
        'malabsorption',
        # Hydration
        'hydration',
        # Edema
        'edema',
        # Physical assessment
        'physical_assessment',
        # Classification terms
        'malnutrition_class',
        # Confidence level
        'confidence',
        # Severity if available
        'severity'
    ]

    # Count feature occurrences per outcome
    yes_predictions = correct_features[correct_features['prediction'] == 'yes']
    no_predictions = correct_features[correct_features['prediction'] == 'no']

    yes_feature_counts = {}
    no_feature_counts = {}

    for feature in key_features:
        if feature in correct_features.columns:
            yes_feature_counts[feature] = sum(yes_predictions[feature].notna() & 
                                            (yes_predictions[feature] != 'not_mentioned'))
            no_feature_counts[feature] = sum(no_predictions[feature].notna() & 
                                           (no_predictions[feature] != 'not_mentioned'))

    # Sort features by frequency
    yes_features_sorted = sorted(yes_feature_counts.items(), key=lambda x: x[1], reverse=True)
    no_features_sorted = sorted(no_feature_counts.items(), key=lambda x: x[1], reverse=True)

    # Filter to only include features that occur at least once
    yes_features_sorted = [(f, c) for f, c in yes_features_sorted if c > 0]
    no_features_sorted = [(f, c) for f, c in no_features_sorted if c > 0]

    # Determine how many features to include in each path (up to 5 or available features)
    yes_path_length = min(5, len(yes_features_sorted))
    no_path_length = min(5, len(no_features_sorted))

    # Add connections for "yes" prediction path
    if yes_features_sorted:
        G.add_edge("START", yes_features_sorted[0][0], weight=yes_features_sorted[0][1])
        for i in range(yes_path_length - 1):
            G.add_edge(yes_features_sorted[i][0], yes_features_sorted[i+1][0],
                      weight=min(yes_features_sorted[i][1], yes_features_sorted[i+1][1]))
        if yes_path_length > 0:
            G.add_edge(yes_features_sorted[yes_path_length-1][0], "YES",
                      weight=yes_features_sorted[yes_path_length-1][1])

    # Add connections for "no" prediction path
    if no_features_sorted:
        G.add_edge("START", no_features_sorted[0][0], weight=no_features_sorted[0][1])
        for i in range(no_path_length - 1):
            G.add_edge(no_features_sorted[i][0], no_features_sorted[i+1][0],
                      weight=min(no_features_sorted[i][1], no_features_sorted[i+1][1]))
        if no_path_length > 0:
            G.add_edge(no_features_sorted[no_path_length-1][0], "NO",
                      weight=no_features_sorted[no_path_length-1][1])

    # Add cross-connections between paths where appropriate
    # Find common features between yes and no paths
    yes_features_set = set([f for f, _ in yes_features_sorted])
    no_features_set = set([f for f, _ in no_features_sorted])
    common_features = yes_features_set.intersection(no_features_set)
    
    # Connect common features with appropriate weights
    for feature in common_features:
        yes_weight = yes_feature_counts[feature]
        no_weight = no_feature_counts[feature]
        
        # Connect to the outcome with higher weight
        if yes_weight > no_weight * 1.5:  # Significantly more common in yes predictions
            if feature in [f for f, _ in yes_features_sorted[:yes_path_length]]:
                G.add_edge(feature, "YES", weight=yes_weight)
        elif no_weight > yes_weight * 1.5:  # Significantly more common in no predictions
            if feature in [f for f, _ in no_features_sorted[:no_path_length]]:
                G.add_edge(feature, "NO", weight=no_weight)

    # Create the figure with a white background
    plt.figure(figsize=(18, 14), facecolor='white')
    
    # Use a hierarchical layout for better visualization of the flow
    try:
        pos = nx.nx_pydot.graphviz_layout(G, prog='dot')
    except:
        try:
            # Try another layout option if graphviz 'dot' fails
            pos = nx.nx_pydot.graphviz_layout(G, prog='fdp') 
        except:
            # Fallback to spring layout if graphviz is not available
            pos = nx.spring_layout(G, seed=42, k=0.5)  # Adjust k for better node separation

    # Create a custom colormap for nodes based on their role
    node_types = {}
    for node in G.nodes():
        if node == "START":
            node_types[node] = "start"
        elif node == "YES":
            node_types[node] = "yes"
        elif node == "NO":
            node_types[node] = "no"
        else:
            # Check if it's in both paths, yes path, or no path
            if node in yes_features_set and node in no_features_set:
                node_types[node] = "shared"
            elif node in yes_features_set:
                node_types[node] = "yes_feature"
            elif node in no_features_set:
                node_types[node] = "no_feature"
            else:
                node_types[node] = "other"

    node_colors = []
    for node in G.nodes():
        if node_types[node] == "start":
            node_colors.append("#3498db")  # Blue
        elif node_types[node] == "yes":
            node_colors.append("#2ecc71")  # Green
        elif node_types[node] == "no":
            node_colors.append("#e74c3c")  # Red
        elif node_types[node] == "shared":
            node_colors.append("#f39c12")  # Orange
        elif node_types[node] == "yes_feature":
            node_colors.append("#a3e4d7")  # Light green
        elif node_types[node] == "no_feature":
            node_colors.append("#f5b7b1")  # Light red
        else:
            node_colors.append("#bdc3c7")  # Light gray

    # Node sizes based on importance and occurrence counts
    node_sizes = []
    for node in G.nodes():
        if node in ["YES", "NO", "START"]:
            node_sizes.append(4000)
        else:
            # Size based on frequency in either yes or no paths
            count = yes_feature_counts.get(node, 0) + no_feature_counts.get(node, 0)
            # Scale from 1500 to 3000 based on count
            max_count = max(
                max(count for f, count in yes_feature_counts.items() if count > 0), 
                max(count for f, count in no_feature_counts.items() if count > 0)
            ) if yes_feature_counts and no_feature_counts else 1
            node_sizes.append(1500 + (count / max_count) * 1500)

    # Get edge weights for line thickness
    edges = G.edges()
    weights = [G[u][v]['weight'] for u, v in edges]
    max_weight = max(weights) if weights else 1
    
    # Edge colors based on source and target
    edge_colors = []
    for u, v in edges:
        if v == "YES":
            edge_colors.append("#2ecc71")  # Green
        elif v == "NO":
            edge_colors.append("#e74c3c")  # Red
        elif node_types.get(u) == "start":
            edge_colors.append("#3498db")  # Blue
        elif node_types.get(u) == "shared":
            edge_colors.append("#f39c12")  # Orange
        elif node_types.get(u) == "yes_feature":
            edge_colors.append("#a3e4d7")  # Light green
        elif node_types.get(u) == "no_feature":
            edge_colors.append("#f5b7b1")  # Light red
        else:
            edge_colors.append("#95a5a6")  # Gray

    # Scale line width based on weights
    normalized_weights = [1.5 + 3.5 * (w / max_weight) for w in weights]
    
    # Draw the network
    nx.draw(G, pos, 
            with_labels=True, 
            node_color=node_colors,
            node_size=node_sizes, 
            font_size=11, 
            font_weight='bold',
            font_color='black',
            width=normalized_weights, 
            edge_color=edge_colors,
            arrows=True, 
            arrowsize=20,
            connectionstyle='arc3,rad=0.1')  # Curved edges for better visibility
            
    # Add edge labels showing counts
    edge_labels = {(u, v): f"{G[u][v]['weight']}" for u, v in G.edges()}
    
    # Position edge labels closer to their edges
    label_pos = nx.draw_networkx_edge_labels(
        G, pos, 
        edge_labels=edge_labels, 
        font_size=9,
        font_weight='bold',
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
        label_pos=0.5  # Position labels at the middle of edges
    )

    # Add a title
    plt.title("Malnutrition Assessment Decision Flow Diagram", fontsize=20, pad=20)
    
    # Add a legend for node types
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor="#3498db", markersize=15, label='Start'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor="#2ecc71", markersize=15, label='Yes (Malnutrition)'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor="#e74c3c", markersize=15, label='No (No Malnutrition)'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor="#f39c12", markersize=15, label='Shared Feature'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor="#a3e4d7", markersize=15, label='Yes Path Feature'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor="#f5b7b1", markersize=15, label='No Path Feature')
    ]
    plt.legend(handles=legend_elements, loc='lower right', fontsize=12)
    
    # Add a description
    plt.figtext(0.5, 0.01, 
                "Node size represents feature frequency. Edge numbers show occurrence count. " +
                "Edges show decision flow patterns.", 
                ha='center', fontsize=12)

    # Remove axis
    plt.axis('off')
    
    # Handle the tight_layout warning by using bbox_inches when saving
    plt.savefig('malnutrition_decision_flow_diagram.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    return G, pos

def analyze_explanations(data_dict):
    """
    Analyze explanations from LLM outputs to understand decision patterns
    """
    # Extract full dataset
    df = data_dict['full_df']

    # First, let's explore word frequencies in explanations
    word_freq_df = visualize_explanation_elements(df)

    # Extract decision features from explanations
    model, feature_names, features_df = build_decision_tree_from_explanations(df)

    # Visualize the decision tree that explains the LLM's process
    plt.figure(figsize=(20, 10))
    plot_tree(model, feature_names=feature_names, class_names=['no', 'yes'],
              filled=True, rounded=True, fontsize=10)
    plt.title("Decision Tree of LLM's Classification Process", fontsize=16)
    plt.savefig('llm_decision_tree.png')
    plt.close()

    # Create a more intuitive decision flow diagram
    decision_graph = create_decision_flow_diagram(features_df)

    # Analyze feature importance for correct vs incorrect predictions
    correct_df = data_dict['correct_predictions']
    incorrect_df = data_dict['incorrect_predictions']

    # Extract features from both sets
    correct_features = []
    for _, row in correct_df.iterrows():
        features = extract_decision_features(row['explanation'])
        features['prediction'] = row['predicted_label']
        correct_features.append(features)

    incorrect_features = []
    for _, row in incorrect_df.iterrows():
        features = extract_decision_features(row['explanation'])
        features['prediction'] = row['predicted_label']
        incorrect_features.append(features)

    # Convert to DataFrames
    correct_features_df = pd.DataFrame(correct_features).fillna('not_mentioned')
    incorrect_features_df = pd.DataFrame(incorrect_features).fillna('not_mentioned')

    # Count feature mentions
    correct_mentions = {col: sum(correct_features_df[col] != 'not_mentioned')
                       for col in correct_features_df.columns if col != 'prediction'}

    incorrect_mentions = {col: sum(incorrect_features_df[col] != 'not_mentioned')
                         for col in incorrect_features_df.columns if col != 'prediction'}

    # Create feature importance comparison
    feature_comparison = pd.DataFrame({
        'feature': list(set(list(correct_mentions.keys()) + list(incorrect_mentions.keys()))),
        'correct_mentions': [correct_mentions.get(f, 0) for f in set(list(correct_mentions.keys()) + list(incorrect_mentions.keys()))],
        'incorrect_mentions': [incorrect_mentions.get(f, 0) for f in set(list(correct_mentions.keys()) + list(incorrect_mentions.keys()))]
    })

    # Calculate percentage of mentions
    feature_comparison['correct_pct'] = feature_comparison['correct_mentions'] / len(correct_df) * 100
    feature_comparison['incorrect_pct'] = feature_comparison['incorrect_mentions'] / len(incorrect_df) * 100 if len(incorrect_df) > 0 else 0

    # Calculate difference
    feature_comparison['difference'] = feature_comparison['correct_pct'] - feature_comparison['incorrect_pct']

    # Sort by absolute difference
    feature_comparison = feature_comparison.sort_values('difference', key=abs, ascending=False)

    # Visualize feature importance difference
    plt.figure(figsize=(12, 8))
    sns.barplot(x='difference', y='feature', data=feature_comparison.head(10))
    plt.title('Feature Importance Difference: Correct vs Incorrect Predictions', fontsize=16)
    plt.axvline(x=0, color='black', linestyle='--')
    plt.xlabel('Difference in Mention Percentage (Correct - Incorrect)')
    plt.savefig('feature_importance_difference.png')
    plt.close()

    return {
        'word_frequencies': word_freq_df,
        'decision_tree_model': model,
        'feature_names': feature_names,
        'features_df': features_df,
        'feature_comparison': feature_comparison,
        'decision_graph': decision_graph
    }

def main(file_path):
    """
    Main function to load data and analyze decision process
    """
    # Load the data
    data_dict = load_and_filter_data(file_path)

    # Analyze explanations
    analysis_results = analyze_explanations(data_dict)

    # Print summary of findings
    print("\n==== LLM Decision Process Analysis ====")
    print(f"Total samples analyzed: {len(data_dict['full_df'])}")
    print(f"Correct predictions: {len(data_dict['correct_predictions'])} ({len(data_dict['correct_predictions'])/len(data_dict['full_df'])*100:.2f}%)")
    print(f"Incorrect predictions: {len(data_dict['incorrect_predictions'])} ({len(data_dict['incorrect_predictions'])/len(data_dict['full_df'])*100:.2f}%)")

    print("\n==== Top Decision Features ====")
    top_features = analysis_results['feature_comparison'].head(5)
    print(top_features[['feature', 'correct_pct', 'incorrect_pct', 'difference']])

    print("\n==== Decision Tree Created ====")
    print("Decision tree visualization saved as 'llm_decision_tree.png'")
    print("Decision flow diagram saved as 'decision_flow_diagram.png'")
    print("Word frequency analysis saved as 'explanation_word_frequencies.png'")
    print("Feature importance comparison saved as 'feature_importance_difference.png'")

    return analysis_results

if __name__ == "__main__":
    # Replace with your actual file path
    file_path = "./llama_zero_shot/predictions.csv"
    results = main(file_path)
