import os
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

def to_binary(series):
    """
    Map various positive labels ('yes', '1', True, 'malnourished') to 1,
    everything else to 0.
    """
    positive = {'1', 1, 'yes', 'y', 'true', True, 'malnourished'}
    return series.apply(lambda x: 1 if str(x).strip().lower() in positive else 0)


def extract_prompt_factors(explanations):
    """
    Extract mentions of specific clinical factors from explanation texts.
    Returns a DataFrame of binary indicators for each factor.
    """
    prompt_factors = {
        'anthropometric': ['bmi','weight','height','weight-for-height','muac',
                           'mid-upper arm circumference','weight loss','weight gain',
                           'underweight','overweight','obese','thin','emaciated',
                           'percentile','z-score','triceps skinfold','muscle mass','body composition'],
        'clinical': ['muscle wasting','fatigue','weakness','lethargy','skin changes',
                     'hair changes','edema','dermatitis','glossitis','stomatitis',
                     'poor wound healing','bruising','pallor','dry skin','brittle nails',
                     'hair loss','muscle atrophy','sarcopenia'],
        'dietary': ['caloric intake','protein intake','diet','supplement','feeding',
                    'appetite','meal','nutrition','nutrient','malnutrition','deficiency',
                    'vitamin','mineral','food insecurity','limited access','poor diet',
                    'inadequate intake','fasting','anorexia','tube feeding','tpn',
                    'parenteral nutrition','enteral nutrition'],
        'medical': ['chronic illness','gastrointestinal','infection','malabsorption',
                    'diarrhea','vomiting','nausea','constipation','dysphagia',
                    'gastroparesis','celiac','crohn','ulcerative colitis',
                    'pancreatic insufficiency','liver disease','cancer','diabetes',
                    'respiratory disease','renal disease','hiv','tuberculosis'],
        'labs': ['albumin','prealbumin','transferrin','total protein','lymphocyte count',
                 'cholesterol','hemoglobin','hematocrit','ferritin','folate','b12',
                 'vitamin d','zinc','magnesium','calcium','nitrogen balance',
                 'blood pressure','bp','wbc','white blood cell','glucose','sodium',
                 'potassium','creatinine','bun','alt','ast','bilirubin','heart rate',
                 'hr','temperature','temp','oxygen','o2','sats','blood sugar'],
        'risk_factors': ['medications','polypharmacy','depression','anxiety',
                         'cognitive impairment','dementia','socioeconomic','poverty',
                         'homelessness','social isolation','elderly','pediatric',
                         'pregnancy','alcohol','substance abuse','surgery',
                         'hospitalization','immobility','disability','history of',
                         'hx of','smoking','smoker','hypertension','hypertensive',
                         'diabetic','family history'],
        'imaging': ['x-ray','xray','ct scan','ctscan','mri','ultrasound','chest x-ray',
                    'cxr','ekg','ecg','angiogram','imaging','scan'],
        'medications': ['mg','tablet','aspirin','metformin','insulin','penicillin',
                        'ibuprofen','acetaminophen','lisinopril','metoprolol',
                        'atorvastatin','drug','medication','dose','pill','therapy']
    }

    factors_df = pd.DataFrame(0, index=range(len(explanations)),
                              columns=list(prompt_factors.keys()))
    for i, exp in enumerate(explanations):
        if pd.isna(exp):
            continue
        exp_lower = exp.lower()
        for factor, keywords in prompt_factors.items():
            if any(kw in exp_lower for kw in keywords):
                factors_df.at[i, factor] = 1
    return factors_df


def train_surrogate_model(factors_df, labels):
    X_train, X_test, y_train, y_test = train_test_split(
        factors_df, labels, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    print("Surrogate Model Performance:")
    print(classification_report(
        y_test, model.predict(X_test),
        labels=[0,1],
        target_names=['No Malnutrition','Malnutrition']
    ))
    cm = confusion_matrix(y_test, model.predict(X_test))
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No Malnutrition','Malnutrition'],
                yticklabels=['No Malnutrition','Malnutrition'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Surrogate Model Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'surrogate_confusion_matrix.png'), dpi=300)
    plt.close()
    return model, X_test


def shap_analysis(model, X_test):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    plt.figure(figsize=(12,6))
    # Align features via numpy and names to avoid shape mismatch
    shap.summary_plot(
        shap_values[1],
        X_test.values,
        feature_names=X_test.columns,
        show=False
    )
    plt.savefig(os.path.join(OUT_DIR, 'shap_summary.png'), dpi=300)
    plt.close()
    for feature in X_test.columns[:5]:
        plt.figure(figsize=(8,5))
        shap.dependence_plot(
            feature,
            shap_values[1],
            X_test.values,
            feature_names=X_test.columns,
            show=False
        )
        plt.savefig(os.path.join(OUT_DIR, f'shap_dependence_{feature}.png'), dpi=300)
        plt.close()


def lime_analysis(model, X_test, sample_indices=None):
    explainer = lime.lime_tabular.LimeTabularExplainer(
        X_test.values,
        feature_names=X_test.columns,
        class_names=['No','Yes'],
        discretize_continuous=True
    )
    if sample_indices is None:
        sample_indices = [0]
    for idx in sample_indices:
        if idx < len(X_test):
            exp = explainer.explain_instance(
                X_test.iloc[idx].values,
                model.predict_proba
            )
            exp.save_to_file(os.path.join(
                OUT_DIR, f'lime_explanation_sample_{idx}.html'))


def analyze_factor_combinations(factors_df, malnutrition_status):
    y = to_binary(malnutrition_status)
    max_factors = min(3, len(factors_df.columns))
    combinations_list = []
    baseline_rate = y.mean()
    for k in range(1, max_factors+1):
        for combo in itertools.combinations(factors_df.columns, k):
            mask = factors_df[list(combo)].all(axis=1)
            if mask.sum() < 5:
                continue
            malnutrition_rate = y[mask].mean()
            lift = malnutrition_rate / baseline_rate if baseline_rate>0 else 0
            combinations_list.append({
                'combination': ' + '.join(combo),
                'num_factors': k,
                'count': mask.sum(),
                'malnutrition_rate': malnutrition_rate,
                'baseline_rate': baseline_rate,
                'lift': lift
            })
    df = pd.DataFrame(combinations_list)
    if not df.empty:
        df = df.sort_values('lift', ascending=False)
    return df


def statistical_testing(correct_factors, incorrect_factors):
    results = []
    for factor in correct_factors.columns:
        c_vals = correct_factors[factor]
        i_vals = incorrect_factors[factor]
        chi2, p_chi2 = stats.chisquare([c_vals.sum(), i_vals.sum()])
        t_stat, p_ttest = stats.ttest_ind(c_vals, i_vals, equal_var=False)
        results.append({
            'factor': factor,
            'correct_mean': c_vals.mean(),
            'incorrect_mean': i_vals.mean(),
            'difference': c_vals.mean() - i_vals.mean(),
            'chi2_p_value': p_chi2,
            'ttest_p_value': p_ttest
        })
    df = pd.DataFrame(results).sort_values('ttest_p_value')
    df.to_csv(os.path.join(OUT_DIR, 'statistical_tests.csv'), index=False)
    print("Statistical tests saved to", OUT_DIR)
    return df


def plot_factor_analysis(factors_df, malnutrition_status, factor_combinations=None):
    y = to_binary(malnutrition_status)
    stats_df = pd.DataFrame(index=factors_df.columns)
    stats_df['prevalence'] = factors_df.mean()
    stats_df['correlation'] = [factors_df[col].corr(y) for col in factors_df.columns]
    stats_df['present_in_malnutrition'] = [factors_df[col][y==1].mean() for col in factors_df.columns]
    stats_df['present_in_normal'] = [factors_df[col][y==0].mean() for col in factors_df.columns]
    stats_df = stats_df.sort_values('correlation', ascending=False)
    # Correlation plot (no palette to avoid FutureWarning)
    plt.figure(figsize=(12,8))
    sns.barplot(x='correlation', y=stats_df.index, data=stats_df.reset_index())
    plt.title('Correlation of Prompt Factors with Malnutrition Status')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'factor_correlation.png'), dpi=300)
    plt.close()
    # Prevalence plot
    plt.figure(figsize=(14,8))
    comp_df = pd.melt(
        stats_df[['present_in_malnutrition','present_in_normal']].reset_index(),
        id_vars=['index'], value_vars=['present_in_malnutrition','present_in_normal'],
        var_name='status', value_name='prevalence'
    )
    sns.barplot(x='index', y='prevalence', hue='status', data=comp_df, palette='Set2')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'factor_prevalence_by_status.png'), dpi=300)
    plt.close()
    if factor_combinations is not None and not factor_combinations.empty:
        plt.figure(figsize=(14,10))
        top = factor_combinations.head(15)
        sns.barplot(x='lift', y='combination', hue='num_factors', data=top, palette='viridis')
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, 'factor_combinations.png'), dpi=300)
        plt.close()
        plt.figure(figsize=(10,6))
        sns.countplot(x='num_factors', data=factor_combinations)
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, 'factor_combination_sizes.png'), dpi=300)
        plt.close()
    return stats_df


def analyze_prediction_types(data_dict):
    preds = {
        'True Positive': data_dict['correct_yes'],
        'True Negative': data_dict['correct_no'],
        'False Positive': data_dict['incorrect_yes'],
        'False Negative': data_dict['incorrect_no']
    }
    prevalences = {}
    for label, df in preds.items():
        if not df.empty:
            prevalences[label] = extract_prompt_factors(df['explanation']).mean()
    all_prev = pd.DataFrame(prevalences)
    plt.figure(figsize=(14,10))
    melted = pd.melt(all_prev.reset_index(), id_vars=['index'], var_name='prediction_type', value_name='prevalence')
    sns.barplot(x='prevalence', y='index', hue='prediction_type', data=melted, palette='Set2')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'factor_by_prediction_type.png'), dpi=300)
    plt.close()
    return all_prev


def analyze_llm_predictions(file_path):
    # Load and initial stats
    data_dict = load_and_filter_data(file_path)
    full_df = data_dict['full_df']
    print(f"Dataset size: {len(full_df)} records")
    print(f"Correct predictions: {len(data_dict['correct_predictions'])} ({len(data_dict['correct_predictions'])/len(full_df):.2%})")
    print(f"Incorrect predictions: {len(data_dict['incorrect_predictions'])} ({len(data_dict['incorrect_predictions'])/len(full_df):.2%})")

    # prepare output dir
    os.makedirs(OUT_DIR, exist_ok=True)

    # Labels
    y_true = to_binary(full_df['true_label'])
    y_pred = to_binary(full_df['predicted_label'])

    # Classification report and confusion matrix
    print("\nClassification Metrics:")
    print(classification_report(
        y_true, y_pred,
        labels=[0,1],
        target_names=['No Malnutrition','Malnutrition']
    ))
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Malnutrition','Malnutrition'], yticklabels=['No Malnutrition','Malnutrition'])
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR,'confusion_matrix.png'), dpi=300)
    plt.close()

    # Factor extraction
    print("\nExtracting prompt factors from explanations...")
    factors_df = extract_prompt_factors(full_df['explanation'])
    full_df_with_factors = pd.concat([full_df, factors_df], axis=1)
    full_df_with_factors.to_csv(os.path.join(OUT_DIR,'full_data_with_factors.csv'), index=False)

    # Factor analysis plots
    print("\nAnalyzing factors across all predictions...")
    factor_stats = plot_factor_analysis(factors_df, full_df['true_label'])
    print("\nTop factors associated with malnutrition:")
    print(factor_stats.head(5))

    # Surrogate model
    print("\nTraining surrogate model for explainability...")
    surrogate_model, X_test = train_surrogate_model(factors_df, to_binary(full_df['true_label']))

    print("\nPerforming SHAP analysis...")
    shap_analysis(surrogate_model, X_test)

    print("\nPerforming LIME analysis...")
    lime_analysis(surrogate_model, X_test, sample_indices=[0,1,2])

    # Factor combinations
    print("\nAnalyzing factor combinations...")
    factor_combinations = analyze_factor_combinations(factors_df, full_df['true_label'])
    if not factor_combinations.empty:
        print("\nTop factor combinations associated with malnutrition:")
        print(factor_combinations.head(5))
        factor_combinations.to_csv(os.path.join(OUT_DIR,'factor_combinations.csv'), index=False)
        plot_factor_analysis(factors_df, full_df['true_label'], factor_combinations)
    else:
        print("Not enough data to analyze factor combinations.")

    # Correct vs incorrect
    print("\nAnalyzing factors in correct vs. incorrect predictions...")
    correct_factors = extract_prompt_factors(data_dict['correct_predictions']['explanation'])
    incorrect_factors = extract_prompt_factors(data_dict['incorrect_predictions']['explanation'])
    stat_results = statistical_testing(correct_factors, incorrect_factors)

    correct_prev = correct_factors.mean()
    incorrect_prev = incorrect_factors.mean()
    factor_diff = (correct_prev - incorrect_prev).sort_values(ascending=False)
    factor_diff.to_csv(os.path.join(OUT_DIR,'factor_difference.csv'))

    plt.figure(figsize=(12,8))
    sns.barplot(x=factor_diff.values, y=factor_diff.index, palette='RdBu_r')
    plt.axvline(0, color='black')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR,'factor_difference.png'), dpi=300)
    plt.close()

    print("\nFactors more common in correct predictions:")
    print(factor_diff[factor_diff>0].head(5))
    print("\nFactors more common in incorrect predictions:")
    print(factor_diff[factor_diff<0].head(5))

    # By prediction type
    prediction_type_prev = analyze_prediction_types(data_dict)
    prediction_type_prev.to_csv(os.path.join(OUT_DIR,'prediction_type_factor_prevalence.csv'))

    # Summary file
    summary_path = os.path.join(OUT_DIR,'analysis_summary.txt')
    with open(summary_path,'w') as f:
        f.write("Malnutrition Prediction Analysis Summary\n")
        f.write("=======================================\n\n")
        f.write(f"Dataset: {file_path}\n")
        f.write(f"Total records: {len(full_df)}\n\n")
        f.write("Top factors associated with malnutrition:\n")
        for factor in factor_stats.head(5).index:
            f.write(f"- {factor}\n")
        f.write("\nFactors more common in correct predictions:\n")
        for factor in factor_diff[factor_diff>0].head(3).index:
            f.write(f"- {factor}\n")
        f.write("\nFactors more common in incorrect predictions:\n")
        for factor in factor_diff[factor_diff<0].head(3).index:
            f.write(f"- {factor}\n")
    print(f"Analysis summary saved to {summary_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Analyze LLM malnutrition predictions')
    parser.add_argument('--file', type=str, default="./predictions.csv", help='Path to the CSV file')
    parser.add_argument('--out_dir', type=str, default="./figures", help='Directory for outputs')
    args = parser.parse_args()
    OUT_DIR = args.out_dir
    analyze_llm_predictions(args.file)
