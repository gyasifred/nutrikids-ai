#!/usr/bin/env python3
import  os
import matplotlib.pyplot as plt
from analysis_utils import (
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
    analyze_measurement_criteria_alignment
)

def main():
    # Load Data
    file_path = "./llama_zero_shot/prediction.csv"
    data = load_and_filter_data(file_path)

    # Create output directories
    os.makedirs('figures', exist_ok=True)
    os.makedirs('results', exist_ok=True)

    # Define Criteria Dictionary
    criteria_dict = {
        'BMI': [
            'BMI', 'body mass index', 'body-mass index', 'kg/m2', 'kg/m^2', 'kg per meter squared',
            'weight/height squared', 'quetelet index', 'kilo per square meter', 'mass index',
            'bmi calculation', 'bmi score', 'bmi value', 'bmi measurement'
        ],
        
        'weight_for_height': [
            'weight for height', 'weight-for-height', 'WHZ', 'WFH', 'W/H', 'W/H ratio',
            'weight-height ratio', 'weight to height ratio', 'weight relative to height',
            'weight proportional to height', 'weight compared to height', 'weight against height standard',
            'weight height proportion', 'weight/height z-score', 'wasted', 'wasting',
            'weight-stature ratio', 'weight for stature'
        ],
        
        'BMI_for_age': [
            'BMI for age', 'BMI-for-age', 'BAZ', 'age-adjusted BMI', 'age appropriate BMI',
            'BMI percentile', 'BMI centile', 'age-specific BMI', 'BMI z-score for age',
            'BMI standard deviation score', 'BMI relative to age', 'age-normalized BMI',
            'BMI compared to reference', 'weight-for-age related to height', 'growth chart BMI',
            'BMI growth curve', 'BMIFA', 'BMI by age'
        ],
        
        'MUAC': [
            'MUAC', 'mid-upper arm circumference', 'mid upper arm circumference', 'arm circumference',
            'MAC', 'upper arm circumference', 'arm perimeter', 'arm girth', 'arm measurement',
            'mid-arm circumference', 'arm anthropometry', 'MUAC tape', 'MUAC measurement',
            'MUAC screening', 'MUAC for age', 'MUAC z-score', 'MUAC percentile', 'arm size',
            'midupper arm measurement', 'circumferential arm measurement', 'arm diameter'
        ],
        
        'weight_loss': [
            'weight loss', 'lost weight', 'losing weight', 'decreased weight', 'declining weight',
            'weight decrease', 'weight reduction', 'weight deficit', 'unintentional weight loss',
            'involuntary weight loss', 'weight drop', 'weight trend down', 'negative weight change',
            'weight deterioration', 'falling weight', 'weight below baseline', 'weight decline',
            'shrinking weight', 'diminished weight', 'weight depletion', 'reduced body mass',
            'shedding weight', 'weight comparing to previous'
        ],
        
        'inadequate_intake': [
            'inadequate intake', 'poor intake', 'reduced intake', 'insufficient intake', 'low intake',
            'suboptimal intake', 'decreased consumption', 'insufficient consumption', 'minimal eating',
            'poor oral intake', 'decreased oral intake', 'suboptimal feeding', 'inadequate nutrition',
            'poor nutritional intake', 'limited food intake', 'restricted intake', 'inadequate food consumption',
            'reduced dietary intake', 'low dietary consumption', 'poor feeding', 'insufficient calories',
            'inadequate caloric intake', 'poor energy intake', 'low nutrient intake', 'decreased food intake',
            'not meeting nutritional needs'
        ],
        
        'reduced_appetite': [
            'reduced appetite', 'poor appetite', 'loss of appetite', 'anorexia', 'decreased hunger',
            'lack of hunger', 'suppressed appetite', 'diminished appetite', 'no appetite', 'minimal appetite',
            'low appetite', 'appetite decline', 'appetite decrease', 'appetite suppression', 'not hungry',
            'early satiety', 'feeling full quickly', 'satiated easily', 'poor interest in food',
            'aversion to food', 'disinterest in eating', 'reluctance to eat', 'no desire to eat',
            'reduced food interest', 'lack of interest in meals'
        ],
        
        'cachexia': [
            'cachexia', 'muscle wasting', 'muscle loss', 'wasting', 'wasting syndrome', 'tissue wasting',
            'severe wasting', 'protein-energy malnutrition', 'protein wasting', 'protein catabolism',
            'catabolic state', 'hypermetabolic wasting', 'pathological wasting', 'disease-related wasting',
            'severe muscle depletion', 'extreme weight loss', 'profound weight loss', 'body wasting',
            'accelerated muscle loss', 'cancer cachexia', 'cardiac cachexia', 'marasmus', 'marasmic',
            'kwashiorkor', 'severe tissue depletion'
        ],
        
        'sarcopenia': [
            'sarcopenia', 'muscle weakness', 'reduced strength', 'muscle atrophy', 'muscle deterioration',
            'decreased muscle mass', 'loss of muscle function', 'diminished muscle strength',
            'skeletal muscle reduction', 'muscle volume loss', 'reduced lean mass', 'decreased lean body mass',
            'muscle degradation', 'reduced muscle quality', 'age-related muscle loss',
            'progressive muscle decline', 'decreased physical performance', 'reduced physical function',
            'poor muscle tone', 'functional impairment', 'poor grip strength', 'frailty', 'muscle frailty'
        ],
        
        'edema': [
            'edema', 'oedema', 'fluid retention', 'swelling', 'peripheral edema', 'dependent edema',
            'pitting edema', 'non-pitting edema', 'bilateral edema', 'pedal edema', 'ankle edema',
            'leg swelling', 'sacral edema', 'ascites', 'anasarca', 'generalized edema',
            'nutritional edema', 'hypoalbuminemic edema', 'protein deficiency edema', 'excess fluid',
            'fluid accumulation', 'interstitial fluid', 'tissue swelling', 'puffy', 'waterlogging'
        ],
        
        'lab_markers': [
            'albumin', 'prealbumin', 'transferrin', 'protein', 'hemoglobin', 'lymphocyte',
            'serum albumin', 'hypoalbuminemia', 'transthyretin', 'retinol binding protein', 'RBP',
            'total protein', 'protein level', 'protein status', 'serum protein', 'low albumin',
            'low prealbumin', 'iron binding capacity', 'TIBC', 'ferritin', 'total lymphocyte count',
            'TLC', 'white blood cells', 'WBC', 'low hemoglobin', 'anemia', 'low hematocrit',
            'cholesterol', 'low cholesterol', 'hypocholesterolemia', 'CRP', 'C-reactive protein',
            'elevated CRP', 'nitrogen balance', 'creatinine height index', 'CHI'
        ],
        
        'growth_parameters': [
            'growth', 'growth curve', 'growth chart', 'growth velocity', 'growth rate',
            'growth percentile', 'growth trajectory', 'stunting', 'stunted', 'linear growth',
            'height velocity', 'length velocity', 'height for age', 'length for age',
            'height-for-age z-score', 'HAZ', 'LAZ', 'growth faltering', 'growth failure',
            'failure to thrive', 'FTT', 'poor growth', 'growth restriction', 'catch-up growth',
            'deceleration in growth', 'growth plateau', 'crossed percentiles', 'crossed centiles'
        ],
        
        'body_composition': [
            'body composition', 'body fat', 'fat mass', 'lean mass', 'fat-free mass', 'FFM',
            'muscle mass', 'skeletal muscle mass', 'SMM', 'body fat percentage', 'lean body mass',
            'LBM', 'fat distribution', 'subcutaneous fat', 'visceral fat', 'adipose tissue',
            'fat depletion', 'reduced fat stores', 'depleted muscle', 'muscle depletion',
            'bioimpedance', 'BIA', 'DEXA', 'DXA', 'anthropometric', 'skinfold', 'triceps skinfold'
        ],
        
        'physical_assessment': [
            'physical assessment', 'clinical assessment', 'physical examination', 'clinical signs',
            'visible ribs', 'protruding bones', 'temporal wasting', 'sunken eyes', 'sunken cheeks',
            'thin limbs', 'loss of subcutaneous fat', 'hollow temples', 'prominent clavicle',
            'prominent scapula', 'visible spine', 'visible pelvis', 'reduced fat pads',
            'skin tenting', 'poor skin turgor', 'dry skin', 'hair changes', 'brittle nails',
            'pressure ulcers', 'pressure sores', 'delayed wound healing', 'poor wound healing'
        ],
        
        'functional_indicators': [
            'functional status', 'functional capacity', 'physical function', 'physical performance',
            'handgrip strength', 'grip dynamometer', 'muscle function', 'walking speed', 'gait speed',
            'chair stand', 'sit-to-stand', 'physical activity', 'activities of daily living', 'ADL',
            'instrumental activities of daily living', 'IADL', 'functional decline',
            'functional impairment', 'reduced mobility', 'bedridden', 'chair-bound', 'fatigue',
            'weakness', 'exhaustion', 'exertional fatigue', 'reduced endurance', 'exercise intolerance'
        ]
}

    # ------------------------------------------------
    # 1. Criteria Mention Analysis
    # ------------------------------------------------

    # Extract Criteria Mentions
    criteria_results = {
        key: extract_criteria_mentions(data[key]['explanation'], criteria_dict)
        for key in ['correct_predictions', 'incorrect_predictions', 'correct_yes', 'correct_no', 'incorrect_yes', 'incorrect_no']
    }

    # Correlation Analysis for Correct & Incorrect Predictions
    correlation_results = {
        key: analyze_criteria_correlation(criteria_results[key], data[key]['true_label'])
        for key in ['correct_predictions', 'incorrect_predictions']
    }

    # Frequency Analysis for the Four Groups
    frequency_results = {
        key: analyze_criteria_frequency(criteria_results[key])
        for key in ['correct_yes', 'correct_no', 'incorrect_yes', 'incorrect_no']
    }

    # Visualize criteria frequency
    criteria_freq_fig = visualize_criteria_frequency(frequency_results)
    criteria_freq_fig.savefig('figures/criteria_frequency_comparison.png', dpi=300, bbox_inches='tight')
    plt.close(criteria_freq_fig)

    # ------------------------------------------------
    # 2. Clinical Measurement Analysis
    # ------------------------------------------------

    # Extract clinical measurements from explanations
    measurement_results = {
        key: extract_clinical_measurements(data[key]['explanation'])
        for key in ['full_df', 'correct_predictions', 'incorrect_predictions', 'correct_yes', 'correct_no', 'incorrect_yes', 'incorrect_no']
    }

    # Analyze measurement thresholds
    threshold_results = {
        key: analyze_measurement_thresholds(measurement_results[key], data[key]['true_label'])
        for key in ['correct_predictions', 'incorrect_predictions']
    }

    # Generate distribution plots
    plot_results = {}
    for key in ['correct_predictions', 'incorrect_predictions']:
        if len(measurement_results[key]) > 0:
            figures = plot_measurement_distributions(measurement_results[key], data[key]['true_label'])

            # Save figures
            for measure, fig in figures.items():
                fig.savefig(f'figures/{key}_{measure}_distribution.png', dpi=300, bbox_inches='tight')
                plt.close(fig)

            plot_results[key] = figures

    # Analyze severity classifications
    severity_results = {
        key: analyze_severity_classifications(measurement_results[key], data[key]['true_label'])
        for key in ['correct_predictions', 'incorrect_predictions']
    }

    # Analyze alignment between measurements and criteria mentions
    alignment_results = {
        key: analyze_measurement_criteria_alignment(
            measurement_results[key],
            criteria_results[key],
            data[key]['true_label']
        )
        for key in ['correct_predictions', 'incorrect_predictions']
    }

    # ------------------------------------------------
    # 3. Generate Summary Report
    # ------------------------------------------------

    # Generate criteria analysis summary
    criteria_summary = []
    criteria_summary.append("CRITERIA MENTION ANALYSIS SUMMARY")
    criteria_summary.append("==============================\n")

    criteria_summary.append("Top Criteria by Correlation with Malnutrition Status:")
    for key in ['correct_predictions', 'incorrect_predictions']:
        criteria_summary.append(f"\n{key.replace('_', ' ').title()}:")
        top_corr = correlation_results[key].head(5)
        for _, row in top_corr.iterrows():
            criteria_summary.append(f"  {row['criteria']}: {row['correlation']:.4f}")

    criteria_summary.append("\nTop Criteria by Frequency in Each Group:")
    for key in ['correct_yes', 'correct_no', 'incorrect_yes', 'incorrect_no']:
        criteria_summary.append(f"\n{key.replace('_', ' ').title()}:")
        top_freq = frequency_results[key].head(5)
        for _, row in top_freq.iterrows():
            criteria_summary.append(f"  {row['criteria']}: {row['frequency']:.4f}")

    # Generate measurement analysis summary
    measurement_summary = ["", "CLINICAL MEASUREMENT ANALYSIS SUMMARY", "=====================================\n"]
    measurement_summary.append(generate_measurement_summary(measurement_results, threshold_results, alignment_results))

    # Combine summaries
    full_summary = "\n".join(criteria_summary + measurement_summary)

    # Save summary to file
    with open('results/malnutrition_analysis_summary.txt', 'w') as f:
        f.write(full_summary)

    # Save DataFrames to CSV
    for key, df in correlation_results.items():
        df.to_csv(f'results/{key}_correlation.csv', index=False)

    for key, df in frequency_results.items():
        df.to_csv(f'results/{key}_frequency.csv', index=False)

    for key, df in measurement_results.items():
        if len(df) > 0:
            df.to_csv(f'results/{key}_measurements.csv', index=False)

    for key, df in alignment_results.items():
        if len(df) > 0:
            df.to_csv(f'results/{key}_alignment.csv', index=False)

    # Display Results
    print(full_summary)

    return {
        'data': data,
        'criteria_results': criteria_results,
        'correlation_results': correlation_results,
        'frequency_results': frequency_results,
        'measurement_results': measurement_results,
        'threshold_results': threshold_results,
        'alignment_results': alignment_results,
        'severity_results': severity_results
    }

if __name__ == "__main__":
    results = main()
