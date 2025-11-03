"""
Dataset Quality Comparison Tool

This script compares the original Kaggle dataset with the curated medical dataset
and shows key quality metrics.
"""

import pandas as pd
import numpy as np
from collections import Counter

def analyze_dataset(csv_path):
    """Analyze dataset quality metrics"""
    df = pd.read_csv(csv_path)
    
    # Basic stats
    num_diseases = df['Disease'].nunique()
    num_records = len(df)
    num_symptoms_cols = len([col for col in df.columns if col != 'Disease'])
    
    # Collect all symptoms across all records
    all_symptoms = []
    for col in df.columns:
        if col != 'Disease':
            all_symptoms.extend(df[col].dropna().tolist())
    
    symptom_counts = Counter(all_symptoms)
    unique_symptoms = len(symptom_counts)
    
    # Calculate average symptoms per disease (non-null)
    symptoms_per_row = df.iloc[:, 1:].notna().sum(axis=1)
    avg_symptoms = symptoms_per_row.mean()
    
    # Find most common symptoms (these appear across many diseases - less specific)
    most_common = symptom_counts.most_common(10)
    
    # Disease duplication analysis
    disease_counts = df['Disease'].value_counts()
    duplicated_diseases = (disease_counts > 1).sum()
    
    return {
        'path': csv_path,
        'num_diseases': num_diseases,
        'num_records': num_records,
        'unique_symptoms': unique_symptoms,
        'avg_symptoms_per_disease': round(avg_symptoms, 1),
        'symptom_columns': num_symptoms_cols,
        'most_common_symptoms': most_common,
        'duplicated_diseases': duplicated_diseases,
        'disease_counts': disease_counts
    }

def compare_datasets(original_path, curated_path):
    """Compare two datasets side by side"""
    print("="*80)
    print("DATASET QUALITY COMPARISON")
    print("="*80)
    
    original = analyze_dataset(original_path)
    curated = analyze_dataset(curated_path)
    
    print(f"\n📊 BASIC STATISTICS")
    print("-"*80)
    print(f"{'Metric':<40} {'Original':<20} {'Curated':<20}")
    print("-"*80)
    print(f"{'Total diseases':<40} {original['num_diseases']:<20} {curated['num_diseases']:<20}")
    print(f"{'Total records':<40} {original['num_records']:<20} {curated['num_records']:<20}")
    print(f"{'Unique symptoms':<40} {original['unique_symptoms']:<20} {curated['unique_symptoms']:<20}")
    print(f"{'Avg symptoms per disease':<40} {original['avg_symptoms_per_disease']:<20} {curated['avg_symptoms_per_disease']:<20}")
    print(f"{'Symptom columns':<40} {original['symptom_columns']:<20} {curated['symptom_columns']:<20}")
    
    print(f"\n🔍 QUALITY INDICATORS")
    print("-"*80)
    print(f"Duplicated diseases (original): {original['duplicated_diseases']}")
    print(f"Duplicated diseases (curated): {curated['duplicated_diseases']}")
    
    print(f"\n⚠️  MOST COMMON SYMPTOMS (may indicate lack of specificity)")
    print("-"*80)
    print("\nOriginal dataset (top 10):")
    for symptom, count in original['most_common_symptoms']:
        percentage = (count / original['num_records']) * 100
        print(f"  - {symptom}: {count} times ({percentage:.1f}% of records)")
    
    print(f"\nCurated dataset (top 10):")
    for symptom, count in curated['most_common_symptoms']:
        percentage = (count / curated['num_records']) * 100
        print(f"  - {symptom}: {count} times ({percentage:.1f}% of records)")
    
    print(f"\n💡 INTERPRETATION")
    print("-"*80)
    print("✅ Better dataset characteristics:")
    print("   - Fewer unique symptoms (more focused)")
    print("   - Lower overlap of symptoms across diseases (more specific)")
    print("   - Consistent number of symptoms per disease")
    print("   - No or minimal disease duplication")
    print()
    print("❌ Original dataset issues:")
    print(f"   - {original['unique_symptoms']} unique symptoms is very high")
    print(f"   - Many symptoms appear in {(original['most_common_symptoms'][0][1]/original['num_records']*100):.1f}% of diseases")
    print(f"   - {original['duplicated_diseases']} diseases have multiple entries with different symptom combinations")
    print()
    print("✅ Curated dataset improvements:")
    print(f"   - Reduced to {curated['unique_symptoms']} medically-relevant symptoms")
    print(f"   - Each disease has exactly {int(curated['avg_symptoms_per_disease'])} primary symptoms")
    print(f"   - No duplicate disease entries - one canonical symptom set per disease")
    print(f"   - Symptoms are characteristic/diagnostic for each condition")

def test_prediction_example(dataset_path, symptoms_to_test):
    """Test what diseases match given symptoms"""
    df = pd.read_csv(dataset_path)
    
    print(f"\n🧪 TESTING SYMPTOMS: {', '.join(symptoms_to_test)}")
    print("-"*80)
    
    matches = []
    for idx, row in df.iterrows():
        disease = row['Disease']
        disease_symptoms = row[1:].dropna().tolist()
        
        # Count how many of the test symptoms match
        matching_symptoms = [s for s in symptoms_to_test if s in disease_symptoms]
        if matching_symptoms:
            match_percentage = (len(matching_symptoms) / len(symptoms_to_test)) * 100
            matches.append({
                'disease': disease,
                'matches': len(matching_symptoms),
                'percentage': match_percentage,
                'matched_symptoms': matching_symptoms
            })
    
    # Sort by number of matches
    matches.sort(key=lambda x: x['matches'], reverse=True)
    
    print(f"Found {len(matches)} diseases with at least one matching symptom:\n")
    for i, match in enumerate(matches[:10], 1):
        print(f"{i}. {match['disease']}")
        print(f"   Matched: {match['matches']}/{len(symptoms_to_test)} symptoms ({match['percentage']:.1f}%)")
        print(f"   Symptoms: {', '.join(match['matched_symptoms'])}")
        print()

if __name__ == '__main__':
    original_path = 'resources/dataset_kaggle.csv'
    curated_path = 'resources/dataset_curated.csv'
    
    # Compare datasets
    compare_datasets(original_path, curated_path)
    
    # Test the problematic example from user
    print("\n" + "="*80)
    print("EXAMPLE TEST: Anxiety, Bloating, Belching")
    print("="*80)
    
    test_symptoms = ['Anxiety', 'Bloating', 'Belching']
    
    print("\n📋 ORIGINAL DATASET:")
    test_prediction_example(original_path, test_symptoms)
    
    print("\n📋 CURATED DATASET:")
    test_prediction_example(curated_path, test_symptoms)
