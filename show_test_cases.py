import pandas as pd

df = pd.read_csv('resources/dataset_curated.csv')

print("=" * 80)
print("GOOD TEST CASES - Diseases with Distinct Symptom Patterns")
print("=" * 80)

# Select diseases with very distinct symptoms
test_cases = [
    ('Heart Disease', [0, 1, 2]),  # Chest pain, Shortness of breath, Fatigue
    ('Asthma', [0, 1, 2]),  # Wheezing, Shortness of breath, Coughing
    ('Diabetes', [0, 1, 2]),  # Frequent urination, Excessive thirst, Fatigue
    ('Migraine', [0, 1, 2]),  # Severe headache, Nausea, Sensitivity to light
    ('Kidney Stones', [0, 1, 2]),  # Severe flank pain, Blood in urine, Nausea
    ('Typhoid', [0, 1, 2]),  # High fever, Abdominal pain, Headache
    ('Stroke', [0, 1, 2]),  # Sudden numbness, Confusion, Severe headache
    ('Appendicitis', [0, 1, 2]),  # Abdominal pain, Nausea, Vomiting
]

for disease_name, symptom_indices in test_cases:
    disease_row = df[df['Disease'] == disease_name].iloc[0]
    all_symptoms = [s for s in disease_row[1:].dropna().values if pd.notna(s)]
    
    print(f"\n{'='*80}")
    print(f"Disease: {disease_name}")
    print(f"{'='*80}")
    print(f"Select these 3 symptoms:")
    for idx in symptom_indices:
        print(f"  ✓ {all_symptoms[idx]}")
    print(f"\nAll symptoms for reference:")
    print(f"  {', '.join(all_symptoms)}")

print("\n" + "="*80)
print("Try these combinations in the app - they should predict correctly!")
print("="*80)
