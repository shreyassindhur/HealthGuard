import pandas as pd

df = pd.read_csv('resources/dataset_kaggle.csv')
symptoms = ['Chest pain', 'Shortness of breath', 'Fatigue']

print(f"\n🔍 Checking diseases that match: {', '.join(symptoms)}\n")
print("="*80)

matches = []
for _, row in df.iterrows():
    disease = row['Disease']
    disease_symptoms = row[1:].values
    matched = [s for s in symptoms if s in disease_symptoms]
    if matched:
        matches.append({
            'disease': disease,
            'matched': matched,
            'count': len(matched)
        })

matches.sort(key=lambda x: x['count'], reverse=True)

if matches:
    print(f"Found {len(matches)} diseases with at least one matching symptom:\n")
    for i, m in enumerate(matches[:15], 1):
        print(f"{i}. {m['disease']}")
        print(f"   Matched: {m['count']}/3 symptoms")
        print(f"   Symptoms: {', '.join(m['matched'])}")
        print()
else:
    print("❌ NO diseases found with these exact symptom names!")
    print("\nThis means these symptoms don't exist in the curated dataset.")
    print("The curated dataset uses different symptom names.\n")

