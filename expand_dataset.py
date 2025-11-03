"""
Expand the curated dataset with additional diseases
This adds more common diseases with medically accurate symptoms
"""
import csv

# Additional diseases to add (each with 10 primary symptoms)
new_diseases = [
    # Respiratory/ENT
    ['Sinusitis', 'Facial pain', 'Nasal congestion', 'Thick nasal discharge', 'Reduced sense of smell', 'Headache', 'Cough', 'Fatigue', 'Dental pain', 'Ear pressure', 'Fever'],
    ['Laryngitis', 'Hoarseness', 'Weak voice', 'Sore throat', 'Dry throat', 'Tickling sensation', 'Cough', 'Difficulty swallowing', 'Fever', 'Swollen lymph nodes', 'Voice loss'],
    ['Tonsillitis', 'Sore throat', 'Difficulty swallowing', 'Swollen tonsils', 'White patches on tonsils', 'Fever', 'Tender lymph nodes', 'Bad breath', 'Headache', 'Ear pain', 'Stiff neck'],
    
    # Cardiovascular
    ['Angina', 'Chest pain', 'Pressure in chest', 'Pain radiating to arm', 'Shortness of breath', 'Nausea', 'Fatigue', 'Sweating', 'Dizziness', 'Pain in jaw', 'Indigestion'],
    ['Atrial Fibrillation', 'Irregular heartbeat', 'Heart palpitations', 'Fatigue', 'Shortness of breath', 'Chest pain', 'Dizziness', 'Lightheadedness', 'Weakness', 'Reduced exercise capacity', 'Confusion'],
    ['Deep Vein Thrombosis', 'Leg pain', 'Swelling in leg', 'Warmth in affected area', 'Red or discolored skin', 'Cramping', 'Tenderness', 'Enlarged veins', 'Heavy feeling in leg', 'Skin changes', 'Pain worsens when walking'],
    
    # Neurological
    ['Meningitis', 'Severe headache', 'Stiff neck', 'High fever', 'Sensitivity to light', 'Confusion', 'Nausea', 'Vomiting', 'Drowsiness', 'Difficulty concentrating', 'Seizures'],
    ['Vertigo', 'Spinning sensation', 'Dizziness', 'Balance problems', 'Nausea', 'Vomiting', 'Sweating', 'Abnormal eye movements', 'Headache', 'Ringing in ears', 'Hearing loss'],
    ['Trigeminal Neuralgia', 'Severe facial pain', 'Electric shock sensation', 'Pain triggered by touch', 'Pain in jaw', 'Pain in cheek', 'Pain in teeth', 'Pain in gums', 'Spasms', 'Pain on one side', 'Brief pain episodes'],
    
    # Endocrine/Metabolic
    ['Hypoglycemia', 'Shakiness', 'Nervousness', 'Sweating', 'Chills', 'Confusion', 'Rapid heartbeat', 'Lightheadedness', 'Hunger', 'Irritability', 'Headache'],
    ['Addison\'s Disease', 'Extreme fatigue', 'Weight loss', 'Low blood pressure', 'Salt craving', 'Darkened skin', 'Muscle weakness', 'Depression', 'Irritability', 'Nausea', 'Diarrhea'],
    ['Cushing\'s Syndrome', 'Weight gain', 'Fatty deposits', 'Purple stretch marks', 'Round face', 'Thinning skin', 'Slow healing', 'Acne', 'Fatigue', 'Muscle weakness', 'High blood pressure'],
    
    # Gastrointestinal
    ['Pancreatitis', 'Upper abdominal pain', 'Pain radiating to back', 'Nausea', 'Vomiting', 'Fever', 'Rapid pulse', 'Tender abdomen', 'Weight loss', 'Oily stools', 'Pain after eating'],
    ['Gallstones', 'Upper right abdominal pain', 'Pain between shoulder blades', 'Right shoulder pain', 'Nausea', 'Vomiting', 'Indigestion', 'Bloating', 'Pain after fatty meals', 'Jaundice', 'Clay-colored stools'],
    ['Diverticulitis', 'Lower left abdominal pain', 'Fever', 'Nausea', 'Vomiting', 'Constipation', 'Diarrhea', 'Bloating', 'Cramping', 'Tender abdomen', 'Change in bowel habits'],
    
    # Urinary/Renal
    ['Urinary Tract Infection', 'Burning during urination', 'Frequent urination', 'Urgent need to urinate', 'Cloudy urine', 'Strong-smelling urine', 'Pelvic pain', 'Lower abdominal pain', 'Blood in urine', 'Fever', 'Fatigue'],
    ['Bladder Infection', 'Painful urination', 'Frequent urination', 'Lower abdominal pressure', 'Pelvic discomfort', 'Urgent urination', 'Blood in urine', 'Cloudy urine', 'Foul-smelling urine', 'Mild fever', 'Back pain'],
    ['Kidney Infection', 'Fever', 'Chills', 'Back pain', 'Groin pain', 'Nausea', 'Vomiting', 'Frequent urination', 'Painful urination', 'Cloudy urine', 'Blood in urine'],
    
    # Dermatological
    ['Psoriasis', 'Red patches', 'Silvery scales', 'Dry skin', 'Cracked skin', 'Itching', 'Burning sensation', 'Thickened skin', 'Pitted nails', 'Joint pain', 'Stiff joints'],
    ['Eczema', 'Itchy skin', 'Red rash', 'Dry skin', 'Cracked skin', 'Thickened skin', 'Small bumps', 'Oozing', 'Crusting', 'Sensitive skin', 'Inflamed skin'],
    ['Shingles', 'Pain', 'Burning sensation', 'Tingling', 'Numbness', 'Red rash', 'Fluid-filled blisters', 'Itching', 'Fever', 'Headache', 'Sensitivity to touch'],
    
    # Musculoskeletal
    ['Fibromyalgia', 'Widespread pain', 'Fatigue', 'Sleep problems', 'Morning stiffness', 'Headaches', 'Memory problems', 'Concentration difficulty', 'Tender points', 'Numbness', 'Tingling'],
    ['Tendinitis', 'Pain at tendon', 'Tenderness', 'Mild swelling', 'Pain worsens with movement', 'Stiffness', 'Weakness', 'Grating sensation', 'Warmth', 'Redness', 'Difficulty moving joint'],
    ['Bursitis', 'Joint pain', 'Tenderness', 'Swelling', 'Warmth', 'Redness', 'Stiffness', 'Pain worsens with movement', 'Limited range of motion', 'Aching', 'Sharp pain with pressure'],
    
    # Infectious
    ['Mononucleosis', 'Extreme fatigue', 'Sore throat', 'Fever', 'Swollen lymph nodes', 'Swollen tonsils', 'Headache', 'Skin rash', 'Loss of appetite', 'Muscle aches', 'Enlarged spleen'],
    ['Chickenpox', 'Itchy rash', 'Red spots', 'Fluid-filled blisters', 'Fever', 'Fatigue', 'Loss of appetite', 'Headache', 'Spots on body', 'Spots on face', 'Spots on scalp'],
    ['Mumps', 'Swollen salivary glands', 'Jaw pain', 'Difficulty chewing', 'Difficulty swallowing', 'Fever', 'Headache', 'Muscle aches', 'Fatigue', 'Loss of appetite', 'Earache'],
    
    # Hematological
    ['Sickle Cell Disease', 'Pain episodes', 'Swelling in hands', 'Swelling in feet', 'Frequent infections', 'Delayed growth', 'Vision problems', 'Fatigue', 'Jaundice', 'Shortness of breath', 'Dizziness'],
    ['Hemophilia', 'Easy bruising', 'Large bruises', 'Bleeding into joints', 'Joint pain', 'Joint swelling', 'Blood in urine', 'Blood in stool', 'Nosebleeds', 'Prolonged bleeding', 'Unexplained bleeding'],
    ['Thrombocytopenia', 'Easy bruising', 'Petechiae', 'Prolonged bleeding', 'Blood in urine', 'Blood in stool', 'Heavy menstrual bleeding', 'Nosebleeds', 'Bleeding gums', 'Fatigue', 'Enlarged spleen'],
]

# Read existing dataset
existing_diseases = []
with open('resources/dataset_curated.csv', 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    header = next(reader)
    for row in reader:
        existing_diseases.append(row)

print(f"Current dataset: {len(existing_diseases)} diseases")
print(f"Adding: {len(new_diseases)} new diseases")
print(f"New total: {len(existing_diseases) + len(new_diseases)} diseases")

# Write expanded dataset
with open('resources/dataset_curated.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(header)  # Write header
    writer.writerows(existing_diseases)  # Write existing data
    writer.writerows(new_diseases)  # Write new data

print("\n✅ Dataset expanded successfully!")
print("\nNew diseases added:")
for i, disease in enumerate(new_diseases, 1):
    print(f"{i}. {disease[0]}")

print("\n⚠️ Important: You need to retrain the model with the expanded dataset!")
print("Run: python train_curated_model.py")
