"""
Retrain the disease prediction model with the curated dataset

This script:
1. Loads the curated dataset
2. Trains a new MLP model with the same architecture
3. Saves the new model as mlp_model_curated.h5
4. Updates the symptom mappings
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import json
import os

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def load_and_prepare_data(csv_path, augment=True, samples_per_disease=10):
    """Load curated dataset and prepare for training with augmentation"""
    print(f"Loading dataset from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    print(f"✓ Loaded {len(df)} diseases")
    print(f"✓ {len(df.columns)-1} symptoms per disease")
    
    # Extract all unique symptoms
    all_symptoms = set()
    for col in df.columns:
        if col != 'Disease':
            all_symptoms.update(df[col].dropna().unique())
    
    symptoms_list = sorted(list(all_symptoms))
    print(f"✓ {len(symptoms_list)} unique symptoms total")
    
    # Create symptom index mapping
    symptom_to_index = {symptom: idx for idx, symptom in enumerate(symptoms_list)}
    
    # Encode diseases
    label_encoder = LabelEncoder()
    
    # Create binary symptom vectors with augmentation
    X = []
    y_diseases = []
    
    for idx, row in df.iterrows():
        disease = row['Disease']
        disease_symptoms = [row[col] for col in df.columns if col != 'Disease' and pd.notna(row[col])]
        
        if augment:
            # Generate multiple samples per disease with WEIGHTED symptom selection
            # Earlier symptoms in the list are more important (primary symptoms)
            # This preserves clinical relevance during augmentation
            
            # Always include the full symptom set first
            symptom_vector = np.zeros(len(symptoms_list))
            for symptom in disease_symptoms:
                if symptom in symptom_to_index:
                    symptom_vector[symptom_to_index[symptom]] = 1
            X.append(symptom_vector)
            y_diseases.append(disease)
            
            # Then create variations with subsets, but prefer earlier (more important) symptoms
            for _ in range(samples_per_disease - 1):
                num_symptoms = np.random.randint(3, min(8, len(disease_symptoms) + 1))
                
                # Weight symptoms by position (earlier = more important)
                weights = np.array([1.0 / (i + 1) for i in range(len(disease_symptoms))])
                weights = weights / weights.sum()
                
                selected_symptoms = np.random.choice(
                    disease_symptoms, 
                    size=num_symptoms, 
                    replace=False,
                    p=weights
                )
                
                symptom_vector = np.zeros(len(symptoms_list))
                for symptom in selected_symptoms:
                    if symptom in symptom_to_index:
                        symptom_vector[symptom_to_index[symptom]] = 1
                
                X.append(symptom_vector)
                y_diseases.append(disease)
        else:
            # Use all symptoms (original approach)
            symptom_vector = np.zeros(len(symptoms_list))
            for symptom in disease_symptoms:
                if symptom in symptom_to_index:
                    symptom_vector[symptom_to_index[symptom]] = 1
            X.append(symptom_vector)
            y_diseases.append(disease)
    
    X = np.array(X)
    encoded_diseases = label_encoder.fit_transform(y_diseases)
    y = keras.utils.to_categorical(encoded_diseases, num_classes=len(label_encoder.classes_))
    
    print(f"✓ Generated {len(X)} training samples ({samples_per_disease} per disease)" if augment else f"✓ Generated {len(X)} samples")
    print(f"✓ Input shape: {X.shape}")
    print(f"✓ Output shape: {y.shape}")
    
    return X, y, symptoms_list, label_encoder, df

def build_model(input_dim, output_dim):
    """Build MLP model with same architecture as original"""
    model = keras.Sequential([
        layers.Dense(256, activation='relu', input_shape=(input_dim,)),
        layers.Dropout(0.3),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(64, activation='relu'),
        layers.Dense(output_dim, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_model(X, y, epochs=100, batch_size=16):
    """Train the model"""
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=np.argmax(y, axis=1)
    )
    
    print(f"\n📊 Data split:")
    print(f"  Training: {len(X_train)} samples")
    print(f"  Testing: {len(X_test)} samples")
    
    # Build model
    model = build_model(X.shape[1], y.shape[1])
    
    print(f"\n🏗️  Model architecture:")
    model.summary()
    
    # Train
    print(f"\n🚀 Training for {epochs} epochs...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )
    
    # Evaluate
    print(f"\n📈 Final Results:")
    train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    
    print(f"  Training accuracy: {train_acc*100:.2f}%")
    print(f"  Test accuracy: {test_acc*100:.2f}%")
    
    return model, history

def save_model_and_mappings(model, symptoms_list, label_encoder, df, output_dir='resources'):
    """Save the trained model and necessary mappings"""
    print(f"\n💾 Saving model and mappings to {output_dir}/")
    
    # Save model
    model_path = os.path.join(output_dir, 'mlp_model_curated.h5')
    model.save(model_path)
    print(f"  ✓ Model saved to {model_path}")
    
    # Save symptom index
    symptom_index = {symptom: idx for idx, symptom in enumerate(symptoms_list)}
    symptom_index_path = os.path.join(output_dir, 'symptom_index_curated.json')
    with open(symptom_index_path, 'w') as f:
        json.dump(symptom_index, f, indent=2)
    print(f"  ✓ Symptom index saved to {symptom_index_path}")
    
    # Save label (disease) index
    label_index = {disease: int(idx) for idx, disease in enumerate(label_encoder.classes_)}
    label_index_path = os.path.join(output_dir, 'label_index_curated.json')
    with open(label_index_path, 'w') as f:
        json.dump(label_index, f, indent=2)
    print(f"  ✓ Label index saved to {label_index_path}")
    
    # Save curated dataset copy
    dataset_path = os.path.join(output_dir, 'dataset_curated.csv')
    if not os.path.exists(dataset_path):
        df.to_csv(dataset_path, index=False)
        print(f"  ✓ Dataset saved to {dataset_path}")
    
    print(f"\n✅ All files saved successfully!")
    print(f"\n📝 To use the curated model in your app:")
    print(f"   1. Update disease_prediction.py to load 'mlp_model_curated.h5'")
    print(f"   2. Update to load 'dataset_curated.csv'")
    print(f"   3. Update to load 'symptom_index_curated.json'")
    print(f"\n   Or simply rename the files to replace the originals (backup first!):")
    print(f"   - mlp_model_curated.h5 → mlp_model.h5")
    print(f"   - dataset_curated.csv → dataset_kaggle.csv")
    print(f"   - symptom_index_curated.json → symptom_index.json")

if __name__ == '__main__':
    # Configuration
    CSV_PATH = 'resources/dataset_curated.csv'
    EPOCHS = 150
    BATCH_SIZE = 8  # Smaller batch for small dataset
    
    print("="*80)
    print("TRAINING DISEASE PREDICTION MODEL WITH CURATED DATASET")
    print("="*80)
    
    # Load and prepare data
    X, y, symptoms_list, label_encoder, df = load_and_prepare_data(CSV_PATH)
    
    # Train model
    model, history = train_model(X, y, epochs=EPOCHS, batch_size=BATCH_SIZE)
    
    # Save everything
    save_model_and_mappings(model, symptoms_list, label_encoder, df)
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)
