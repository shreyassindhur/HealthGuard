"""Retrain a disease prediction model from the included dataset.

Usage:
    python train_model.py

Outputs written to ./resources/:
 - mlp_model.h5           : Keras model
 - symptom_index.json     : list of symptoms (index -> symptom)
 - label_index.json       : mapping index -> disease label
 - training_history.json  : training loss/accuracy history

Notes:
 - This script is intended to be run locally in a virtualenv where
   dependencies from requirements.txt are installed.
 - The dataset is expected at resources/dataset_kaggle.csv
"""

import argparse
import json
import os
import random
from collections import Counter

import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
import joblib

try:
    import tensorflow as tf
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, InputLayer
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False


ROOT = os.path.dirname(__file__)
RESOURCES = os.path.join(ROOT, "resources")
DATA_PATH = os.path.join(RESOURCES, "dataset_kaggle.csv")


def load_dataset(path):
    df = pd.read_csv(path)
    # Symptom columns often named Symptom_1..Symptom_17
    symptom_cols = [c for c in df.columns if c.lower().startswith("symptom")]
    # Build list of symptoms per row, dropping NaNs and empty strings
    symptoms = []
    for _, row in df[symptom_cols].iterrows():
        vals = [str(v).strip() for v in row.values if pd.notna(v) and str(v).strip()]
        symptoms.append(vals)
    diseases = df['Disease'].astype(str).str.strip().tolist()
    return diseases, symptoms


def normalize_symptom(s: str) -> str:
    # Basic normalization: lowercase, strip, remove duplicate spaces
    return ' '.join(s.lower().strip().split())


def clean_symptoms(symptom_lists):
    return [[normalize_symptom(s) for s in row if s and str(s).strip()] for row in symptom_lists]


def build_matrices(diseases, symptoms, min_freq=1):
    # Optionally filter out rare symptoms
    flat = [s for row in symptoms for s in row]
    counts = Counter(flat)
    allowed = {s for s, c in counts.items() if c >= min_freq}
    filtered = [[s for s in row if s in allowed] for row in symptoms]

    mlb = MultiLabelBinarizer(sparse_output=False)
    X = mlb.fit_transform(filtered)

    # Encode disease labels
    le = LabelEncoder()
    y = le.fit_transform(diseases)

    # Convert y to categorical for Keras when TF available
    if TF_AVAILABLE:
        y_cat = tf.keras.utils.to_categorical(y, num_classes=len(le.classes_))
    else:
        y_cat = None
    return X, y, y_cat, mlb, le, counts


def augment_dataset(diseases, symptoms, augment_per_row=0, random_seed=42):
    # Simple augmentation: for each disease, create additional synthetic rows by sampling
    # symptoms from the disease's union set.
    if augment_per_row <= 0:
        return diseases, symptoms
    random.seed(random_seed)
    new_diseases = list(diseases)
    new_symptoms = [list(s) for s in symptoms]

    # Build per-disease symptom pools and typical lengths
    pools = {}
    lengths = {}
    for d, s in zip(diseases, symptoms):
        pools.setdefault(d, set()).update(s)
        lengths.setdefault(d, []).append(len(s))

    for d in pools:
        pools[d] = list(pools[d])

    for i, (d, s) in enumerate(zip(diseases, symptoms)):
        pool = pools.get(d, list(s))
        if not pool:
            continue
        # generate augment_per_row synthetic samples
        for _ in range(augment_per_row):
            # sample a length similar to existing examples for that disease
            target_len = int(round(np.random.choice(lengths.get(d, [max(1, len(s))]))))
            target_len = max(1, target_len)
            # sample without replacement where possible
            if target_len >= len(pool):
                sample = list(pool)
            else:
                sample = random.sample(pool, target_len)
            new_diseases.append(d)
            new_symptoms.append(sample)

    return new_diseases, new_symptoms


def build_model(input_dim, num_classes):
    if not TF_AVAILABLE:
        raise RuntimeError("TensorFlow is not available in this environment. Install TF to use the MLP trainer.")
    model = Sequential([
        InputLayer(input_shape=(input_dim,)),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.4),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def train_logistic(X_train, y_train, X_val, y_val, label_encoder):
    print("Training LogisticRegression (efficient)...")
    clf = LogisticRegression(max_iter=1000, solver='saga', multi_class='multinomial')
    clf.fit(X_train, y_train)
    preds = clf.predict(X_val)
    print(classification_report(y_val, preds, target_names=label_encoder.classes_, zero_division=0))
    return clf


def main():
    parser = argparse.ArgumentParser(description='Retrain disease prediction model with augmentation')
    parser.add_argument('--model', choices=['mlp', 'logistic'], default='logistic',
                        help='Which model to train (mlp requires TensorFlow)')
    parser.add_argument('--augment', type=int, default=0, help='Synthetic samples to generate per row')
    parser.add_argument('--min-symptom-freq', type=int, default=1, help='Remove symptoms appearing fewer times')
    parser.add_argument('--test-size', type=float, default=0.15, help='Validation set size fraction')
    args = parser.parse_args()

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    print("Loading dataset...")
    diseases, symptoms = load_dataset(DATA_PATH)
    print(f"Rows: {len(diseases)}  Unique diseases: {len(set(diseases))}")

    # Clean and normalize
    symptoms = clean_symptoms(symptoms)

    # Augment
    if args.augment > 0:
        print(f"Augmenting dataset: {args.augment} synthetic samples per row...")
        diseases, symptoms = augment_dataset(diseases, symptoms, augment_per_row=args.augment)
        print(f"New row count: {len(diseases)}")

    print("Building matrices...")
    X, y, y_cat, mlb, le, counts = build_matrices(diseases, symptoms, min_freq=args.min_symptom_freq)

    n_samples, input_dim = X.shape
    num_classes = len(le.classes_)
    print(f"Input dim: {input_dim}  Num classes: {num_classes}  Samples: {n_samples}")

    # Split
    if args.model == 'mlp' and not TF_AVAILABLE:
        raise RuntimeError('Requested mlp model but TensorFlow is not installed or importable')

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=args.test_size, random_state=42, stratify=y)

    # Class weights optionally used by MLP
    classes = np.unique(y_train)
    class_weight = None
    if args.model == 'mlp':
        class_weights_raw = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
        class_weight = {int(cls): float(w) for cls, w in zip(classes, class_weights_raw)}
        print("Class weights computed for MLP.")

    # Train requested model
    os.makedirs(RESOURCES, exist_ok=True)

    if args.model == 'logistic':
        clf = train_logistic(X_train, y_train, X_val, y_val, le)
        model_path = os.path.join(RESOURCES, 'logistic_model.joblib')
        print(f"Saving logistic model to {model_path}")
        joblib.dump(clf, model_path)
    else:
        model = build_model(input_dim, num_classes)
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)
        ]

        print("Starting training of MLP...")
        history = model.fit(
            X_train, y_cat[y_train],
            validation_data=(X_val, y_cat[[int(i) for i in y_val]]),
            epochs=50,
            batch_size=32,
            callbacks=callbacks,
            class_weight=class_weight,
            verbose=2
        )

        print("Evaluating on validation set...")
        preds = model.predict(X_val)
        y_pred = np.argmax(preds, axis=1)
        print(classification_report(y_val, y_pred, target_names=le.classes_, zero_division=0))

        model_path = os.path.join(RESOURCES, 'mlp_model.h5')
        print(f"Saving model to {model_path}")
        model.save(model_path)

        history_path = os.path.join(RESOURCES, 'training_history.json')
        print(f"Saving training history to {history_path}")
        history_dict = {k: [float(x) for x in v] for k, v in history.history.items()}
        with open(history_path, 'w', encoding='utf-8') as f:
            json.dump(history_dict, f, ensure_ascii=False, indent=2)

    # Save artifact mappings (common for both models)
    symptom_index_path = os.path.join(RESOURCES, 'symptom_index.json')
    print(f"Saving symptom index to {symptom_index_path}")
    with open(symptom_index_path, 'w', encoding='utf-8') as f:
        json.dump(list(mlb.classes_), f, ensure_ascii=False, indent=2)

    label_index_path = os.path.join(RESOURCES, 'label_index.json')
    label_map = {int(i): label for i, label in enumerate(list(le.classes_))}
    print(f"Saving label index to {label_index_path}")
    with open(label_index_path, 'w', encoding='utf-8') as f:
        json.dump(label_map, f, ensure_ascii=False, indent=2)

    # Save symptom frequency for debugging
    freq_path = os.path.join(RESOURCES, 'symptom_freq.json')
    with open(freq_path, 'w', encoding='utf-8') as f:
        json.dump(counts, f, ensure_ascii=False, indent=2)

    print("Training complete.")


if __name__ == '__main__':
    main()
