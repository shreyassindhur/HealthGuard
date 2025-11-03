"""CLI helper to test OCR and model predictions on a single image.

Usage:
  python ocr_predict.py path/to/report.jpg

This script will:
 - load the image
 - attempt OCR (requires pytesseract + Tesseract engine)
 - extract symptom names using fuzzy matching
 - if at least 3 symptoms found and symptom model present, run the symptom model and print top-5
 - if image model present, run image classifier and print top-5
"""
import sys
import os
import json
import numpy as np
from PIL import Image, ImageOps

ROOT = os.path.dirname(__file__)
RESOURCES = os.path.join(ROOT, 'resources')

# Reuse helpers from disease_prediction.py by importing it as a module
try:
    import disease_prediction as dp
except Exception as e:
    print('Failed to import disease_prediction module:', e)
    dp = None


def load_models():
    symptom_model = None
    df = None
    image_model = None
    image_labels = None

    try:
        model_path = os.path.join(RESOURCES, 'mlp_model.h5')
        if os.path.exists(model_path):
            from tensorflow.keras.models import load_model
            symptom_model = load_model(model_path)
    except Exception as e:
        print('Could not load symptom model:', e)

    try:
        data_path = os.path.join(RESOURCES, 'dataset_kaggle.csv')
        if os.path.exists(data_path):
            import pandas as pd
            df = pd.read_csv(data_path)
    except Exception as e:
        print('Could not load dataset:', e)

    try:
        img_model_path = os.path.join(RESOURCES, 'image_model.h5')
        if os.path.exists(img_model_path):
            from tensorflow.keras.models import load_model
            image_model = load_model(img_model_path)
    except Exception as e:
        print('Could not load image model:', e)

    try:
        labels_path = os.path.join(RESOURCES, 'image_label_index.json')
        if os.path.exists(labels_path):
            with open(labels_path, 'r', encoding='utf-8') as f:
                image_labels = json.load(f)
    except Exception as e:
        print('Could not load image labels:', e)

    return symptom_model, df, image_model, image_labels


def preprocess_image_for_image_model(pil_image, target_size=(224, 224)):
    img = ImageOps.exif_transpose(pil_image.convert('RGB'))
    img = ImageOps.fit(img, target_size, Image.LANCZOS)
    arr = np.asarray(img).astype('float32') / 255.0
    return np.expand_dims(arr, axis=0)


def main():
    if len(sys.argv) < 2:
        print('Usage: python ocr_predict.py path/to/report.jpg')
        return

    img_path = sys.argv[1]
    if not os.path.exists(img_path):
        print('Image not found:', img_path)
        return

    pil = Image.open(img_path)
    print('Loaded image:', img_path)

    # OCR
    text, err = dp.ocr_extract_text(pil) if dp else (None, 'disease_prediction module not importable')
    if err:
        print('OCR not available:', err)
    else:
        print('\n--- OCR extracted text (first 1000 chars) ---\n')
        print((text or '')[:1000])

    # Extract symptoms
    symptoms = []
    if dp and text:
        symptoms = dp.extract_symptoms_from_text(text, dp.symptoms_list)
        print('\nDetected symptoms from OCR:', symptoms)

    # Load models
    symptom_model, df, image_model, image_labels = load_models()

    # If symptoms found and symptom model present, run prediction
    if symptoms and len(symptoms) >= 3 and symptom_model is not None and df is not None:
        print('\nRunning symptom-model prediction...')
        dp.predict_and_display(symptoms, symptom_model, df, output_column=None)
    else:
        if len(symptoms) < 3:
            print('\nNot enough unique symptoms detected (need >=3) to run symptom-model automatically.')

    # Image classifier
    if image_model is not None:
        print('\nRunning image classifier...')
        x = preprocess_image_for_image_model(pil)
        preds = image_model.predict(x)
        if preds.ndim == 2 and preds.shape[0] == 1:
            probs = preds[0]
        elif preds.ndim == 1:
            probs = preds
        else:
            probs = np.array(preds).flatten()

        if isinstance(image_labels, dict):
            labels_list = [image_labels.get(str(i), f'label_{i}') for i in range(len(probs))]
        elif isinstance(image_labels, list):
            labels_list = image_labels
        else:
            labels_list = [f'label_{i}' for i in range(len(probs))]

        pairs = sorted(list(zip(labels_list, probs)), key=lambda x: x[1], reverse=True)[:5]
        print('\nTop image-classifier predictions:')
        for lbl, p in pairs:
            print(f'- {lbl}: {p*100:.1f}%')
    else:
        print('\nNo image classifier model found in resources/.')


if __name__ == '__main__':
    main()
