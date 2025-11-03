"""Train an image classifier from images placed in resources/images/<label>/*.jpg

Directory layout expected:
  resources/images/<label_1>/*.jpg
  resources/images/<label_2>/*.jpg

Outputs:
  resources/image_model.h5
  resources/image_label_index.json

Run: python train_image_model.py
"""

import os
import json
import argparse

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

ROOT = os.path.dirname(__file__)
RESOURCES = os.path.join(ROOT, 'resources')
IMAGE_DIR = os.path.join(RESOURCES, 'images')


def build_simple_cnn(input_shape, num_classes):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.4),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img-size', type=int, default=224)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=20)
    args = parser.parse_args()

    if not os.path.isdir(IMAGE_DIR):
        raise FileNotFoundError(f"Image directory not found: {IMAGE_DIR}. Create resources/images/<label>/... and populate images.")

    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.15,
                                 rotation_range=20, width_shift_range=0.1, height_shift_range=0.1,
                                 horizontal_flip=True, zoom_range=0.1)

    train_gen = datagen.flow_from_directory(IMAGE_DIR, target_size=(args.img_size, args.img_size),
                                            batch_size=args.batch_size, class_mode='categorical', subset='training')

    val_gen = datagen.flow_from_directory(IMAGE_DIR, target_size=(args.img_size, args.img_size),
                                          batch_size=args.batch_size, class_mode='categorical', subset='validation')

    num_classes = len(train_gen.class_indices)
    model = build_simple_cnn((args.img_size, args.img_size, 3), num_classes)

    callbacks = [EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
                 ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)]

    history = model.fit(train_gen, validation_data=val_gen, epochs=args.epochs, callbacks=callbacks)

    os.makedirs(RESOURCES, exist_ok=True)
    model_path = os.path.join(RESOURCES, 'image_model.h5')
    model.save(model_path)

    labels_path = os.path.join(RESOURCES, 'image_label_index.json')
    with open(labels_path, 'w', encoding='utf-8') as f:
        # save as list where index -> label
        labels = {str(i): label for label, i in train_gen.class_indices.items()}
        json.dump(labels, f, ensure_ascii=False, indent=2)

    print('Image model trained and saved to resources/')


if __name__ == '__main__':
    main()
