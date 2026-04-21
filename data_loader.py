import os
import numpy as np
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
import cv2

# Klassen definieren
CLASSES = ['A', 'B', 'C', 'L', 'V', 'W', 'O', 'Y']
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

def load_data(data_dir):
    """
    Lädt Bilder aus dem data_dir und strukturiert sie in X und y.
    Annahme: data_dir hat Unterordner pro Klasse, z.B. data/A/, data/B/, etc.
    """
    X = []
    y = []
    for idx, cls in enumerate(CLASSES):
        cls_dir = os.path.join(data_dir, cls)
        if not os.path.exists(cls_dir):
            print(f"Warnung: Ordner {cls_dir} existiert nicht.")
            continue
        for img_file in os.listdir(cls_dir):
            img_path = os.path.join(cls_dir, img_file)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, IMG_SIZE)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # BGR zu RGB
                X.append(img)
                y.append(idx)
    X = np.array(X)
    y = np.array(y)
    return X, y

def split_data(X, y, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    Splitte Daten in Train, Validation und Test.
    """
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=(val_ratio + test_ratio), random_state=42)
    val_size = val_ratio / (val_ratio + test_ratio)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=(1 - val_size), random_state=42)
    return X_train, X_val, X_test, y_train, y_val, y_test

def create_data_generators(X_train, y_train, X_val, y_val, X_test, y_test):
    """
    Erstellt DataGenerators für Training, Validation und Test mit Augmentation.
    """
    # Normalisierung: 0-1
    X_train = X_train.astype('float32') / 255.0
    X_val = X_val.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0

    # Data Augmentation für Training
    train_datagen = ImageDataGenerator(
        horizontal_flip=True,  # Wichtig wegen Spiegelung
        rotation_range=15,
        zoom_range=0.1,
        brightness_range=[0.8, 1.2],
        width_shift_range=0.1,
        height_shift_range=0.1
    )

    val_datagen = ImageDataGenerator()  # Keine Augmentation für Val/Test
    test_datagen = ImageDataGenerator()

    train_generator = train_datagen.flow(X_train, y_train, batch_size=BATCH_SIZE, shuffle=True)
    val_generator = val_datagen.flow(X_val, y_val, batch_size=BATCH_SIZE, shuffle=False)
    test_generator = test_datagen.flow(X_test, y_test, batch_size=BATCH_SIZE, shuffle=False)

    return train_generator, val_generator, test_generator