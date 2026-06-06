"""
Daten-Lader und Daten-Splitter für Trainingsmodelle

Dieses Modul stellt Funktionen bereit zum Laden von Bildern aus Ordnerstrukturen,
Splitten in Train/Val/Test-Sets und Erstellen von Data Generators mit Augmentation.

Hinweis: Dieses Modul ist für ältere Trainings-Pipelines. Das Haupttraining nutzt
train_simple.py mit direktem Datenladen (nicht Generatoren).
"""

import os
import numpy as np
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
import cv2

# ============================================================================
# KONFIGURATION
# ============================================================================

# Gebärdensprachen-Klassen
CLASSES = ['A', 'B', 'C', 'L', 'V', 'W', 'O', 'Y']

# Standard-Bildgrösse (muss mit Modell-Input übereinstimmen)
IMG_SIZE = (224, 224)

# Batch-Grösse für Training
BATCH_SIZE = 32

# ============================================================================
# DATENLADEN
# ============================================================================

def load_data(data_dir):
    """
    Lädt alle Bilder aus einem Ordner mit Klassen-Unterordnern.

    Erwartet Struktur:
    ```
    data_dir/
      A/
        image1.jpg
        image2.jpg
        ...
      B/
        ...
    ```

    Args:
        data_dir (str): Pfad zum Daten-Ordner

    Returns:
        tuple: (X, y) - Arrays mit Bildern und Labels
               X: Shape (N, 224, 224, 3), dtype uint8
               y: Shape (N,), Labels 0-7
    """
    X = []
    y = []

    # Durchsuche alle Klassen
    for idx, cls in enumerate(CLASSES):
        cls_dir = os.path.join(data_dir, cls)
        if not os.path.exists(cls_dir):
            print(f"Warnung: Ordner {cls_dir} existiert nicht.")
            continue

        # Lade alle Bilder in dieser Klasse
        for img_file in os.listdir(cls_dir):
            img_path = os.path.join(cls_dir, img_file)
            img = cv2.imread(img_path)
            if img is not None:
                # Resize auf Standard-Grösse
                img = cv2.resize(img, IMG_SIZE)
                # BGR zu RGB konvertieren
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                X.append(img)
                y.append(idx)

    X = np.array(X)
    y = np.array(y)
    return X, y

# ============================================================================
# DATEN-SPLITTUNG
# ============================================================================

def split_data(X, y, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    Splittet Datensatz in Train, Validation und Test-Sets.

    Workflow:
    1. Erste Splittung: Trennung des Test-Sets
    2. Zweite Splittung: Trennung des Val-Sets vom verbleibenden

    Args:
        X (np.ndarray): Bilder-Array
        y (np.ndarray): Label-Array
        train_ratio (float): Anteil für Training (Standard: 70%)
        val_ratio (float): Anteil für Validation (Standard: 15%)
        test_ratio (float): Anteil für Test (Standard: 15%)

    Returns:
        tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    # Erste Splittung: Trenne Test-Set
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(val_ratio + test_ratio), random_state=42
    )

    # Zweite Splittung: Trenne Val/Test aus dem Rest
    val_size = val_ratio / (val_ratio + test_ratio)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=(1 - val_size), random_state=42
    )

    return X_train, X_val, X_test, y_train, y_val, y_test

# ============================================================================
# DATA GENERATORS MIT AUGMENTATION
# ============================================================================

def create_data_generators(X_train, y_train, X_val, y_val, X_test, y_test):
    """
    Erstellt Keras Data Generators mit Augmentation für Training.

    Wichtig:
    - Training nutzt Augmentation (Flip, Rotation, Zoom, etc.)
    - Validation/Test nutzen KEINE Augmentation
    - Alle Daten werden normalisiert auf [0, 1]

    Args:
        X_train, y_train: Training-Daten
        X_val, y_val: Validation-Daten
        X_test, y_test: Test-Daten

    Returns:
        tuple: (train_generator, val_generator, test_generator)
    """
    # ========== Normalisierung ==========
    # Skaliere Pixel-Werte von [0, 255] zu [0, 1]
    X_train = X_train.astype('float32') / 255.0
    X_val = X_val.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0

    # ========== Augmentation für Training ==========
    # Ziel: Robustheit gegenüber Variationen
    train_datagen = ImageDataGenerator(
        horizontal_flip=True,           # Wichtig: Webcam kann gespiegelt sein
        rotation_range=15,              # ±15° Rotation
        zoom_range=0.1,                 # 10% Zoom
        brightness_range=[0.8, 1.2],    # Helligkeit 80-120%
        width_shift_range=0.1,          # 10% horizontales Shift
        height_shift_range=0.1          # 10% vertikales Shift
    )

    # ========== Keine Augmentation für Val/Test ==========
    # Validierung/Test sollen Original-Daten evaluieren
    val_datagen = ImageDataGenerator()
    test_datagen = ImageDataGenerator()

    # ========== Erstelle Generatoren ==========
    train_generator = train_datagen.flow(
        X_train, y_train, batch_size=BATCH_SIZE, shuffle=True
    )
    val_generator = val_datagen.flow(
        X_val, y_val, batch_size=BATCH_SIZE, shuffle=False
    )
    test_generator = test_datagen.flow(
        X_test, y_test, batch_size=BATCH_SIZE, shuffle=False
    )

    return train_generator, val_generator, test_generator