"""
Trainingsmodul für Gebärdensprachen-Buchstaben-Erkennung

Dieses Modul trainiert ein CNN-Modell basierend auf MobileNetV2 mit Transfer Learning.
Die Trainingslogik ist optimiert für schnelle Konvergenz und gute Generalisierung.
Das Modell wird trainiert auf bereinigten Daten (data_cleaned/) und anschließend
automatisch mit externe Testdaten (external_test/) evaluiert.
"""

import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from utils import get_timestamp

# ============================================================================
# KONFIGURATION
# ============================================================================

# Gebärdensprachen-Klassen (8 Buchstaben)
CLASSES = ['A', 'B', 'C', 'L', 'V', 'W', 'O', 'Y']
NUM_CLASSES = len(CLASSES)

# Hyperparameter
IMG_SIZE = (224, 224, 3)  # MobileNetV2 Input-Grösse
BATCH_SIZE = 32            # Batch-Grösse für Training
EPOCHS = 40                # Maximale Anzahl Epochen (mit EarlyStopping)

# MobileNetV2-Gewichte (vortrainiert auf ImageNet)
LOCAL_MOBILENET_WEIGHTS = os.path.expanduser(
    "~/.keras/models/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224_no_top.h5"
)

# ============================================================================
# DATEN-LADEN UND AUGMENTATION
# ============================================================================

def load_data(data_dir):
    """
    Lädt Bilddaten aus die bereinigten Trainingsdaten.

    Args:
        data_dir (str): Pfad zum Datenordner (z.B. 'data_cleaned/')

    Returns:
        tuple: (X, y) - Arrays mit Bildern und Labels (0-7 für 8 Klassen)
    """
    X = []
    y = []
    for idx, cls in enumerate(CLASSES):
        cls_dir = os.path.join(data_dir, cls)
        if not os.path.exists(cls_dir):
            print(f"Warnung: Ordner {cls_dir} existiert nicht.")
            continue

        # Lade alle Bilder aus dem Klassenordner
        for img_file in os.listdir(cls_dir):
            img_path = os.path.join(cls_dir, img_file)
            img = cv2.imread(img_path)
            if img is not None:
                # Resize auf Standard-Grösse und in RGB konvertieren
                img = cv2.resize(img, (224, 224))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                X.append(img)
                y.append(idx)

    X = np.array(X)
    y = np.array(y)
    return X, y

def augment_image(img):
    """
    Augmentiert Bilder während des Trainings für bessere Robustheit.

    Angewendete Transformationen:
    - Horizontales Flip (50% Wahrscheinlichkeit)
    - Rotation ±15° (30% Wahrscheinlichkeit)
    - Helligkeitsanpassung 0.8-1.2x (30% Wahrscheinlichkeit)

    Args:
        img (np.ndarray): Eingangsbild

    Returns:
        np.ndarray: Augmentiertes Bild
    """
    # Zufälliges horizontales Flip
    if np.random.rand() > 0.5:
        img = cv2.flip(img, 1)

    # Zufällige Rotation
    if np.random.rand() > 0.7:
        angle = np.random.uniform(-15, 15)
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
        img = cv2.warpAffine(img, M, (w, h))

    # Zufällige Helligkeitsanpassung
    if np.random.rand() > 0.7:
        brightness = np.random.uniform(0.8, 1.2)
        img = cv2.convertScaleAbs(img * brightness)

    return img

# ============================================================================
# MODELLAUFBAU
# ============================================================================

def build_model():
    """
    Baut das CNN-Modell mit Transfer Learning (MobileNetV2-Basis).

    Architektur:
    - MobileNetV2 (ImageNet vortrainiert, Top-Schichten entfernt)
    - GlobalAveragePooling2D
    - Dense(128) + ReLU + Dropout(0.5)
    - Dense(NUM_CLASSES) + Softmax (Ausgabe)

    Returns:
        Model: Das Keras-Modell bereit zum Training
    """
    # Verwende lokale MobileNetV2-Gewichte, falls vorhanden; sonst ImageNet-Download
    weights = LOCAL_MOBILENET_WEIGHTS if os.path.exists(LOCAL_MOBILENET_WEIGHTS) else 'imagenet'
    if weights != 'imagenet':
        print(f"Verwende lokale MobileNetV2-Gewichte: {weights}")
    else:
        print("Keine lokalen MobileNetV2-Gewichte gefunden, verwende ImageNet-Download.")

    # Lade vortrainierte MobileNetV2 ohne Top-Schichten
    base_model = MobileNetV2(weights=weights, include_top=False, input_shape=IMG_SIZE)
    base_model.trainable = False  # Einfrieren der Base-Schichten (Transfer Learning)

    # Füge eigene Klassifizierungs-Schichten hinzu
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)  # Prevent Overfitting
    predictions = Dense(NUM_CLASSES, activation='softmax')(x)

    # Erstelle Modell
    model = Model(inputs=base_model.input, outputs=predictions)
    return model

# ============================================================================
# TRAINING
# ============================================================================

def train_model(data_dir, model_save_path=None):
    """
    Trainiert das Modell auf bereinigten Trainingsdaten.

    Workflow:
    1. Lade Daten aus data_cleaned/
    2. Split: 80% Training, 20% Validation
    3. Normalisiere Pixel-Werte (0-1)
    4. Trainiere mit EarlyStopping, LR-Scheduler und Checkpointing
    5. Speichere das beste Modell
    6. Speichere Trainingskurven und Metriken
    7. Starte automatisch evaluate.py mit dem trainierten Modell

    Args:
        data_dir (str): Pfad zu bereinigte Trainingsdaten (Standard: 'data_cleaned')
        model_save_path (str, optional): Pfad zum Speichern des Modells

    Returns:
        tuple: (model, history, model_save_path)
    """
    timestamp = get_timestamp()
    if model_save_path is None:
        model_save_path = f'models/sign_language_model_{timestamp}.h5'
    
    print(f"\n=== Gebärdensprachen-Erkennung Training ===")
    print(f"Lade Daten von: {data_dir}")
    print(f"Klassen: {CLASSES}")
    print(f"Zeitstempel: {timestamp}\n")
    
    # ========== Datenladen ==========
    X, y = load_data(data_dir)
    print(f"Geladen: {len(X)} Bilder")
    
    # ========== Train/Validation Split ==========
    # Nur Train/Validation Split (kein interner Test)
    # Externe Testdaten werden später mit evaluate.py genutzt
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.20, random_state=42)

    # ========== Normalisierung ==========
    # Skaliere Pixel-Werte von [0, 255] auf [0, 1]
    X_train = X_train.astype('float32') / 255.0
    X_val = X_val.astype('float32') / 255.0

    print(f"Train: {len(X_train)}, Val: {len(X_val)}\n")

    # ========== Modellaufbau und Kompilierung ==========
    model = build_model()
    model.compile(optimizer=Adam(learning_rate=0.001), 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])
    
    # ========== Callbacks Setup ==========
    os.makedirs('models', exist_ok=True)
    os.makedirs('logs/training_metrics', exist_ok=True)
    os.makedirs('logs/training_curves', exist_ok=True)

    # Early Stopping bei Stagnation (val_loss verbessert sich nicht mehr)
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    # Model Checkpoint: speichere beste Modell basierend auf val_accuracy
    checkpoint = ModelCheckpoint(model_save_path, monitor='val_accuracy', save_best_only=True)

    # Learning Rate Reducer: reduziere LR wenn val_loss stagniert
    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)

    # CSV Logger für Epoch-Metriken
    csv_logger = CSVLogger(f'logs/training_metrics/epoch_metrics_{timestamp}.csv')

    # ========== TRAINING ==========
    print("=== Training startet ===\n")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stop, checkpoint, lr_scheduler, csv_logger],
        verbose=1
    )
    
    # ========== Speicherung ==========
    # Speichere das Modell in mehreren Formaten:
    # 1. Zeitgestempelt (für archivierung)
    # 2. Kanonische Datei (aktuellstes Modell für Reports/Clones)
    model.save(model_save_path)
    model.save('models/sign_language_model.keras')
    model.save('models/sign_language_model.h5')
    print(f"\n✓ Modell gespeichert: {model_save_path}")
    
    # ========== Trainingskurven Plot ==========
    print("\nSpeichere Trainingskurven...")
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Loss')
    plt.legend()
    plt.grid()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title('Accuracy')
    plt.legend()
    plt.grid()
    
    plt.tight_layout()
    plt.savefig(f'logs/training_curves/training_curves_{timestamp}.png', dpi=150)
    plt.close()
    print(f"✓ Trainingskurven gespeichert: logs/training_curves/training_curves_{timestamp}.png")
    print(f"✓ Epoch-Metriken gespeichert: logs/training_metrics/epoch_metrics_{timestamp}.csv")

    print("\n" + "="*60)
    print("✅ Training abgeschlossen!")
    print("="*60)
    print(f"Modell gespeichert: {model_save_path}")
    print("Für FINALE EVALUATION (mit external_test) verwende:")
    print(f"  python evaluate.py --model {model_save_path}")
    print("="*60)

    return model, history, model_save_path


# ============================================================================
# HAUPTEINSTIEG
# ============================================================================

if __name__ == "__main__":
    import subprocess

    # Training starten
    data_dir = 'data_cleaned'  # Bereinigte Daten verwenden (nicht data_raw!)
    model, history, model_save_path = train_model(data_dir)

    print("\n" + "="*60)
    print("🚀 Starte automatische finale Evaluation...")
    print("="*60)

    # Automatisch evaluate.py ausführen mit dem trainierten Modell
    try:
        subprocess.run(
            ['python', 'evaluate.py', '--model', model_save_path],
            check=True
        )
    except subprocess.CalledProcessError as e:
        print(f"⚠️ Fehler bei der automatischen Evaluation: {e}")
        print(f"Führe manuell aus: python evaluate.py --model {model_save_path}")
