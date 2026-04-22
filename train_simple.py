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

# Klassen definieren
CLASSES = ['A', 'B', 'C', 'L', 'V', 'W', 'O', 'Y']
NUM_CLASSES = len(CLASSES)
IMG_SIZE = (224, 224, 3)
BATCH_SIZE = 32
EPOCHS = 40
LOCAL_MOBILENET_WEIGHTS = os.path.expanduser(
    "~/.keras/models/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224_no_top.h5"
)

def load_data(data_dir):
    """Lädt Bilder aus dem data_dir."""
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
                img = cv2.resize(img, (224, 224))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                X.append(img)
                y.append(idx)
    X = np.array(X)
    y = np.array(y)
    return X, y

def augment_image(img):
    """Einfache Augmentation."""
    if np.random.rand() > 0.5:
        img = cv2.flip(img, 1)  # Horizontal flip
    if np.random.rand() > 0.7:
        angle = np.random.uniform(-15, 15)
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
        img = cv2.warpAffine(img, M, (w, h))
    if np.random.rand() > 0.7:
        brightness = np.random.uniform(0.8, 1.2)
        img = cv2.convertScaleAbs(img * brightness)
    return img

def build_model():
    """Baut das Modell mit Transfer Learning (MobileNetV2)."""
    weights = LOCAL_MOBILENET_WEIGHTS if os.path.exists(LOCAL_MOBILENET_WEIGHTS) else 'imagenet'
    if weights != 'imagenet':
        print(f"Verwende lokale MobileNetV2-Gewichte: {weights}")
    else:
        print("Keine lokalen MobileNetV2-Gewichte gefunden, verwende ImageNet-Download.")

    base_model = MobileNetV2(weights=weights, include_top=False, input_shape=IMG_SIZE)
    base_model.trainable = False  # Freeze base layers

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(NUM_CLASSES, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    return model

def train_model(data_dir, model_save_path=None):
    """Trainiert das Modell."""
    timestamp = get_timestamp()
    if model_save_path is None:
        model_save_path = f'models/sign_language_model_{timestamp}.h5'
    
    print(f"\n=== Gebärdensprachen-Erkennung Training ===")
    print(f"Lade Daten von: {data_dir}")
    print(f"Klassen: {CLASSES}")
    print(f"Zeitstempel: {timestamp}\n")
    
    # Daten laden
    X, y = load_data(data_dir)
    print(f"Geladen: {len(X)} Bilder")
    
    # Split: nur Train + Validation (80/20)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.20, random_state=42)

    # Normalisieren
    X_train = X_train.astype('float32') / 255.0
    X_val = X_val.astype('float32') / 255.0

    print(f"Train: {len(X_train)}, Val: {len(X_val)}\n")

    # Modell bauen
    model = build_model()
    model.compile(optimizer=Adam(learning_rate=0.001), 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])
    
    # Callbacks
    os.makedirs('models', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    checkpoint = ModelCheckpoint(model_save_path, monitor='val_accuracy', save_best_only=True)
    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)
    csv_logger = CSVLogger(f'logs/training_metrics/epoch_metrics_{timestamp}.csv')

    # Training mit verbose=1 für Live-Output
    print("=== Training startet ===\n")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stop, checkpoint, lr_scheduler, csv_logger],
        verbose=1
    )
    
    # Speichern
    model.save(model_save_path)
    model.save('models/sign_language_model.keras')
    model.save('models/sign_language_model.h5')
    print(f"\n✓ Modell gespeichert: {model_save_path}")
    
    # Plot Trainingskurven
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



if __name__ == "__main__":
    import subprocess

    data_dir = 'data_cleaned'  # Bereinigte Daten verwenden
    model, history, model_save_path = train_model(data_dir)

    print("\n" + "="*60)
    print("🚀 Starte automatische finale Evaluation...")
    print("="*60)

    # Automatisch evaluate.py ausführen
    try:
        subprocess.run(
            ['python', 'evaluate.py', '--model', model_save_path],
            check=True
        )
    except subprocess.CalledProcessError as e:
        print(f"⚠️ Fehler bei der automatischen Evaluation: {e}")
        print(f"Führe manuell aus: python evaluate.py --model {model_save_path}")
