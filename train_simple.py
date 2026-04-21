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
    
    # Split
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42)
    
    # Normalisieren
    X_train = X_train.astype('float32') / 255.0
    X_val = X_val.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0
    
    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}\n")
    
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
    csv_logger = CSVLogger(f'logs/epoch_metrics_{timestamp}.csv')
    
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
    plt.savefig(f'logs/training_curves_{timestamp}.png', dpi=150)
    plt.close()
    print(f"✓ Trainingskurven gespeichert: logs/training_curves_{timestamp}.png")
    print(f"✓ Epoch-Metriken gespeichert: logs/epoch_metrics_{timestamp}.csv")
    
    # Automatische Evaluation auf Testdaten
    print("\n=== Evaluation auf Testdaten ===")
    evaluate_on_test_data(model, X_test, y_test, timestamp)
    
    return model, history, X_test, y_test

def evaluate_on_test_data(model, X_test, y_test, timestamp):
    """
    Evaluiert das Modell auf Testdaten und speichert Confusion Matrix + Classification Report.
    """
    # Predictions
    y_pred = model.predict(X_test, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    # Test Accuracy
    test_acc = np.mean(y_pred_classes == y_test)
    print(f"Test Accuracy: {test_acc:.4f}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred_classes)
    
    # Plot Confusion Matrix mit matplotlib
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, cmap='Blues')
    
    # Labels
    ax.set_xticks(np.arange(len(CLASSES)))
    ax.set_yticks(np.arange(len(CLASSES)))
    ax.set_xticklabels(CLASSES)
    ax.set_yticklabels(CLASSES)
    
    # Text annotations
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            text = ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
    
    ax.set_title('Confusion Matrix (Test Set)')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(f'logs/confusion_matrix_{timestamp}.png', dpi=150)
    plt.close()
    print(f"✓ Confusion Matrix gespeichert: logs/confusion_matrix_{timestamp}.png")
    
    # Classification Report
    report = classification_report(y_test, y_pred_classes, target_names=CLASSES)
    with open(f'logs/classification_report_{timestamp}.txt', 'w') as f:
        f.write(f"Sign Language Recognition - Classification Report\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Test Accuracy: {test_acc:.4f}\n")
        f.write("=" * 60 + "\n\n")
        f.write(report)
    print(f"✓ Classification Report gespeichert: logs/classification_report_{timestamp}.txt")

if __name__ == "__main__":
    data_dir = 'data_raw'  # ursprünglicher Datensatz mit allen Bildern
    model, history, X_test, y_test = train_model(data_dir)
    print("\n=== Training abgeschlossen ===")
    print("Alle Ergebnisse wurden mit Zeitstempel gespeichert.")
