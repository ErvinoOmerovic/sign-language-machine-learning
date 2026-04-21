import tensorflow as tf
from keras.applications import MobileNetV2
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import os
from data_loader import create_data_generators, load_data, split_data

CLASSES = ['A', 'B', 'C', 'L', 'V', 'W', 'O', 'Y']
NUM_CLASSES = len(CLASSES)
IMG_SIZE = (224, 224, 3)
BATCH_SIZE = 32
EPOCHS = 40

def build_model():
    """
    Baut das Modell mit Transfer Learning (MobileNetV2).
    """
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=IMG_SIZE)
    base_model.trainable = False  # Freeze base layers

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(NUM_CLASSES, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    return model

def train_model(data_dir, model_save_path='models/sign_language_model.h5'):
    """
    Trainiert das Modell.
    """
    # Daten laden und splitten
    X, y = load_data(data_dir)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    train_gen, val_gen, test_gen = create_data_generators(X_train, y_train, X_val, y_val, X_test, y_test)

    # Modell bauen
    model = build_model()
    model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Callbacks
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    checkpoint = ModelCheckpoint(model_save_path, monitor='val_accuracy', save_best_only=True)
    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)

    # Training - verbose=1 für Live-Output
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=[early_stop, checkpoint, lr_scheduler],
        verbose=1
    )

    # Speichere finales Modell
    model.save(model_save_path)

    # Plot Trainingskurven
    plot_history(history)

    return model, history

def plot_history(history):
    """
    Plottet Loss und Accuracy.
    """
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title('Accuracy')
    plt.legend()

    plt.savefig('logs/training_curves.png')
    plt.show()

if __name__ == "__main__":
    import sys
    print(f"Python Version: {sys.version}")
    print(f"TensorFlow: {tf.__version__}")
    
    data_dir = 'data'
    os.makedirs('models', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    print(f"\n=== Gebärdensprachen-Erkennung Training ===")
    print(f"Lade Daten von: {data_dir}")
    print(f"Klassen: {CLASSES}")
    
    train_model(data_dir)