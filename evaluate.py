import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from data_loader import load_data
from sklearn.model_selection import train_test_split
from utils import get_timestamp
import os

CLASSES = ['A', 'B', 'C', 'L', 'V', 'W', 'O', 'Y']

def evaluate_model(model_path, data_dir):
    """
    Evaluiert das Modell auf Testdaten und speichert Confusion Matrix und Classification Report.
    """
    timestamp = get_timestamp()
    
    # Modell laden
    model = load_model(model_path)
    print(f"Modell geladen: {model_path}")

    # Daten laden (gleiche Split wie im Training)
    X, y = load_data(data_dir)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42)
    
    # Normalisieren
    X_test = X_test.astype('float32') / 255.0
    
    print(f"Testdaten: {len(X_test)} Bilder")

    # Predictions
    y_pred = model.predict(X_test, verbose=1)
    y_pred_classes = np.argmax(y_pred, axis=1)

    # Test Accuracy
    acc = accuracy_score(y_test, y_pred_classes)
    print(f"\nTest Accuracy: {acc:.4f}")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred_classes)
    plot_confusion_matrix(cm, timestamp)

    # Classification Report
    report = classification_report(y_test, y_pred_classes, target_names=CLASSES)
    save_classification_report(report, timestamp)
    
    print(f"\nClassification Report gespeichert: logs/classification_report_{timestamp}.txt")
    print(f"Confusion Matrix gespeichert: logs/confusion_matrix_{timestamp}.png")

    return acc

def plot_confusion_matrix(cm, timestamp):
    """
    Plottet und speichert Confusion Matrix mit matplotlib.
    """
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
            text = ax.text(j, i, cm[i, j], ha="center", va="center", 
                          color="white" if cm[i, j] > cm.max() / 2 else "black")
    
    ax.set_title('Confusion Matrix')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    plt.colorbar(im, ax=ax)
    
    os.makedirs('logs', exist_ok=True)
    plt.savefig(f'logs/confusion_matrix_{timestamp}.png', dpi=150)
    plt.close()

def save_classification_report(report, timestamp):
    """
    Speichert den Classification Report als TXT.
    """
    os.makedirs('logs', exist_ok=True)
    with open(f'logs/classification_report_{timestamp}.txt', 'w') as f:
        f.write(report)

if __name__ == "__main__":
    model_path = 'models/sign_language_model.h5'  # Oder der neueste mit Zeitstempel
    data_dir = 'data'
    evaluate_model(model_path, data_dir)