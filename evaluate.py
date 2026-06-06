"""
Evaluationsmodul für finale Test-Performance

Dieses Modul lädt ein trainiertes Modell und evaluiert es auf externe Testdaten (external_test/).
Die external_test-Daten werden NICHT für Training verwendet.
Sie enthalten echte, ungesehene Test-Bilder aus mehreren Quellen (dataset2, dataset3, ...).

Das Skript kann standalone ausgeführt oder automatisch von train_simple.py aufgerufen werden.
"""

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from utils import get_timestamp
import os
import cv2
import argparse
from pathlib import Path

# Gebärdensprachen-Klassen (muss mit Training übereinstimmen)
CLASSES = ['A', 'B', 'C', 'L', 'V', 'W', 'O', 'Y']

# ============================================================================
# DATEN-LADEN
# ============================================================================
def load_external_test_data(external_test_dir='external_test'):
    """
    Lädt alle Bilder aus external_test/ (finale Testdaten).

    Die external_test-Daten werden NICHT für Training verwendet.
    Sie enthalten echte, ungesehene Test-Bilder aus mehreren Quellen (dataset2, dataset3, ...).

    Args:
        external_test_dir (str): Pfad zum Ordner mit externen Test-Datensätzen

    Returns:
        tuple: (X, y) - Arrays mit Testbildern und Labels (0-7 für 8 Klassen)

    Raises:
        FileNotFoundError: Falls external_test/ nicht existiert
    """
    X = []
    y = []

    # Prüfe ob external_test-Ordner existiert
    if not os.path.exists(external_test_dir):
        raise FileNotFoundError(f"Ordner {external_test_dir} nicht gefunden!")

    # Finde alle Datensätze in external_test/ (dataset2, dataset3, etc.)
    dataset_dirs = [d for d in os.listdir(external_test_dir)
                    if os.path.isdir(os.path.join(external_test_dir, d))]

    print(f"Gefundene externe Datensätze: {dataset_dirs}")

    # Lade Bilder aus jedem Datensatz
    for dataset_name in sorted(dataset_dirs):
        dataset_path = os.path.join(external_test_dir, dataset_name)
        print(f"\nLade Datensatz: {dataset_name}")

        # Durchsuche alle Klassen im Datensatz
        for idx, cls in enumerate(CLASSES):
            cls_dir = os.path.join(dataset_path, cls)
            if not os.path.exists(cls_dir):
                print(f"  Warnung: {cls_dir} existiert nicht")
                continue

            img_count = 0
            for img_file in os.listdir(cls_dir):
                # Ignoriere Hidden Files (z.B. .DS_Store)
                if img_file.startswith('.'):
                    continue
                img_path = os.path.join(cls_dir, img_file)
                img = cv2.imread(img_path)
                if img is not None:
                    # Resize auf Standard-Grösse und in RGB konvertieren
                    img = cv2.resize(img, (224, 224))
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    X.append(img)
                    y.append(idx)
                    img_count += 1

            print(f"  Klasse {cls}: {img_count} Bilder")

    X = np.array(X)
    y = np.array(y)
    print(f"\nGesamt externe Testdaten: {len(X)} Bilder")
    return X, y

# ============================================================================
# EVALUATION
# ============================================================================
def evaluate_model(model_path, external_test_dir='external_test'):
    """
    Evaluiert das Modell auf EXTERNEN Testdaten.

    Dies ist die FINALE EVALUATION mit allen separaten, unsichtbaren Testdatensätzen.
    Metriken werden berechnet und visualisiert:
    - Gesamtaccuracy, Precision, Recall, F1-Score
    - Confusion Matrix (grafisch)
    - Classification Report (pro Klasse + gesamt)

    Alle Ergebnisse werden in den logs/ Ordner(-struktur) gespeichert.

    Args:
        model_path (str): Pfad zum trainierten Modell (.h5 oder .keras)
        external_test_dir (str): Pfad zum external_test Ordner

    Returns:
        float: Accuracy auf externen Testdaten
    """
    timestamp = get_timestamp()
    
    # ========== Modell laden ==========
    model = load_model(model_path)
    print(f"Modell geladen: {model_path}")
    print("\n" + "="*60)
    print("🧪 FINALE EVALUATION auf EXTERNEN Testdaten")
    print("="*60)

    # ========== Externe Test-Daten laden ==========
    # Wichtig: NICHT aus data_raw! Nur aus external_test (ungesehene Daten)
    X_test, y_test = load_external_test_data(external_test_dir)

    # ========== Normalisierung ==========
    # Skaliere Pixel-Werte von [0, 255] auf [0, 1] (wie beim Training)
    X_test = X_test.astype('float32') / 255.0
    
    print(f"\n✓ Externe Testdaten bereit: {len(X_test)} Bilder")

    # ========== Vorhersagen ==========
    print("\nMache Predictions...")
    y_pred = model.predict(X_test, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)  # Argmax: beste Klasse pro Bild

    # ========== Metriken Berechnung ==========
    acc = accuracy_score(y_test, y_pred_classes)
    precision = precision_score(y_test, y_pred_classes, average='weighted')
    recall = recall_score(y_test, y_pred_classes, average='weighted')
    f1 = f1_score(y_test, y_pred_classes, average='weighted')

    print("\n" + "="*60)
    print("📊 FINALE ERGEBNISSE")
    print("="*60)
    print(f"Test Accuracy:  {acc:.4f}")
    print(f"Precision:      {precision:.4f}")
    print(f"Recall:         {recall:.4f}")
    print(f"F1-Score:       {f1:.4f}")

    # ========== Confusion Matrix ==========
    cm = confusion_matrix(y_test, y_pred_classes)
    plot_confusion_matrix(cm, timestamp)

    # ========== Classification Report ==========
    report = classification_report(y_test, y_pred_classes, target_names=CLASSES)
    save_classification_report(report, timestamp, acc, precision, recall, f1)

    print(f"\n✓ Classification Report gespeichert: logs/classification_reports/classification_report_{timestamp}.txt")
    print(f"✓ Confusion Matrix gespeichert: logs/confusion_matrices/confusion_matrix_{timestamp}.png")
    print("="*60)

    return acc

# ============================================================================
# VISUALISIERUNG UND SPEICHERUNG
# ============================================================================
def plot_confusion_matrix(cm, timestamp):
    """
    Plottet und speichert die Confusion Matrix als Heatmap.

    Die Confusion Matrix zeigt, wie oft jede Klasse korrekt erkannt wird
    und wie oft sie mit anderen Klassen verwechselt wird.

    Args:
        cm (np.ndarray): Confusion Matrix (8x8 für 8 Klassen)
        timestamp (str): Zeitstempel für Dateinamen
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    # Heatmap mit Blues Colormap (dunkel = viele Anzahl)
    im = ax.imshow(cm, cmap='Blues')
    
    # ========== Achsen-Labels ==========
    ax.set_xticks(np.arange(len(CLASSES)))
    ax.set_yticks(np.arange(len(CLASSES)))
    ax.set_xticklabels(CLASSES)
    ax.set_yticklabels(CLASSES)
    
    # ========== Text-Annotationen (Zahlenwerte) ==========
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            # Wähle Textfarbe je nach Hintergrund-Helligkeit
            text = ax.text(j, i, cm[i, j], ha="center", va="center",
                          color="white" if cm[i, j] > cm.max() / 2 else "black",
                          fontsize=10, fontweight='bold')

    ax.set_title('Confusion Matrix - External Test Data', fontsize=14, fontweight='bold')
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('True', fontsize=12)
    plt.colorbar(im, ax=ax)
    plt.tight_layout()

    # Speichere Plot
    os.makedirs('logs/confusion_matrices', exist_ok=True)
    plt.savefig(f'logs/confusion_matrices/confusion_matrix_{timestamp}.png', dpi=150)
    plt.close()

def save_classification_report(report, timestamp, acc=None, precision=None, recall=None, f1=None):
    """
    Speichert den Classification Report als Text-Datei.

    Der Report enthält:
    - Precision, Recall, F1-Score pro Klasse
    - Support (Anzahl Test-Samples pro Klasse)
    - Gesamtmetriken (gewichtet)

    Args:
        report (str): Classification Report von sklearn
        timestamp (str): Zeitstempel für Dateinamen
        acc (float, optional): Accuracy gesamt
        precision (float, optional): Precision gesamt (gewichtet)
        recall (float, optional): Recall gesamt (gewichtet)
        f1 (float, optional): F1-Score gesamt (gewichtet)
    """
    os.makedirs('logs/classification_reports', exist_ok=True)
    with open(f'logs/classification_reports/classification_report_{timestamp}.txt', 'w') as f:
        # Header
        f.write("="*60 + "\n")
        f.write("EXTERNE TEST EVALUATION (Final Test Set)\n")
        f.write("="*60 + "\n\n")

        # Zusammenfassung
        if acc is not None:
            f.write(f"Accuracy:  {acc:.4f}\n")
            f.write(f"Precision: {precision:.4f}\n")
            f.write(f"Recall:    {recall:.4f}\n")
            f.write(f"F1-Score:  {f1:.4f}\n")
            f.write("-"*60 + "\n\n")

        # Detail-Report pro Klasse
        f.write(report)

# ============================================================================
# HAUPTEINSTIEG
# ============================================================================
if __name__ == "__main__":
    # ========== Command Line Arguments ==========
    parser = argparse.ArgumentParser(description='Final evaluation on external test sets')
    parser.add_argument('--model', type=str, default='models/sign_language_model.h5',
                        help='Pfad zum Modell (.h5 oder .keras). Standard: models/sign_language_model.h5')
    parser.add_argument('--external_test', type=str, default='external_test',
                        help='Pfad zum external_test Ordner')
    args = parser.parse_args()

    model_path = args.model
    external_test_dir = args.external_test

    # Evaluiere Modell
    evaluate_model(model_path, external_test_dir)
