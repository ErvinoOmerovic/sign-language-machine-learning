"""
Utility-Funktionen für Modell-Testing, Prediction und Debugging

Dieses Modul enthält hilfreiche Funktionen für:
- Zeitstempel-Generierung
- Single-Image-Testing
- Prediction mit Top-N Klassen
- Vergleich mit Trainingsbildern (Debug)
"""

import cv2
import numpy as np
from preprocess import preprocess_for_webcam
from tensorflow.keras.models import load_model
import os
from datetime import datetime

# Gebärdensprachen-Klassen
CLASSES = ['A', 'B', 'C', 'L', 'V', 'W', 'O', 'Y']

# ============================================================================
# ZEITSTEMPEL UND LOGGING
# ============================================================================

def get_timestamp():
    """
    Generiert einen Zeitstempel im Format YYYY-MM-DD_HH-MM-SS.

    Verwendung: Für eindeutige Benennungskonventionen von Modellen, Logs, etc.

    Returns:
        str: Zeitstempel (z.B. '2026-06-05_14-32-45')
    """
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# ============================================================================
# DEBUGGING UND VERGLEICH
# ============================================================================

def compare_with_training_images(model, frame, data_dir, flip=False):
    """
    Vergleicht Vorhersage mit ähnlichen Trainingsbildern (Debug-Funktion).

    Workflow:
    1. Mache Vorhersage auf Input-Frame
    2. Lade einige Beispielbilder der vorhergesagten Klasse
    3. Zeige sie der Reihe nach an (je 1 Sekunde)

    Hinweis: Dies ist eine einfache Debug-Funktion, nicht für Produktion.

    Args:
        model: Geladenes Modell
        frame (np.ndarray): Webcam-Frame
        data_dir (str): Pfad zu Trainings-Datenordner
        flip (bool): Ob Frame flippen vor Preprocessing
    """
    # Mache Vorhersage
    processed = preprocess_for_webcam(frame, flip=flip)
    pred = model.predict(processed, verbose=0)[0]
    pred_class = CLASSES[np.argmax(pred)]

    # Lade einige Trainingsbilder der vorhergesagten Klasse
    cls_dir = os.path.join(data_dir, pred_class)
    if os.path.exists(cls_dir):
        images = os.listdir(cls_dir)[:5]  # Erste 5 Bilder
        for img_file in images:
            img_path = os.path.join(cls_dir, img_file)
            img = cv2.imread(img_path)
            if img is not None:
                # Zeige Bild
                cv2.imshow(f"Training {pred_class}: {img_file}", img)
                cv2.waitKey(1000)  # 1 Sekunde Anzeigedauer
                cv2.destroyWindow(f"Training {pred_class}: {img_file}")

# ============================================================================
# PREDICTION
# ============================================================================

def test_single_image(model, img_path, flip=False):
    """
    Testet ein einzelnes Bild und zeigt Vorhersage an.

    Args:
        model: Geladenes Modell
        img_path (str): Pfad zu Test-Bild
        flip (bool): Ob Bild flippen vor Vorhersage
    """
    # Lade Bild
    img = cv2.imread(img_path)
    if img is None:
        print("Bild nicht gefunden.")
        return

    # Mache Vorhersage
    pred_class, prob, _, _ = predict_frame(model, img, flip=flip)

    # Ausgabe
    print(f"Prediction: {pred_class} with {prob:.2f}")

    # Zeige Bild
    cv2.imshow('Test Image', img)
    cv2.waitKey(0)  # Warte auf Tastendruck
    cv2.destroyAllWindows()


def predict_frame(model, frame, flip=False):
    """
    Macht Vorhersage auf einem Frame und gibt Top-3 Klassen zurück.

    Workflow:
    1. Vorverarbeite Frame (Resize, Normalisierung)
    2. Mache Vorhersage mit Modell
    3. Extrahiere Top-3 Klassen + Confidence

    Args:
        model: Geladenes Keras-Modell
        frame (np.ndarray): Input-Frame (BGR)
        flip (bool): Ob horizontal flippen

    Returns:
        tuple: (top_class, top_prob, top_classes_list, top_probs_list)
               - top_class: beste Klasse (str)
               - top_prob: Konfidenz beste Klasse (float 0-1)
               - top_classes_list: Top-3 Klassen
               - top_probs_list: Top-3 Konfidenzwerte
    """
    # Vorverarbeite Frame
    processed = preprocess_for_webcam(frame, flip=flip)

    # Mache Vorhersage (Output: Wahrscheinlichkeiten für jede Klasse)
    pred = model.predict(processed, verbose=0)[0]

    # Extrahiere Top-3 Klassen
    # argsort: sortiert indizes nach Wert, [-3:] = letzte 3 (höchste), [::-1] = umgekehrte Reihenfolge
    top_indices = np.argsort(pred)[-3:][::-1]
    top_classes = [CLASSES[i] for i in top_indices]
    top_probs = [pred[i] for i in top_indices]

    return top_classes[0], top_probs[0], top_classes, top_probs