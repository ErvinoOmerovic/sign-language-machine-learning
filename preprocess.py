"""
Preprocessing-Modul für Bilder und Webcam-Frames

Dieses Modul enthält Funktionen für konsistente Vorverarbeitung von Bildern
vor der Modell-Vorhersage (Resize, Normalisierung, Flip-Handling).
"""

import cv2
import numpy as np

# Standard-Bildgrösse (muss mit Trainingsmodell übereinstimmen)
IMG_SIZE = (224, 224)

def preprocess_image(img, flip=False):
    """
    Vorverarbeitet ein Bild für Modell-Vorhersage.

    Schritte:
    1. Optional: Horizontales Flip (für Webcam-Spiegelung)
    2. Resize auf Standard-Grösse (224x224)
    3. BGR→RGB Konvertierung
    4. Normalisierung (Pixel 0-1)
    5. Batch-Dimensionalität hinzufügen (1, 224, 224, 3)

    Args:
        img (np.ndarray): Input-Bild (BGR, beliebige Grösse)
        flip (bool): Ob horizontal flippen

    Returns:
        np.ndarray: Vorverarbeitetes Bild mit Batch-Dimension (1, 224, 224, 3)
    """
    # Optional: Flip für Webcam-Spiegelung
    if flip:
        img = cv2.flip(img, 1)  # Horizontales Flip (1 = horizontal)

    # Resize auf Standard-Grösse
    img = cv2.resize(img, IMG_SIZE)

    # BGR (OpenCV) zu RGB konvertieren
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Normalisiere Pixel-Werte: [0, 255] → [0, 1]
    img = img.astype('float32') / 255.0

    # Füge Batch-Dimension hinzu für Modell-Input
    # Shape: (H, W, C) → (1, H, W, C) für Batch-Verarbeitung
    img = np.expand_dims(img, axis=0)

    return img

def preprocess_for_webcam(frame, flip=False):
    """
    Wrapper für Webcam-Frame-Preprocessing.

    Kurz: ruft preprocess_image auf (könnte später erweitert werden
    für webcam-spezifische Transformationen).

    Args:
        frame (np.ndarray): Webcam-Frame (BGR)
        flip (bool): Ob horizontal flippen

    Returns:
        np.ndarray: Vorverarbeiteter Frame mit Batch-Dimension
    """
    return preprocess_image(frame, flip=flip)