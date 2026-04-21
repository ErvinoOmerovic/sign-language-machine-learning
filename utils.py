import cv2
import numpy as np
from preprocess import preprocess_for_webcam
from tensorflow.keras.models import load_model
import os
from datetime import datetime

CLASSES = ['A', 'B', 'C', 'L', 'V', 'W', 'O', 'Y']

def get_timestamp():
    """Gibt einen Zeitstempel im Format YYYY-MM-DD_HH-MM-SS zurück."""
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def compare_with_training_images(model, frame, data_dir, flip=False):
    """
    Vergleicht Prediction mit ähnlichen Trainingsbildern.
    Einfache Implementierung: Zeige Bilder der vorhergesagten Klasse.
    """
    processed = preprocess_for_webcam(frame, flip=flip)
    pred = model.predict(processed, verbose=0)[0]
    pred_class = CLASSES[np.argmax(pred)]

    # Lade einige Bilder der Klasse
    cls_dir = os.path.join(data_dir, pred_class)
    if os.path.exists(cls_dir):
        images = os.listdir(cls_dir)[:5]  # Erste 5
        for img_file in images:
            img_path = os.path.join(cls_dir, img_file)
            img = cv2.imread(img_path)
            if img is not None:
                cv2.imshow(f"Training {pred_class}: {img_file}", img)
                cv2.waitKey(1000)  # 1 Sekunde pro Bild
                cv2.destroyWindow(f"Training {pred_class}: {img_file}")

def test_single_image(model, img_path, flip=False):
    """
    Testet ein einzelnes Bild.
    """
    img = cv2.imread(img_path)
    if img is None:
        print("Bild nicht gefunden.")
        return
    pred_class, prob, _, _ = predict_frame(model, img, flip=flip)
    print(f"Prediction: {pred_class} with {prob:.2f}")
    cv2.imshow('Test Image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def predict_frame(model, frame, flip=False):
    """
    Hilfsfunktion für Prediction.
    """
    processed = preprocess_for_webcam(frame, flip=flip)
    pred = model.predict(processed, verbose=0)[0]
    top_indices = np.argsort(pred)[-3:][::-1]
    top_classes = [CLASSES[i] for i in top_indices]
    top_probs = [pred[i] for i in top_indices]
    return top_classes[0], top_probs[0], top_classes, top_probs