import cv2
import numpy as np

IMG_SIZE = (224, 224)

def preprocess_image(img, flip=False):
    """
    Preprocessing für ein einzelnes Bild: Resize, RGB, Normalisierung.
    Optional: Horizontal Flip für Spiegelproblem.
    """
    if flip:
        img = cv2.flip(img, 1)  # Horizontal flip
    img = cv2.resize(img, IMG_SIZE)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)  # Für Modell-Input
    return img

def preprocess_for_webcam(frame, flip=False):
    """
    Preprocessing für Webcam-Frame.
    """
    return preprocess_image(frame, flip=flip)