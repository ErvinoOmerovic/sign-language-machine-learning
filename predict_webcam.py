"""
Echtzeitvorhersage über Webcam für Gebärdensprachen-Buchstaben

Dieses Modul nutzt die Webcam um live Gebärdensprachen-Buchstaben zu erkennen.
Es lädt ein trainiertes Modell und macht Vorhersagen für jeden Kamera-Frame.

Optional ist MediaPipe-Integration für Hand-Segmentierung verfügbar,
um bessere Fokussierung auf die Handgesten zu erreichen.

Steuerung:
- q: Beenden
- s: Frame speichern (für Debugging)
- f: Flip toggeln (Webcam-Spiegelung)
"""

import os
import glob
import tempfile

# Matplotlib-Config vor Imports setzen (für Headless Umgebungen)
os.environ.setdefault(
    "MPLCONFIGDIR",
    os.path.join(tempfile.gettempdir(), "mplconfig_signlang"),
)

import cv2
import numpy as np
from tensorflow.keras.models import load_model
from preprocess import preprocess_for_webcam

# ============================================================================
# MediaPipe Setup (optional für Hand-Tracking)
# ============================================================================
try:
    import mediapipe as mp
    mp_hands_module = mp.solutions.hands
    mp_hands = mp_hands_module.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    mediapipe_available = True
except Exception as e:
    print(f"MediaPipe nicht verfügbar: {e}")
    mp = None
    mp_hands = None
    mediapipe_available = False

# Gebärdensprachen-Klassen
CLASSES = ['A', 'B', 'C', 'L', 'V', 'W', 'O', 'Y']

# ============================================================================
# Modell-Auswahl
# ============================================================================
def find_model_candidates():
    """
    Findet alle verfügbaren Modell-Dateien im models/-Ordner.

    Sucht nach .keras und .h5 Dateien (bei Namensstart 'sign_language_model*').
    Sortiert sie nach Änderungszeit (neueste zuerst).

    Returns:
        list: Sortierte Liste von Modell-Datei-Pfaden (neueste zuerst)

    Raises:
        FileNotFoundError: Falls keine Modell-Dateien gefunden
    """
    # Suche nach .keras und .h5 Dateien
    keras_files = glob.glob('models/sign_language_model*.keras')
    h5_files = glob.glob('models/sign_language_model*.h5')
    model_files = keras_files + h5_files

    if not model_files:
        raise FileNotFoundError("Kein Modell in models/ gefunden (.keras oder .h5).")

    # Sortiere nach Änderungszeit (neueste zuerst)
    model_files.sort(key=os.path.getmtime, reverse=True)
    return model_files

# ========== Global Flip-Flag ==========
# Wird mit 'f' toggelt während Webcam läuft
flip_flag = [False]

# ============================================================================
# Hand-Erkennung (MediaPipe)
# ============================================================================
def detect_hand_bbox(frame, padding=0.2):
    """
    Erkennt Hand-Position mittels MediaPipe und gibt Bounding-Box zurück.

    MediaPipe erkennt Hand-Landmarks (21 Gelenke pro Hand).
    Aus diesen wird eine Bounding-Box(x_min, y_min, x_max, y_max) berechnet,
    mit optionalem Padding für Kontext.

    Args:
        frame (np.ndarray): Kamera-Frame (BGR)
        padding (float): Relative Padding um Hand herum (0.2 = 20%)

    Returns:
        tuple or None: (x_min, y_min, x_max, y_max) oder None falls keine Hand erkannt
    """
    if not mediapipe_available or mp_hands is None:
        return None

    # Konvertiere von BGR zu RGB für MediaPipe
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = mp_hands.process(rgb)

    if not results.multi_hand_landmarks:
        return None

    # Extrahiere Hand-Landmarks
    h, w = frame.shape[:2]
    xs = [lm.x * w for lm in results.multi_hand_landmarks[0].landmark]
    ys = [lm.y * h for lm in results.multi_hand_landmarks[0].landmark]

    # Berechne Bounding-Box
    x_min = int(max(0, min(xs)))
    x_max = int(min(w, max(xs)))
    y_min = int(max(0, min(ys)))
    y_max = int(min(h, max(ys)))

    # Ignoriere zu kleine Hände
    if x_max - x_min < 20 or y_max - y_min < 20:
        return None

    # Füge Padding hinzu
    margin = int(max(x_max - x_min, y_max - y_min) * padding)
    x_min = max(0, x_min - margin)
    y_min = max(0, y_min - margin)
    x_max = min(w, x_max + margin)
    y_max = min(h, y_max + margin)

    return x_min, y_min, x_max, y_max

# ============================================================================
# Vorhersage
# ============================================================================
def predict_frame(model, frame, flip=False):
    """
    Macht Vorhersage für einen Kamera-Frame.

    Der Frame wird vorverarbeitet (resize, normalisierung),
    an das Modell übergeben und die beste Klasse + Konfidenz zurückgegeben.

    Args:
        model: Geladenes Keras-Modell
        frame (np.ndarray): Kamera-Frame (BGR)
        flip (bool): Ob Frame horizontal flippen vor Verarbeitung

    Returns:
        tuple: (predicted_class, probability)
               z.B. ('A', 0.95)
    """
    # Vorverarbeitung
    processed = preprocess_for_webcam(frame, flip=flip)

    # Vorhersage
    pred = model.predict(processed, verbose=0)[0]  # Batch von 1

    # Extrahiere beste Klasse
    best_idx = np.argmax(pred)
    pred_class = CLASSES[best_idx]
    prob = pred[best_idx]

    return pred_class, prob

# ============================================================================
# Visualisierung
# ============================================================================
def draw_predictions(frame, pred_class, prob, mediapipe_active):
    """
    Zeichnet Vorhersage und Status auf den Frame.

    Args:
        frame (np.ndarray): Ausgabe-Frame
        pred_class (str): Vorhergesagte Klasse
        prob (float): Konfidenzwert (0-1)
        mediapipe_active (bool): Ob MediaPipe gerade aktiv ist

    Returns:
        np.ndarray: Frame mit gezeichneten Texten
    """
    color = (0, 255, 0) if mediapipe_active else (0, 165, 255)
    mode_text = "MediaPipe: ON" if mediapipe_active else "MediaPipe: OFF"

    # MediaPipe-Status
    cv2.putText(frame, mode_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # Vorhersage
    cv2.putText(frame, f"{pred_class}: {prob:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return frame

def save_frame(frame, filename):
    """
    Speichert einen Frame für Debugging-Zwecke.

    Args:
        frame (np.ndarray): Frame zum speichern
        filename (str): Dateiname (z.B. 'frame_0.jpg')
    """
    os.makedirs('saved_frames', exist_ok=True)
    cv2.imwrite(os.path.join('saved_frames', filename), frame)

# ============================================================================
# HAUPTPROGRAMM
# ============================================================================
def main():
    """
    Hauptschleife für Webcam-Echtzeit-Vorhersage.

    Workflow:
    1. Suche und lade das neueste trainierte Modell
    2. Öffne Webcam
    3. Für jeden Frame:
       a. Versuche Hand zu erkennen (optional MediaPipe)
       b. Mache Vorhersage auf Hand-Region oder Vollbild
       c. Zeichne Vorhersage auf Frame
       d. Zeige Frame an
       e. Verarbeite Tastenanschläge
    4. Beende bei 'q'
    """
    # ========== Modell laden ==========
    last_error = None
    model = None
    model_path = None
    for candidate in find_model_candidates():
        try:
            # Versuche Modell zu laden
            model = load_model(candidate, compile=False)
            model_path = candidate
            break
        except Exception as exc:
            last_error = exc
            print(f"Modell konnte nicht geladen werden: {candidate}")

    if model is None:
        raise RuntimeError(f"Kein Modell ladbar. Letzter Fehler: {last_error}")

    print(f"Verwende Modell: {model_path}")

    # MediaPipe-Status anzeigen
    if not mediapipe_available:
        print("MediaPipe nicht verfügbar. Fallback: Vollbild-Prediction ohne Hand-Crop.")

    # ========== Webcam öffnen ==========
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Webcam nicht gefunden.")
        return

    frame_count = 0

    # ========== HAUPTSCHLEIFE ==========
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # ========== Hand-Erkennung ==========
        bbox = detect_hand_bbox(frame)
        mediapipe_active_now = bbox is not None

        if bbox is not None:
            x1, y1, x2, y2 = bbox
            cropped_frame = frame[y1:y2, x1:x2]
            # Fallback falls Crop fehlschlägt
            if cropped_frame.size == 0:
                cropped_frame = frame
            # Zeichne Bounding-Box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        else:
            # Kein Hand erkannt: nutze Vollbild
            cropped_frame = frame

        # ========== Vorhersage ==========
        pred_class, prob = predict_frame(
            model, cropped_frame, flip=flip_flag[0]
        )

        # Zeichne auf Frame
        frame = draw_predictions(frame, pred_class, prob, mediapipe_active_now)

        # Zeige Frame an
        cv2.imshow('Sign Language Recognition', frame)

        # ========== Tastenanschläge ==========
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            # Beenden
            break
        elif key == ord('s'):
            # Frame speichern
            save_frame(frame, f"frame_{frame_count}.jpg")
            print(f"Frame gespeichert: saved_frames/frame_{frame_count}.jpg")
            frame_count += 1
        elif key == ord('f'):
            # Flip toggeln
            flip_flag[0] = not flip_flag[0]
            print(f"Flip toggled: {flip_flag[0]}")

    # ========== Cleanup ==========
    cap.release()
    cv2.destroyAllWindows()

# ============================================================================
# EINSTIEG
# ============================================================================
if __name__ == "__main__":
    main()
