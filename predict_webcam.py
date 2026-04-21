import os
import glob
import tempfile

os.environ.setdefault(
    "MPLCONFIGDIR",
    os.path.join(tempfile.gettempdir(), "mplconfig_signlang"),
)

import cv2
import numpy as np
from tensorflow.keras.models import load_model
from preprocess import preprocess_for_webcam

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

CLASSES = ['A', 'B', 'C', 'L', 'V', 'W', 'O', 'Y']

def find_model_candidates():
    """Findet Modell-Kandidaten im models/-Ordner, bevorzugt .keras."""
    keras_files = glob.glob('models/sign_language_model*.keras')
    h5_files = glob.glob('models/sign_language_model*.h5')
    model_files = keras_files + h5_files

    if not model_files:
        raise FileNotFoundError("Kein Modell in models/ gefunden (.keras oder .h5).")

    model_files.sort(key=os.path.getmtime, reverse=True)
    return model_files

# Global flag für Flip
flip_flag = [False]  # Als Liste, damit es in Funktionen modifizierbar ist

def detect_hand_bbox(frame, padding=0.2):
    """Ermittelt die Hand-Bounding-Box mit MediaPipe, sonst None."""
    if not mediapipe_available or mp_hands is None:
        return None

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = mp_hands.process(rgb)
    if not results.multi_hand_landmarks:
        return None

    h, w = frame.shape[:2]
    xs = [lm.x * w for lm in results.multi_hand_landmarks[0].landmark]
    ys = [lm.y * h for lm in results.multi_hand_landmarks[0].landmark]
    x_min = int(max(0, min(xs)))
    x_max = int(min(w, max(xs)))
    y_min = int(max(0, min(ys)))
    y_max = int(min(h, max(ys)))

    if x_max - x_min < 20 or y_max - y_min < 20:
        return None

    margin = int(max(x_max - x_min, y_max - y_min) * padding)
    x_min = max(0, x_min - margin)
    y_min = max(0, y_min - margin)
    x_max = min(w, x_max + margin)
    y_max = min(h, y_max + margin)

    return x_min, y_min, x_max, y_max

def predict_frame(model, frame, flip=False):
    """
    Macht Prediction für einen Frame und gibt nur die beste Klasse zurück.
    """
    processed = preprocess_for_webcam(frame, flip=flip)
    pred = model.predict(processed, verbose=0)[0]
    best_idx = np.argmax(pred)
    pred_class = CLASSES[best_idx]
    prob = pred[best_idx]
    return pred_class, prob

def draw_predictions(frame, pred_class, prob, mediapipe_active):
    color = (0, 255, 0) if mediapipe_active else (0, 165, 255)
    mode_text = "MediaPipe: ON" if mediapipe_active else "MediaPipe: OFF"

    cv2.putText(frame, mode_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv2.putText(frame, f"{pred_class}: {prob:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return frame

def save_frame(frame, filename):
    """
    Speichert Frame für Debugging.
    """
    os.makedirs('saved_frames', exist_ok=True)
    cv2.imwrite(os.path.join('saved_frames', filename), frame)

def main():
    last_error = None
    model = None
    model_path = None
    for candidate in find_model_candidates():
        try:
            model = load_model(candidate, compile=False)
            model_path = candidate
            break
        except Exception as exc:
            last_error = exc
            print(f"Modell konnte nicht geladen werden: {candidate}")

    if model is None:
        raise RuntimeError(f"Kein Modell ladbar. Letzter Fehler: {last_error}")

    print(f"Verwende Modell: {model_path}")
    if not mediapipe_available:
        print("MediaPipe nicht verfügbar. Fallback: Vollbild-Prediction ohne Hand-Crop.")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Webcam nicht gefunden.")
        return

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        bbox = detect_hand_bbox(frame)
        mediapipe_active_now = bbox is not None
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            cropped_frame = frame[y1:y2, x1:x2]
            if cropped_frame.size == 0:
                cropped_frame = frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        else:
            cropped_frame = frame

        pred_class, prob = predict_frame(
            model, cropped_frame, flip=flip_flag[0]
        )
        frame = draw_predictions(frame, pred_class, prob, mediapipe_active_now)

        cv2.imshow('Sign Language Recognition', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):  # Save frame
            save_frame(frame, f"frame_{frame_count}.jpg")
            frame_count += 1
        elif key == ord('f'):  # Toggle flip
            flip_flag[0] = not flip_flag[0]
            print(f"Flip toggled: {flip_flag[0]}")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
