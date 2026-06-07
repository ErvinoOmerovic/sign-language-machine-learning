"""
Webcam-Bilderfassung für Custom Gebärdensprachen-Datensatz

Dieses Skript ermöglicht die Erfassung von Webam-Bildern für einzelne
Gebärdensprachen-Klassen. Die Bilder werden direkt in die bestehende
Datenstruktur (data_raw/<KLASSE>/) gespeichert.

Steuerung:
- s: Einzelnes Bild speichern (manueller Modus)
- a: Auto-Modus starten/stoppen (alle 0.3 Sekunden ein Bild)
- f: Flip toggeln (Webcam-Spiegelung)
- q: Beenden
"""

import os
import cv2
import time

# Verfügbare Klassen
CLASSES = ['A', 'B', 'C', 'L', 'V', 'W', 'O', 'Y']

# ROI-Größe (Rechteck für Handposition)
ROI_WIDTH = 300
ROI_HEIGHT = 300

# Farben (BGR)
COLOR_ROI = (0, 255, 0)      # Grün für ROI-Rechteck
COLOR_TEXT = (255, 255, 255) # Weiß für Text
COLOR_STATUS = (0, 165, 255) # Orange für Status


def get_class_from_user():
    """
    Fragt den Benutzer nach der Klasse, für die Bilder aufgenommen werden sollen.

    Returns:
        str: Die ausgewählte Klasse (z.B. 'V')
    """
    print("\n" + "="*60)
    print("WEBCAM-BILDERFASSUNG für Gebärdensprachen")
    print("="*60)
    print("\nVerfügbare Klassen:")
    for i, cls in enumerate(CLASSES, 1):
        print(f"  {i}. {cls}")

    while True:
        user_input = input("\nWähle eine Klasse (Buchstabe oder Nummer): ").strip().upper()

        # Akzeptiere Buchstaben oder Nummern
        if user_input in CLASSES:
            return user_input

        try:
            idx = int(user_input) - 1
            if 0 <= idx < len(CLASSES):
                return CLASSES[idx]
        except ValueError:
            pass

        print("Ungültige Eingabe! Bitte versuche es erneut.")


def get_next_image_number(class_dir):
    """
    Findet die nächste Bildnummer für eine Klasse.

    Sucht nach bestehenden Dateien im Format '<KLASSE>_webcam_<NR>.png'
    und gibt die nächste verfügbare Nummer zurück.

    Args:
        class_dir (str): Pfad zum Klassen-Ordner

    Returns:
        int: Die nächste Bildnummer
    """
    if not os.path.exists(class_dir):
        return 1

    # Finde alle Bilder in diesem Ordner
    max_num = 0
    for filename in os.listdir(class_dir):
        if filename.endswith('.png') or filename.endswith('.jpg'):
            # Versuche Nummer aus dem Dateinamen zu extrahieren
            parts = filename.split('_')
            if len(parts) >= 3:
                try:
                    num = int(parts[-1].split('.')[0])
                    max_num = max(max_num, num)
                except ValueError:
                    pass

    return max_num + 1


def draw_roi(frame, roi_x, roi_y):
    """
    Zeichnet das ROI-Rechteck auf den Frame.

    Args:
        frame (np.ndarray): Der Kamera-Frame
        roi_x (int): X-Koordinate der oberen linken Ecke des ROI
        roi_y (int): Y-Koordinate der oberen linken Ecke des ROI

    Returns:
        np.ndarray: Frame mit gezeichnetem ROI
    """
    # Berechne Eckpunkte des Rechtecks
    x1, y1 = roi_x, roi_y
    x2, y2 = roi_x + ROI_WIDTH, roi_y + ROI_HEIGHT

    # Zeichne Rechteck
    cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_ROI, 2)

    # Zeichne Eckmarkierungen (kleine Viereck an den Ecken)
    corner_size = 10
    corners = [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]
    for cx, cy in corners:
        cv2.rectangle(frame, (cx - corner_size, cy - corner_size),
                     (cx + corner_size, cy + corner_size), COLOR_ROI, 2)


    return frame


def draw_status(frame, status_text, image_count, auto_mode):
    """
    Zeichnet Statusinformationen auf den Frame.

    Args:
        frame (np.ndarray): Der Kamera-Frame
        status_text (str): Statustext (z.B. "Bild gespeichert!")
        image_count (int): Anzahl der bisher gespeicherten Bilder
        auto_mode (bool): Ob Auto-Modus aktiviert ist

    Returns:
        np.ndarray: Frame mit gezeichnetem Status
    """
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Bilder-Zähler (oben rechts)
    count_text = f"Bilder: {image_count}"
    cv2.putText(frame, count_text, (10, 30), font, 0.8, COLOR_TEXT, 2)

    # Auto-Modus-Status (oben Mitte)
    if auto_mode:
        mode_text = "AUTO-MODUS: AN"
        color = (0, 255, 0)  # Grün
    else:
        mode_text = "MANUELLER MODUS"
        color = (0, 165, 255)  # Orange

    cv2.putText(frame, mode_text, (10, 60), font, 0.7, color, 2)

    # Status-Nachricht (oben links, temporär)
    if status_text:
        cv2.putText(frame, status_text, (10, 90), font, 0.8, (0, 255, 0), 2)

    # Hilfe-Text (unten)
    help_text = "s: Speichern | a: Auto-Modus | f: Flip | q: Quit"
    cv2.putText(frame, help_text, (10, frame.shape[0] - 15), font, 0.5, COLOR_TEXT, 1)

    return frame


def save_roi_image(frame, roi_x, roi_y, class_name, class_dir, filename):
    """
    Schneitet den ROI-Bereich aus und speichert ihn als Bild.

    Args:
        frame (np.ndarray): Der Kamera-Frame
        roi_x (int): X-Koordinate des ROI
        roi_y (int): Y-Koordinate des ROI
        class_name (str): Die Klasse (z.B. 'V')
        class_dir (str): Pfad zum Klassen-Ordner
        filename (str): Dateiname für das Bild

    Returns:
        bool: True wenn erfolgreich gespeichert, False sonst
    """
    try:
        # Extrahiere ROI aus Frame
        roi = frame[roi_y:roi_y + ROI_HEIGHT, roi_x:roi_x + ROI_WIDTH]

        # Stelle sicher, dass Ordner existiert
        os.makedirs(class_dir, exist_ok=True)

        # Speichere Bild
        filepath = os.path.join(class_dir, filename)
        cv2.imwrite(filepath, roi)

        return True
    except Exception as e:
        print(f"Fehler beim Speichern: {e}")
        return False


def main():
    """
    Hauptprogramm für die Webcam-Bilderfassung.

    Workflow:
    1. Frage Benutzer nach Klasse
    2. Öffne Webcam
    3. Zeige Livestream mit ROI-Rechteck
    4. Speichere Bilder basierend auf Benutzer-Input
    5. Beende bei 'q'
    """
    # ========== Klasse vom Benutzer erfragen ==========
    class_name = get_class_from_user()
    print(f"\n✓ Klasse ausgewählt: {class_name}")

    # ========== Pfade und Startparameter ==========
    class_dir = os.path.join('data_raw', class_name)
    next_num = get_next_image_number(class_dir)
    image_count = next_num - 1

    print(f"✓ Speicherort: {class_dir}/")
    print(f"✓ Nächste Bildnummer: {next_num}")
    print("\nWebcam wird gestartet...\n")

    # ========== Webcam öffnen ==========
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Webcam konnte nicht geöffnet werden!")
        return

    # ========== Parameter ==========
    flip_enabled = False
    auto_mode = False
    auto_interval = 0.3  # Sekunden zwischen Auto-Bildern
    last_auto_capture = time.time()
    status_message = ""
    status_time = 0

    # Berechne ROI-Position (Mitte des Frames)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    roi_x = (frame_width - ROI_WIDTH) // 2
    roi_y = (frame_height - ROI_HEIGHT) // 2

    print(f"ℹ️  Frame-Größe: {frame_width}x{frame_height}")
    print(f"ℹ️  ROI-Position: ({roi_x}, {roi_y}) bis ({roi_x + ROI_WIDTH}, {roi_y + ROI_HEIGHT})")

    # ========== HAUPTSCHLEIFE ==========
    print("▶ Webcam läuft. Drücke eine Taste:\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Fehler beim Lesen des Frames!")
            break

        # ========== Flip anwenden wenn aktiviert ==========
        if flip_enabled:
            frame = cv2.flip(frame, 1)

        # ========== Auto-Modus Bildaufnahme ==========
        if auto_mode:
            current_time = time.time()
            if current_time - last_auto_capture >= auto_interval:
                filename = f"{class_name}_webcam_{next_num:03d}.png"
                success = save_roi_image(frame, roi_x, roi_y, class_name, class_dir, filename)

                if success:
                    image_count += 1
                    next_num += 1
                    status_message = f"✓ {filename}"
                    status_time = time.time()
                    print(f"  ✓ {filename} gespeichert")

                last_auto_capture = current_time

        # ========== Status-Nachricht nach Timeout ausblenden ==========
        if status_message and time.time() - status_time > 1.0:
            status_message = ""

        # ========== Zeichne Visualisierungen ==========
        frame = draw_roi(frame, roi_x, roi_y)
        frame = draw_status(frame, status_message, image_count, auto_mode)

        # Zeige Frame an
        cv2.imshow('Bilderfassung - Gebärdensprachen', frame)

        # ========== Tastenanschläge ==========
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            # Beenden
            print("\n❌ Beendet von Benutzer")
            break

        elif key == ord('s'):
            # Einzelnes Bild speichern (manueller Modus)
            if auto_mode:
                print("  ⚠️  Auto-Modus ist aktiv. Deaktiviere mit 'a'.")
            else:
                filename = f"{class_name}_webcam_{next_num:03d}.png"
                success = save_roi_image(frame, roi_x, roi_y, class_name, class_dir, filename)

                if success:
                    image_count += 1
                    next_num += 1
                    status_message = f"✓ {filename}"
                    status_time = time.time()
                    print(f"  ✓ {filename} gespeichert")
                else:
                    status_message = "❌ Fehler beim Speichern!"
                    status_time = time.time()
                    print(f"  ❌ Fehler beim Speichern von {filename}")

        elif key == ord('a'):
            # Auto-Modus toggle
            auto_mode = not auto_mode
            last_auto_capture = time.time()

            if auto_mode:
                print(f"  ▶ Auto-Modus GESTARTET (Intervall: {auto_interval}s)")
                status_message = "Auto: AN"
            else:
                print(f"  ⏹ Auto-Modus GESTOPPT")
                status_message = "Auto: AUS"

            status_time = time.time()

        elif key == ord('f'):
            # Flip toggle
            flip_enabled = not flip_enabled
            flip_text = "AN" if flip_enabled else "AUS"
            print(f"  ➜ Flip: {flip_text}")
            status_message = f"Flip: {flip_text}"
            status_time = time.time()

    # ========== Cleanup ==========
    cap.release()
    cv2.destroyAllWindows()

    print("\n" + "="*60)
    print(f"ZUSAMMENFASSUNG:")
    print(f"  Klasse: {class_name}")
    print(f"  Bilder aufgenommen: {image_count}")
    print(f"  Speicherort: {class_dir}/")
    print("="*60 + "\n")

    if image_count > 0:
        print("✓ Bilder wurden erfolgreich aufgenommen!")
        print("  Führe anschließend folgende Schritte durch:")
        print("    1. python clean_dataset.py  (Daten bereinigen)")
        print("    2. python train_simple.py   (Modell trainieren)")
    else:
        print("⚠️  Keine Bilder wurden aufgenommen.")


# ============================================================================
# EINSTIEG
# ============================================================================
if __name__ == "__main__":
    main()


