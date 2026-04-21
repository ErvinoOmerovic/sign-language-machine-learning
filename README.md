# Gebärdensprachen-Buchstaben Erkennung

Dieses Projekt implementiert ein Machine-Learning-Modell zur Erkennung von 8 Gebärdensprachen-Buchstaben (A, B, C, L, V, W, O, Y) in Echtzeit über die Webcam.

## Setup

1. Stelle sicher, dass Python 3.8+ installiert ist.
2. Erstelle ein Virtual Environment: `python -m venv .venv`
3. Aktiviere es: `source .venv/bin/activate` (macOS/Linux)
4. Installiere Abhängigkeiten: `pip install -r requirements.txt`

## Datenstruktur

Lege deine Bilder in den `data/` Ordner, mit Unterordnern pro Klasse:
- `data/A/`
- `data/B/`
- etc.

## Training

Führe `python train.py` aus, um das Modell zu trainieren. Das beste Modell wird in `models/sign_language_model.h5` gespeichert.

## Evaluation

Nach Training: `python evaluate.py` für Metriken und Confusion Matrix.

## Webcam Prediction

Führe `python predict_webcam.py` aus. Drücke 'q' zum Beenden, 's' zum Speichern eines Frames, 'f' zum Togglen des Flips.

## Debugging

Verwende `utils.py` für Vergleiche mit Trainingsbildern oder Test einzelner Bilder.

## Bericht

Siehe `report.md` für den wissenschaftlichen Bericht.