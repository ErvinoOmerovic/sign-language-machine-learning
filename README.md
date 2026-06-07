# Gebärdensprachen-Buchstaben-Erkennung mit Machine Learning

Dieses Projekt implementiert ein Machine-Learning-System zur Erkennung ausgewählter Gebärdensprachen-Buchstaben. Ziel ist es, ein CNN-basiertes Modell zu trainieren, zu evaluieren und anschließend für eine einfache Echtzeit-Erkennung über die Webcam nutzbar zu machen.

Das Projekt wurde im Rahmen eines Machine-Learning-Moduls umgesetzt und umfasst den gesamten Ablauf von der Datenbeschaffung über Datenbereinigung, Training und Evaluation bis hin zur praktischen Webcam-Anwendung.

> Hinweis: Diese README dient als technische Übersicht und Einstieg in das Projekt. Der vollständige wissenschaftliche Bericht bzw. das Portfolio befindet sich in der Datei `report.md`.

## Projektziel

Das Ziel des Projekts besteht darin, ausgewählte Gebärdensprachen-Buchstaben automatisiert anhand von Bild- und Webcam-Daten zu klassifizieren. Dafür wird ein CNN-basiertes Modell mit Transfer Learning eingesetzt. Die Modellleistung wird anhand geeigneter Evaluationsmetriken wie Accuracy, Precision, Recall, F1-Score und Confusion Matrix bewertet.

## Unterstützte Klassen

Das Modell unterscheidet aktuell acht ausgewählte Klassen:

```text
A, B, C, L, O, V, W, Y
```

Die Beschränkung auf acht Klassen dient einer kontrollierten Projektumsetzung und ermöglicht eine gezielte Analyse der Modellleistung.

## Projektüberblick

Das Projekt besteht aus folgenden zentralen Bestandteilen:

- Datenbeschaffung aus externen Quellen
- Datenbereinigung und Qualitätssicherung
- Training eines CNN-basierten Klassifikationsmodells
- Evaluation auf externen Testdaten
- Visualisierung der Ergebnisse
- Echtzeit-Erkennung über Webcam
- wissenschaftlicher Projektbericht

## Projektstruktur

```text
project/
├── data_raw/              # Rohdaten für das Training (nicht im Repository)
├── data_cleaned/          # Bereinigte Trainingsdaten (nicht im Repository)
├── data_downloads/        # Lokale Downloads der Datensätze (nicht im Repository)
├── external_test/         # Externe Testdaten (nicht im Repository)
├── logs/                  # Lokal erzeugte Trainings- und Evaluationslogs
├── models/                # Gespeichertes finales Modell
├── report_assets/         # Abbildungen für den Projektbericht
├── analyze_distribution.py
├── clean_dataset.py
├── data_loader.py
├── evaluate.py
├── predict_webcam.py
├── preprocess.py
├── setup_data.py
├── setup_environment.py
├── train.py
├── train_simple.py
├── requirements.txt
├── report.md
└── README.md
```

Hinweis: Große Datensätze und lokale Logs werden nicht im Repository mitgeführt, um die Repository-Größe gering zu halten. Die für den Bericht relevanten Abbildungen befinden sich im Ordner `report_assets/`.

## Installation

## Python-Interpreter auswählen

Für das Projekt sollte in PyCharm die Conda-Umgebung `ml_train` als Python-Interpreter ausgewählt werden. Wichtig ist, dass die Abhängigkeiten aus `requirements.txt` in genau dieser Umgebung installiert wurden.

Falls PyCharm eine andere Python-Version verwendet, kann dies unter `Settings → Project → Python Interpreter` angepasst werden. Nach der Auswahl sollte unten rechts in PyCharm `ml_train` angezeigt werden.

### 1. Repository klonen

```bash
git clone <repository-url>
cd <repository-name>
```

### 2. Virtuelle Umgebung erstellen

```bash
python -m venv .venv
source .venv/bin/activate
```

Für Windows:

```bash
.venv\Scripts\activate
```

### 3. Abhängigkeiten installieren

```bash
pip install -r requirements.txt
```

## Schnellstart: Webcam-Erkennung

Wenn das trainierte Modell im Ordner `models/` enthalten ist, kann die Webcam-Erkennung direkt gestartet werden:

```bash
python predict_webcam.py
```

Das Skript lädt das gespeicherte Modell, öffnet die Webcam und zeigt die erkannte Klasse direkt im Kamerafenster an.

### Steuerung der Webcam-Anwendung

| Taste | Funktion |
|---|---|
| `q` | Programm beenden |
| `s` | aktuellen Frame speichern |
| `f` | horizontales Spiegeln aktivieren/deaktivieren |

## Trainiertes Modell

Das finale Modell soll unter folgendem Pfad liegen:

```text
models/sign_language_model.h5
```

Wenn diese Datei im Repository enthalten ist, muss das Modell nach dem Klonen nicht erneut trainiert werden. Ein erneutes Training ist nur erforderlich, wenn neue Trainingsdaten verwendet oder ein neues Modell erzeugt werden soll.

Falls das Modell nicht vorhanden ist, kann es über den Trainingsprozess neu erstellt werden.

## Datenquellen

Für Training und Evaluation werden externe Datensätze verwendet. Die Datensätze sind aus Speicher- und Lizenzgründen nicht vollständig im Repository enthalten.

Verwendete Quellen:

1. ASL Alphabet Dataset (Kaggle)  
   https://www.kaggle.com/datasets/debashishsau/aslamerican-sign-language-aplhabet-dataset

2. ASL Dataset (Zenodo)  
   https://zenodo.org/records/14635573

3. Synthetic ASL Alphabet Dataset (Kaggle)  
   https://www.kaggle.com/datasets/lexset/synthetic-asl-alphabet

Aus den Datensätzen werden ausschließlich die Klassen `A`, `B`, `C`, `L`, `O`, `V`, `W` und `Y` verwendet.

## Datensätze einrichten

Für den automatisierten Datenimport steht das Skript `setup_data.py` zur Verfügung.

```bash
python setup_data.py
```

Das Skript lädt beziehungsweise verarbeitet die definierten Datensätze und legt die Daten in der vorgesehenen Projektstruktur ab.

### Voraussetzungen für Kaggle

Für Kaggle-Datensätze wird die Kaggle CLI benötigt:

```bash
pip install kaggle
```

Zusätzlich muss eine Kaggle-API-Datei vorhanden sein:

```text
~/.kaggle/kaggle.json
```

Diese Datei enthält den persönlichen Kaggle API Key und darf nicht in das Repository hochgeladen werden.

## Datenbereinigung

Vor dem Training können die Rohdaten mit folgendem Skript bereinigt werden:

```bash
python clean_dataset.py
```

Die Datenbereinigung umfasst unter anderem:

- Erkennung und Entfernung von Duplikaten
- Filterung unscharfer Bilder
- Entfernung beschädigter oder nicht lesbarer Dateien
- Filterung extremer Bildinhalte
- Erstellung von Datenverteilungsanalysen

Die bereinigten Daten werden in `data_cleaned/` gespeichert und dienen als Grundlage für das Training.

## Training

Das empfohlene Trainingsskript lautet:

```bash
python train_simple.py
```

Das Skript trainiert das Modell auf den bereinigten Daten und speichert das resultierende Modell im Ordner `models/`.

Der stabile Modellpfad für die spätere Nutzung lautet:

```text
models/sign_language_model.h5
```

## Evaluation

Die Evaluation kann über folgendes Skript durchgeführt werden:

```bash
python evaluate.py
```

Dabei werden die Modellmetriken berechnet und eine Confusion Matrix erzeugt. Die wichtigsten im Projektbericht verwendeten Abbildungen werden im Ordner `report_assets/` bereitgestellt, damit sie im Repository sichtbar und im Bericht korrekt eingebunden sind.

Bewertete Metriken:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

## Projektbericht

Der wissenschaftliche Projektbericht befindet sich in:

```text
report.md
```

Die im Bericht verwendeten Abbildungen befinden sich in:

```text
report_assets/
```

Dazu gehören insbesondere:

- Trainings- und Validierungsverlauf
- Confusion Matrix der externen Evaluation
- Datenverteilung nach der Bereinigung

## Reproduzierbarkeit

Das Projekt ist so aufgebaut, dass zentrale Schritte nachvollziehbar ausgeführt werden können:

1. Datensätze einrichten
2. Rohdaten bereinigen
3. Modell trainieren
4. Modell evaluieren
5. Webcam-Erkennung starten

Die lokalen Logs und großen Datensätze werden nicht vollständig versioniert. Relevante Ergebnisabbildungen für den Bericht werden jedoch im Ordner `report_assets/` bereitgestellt.

## Hinweise zum Repository

Folgende Inhalte sind bewusst nicht Bestandteil des Repositorys:

- vollständige Rohdatensätze
- bereinigte Trainingsdaten
- lokale Download-Archive
- lokale Trainingslogs
- virtuelle Umgebung `.venv/`
- Cache-Dateien wie `__pycache__/`

Das finale Modell kann im Ordner `models/` mitgeführt werden, damit die Webcam-Demo nach dem Klonen direkt nutzbar ist.

## Technologien

- Python
- TensorFlow / Keras
- OpenCV
- NumPy
- scikit-learn
- Matplotlib
- Pillow

## Mögliche Erweiterungen

- Erweiterung auf das vollständige Gebärdensprachen-Alphabet
- Verbesserung der Robustheit bei unterschiedlichen Lichtverhältnissen
- Integration einer Handsegmentierung
- Erweiterung zu einer Web-Anwendung
- Optimierung für mobile Geräte
