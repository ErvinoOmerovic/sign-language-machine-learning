# Gebärdensprachen-Buchstaben Erkennung 🤟

Dieses Projekt implementiert ein Machine-Learning-Modell zur Erkennung von Gebärdensprachen-Buchstaben in Echtzeit über eine Webcam.

## Unterstützte Klassen

A, B, C, L, V, W, O, Y

---

## 🚀 Features

- Echtzeit-Erkennung über Webcam
- Bild-Preprocessing Pipeline
- CNN-basiertes Modell (TensorFlow / Keras)
- Debugging-Tools zur Analyse von Trainingsdaten

---

## ⚙️ Installation

```bash
python -m venv .venv
source .venv/bin/activate   # macOS / Linux
# Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

---

## 📁 Projektstruktur

```
project/
├── data_raw/            # Rohdaten (nicht im Repo)
├── data_cleaned/        # Vorverarbeitete Daten (nicht im Repo)
├── external_test/       # Externer Testdatensatz (nicht im Repo)
├── models/              # Gespeicherte Modelle (nicht im Repo)
├── logs/                # Trainingslogs
│   ├── classification_reports/  # Classification Reports
│   ├── confusion_matrices/      # Confusion Matrices
│   ├── training_metrics/        # Epoch-Metriken (CSV)
│   ├── training_curves/         # Trainingskurven (PNG)
│   ├── data_analysis/           # Datenanalyse (Plots)
│   └── cleaning_logs/           # Cleaning-Logs (TXT)
├── preprocess.py
├── train_simple.py
├── train.py
├── evaluate.py
├── predict_webcam.py
├── utils.py
├── requirements.txt
├── report.md
└── README.md
```

---

## 📦 Dataset

Für das Training und die Evaluation des Modells werden externe Datensätze verwendet.

⚠️ Die Datensätze sind **nicht im Repository enthalten** und müssen manuell heruntergeladen werden.

---

### 🔹 Empfohlene Datensätze

#### 1. ASL Alphabet Dataset (Kaggle)

👉 https://www.kaggle.com/datasets/debashishsau/aslamerican-sign-language-aplhabet-dataset

- Enthält Bilder des amerikanischen Gebärdensprachen-Alphabets  
- Struktur: Bilder sind bereits in Klassenordnern organisiert (z. B. A, B, C, …) :contentReference[oaicite:0]{index=0}  
- Bilder werden u.a. als Datenaugmentierung verwendet, um die Modellrobustheit zu verbessern (Hintergrund zu sehen, Mensch zu sehen, nicht nur die Hand)

---

#### 2. ASL Dataset (Zenodo)

👉 https://zenodo.org/records/14635573

- Enthält farbige (RGB) Bilder von Handgesten für das Alphabet  
- Fokus auf Finger- und Gelenkpositionen für präzisere Erkennung :contentReference[oaicite:1]{index=1}  
- Gut geeignet für Experimente und Modellverbesserung

---

## 📁 Verwendung im Projekt

Die Datensätze werden **nicht automatisch** geladen. Für den manuellen Import gibt es das Skript:

```bash
python setup_data.py
```

Das Skript:
- lädt die definierten Kaggle- und Zenodo-Datensätze manuell herunter
- entpackt die Archive lokal
- übernimmt nur die Klassen `A, B, C, L, O, V, W, Y`
- schreibt Trainingsdaten nach `data_raw/`
- schreibt externe Testdaten nach `external_test/dataset2/` und `external_test/dataset3/`
- überspringt bereits importierte Daten standardmäßig
- lädt mit `--force` die betreffenden Datensätze neu

### Manuelle Nutzung

Alle Datensätze:

```bash
python setup_data.py
```

Nur ausgewählte Datensätze:

```bash
python setup_data.py --datasets kaggle_asl_2 zenodo_asl
```

Nur prüfen, ohne Dateien zu ändern:

```bash
python setup_data.py --dry-run
```

Erneut herunterladen und bereits importierte Dateien dieses Skripts ersetzen:

```bash
python setup_data.py --force
```

### Voraussetzungen für Kaggle

- Kaggle CLI installiert: `pip install kaggle`
- Auth-Datei vorhanden unter `~/.kaggle/kaggle.json`

Wenn die Authentifizierung fehlt, beendet sich das Skript mit einer verständlichen Fehlermeldung.

### Zielstruktur

```
data_raw/
├── A/
├── B/
├── C/
├── L/
├── O/
├── V/
├── W/
└── Y/

external_test/
├── dataset2/
│   ├── A/
│   ├── B/
│   ├── C/
│   ├── L/
│   ├── O/
│   ├── V/
│   ├── W/
│   └── Y/
└── dataset3/
    ├── A/
    ├── B/
    ├── C/
    ├── L/
    ├── O/
    ├── V/
    ├── W/
    └── Y/
```

### Beispiel Summary

```text
========================================================================
SETUP DATA SUMMARY
========================================================================
kaggle_asl_1: ERFOLG
  Quelle: Kaggle ASL Dataset 1
  Training nach data_raw/: 1880 Dateien
  Verteilung: {"A": 235, "B": 235, "C": 235, "L": 235, "O": 235, "V": 235, "W": 235, "Y": 235}
  Validierung: Training validiert: 1880 Dateien in data_raw

kaggle_asl_2: ERFOLG
  Quelle: Kaggle ASL Dataset 2
  Training nach data_raw/: 1600 Dateien
  Verteilung: {"A": 200, "B": 200, "C": 200, "L": 200, "O": 200, "V": 200, "W": 200, "Y": 200}
  Test nach external_test/dataset2/: 640 Dateien
  Verteilung: {"A": 80, "B": 80, "C": 80, "L": 80, "O": 80, "V": 80, "W": 80, "Y": 80}
  Validierung: Training validiert: 1600 Dateien in data_raw
  Validierung: Test validiert: 640 Dateien in external_test/dataset2

zenodo_asl: ERFOLG
  Quelle: Zenodo ASL Dataset
  Test nach external_test/dataset3/: 800 Dateien
  Verteilung: {"A": 100, "B": 100, "C": 100, "L": 100, "O": 100, "V": 100, "W": 100, "Y": 100}
  Validierung: Test validiert: 800 Dateien in external_test/dataset3

Logdatei: logs/setup_data.log
```


## 🧠 Training

### Einfaches Training (empfohlen)

```bash
python train_simple.py
```

Das trainierte Modell wird gespeichert unter:

```
models/sign_language_model.h5
```


## 📊 Evaluation

Nach dem Training werden alle relevanten Metriken automatisch berechnet und in den Logs gespeichert.

👉 Es ist **kein separates Skript erforderlich**.

Die Ergebnisse findest du in:

```
logs/
```

Dort enthalten:
- Accuracy
- Confusion Matrix
- weitere Trainingsmetriken

---

## 🎥 Webcam Prediction

Starte die Echtzeit-Erkennung:

```bash
python predict_webcam.py
```

### Controls

| Taste | Funktion |
|------|--------|
| q | Beenden |
| s | Frame speichern |
| f | Flip toggeln |


## 📄 Bericht

Der wissenschaftliche Bericht befindet sich in:

```
report.md
```

---

## 🧠 Technologien

- Python
- TensorFlow / Keras
- OpenCV
- NumPy

---

## ⚠️ Hinweis

Folgende Inhalte sind bewusst **nicht im Repository enthalten**:

- Datensätze (`data_raw`, `data_cleaned`, `external_test`)
- Modelle (`models/`)
- virtuelle Umgebung (`.venv`)

Diese sind in `.gitignore` definiert.

---

## 🧹 Datenbereinigung

Bereinigt Rohdaten aus `data_raw/` und erstellt automatisch eine Datenverteilungs-Analyse:

```bash
python clean_dataset.py
```

**Features:**
- Duplikat-Erkennung
- Unschärfe-Filterung
- Kaputte Bilder entfernen
- Extrem schlechte Bilder filtern
- **Automatische Datenanalyse** mit Diagrammen und Statistiken

**Output:**
- Bereinigte Bilder in `data_cleaned/`
- Analyse-Diagramm: `logs/data_analysis/data_distribution_TIMESTAMP.png`
- Analyse-Tabelle: `logs/data_analysis/data_distribution_TIMESTAMP.txt`

---

## 🚀 Zukunft / Erweiterungen

- Unterstützung für das komplette Alphabet
- Verbesserung der Modellgenauigkeit
- Deployment als Web-App (Flask / FastAPI)
- Mobile Integration
