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

Nach dem Download:

1. Entpacke die Datensätze
2. Lege die Bilder in folgende Struktur:

### Erwartete Struktur

```
data_raw/
├── A/
├── B/
├── C/
...
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

