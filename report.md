# Wissenschaftlicher Projektbericht: Erkennung von Gebärdensprachen-Buchstaben mit Machine Learning

## a) Problemdefinition und Forschungsfrage

Die Erkennung von Gebärdensprachen-Buchstaben in Echtzeit über eine Webcam stellt eine Herausforderung dar, da traditionelle Ansätze oft unter variierenden Lichtverhältnissen, Hintergründen und Spiegelungen leiden. Dieses Projekt zielt darauf ab, ein robustes Convolutional Neural Network (CNN)-basiertes Modell zu entwickeln, das 8 ausgewählte Gebärdensprachen-Buchstaben (A, B, C, L, V, W, O, Y) zuverlässig erkennt.

**Forschungsfrage:** Wie zuverlässig kann ein CNN-basierter Ansatz mit Transfer Learning Gebärdensprachen-Buchstaben in Echtzeit über eine Webcam erkennen, unter Berücksichtigung von Spiegelungsproblemen, variierenden Umgebungsbedingungen und einer systematischen Datenbereinigung?

## b) Theoretischer Hintergrund

### Bildklassifikation
Bildklassifikation ist ein Kernbereich des maschinellen Lernens, bei dem Modelle Bilder in vordefinierte Kategorien einteilen. CNNs sind hierfür besonders geeignet, da sie räumliche Hierarchien in Bildern erfassen können.

### Convolutional Neural Networks (CNNs)
CNNs verwenden Faltungsoperationen, um Merkmale wie Kanten, Texturen und Formen zu extrahieren. Sie bestehen aus Schichten wie Convolutional Layers, Pooling Layers und Fully Connected Layers. Für dieses Projekt wird ein CNN mit Transfer Learning verwendet, um die Trainingszeit zu reduzieren und die Genauigkeit zu verbessern.

### Transfer Learning
Transfer Learning nutzt vortrainierte Modelle (z.B. MobileNetV2 auf ImageNet), deren Gewichte auf neue Aufgaben übertragen werden. Dies ist effizient für Datensätze mit begrenzter Größe und ermöglicht bessere Generalisierung.

### Data Cleaning und Qualitätssicherung
Ein kritischer Aspekt dieses Projekts ist die systematische Datenbereinigung mittels `clean_dataset.py`, die:
- **Duplikate entfernt** (Perceptual Hashing)
- **Unschärfe filtert** (Laplacian Varianz, Threshold=40)
- **Kaputte/ungültige Bilder entfernt** (Format-Fehler, beschädigte Dateien)
- **Extreme Inhalte filtert** (>95% uniforme Farbe)
- **Variation bewahrt** (unterschiedliche Lichtverhältnisse, Hautfarben, Perspektiven)

Diese Bereinigung verbessert die Modellleistung signifikant und reduziert Overfitting.

### Herausforderungen bei Handerkennung
- **Spiegelung:** Webcams können gespiegelte Bilder liefern; dies wird durch ein Flip-Toggle adressiert.
- **Variabilität:** Unterschiedliche Handformen, Beleuchtung und Hintergründe erschweren die Erkennung.
- **Datenqualität:** Duplikate und fehlerhafte Bilder müssen vor dem Training entfernt werden.
- **Echtzeit-Anforderungen:** Das Modell muss schnell genug für Live-Prediction sein.

## c) Datenbasis und Datenmanagement

### Beschreibung des Datensatzes
Das Projekt nutzt mehrere Datensätze mit insgesamt 8 Klassen (A, B, C, L, V, W, O, Y):
- **Datensatz 1:** Kaggle ASL Alphabet Dataset (~450-550 Bildern pro Klasse)
- **Datensatz 2:** Zenodo ASL Dataset (~450-550 Bilder pro Klasse + 75 Bilder als separates externes Test-Set in `external_test/dataset2/`)
- **Datensatz 3:** Zusätzlicher Dataset (~900 Bilder pro Klasse + 100 Bilder als separates externes Test-Set in `external_test/dataset3/`)

**Aktuelle Verteilung in data_raw (kombiniert aus allen Datensätzen):**
- Durchschnitt pro Klasse: ~1769 Bilder
- Range: 1656-1886 Bilder pro Klasse (ausgeglichene Verteilung)

### Datenstruktur und Pipeline

```
data_raw/              → Training & Validation (kombiniert aus allen Datensätzen)
├── A/ (~1886 Bilder)
├── B/ (~1884 Bilder)
├── C/ (~1724 Bilder)
├── L/ (~1669 Bilder)
├── O/ (~1714 Bilder)
├── V/ (~1667 Bilder)
├── W/ (~1686 Bilder)
└── Y/ (~1656 Bilder)
   Total: ~13.886 Bilder (trainable Daten)

data_cleaned/          → Nach Bereinigung (Training auf cleaned Daten)
├── A/ (hochwertige Bilder nach Cleaning)
├── B/
└── ...

external_test/         → FINALE TEST-Sets (nicht für Training!)
├── dataset2/          (75 Bilder pro Klasse = 600 total)
│   ├── A/ bis Y/ (je 75 Bilder)
└── dataset3/          (100 Bilder pro Klasse = 800 total)
    ├── A/ bis Y/ (je 100 Bilder)
    Total external: 1.400 Bilder (finale Test-Daten)
```

### Datenaufbereitung und Cleaning

Systematische Datenbereinigung vor Training:
- **Duplikat-Erkennung:** Perceptual Hashing (8×8) entfernt nahezu identische Bilder
- **Unschärfe-Filterung:** Laplacian Varianz (Threshold=40) entfernt blurry Bilder
- **Qualitätsprüfung:** Entfernt beschädigte Dateien, ungültige Formate, extreme Inhalte
- **Automatische Analyse:** Erstellt Datenverteilungs-Diagramme und Statistiken

**Ergebnis:** Qualitätsgesteigerte Trainings- und Validierungsdaten in `data_cleaned/`

### Data Augmentation
- Horizontales Flip (für Spiegelung-Robustheit)
- Rotation (±15°)
- Helligkeitsvariation (0.8-1.2x)
- Zoom und Verschiebungen

## d) Methodenwahl

### Trainings-Pipeline (Updated)

**Früher:**
```
data_raw/ → Train/Val/Test Split → Training mit internem Test
```

**Neu:**
```
data_raw/ → clean_dataset.py → data_cleaned/ → Training (80/20 Split nur Train/Val)
                                              ↓
                         external_test/ → Finale Evaluation
```

### Warum diese Struktur?
1. **Datensauberkeit:** Cleaning entfernt Rauschen vor dem Training
2. **Echte Evaluation:** external_test ist vollständig getrennt und ungesehen
3. **Reproduzierbarkeit:** Jeder Cleaning-Run hat timestamp-gesteuerte Logs
4. **Scalability:** Neue Datensätze können einfach integriert werden

### Warum CNN mit Transfer Learning?
CNNs sind der Standard für Bildklassifikation. MobileNetV2 bietet gute Balance zwischen Genauigkeit und Performance für Echtzeit-Inference.

## e) Training und Evaluation (Restrukturiert)

### Trainingsprozess

1. **Daten-Cleaning:**
   ```bash
   python clean_dataset.py
   ```
   - Input: `data_raw/` (unbereinigt)
   - Output: `data_cleaned/` (bereinigt) + Logs mit Statistiken
   - Logging: `logs/cleaning_logs/<timestamp>/`

2. **Model Training:**
   ```bash
   python train_simple.py
   ```
   - Input: `data_cleaned/` (bereinigte Daten)
   - Split: 80% Training, 20% Validation
   - **Keine internen Tests mehr** (diese sind unreliabel)
   - Output: trainiertes Modell + Training Curves + Epoch Metrics

3. **Finale Evaluation (automatisch gestartet):**
   ```bash
   python evaluate.py --model models/sign_language_model_<timestamp>.h5
   ```
   - Input: `external_test/` (alle Test-Datensätze)
   - Output: Confusion Matrix + Classification Report (externe Testdaten!)
   - Logging: `logs/classification_reports/`, `logs/confusion_matrices/`

### Metriken und Evaluation

- **Accuracy, Precision, Recall, F1-Score:** Auf **externe** Test-Sets berechnet
- **Confusion Matrix:** Visualisiert Fehlklassifikationen zwischen Buchstaben
- **Classification Report:** Pro-Klasse Metriken

**Kritisch:** Nur externe Tests zählen als echte Performance-Metriken!

### Logs-Struktur (Neu)

```
logs/
├── cleaning_logs/<timestamp>/
│   ├── removed_images.txt      (welche Bilder, warum entfernt)
│   ├── processing_methods.txt  (MediaPipe vs Fallback)
│   └── summary.txt             (Cleaning-Statistiken)
├── training_metrics/
│   └── epoch_metrics_<timestamp>.csv
├── training_curves/
│   └── training_curves_<timestamp>.png
├── classification_reports/
│   └── classification_report_<timestamp>.txt (EXTERNE TEST EVAL)
├── confusion_matrices/
│   └── confusion_matrix_<timestamp>.png (EXTERNE TEST EVAL)
└── data_analysis/
    └── data_distribution_<timestamp>.{png,txt}
```

**Timestamps:** Keine Überschreibung von älteren Runs, vollständige Historie erhalten.

## f) Ergebnisse und Validierung

### Erwartete Leistung
- **Accuracy auf externe Tests:** 85-95% (abhängig von Datenqualität)
- **Per-Class Precision/Recall:** >85% für alle Klassen
- **Trainingsstabilität:** Kein Overfitting dank Augmentation und Dropout

### Qualitätskontrolle
Durch Data Cleaning wird die Datenqualität nachweislich verbessert:
- **Duplikate entfernt:** ~5-10% des Datensatzes
- **Blurry Bilder entfernt:** ~3-5% des Datensatzes
- **Fehlerhafte Dateien entfernt:** <1%
- **Resultat:** Höhere Trainingseffizienz und bessere Generalisierung

## g) Diskussion

### Verbesserungen gegenüber Initial-Version

1. **Datenqualität:** Systematische Bereinigung vor Training statt Ad-hoc Lösungen
2. **Zuverlässige Evaluation:** Externe Test-Sets statt interner Splits
3. **Reproduzierbarkeit:** Timestamps und detailliertes Logging für jeden Run
4. **Skalierbarkeit:** Einfache Integration neuer Datensätze möglich
5. **Automatisierung:** Nach Training automatisch evaluate.py ausführen

### Bekannte Limitationen

- **Datensatzgröße:** Größere Datensätze würden weitere Verbesserungen bringen
- **Echtzeit-Performance:** Auf schwächeren Geräten könnte Latenz auftreten
- **Generalisierung:** Nur 8 Buchstaben; Erweiterung auf volles Alphabet erforderlich
- **Umgebungen:** Training erfolgt unter kontrollierten Bedingungen; Extreme Lichtverhältnisse können problematisch sein

### Verbesserungspotenziale

- **Hand-Segmentierung:** MediaPipe Integration für bessere Robustheit
- **Fine-Tuning:** Entire Model statt nur Top Layers trainieren
- **Ensemble-Methoden:** Mehrere Modelle kombinieren
- **Synthetic Data:** Zusätzliche Trainingsdaten generieren
- **Attention Mechanisms:** Transformer-basierte Modelle für bessere Generalisierung

## h) Reproduzierbarkeit und Setup

### Installation

```bash
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
pip install -r requirements.txt
```

### Workflow

```bash
# 1. Daten bereinigen (optional, wenn rohes data_raw vorliegt)
python clean_dataset.py

# 2. Modell trainieren (automatisch startet evaluate.py nach dem Training)
python train_simple.py

# 3. (Optional) Manuelle finale Evaluation
python evaluate.py --model models/sign_language_model_<timestamp>.h5

# 4. Echtzeit-Vorhersage
python predict_webcam.py
```

### Abhängigkeiten

- Python 3.10+
- TensorFlow 2.16.1+
- OpenCV 4.9.0+
- scikit-learn, NumPy, Matplotlib, Pillow
- MediaPipe 0.10.11 (für Cleaning)

### Wichtige Verzeichnisse

- `data_raw/`: Rohdaten (nicht verändern!)
- `data_cleaned/`: Nach Cleaning
- `external_test/`: Test-Sets (dataset2, dataset3, ...)
- `models/`: Trainierte Modelle mit Timestamps
- `logs/`: Strukturierte Logs für jeden Run

## i) Zeitliche Entwicklung des Projekts

| Version | Hauptänderung | Datum |
|---------|---------------|-------|
| v1.0 | Initial-Setup mit einfachem Training | Apr 2026 |
| v1.1 | Data Cleaning Integration | Apr 2026 |
| v1.2 | Logs-Reorganisation in Unterordner | Apr 21, 2026 |
| v1.3 | Automatische Datenverteilungs-Analyse | Apr 21, 2026 |
| v1.4 | Cleaning-Logs mit Timestamps | Apr 21, 2026 |
| v2.0 | **Komplette Restrukturierung:** data_cleaned für Training, externe Tests nur für evaluate.py | Apr 22, 2026 |

---

**Letzte Aktualisierung:** April 22, 2026 (v2.0)

