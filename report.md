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

## 📊 Verwendete Datensätze (Fakten)

Das Projekt verwendet die folgenden externen Datensätze. Die Links verweisen auf die Originalquellen:

- Datensatz 1 (Kaggle): [ASL Alphabet Dataset](https://www.kaggle.com/datasets/debashishsau/aslamerican-sign-language-aplhabet-dataset)
- Datensatz 2 (Zenodo): [ASL Dataset](https://zenodo.org/records/14635573)
- Datensatz 3 (Kaggle - Synthetic): [Synthetic ASL Alphabet Dataset](https://www.kaggle.com/datasets/lexset/synthetic-asl-alphabet)

Extrahiert werden ausschließlich die Klassen: A, B, C, L, V, W, O, Y.

Aktuelle, projektinterne Fakten (aus den projektweiten Logs):
- Anzahl Klassen: 8
- Externe Test-Sets (gespeichert unter `external_test/`):
  - `dataset2/` (Zenodo) — 75 Bilder pro Klasse ⇒ 600 Bilder total
  - `dataset3/` (Synthetic) — 100 Bilder pro Klasse ⇒ 800 Bilder total

Die Trainingsdaten für das Modell werden aus den bereinigten Bildern in `data_cleaned/` geladen. Die Rohdaten sind in `data_raw/` vorhanden und werden von den Cleaning-Skripten gelesen, aber nicht verändert.

### Datenstruktur (tatsächlicher Zustand im Projekt)

```
data_raw/        # Rohdaten (nur lesend verwendet)
data_cleaned/    # Bereinigte Bilder, Eingang für Training
external_test/   # Finale Test-Sets (dataset2/, dataset3/)
```

Die exakten Bildzahlen pro Klasse in `data_raw/` können variieren, die aktuell verwendeten finalen Test-Sets sind jedoch die oben genannten (600 + 800 Bilder).

### Datenaufbereitung und Cleaning (Fakten)

Die Bereinigung erfolgt mit `clean_dataset.py` und umfasst:
- Perceptual Hashing zur Duplikatserkennung (8×8 Hash)
- Laplacian-Varianz (Threshold = 40) zur Erkennung starker Unschärfe
- Entfernung beschädigter oder nicht ladbarer Dateien
- Filterung nahezu einfarbiger (extremer) Bilder
- Pro-Run-Logs und Datenverteilungs-Analysen werden in `logs/cleaning_logs/<timestamp>/` abgelegt

Die bereinigten Bilder werden in `data_cleaned/` gespeichert und bilden die Grundlage für das Training.

### Data Augmentation (Kurz, Fakten)

- Horizontaler Flip
- Rotation (±15°)
- Helligkeitsvariation (0.8–1.2×)
- Gelegentliche Zoom- und Verschiebungsoperationen

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

## e) Training und Evaluation (Fakten und Interpretation)

Die folgende Darstellung trennt sachliche, reproduzierbare Ergebnisse (Fakten) von anschließenden Interpretationen.

### e.1 Fakten — Training

Die Trainingsmetriken werden epochal in `logs/training_metrics/` abgelegt. Für das zuletzt trainierte Modell (`models/sign_language_model_2026-04-21_21-43-41.h5`) ergeben sich aus den epochalen Logs (Datei `epoch_metrics_2026-04-21_21-43-41.csv`) folgende, direkt messbare Werte:

- Maximale Validierungsgenauigkeit (val_accuracy) in den Logs: 0.9874706268310547 (Epoch 12)
- Val_accuracy zum Ende des Trainingslaufes (letzte aufgezeichnete Epoche): 0.9851213693618774 (Epoch 19)
- Finale Trainingsaccuracy (letzte Epoche): 0.9962799549102783
- In den epochalen Logs sind Loss- und Learning-Rate-Verläufe für alle Epochen gespeichert (siehe `logs/training_metrics/epoch_metrics_2026-04-21_21-43-41.csv`).

### e.2 Fakten — Finale (externe) Evaluation

Die finale Evaluation wurde auf den zusammengeführten externen Test-Sets durchgeführt. Das aktuellste und maßgebliche Classification-Report-Log ist `logs/classification_reports/classification_report_2026-04-22_13-37-47.txt` (EXTERNE TEST EVALUATION). Aus diesem Log gelten folgende gemessene Werte:

- Accuracy (externer Test, gesamt): 0.9171
- Precision (macro/weighted in Log gerundet): 0.9204 (als Gesamtwert im Log angegeben)
- Recall (gesamt): 0.9171
- F1-Score (gesamt): 0.9174
- Support (Anzahl Testbeispiele gesamt): 1400 (175 pro Klasse)

Per-Klasse (Precision / Recall / F1 / Support) — Werte aus dem Classification Report:

- A:  precision=0.95  recall=0.89  f1-score=0.91  support=175
- B:  precision=0.99  recall=0.98  f1-score=0.98  support=175
- C:  precision=0.92  recall=0.99  f1-score=0.95  support=175
- L:  precision=0.87  recall=0.91  f1-score=0.89  support=175
- V:  precision=0.93  recall=0.88  f1-score=0.91  support=175
- W:  precision=0.91  recall=0.89  f1-score=0.90  support=175
- O:  precision=0.97  recall=0.87  f1-score=0.92  support=175
- Y:  precision=0.82  recall=0.94  f1-score=0.87  support=175

Die zugehörige Confusion-Matrix-Abbildung ist im Projekt unter `logs/confusion_matrices/confusion_matrix_2026-04-22_13-37-47.png` abgelegt; die Matrix visualisiert die Verteilung der Fehlklassifikationen zwischen den acht Klassen (siehe Interpretation unten).

### e.3 Interpretation — wissenschaftliche Einordnung der Ergebnisse

- Die Trainings- und Validierungsmetriken (val_accuracy bis 0.987) zeigen, dass das Modell auf den bereinigten Trainingsdaten sehr hohe Validierungsgenauigkeiten erreicht. Der Unterschied zwischen Validierungs- und externer Test-Accuracy (0.987 vs. 0.917) weist auf eine deutliche Generalisierungsdifferenz hin, die im Abschnitt "Diskussion" weiter analysiert wird.
- Die per-Klasse-Werte aus dem Classification Report erlauben eine differenzierte Betrachtung: Klasse B und C zeigen besonders hohe Präzision/Recall-Werte, während Klasse Y eine vergleichsweise niedrige Präzision (0.82) kombiniert mit hohem Recall (0.94) aufweist; dies deutet auf eine erhöhte Rate an False-Positives bei Y hin (andere Klassen werden fälschlich als Y klassifiziert).
- Klassen mit hoher Präzision, aber niedrigerem Recall (z. B. O: precision 0.97, recall 0.87) werden vergleichsweise selten fälschlich vorhergesagt, jedoch öfter übersehen (False-Negatives).

### e.4 Reproduzierbarkeit und Logs

Alle für diese Ergebnisse relevanten Dateien sind im Repository abgelegt und zeitgestempelt:

- Trainingsmetriken: `logs/training_metrics/epoch_metrics_2026-04-21_21-43-41.csv`
- Finale Classification Report: `logs/classification_reports/classification_report_2026-04-22_13-37-47.txt`
- Confusion Matrix (Abbildung): `logs/confusion_matrices/confusion_matrix_2026-04-22_13-37-47.png`

Diese Dateien enthalten die numerischen Werte, Plots und Tabellen, die zur Reproduktion der hier dargestellten Fakten erforderlich sind.

## f) Ergebnisse und Validierung (Fakten)

In diesem Abschnitt werden die tatsächlichen, aus den Logs abgeleiteten Resultate zusammengefasst. Interpretative Aussagen folgen im nächsten Abschnitt.

Fakten aus den Projektlogs (aktuellste, zeitgestempelte Dateien):

- Finale externe Test-Accuracy: 0.9171 (siehe `logs/classification_reports/classification_report_2026-04-22_13-37-47.txt`)
- Finale externe Precision (gesamt): 0.9204
- Finale external Recall (gesamt): 0.9171
- Finale external F1-Score (gesamt): 0.9174
- Test-Support: 1.400 Bilder (8 Klassen × 175 Support pro Klasse)
- Per-Klasse-Metriken sind unter e.2 dokumentiert (Precision/Recall/F1 pro Klasse).

Ergebnisse der Datenbereinigung (aktuellster Cleaning-Run `logs/cleaning_logs/2026-04-21_21-12-50`):

- Entfernte Einträge (aus `removed_images.txt`): 1.117 Bilder insgesamt
  - davon markiert als `duplikat`: 426
  - davon markiert als `unscharf` (Laplacian-Varianz Threshold=40): 691

Trainingsverlauf (aus `logs/training_metrics/epoch_metrics_2026-04-21_21-43-41.csv`):

- Höchste in-Log gemessene Validierungsgenauigkeit: 0.9874706268310547 (Epoch 12)
- Val_accuracy am Ende des Laufs: 0.9851213693618774 (Epoch 19)
- Finale Trainingsaccuracy (letzte Epoche): 0.9962799549102783

Diese Fakten sind vollständig in den angegebenen Log-Dateien dokumentiert und ermöglichen eine reproduzierbare Nachprüfung.

## g) Diskussion und Interpretation (auf Basis der Fakten)

Die folgenden Aussagen sind interpretativ und stützen sich ausschließlich auf die oben dokumentierten Fakten (Training-Logs, Cleaning-Logs, Classification Report, Confusion-Matrix-Abbildung).

- Modellleistung und Generalisierung: Das Modell erreicht auf den bereinigten Trainings-/Validierungsdaten sehr hohe Validierungsaccuracies (bis 0.987). Die externe Test-Accuracy (0.917) liegt jedoch deutlich darunter, was auf eine Generalisierungsdifferenz hinweist. Diese Differenz kann verschiedene Ursachen haben (siehe unten), ist aber empirisch aus den vorliegenden Logs belegt.

- Per-Klasse-Analyse: Klassen B und C zeigen exzellente Performa (B: precision=0.99 / recall=0.98, C: precision=0.92 / recall=0.99). Klasse Y zeichnet sich durch einen hohen Recall (0.94) bei vergleichsweise niedriger Präzision (0.82) aus; dies bedeutet, dass Y selten übersehen wird, aber viele Fehlzuweisungen (False-Positives) in die Klasse Y erfolgen. Klassen wie O (precision=0.97, recall=0.87) werden vergleichsweise konservativ vorhergesagt (wenige False-Positives), aber häufiger übersehen (höherer Anteil False-Negatives).

- Confusion Matrix (qualitative Interpretation): Die in `logs/confusion_matrices/confusion_matrix_2026-04-22_13-37-47.png` abgelegte Matrix korreliert mit den per-Klasse-Metriken: Es ist erkennbar, dass bestimmte Klassen (insbesondere solche mit ähnlicher Handform oder ähnlicher Silhouette) häufiger gegenseitig verwechselt werden. Konkret zeigt die Matrix erhöhte Einträge, die andere Klassen in Y einordnen (vereinbar mit Ys niedriger Präzision), während B und C weitgehend entlang der Diagonalen konzentriert sind (wenige Fehlklassifikationen).

- Einfluss der Datenbereinigung: Der Cleaning-Run entfernte 1.117 Bilder, davon 426 Duplikate und 691 unscharfe Bilder. Dieser Schritt reduziert offensichtliches Rauschen und redundante Muster in den Trainingsdaten. Die sehr hohen Validierungswerte auf den bereinigten Daten deuten darauf hin, dass das Cleaning die Trainingsstabilität verbessert hat; die Differenz zur externen Test-Accuracy legt jedoch nahe, dass verbleibende Domänenunterschiede zwischen Trainings- und Testdaten bestehen.

- Einfluss von Data Augmentation und Transfer Learning: Die Verwendung von Data Augmentation (Flip, Rotation, Helligkeitsvariation, Zoom) und Transfer Learning (MobileNetV2-Basis) ist ein plausibler Grund für die schnelle Konvergenz und die hohen Validierungswerte. Augmentation trägt zur Robustheit gegenüber Bildvariationen bei, Transfer Learning liefert vortrainierte Repräsentationen, die mit begrenzter Datenmenge effektiv adaptiert werden können.

- Mögliche Ursachen für Fehlklassifikationen: Basierend auf den Fakten sind wahrscheinliche Ursachen:
  - Domänenverschiebung zwischen Trainingsdaten (bereinigt) und externen Test-Sets
  - Visuelle Ähnlichkeiten zwischen bestimmten Handgesten (Form, Fingerstellung)
  - Unterschiedliche Bildqualität oder Perspektiven in den externen Testdaten
  - Klassenbalance- oder Sampling-Effekte beim Zusammenfügen mehrerer Quellen

Diese Punkte sollten als Hypothesen für weitere kontrollierte Experimente betrachtet werden; alle hier genannten Aussagen basieren ausschließlich auf den im Projekt vorhandenen Logs und Artefakten.

## Praktische Anwendung

Zusätzlich zur Trainings- und Evaluations-Pipeline wurde eine Echtzeit-Anwendung implementiert, die folgendes umfasst (Tatsachen, implementiert im Projektcode):

- Laden des trainierten Modells: Das Skript `predict_webcam.py` lädt ein gespeichertes Modell aus `models/` (z. B. `models/sign_language_model.h5` oder zeitgestempelte Varianten) mittels TensorFlow/Keras.
- Erfassung der Gesten über Webcam: `predict_webcam.py` nutzt OpenCV zur Kamerainitialisierung und kontinuierlichen Frame-Erfassung.
- Vorhersage in Echtzeit: Jedes Kameraframe wird vorverarbeitet (Größenanpassung, Normalisierung) und an das geladene Modell übergeben; das Modell liefert Wahrscheinlichkeitswerte für die acht Klassen.
- Anzeige des erkannten Buchstabens: Das erkannte Label (Max-Wahrscheinlichkeit) wird in das Kamerafenster gerendert, sodass die Vorhersage unmittelbar sichtbar ist.

Die Implementierung in `predict_webcam.py` demonstriert die praktische Anwendbarkeit des entwickelten Modells in Echtzeit-Szenarien; Details zur Nutzung stehen in der Datei selbst und der zugehörigen README-Abschnitte.

## Schlussreflexion

Die abschließende Reflexion fasst zentrale Erkenntnisse des Projekts zusammen und bewertet deren Bedeutung für zukünftige Arbeiten.

- Datenqualität ist zentral: Die Bereinigung (1.117 entfernte Bilder im relevanten Run) hat die Trainingsstabilität und Metriken auf den Validierungsdaten deutlich verbessert. Saubere, nicht redundante Daten sind eine Grundvoraussetzung für robuste Modelle.
- Externe Tests sind notwendig: Der Abstand zwischen Validierungs- und externer Test-Performance zeigt, dass interne Validierungsmetriken allein nicht ausreichend sind, um Generalisierungsfähigkeit zu beurteilen. Externe, ungesehene Testsets liefern ein realistischeres Bild der Modellqualität.
- Methodische Balance: Transfer Learning und Data Augmentation haben den Trainingsprozess effizient gemacht und zu schnellen Fortschritten geführt; sie ersetzen jedoch nicht die Notwendigkeit, Domänenunterschiede und Datenrepräsentativität systematisch zu adressieren.
- Herausforderungen bei Handgesten: Visuelle Ähnlichkeiten, variierende Perspektiven und Bildqualität sind schwer zu eliminierende Fehlerquellen. Eine Kombination aus besserer Segmentierung (z. B. MediaPipe), zusätzlichen Trainingsbeispielen und gezielter Augmentation könnte hier Abhilfe schaffen.
- Reproduzierbare Pipelines sind essenziell: Zeitgestempelte Logs, gespeicherte Modelle und klar dokumentierte Cleaning-Schritte ermöglichen nachvollziehbare Experimente und vereinfachen Fehleranalyse und iterative Verbesserung.

Insgesamt liefert das Projekt eine reproduzierbare, praxisnahe Lösung mit guter Basisleistung (externe Accuracy 0.917), deren weitere Optimierung vor allem an der Reduktion von Domänenunterschieden und gezielter Fehleranalyse ansetzen sollte.

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


**Letzte Aktualisierung:** 04. Juni 2026

