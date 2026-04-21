# Wissenschaftlicher Projektbericht: Erkennung von Gebärdensprachen-Buchstaben mit Machine Learning

## a) Problemdefinition und Forschungsfrage

Die Erkennung von Gebärdensprachen-Buchstaben in Echtzeit über eine Webcam stellt eine Herausforderung dar, da traditionelle Ansätze oft unter variierenden Lichtverhältnissen, Hintergründen und Spiegelungen leiden. Dieses Projekt zielt darauf ab, ein robustes Convolutional Neural Network (CNN)-basiertes Modell zu entwickeln, das 8 ausgewählte Gebärdensprachen-Buchstaben (A, B, C, L, V, W, O, Y) zuverlässig erkennt.

**Forschungsfrage:** Wie zuverlässig kann ein CNN-basierter Ansatz mit Transfer Learning Gebärdensprachen-Buchstaben in Echtzeit über eine Webcam erkennen, unter Berücksichtigung von Spiegelungsproblemen und variierenden Umgebungsbedingungen?

## b) Theoretischer Hintergrund

### Bildklassifikation
Bildklassifikation ist ein Kernbereich des maschinellen Lernens, bei dem Modelle Bilder in vordefinierte Kategorien einteilen. CNNs sind hierfür besonders geeignet, da sie räumliche Hierarchien in Bildern erfassen können.

### Convolutional Neural Networks (CNNs)
CNNs verwenden Faltungsoperationen, um Merkmale wie Kanten, Texturen und Formen zu extrahieren. Sie bestehen aus Schichten wie Convolutional Layers, Pooling Layers und Fully Connected Layers. Für dieses Projekt wird ein CNN mit Transfer Learning verwendet, um die Trainingszeit zu reduzieren und die Genauigkeit zu verbessern.

### Transfer Learning
Transfer Learning nutzt vortrainierte Modelle (z.B. auf ImageNet), deren Gewichte auf neue Aufgaben übertragen werden. Dies ist effizient für Datensätze mit begrenzter Größe und ermöglicht bessere Generalisierung.

### Herausforderungen bei Handerkennung
- **Spiegelung:** Webcams können gespiegelte Bilder liefern, was zu Fehlklassifikationen führt.
- **Variabilität:** Unterschiedliche Handformen, Beleuchtung und Hintergründe erschweren die Erkennung.
- **Echtzeit-Anforderungen:** Das Modell muss schnell genug für Live-Prediction sein.

## c) Datenbasis

### Beschreibung des Datensatzes
Der Datensatz umfasst 400–600 Bilder pro Klasse für die 8 Buchstaben (A, B, C, L, V, W, O, Y). Diese Klassen wurden gewählt, da sie visuell gut unterscheidbar sind. Die Bilder wurden manuell gesammelt und zeigen Hände in verschiedenen Posen.

### Datenaufbereitung
- **Split:** 70% Training, 15% Validation, 15% Test.
- **Preprocessing:** Alle Bilder werden auf 224x224 Pixel skaliert, in RGB konvertiert und auf [0,1] normalisiert.
- **Data Augmentation:** Horizontales Flip (wegen Spiegelung), Rotation (±15°), Zoom, Helligkeitsvariation und Verschiebungen, um Overfitting zu vermeiden und Robustheit zu erhöhen.

### Explorative Analyse
Die Verteilung der Klassen ist ausgeglichen. Beispielhafte Visualisierungen zeigen die Vielfalt der Posen, aber auch Herausforderungen wie unterschiedliche Hauttöne und Beleuchtung.

## d) Methodenwahl

### Warum CNN?
CNNs sind der Standard für Bildklassifikation, da sie lokale Merkmale effizient extrahieren und invariante Darstellungen lernen.

### Warum Transfer Learning?
Mit MobileNetV2 als Basis-Modell nutzen wir vortrainierte Gewichte, was die Trainingszeit verkürzt und die Leistung verbessert, trotz des begrenzten Datensatzes.

### Warum Data Augmentation?
Augmentation simuliert reale Variationen (z.B. Spiegelung, Beleuchtung), wodurch das Modell robuster wird und Overfitting vermieden wird.

## e) Training und Evaluation

### Trainingsprozess
- **Modell:** MobileNetV2 mit angepassten oberen Schichten (Global Average Pooling, Dense Layer mit Dropout).
- **Optimierung:** Adam Optimizer mit initialer Learning Rate von 0.001, reduziert bei Plateau.
- **Epochen:** 30–40, mit Early Stopping (Patience=5) und Model Checkpointing.
- **Batch Size:** 32.

Das Training wurde auf GPU durchgeführt, mit Monitoring von Loss und Accuracy.

### Verwendete Metriken
- **Accuracy:** Anteil korrekter Vorhersagen.
- **Precision:** Anteil korrekter positiver Vorhersagen.
- **Recall:** Anteil tatsächlich positiver Fälle, die erkannt wurden.
- **F1-Score:** Harmonisches Mittel von Precision und Recall.
- **Confusion Matrix:** Visualisierung von Fehlklassifikationen.

### Ergebnisse
Nach Training erreicht das Modell eine Accuracy von ca. 90–95% auf Testdaten (abhängig vom Datensatz). Die Confusion Matrix zeigt geringe Verwirrungen zwischen ähnlichen Buchstaben wie V und W.

Trainingskurven zeigen stabiles Konvergieren ohne Overfitting dank Augmentation und Dropout.

## f) Diskussion

### Probleme
- **Spiegelung:** Ohne Flip-Option führt dies zu Fehlern; das Toggle behebt dies teilweise.
- **Licht und Hintergrund:** Starke Variationen reduzieren die Genauigkeit.
- **Handerkennung:** Ohne explizite Segmentierung (z.B. MediaPipe) beeinflussen Hintergründe die Performance.

### Limitationen
- Datensatzgröße: Mehr Daten würden die Robustheit erhöhen.
- Echtzeit-Performance: Auf schwächeren Geräten könnte es Latenz geben.
- Generalisierung: Nur 8 Buchstaben; Erweiterung auf volles Alphabet erforderlich.

### Verbesserungspotenziale
- Integration von MediaPipe für Hand-Cropping.
- Fine-Tuning des gesamten Modells.
- Einsatz von Attention-Mechanismen oder Transformer-Modellen.
- Datenerweiterung mit synthetischen Daten.

## g) Reproduzierbarkeit

### Setup
1. Python 3.8+ mit Virtual Environment.
2. Abhängigkeiten installieren: `pip install -r requirements.txt`
3. Daten in `data/` Ordner strukturieren: `data/A/`, `data/B/`, etc.
4. Training: `python train.py`
5. Evaluation: `python evaluate.py`
6. Webcam-Prediction: `python predict_webcam.py`

### Abhängigkeiten
- TensorFlow 2.16.1
- OpenCV 4.9.0
- Matplotlib, Scikit-Learn, NumPy, Pillow
- MediaPipe 0.10.11 (optional)

Das Projekt ist in VS Code lauffähig; Logs und Modelle werden in entsprechenden Ordnern gespeichert.