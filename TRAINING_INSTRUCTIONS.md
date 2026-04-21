# Training des Sign Language Recognition Models

## Codefehler behoben ✓

Die folgenden Fehler wurden behoben:
1. **predict_webcam.py**: Global-Deklarationsfehler bei FLIP-Flag - nun als Liste implementiert
2. **utils.py**: Duplizierte Funktionsdefinition entfernt
3. **Seaborn-Abhängigkeit**: Entfernt, verwende nun matplotlib für Confusion Matrix

## So starten Sie das Training

Öffnen Sie ein Terminal und führen Sie aus:

```bash
cd "/Users/ervin2/Machine Learning MWI/Neues Projekt ML"
conda run -n ml_train python train_simple.py
```

## Was wird trainiert?

Das Script trainiert ein CNN-Modell mit folgenden Parametern:

### Daten
- **Klassen**: A, B, C, L, V, W, O, Y (8 Gebärdensprachen-Buchstaben)
- **Bildgröße**: 224×224 Pixel (RGB)
- **Datensatz-Aufteilung**: 70% Training, 15% Validation, 15% Test
- **Normalisierung**: 0-1 (durch Division durch 255)

### Augmentation
- Horizontales Spiegeln (50% Wahrscheinlichkeit)
- Rotation ±15° (30% Wahrscheinlichkeit)
- Helligkeitsanpassung 0.8-1.2× (30% Wahrscheinlichkeit)

### Modellarchitektur
- **Base**: MobileNetV2 (vortrainiert auf ImageNet)
- **Fine-tuning**: Base-Layer gefroren, Custom-Layers für die 8 Klassen
- **Custom Layers**:
  - GlobalAveragePooling2D
  - Dense(128, activation='relu')
  - Dropout(0.5)
  - Dense(8, activation='softmax')

### Trainingsparameter
- **Optimizer**: Adam (lr=0.001)
- **Loss**: sparse_categorical_crossentropy
- **Epochs**: 40 (mit EarlyStopping bei 5 Epochen ohne Verbesserung)
- **Batch Size**: 32
- **Learning Rate Scheduler**: ReduceLROnPlateau

## Gespeicherte Dateien

Nach dem Training werden folgende Dateien mit Zeitstempel gespeichert:

```
models/
  └── sign_language_model_YYYY-MM-DD_HH-MM-SS.h5

logs/
  ├── training_curves_YYYY-MM-DD_HH-MM-SS.png         (Loss & Accuracy Kurven)
  ├── epoch_metrics_YYYY-MM-DD_HH-MM-SS.csv           (Metriken pro Epoche)
  ├── confusion_matrix_YYYY-MM-DD_HH-MM-SS.png        (Test-Confusion Matrix)
  └── classification_report_YYYY-MM-DD_HH-MM-SS.txt   (Precision, Recall, F1-Score)
```

## Nach dem Training

### Evaluation eines gespeicherten Modells

```bash
python evaluate.py
```

### Live-Webcam-Prediction

```bash
python predict_webcam.py
```

## Features der Implementierung

✓ **Zeitstempel bei allen Dateien** - Alte Ergebnisse bleiben erhalten  
✓ **Automatische Confusion Matrix** - Nach dem Training sofort verfügbar  
✓ **Classification Report** - Precision, Recall, F1-Score pro Klasse  
✓ **CSV-Logging** - Alle Epoch-Metriken im CSV-Format  
✓ **Live-Output** - Trainingsprogress wird in Echtzeit angezeigt  
✓ **Callback-System** - Early Stopping, LR Scheduler, Checkpoint  

## Klassenreihenfolge

**WICHTIG**: Die Klassenreihenfolge ist fest definiert und muss überall identisch sein:

```python
CLASSES = ['A', 'B', 'C', 'L', 'V', 'W', 'O', 'Y']
```

Diese Reihenfolge wird konsistent in:
- `train_simple.py`
- `evaluate.py`
- `predict_webcam.py`
- `utils.py`

verwendet.
