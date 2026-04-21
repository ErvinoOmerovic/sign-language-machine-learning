# 🚨 ENVIRONMENT SETUP ERFORDERLICH

## Problem
Die `ml_train` conda-Umgebung hat nicht alle erforderlichen Pakete. Besonders **opencv-python (cv2)** fehlt.

## Lösung

### Option 1: Automatisches Setup-Script (Empfohlen)

Öffnen Sie ein Terminal und führen Sie aus:

```bash
cd "/Users/ervin2/Machine Learning MWI/Neues Projekt ML"
conda run -n ml_train python setup_environment.py
```

Dieses Script:
- ✓ Prüft alle installierten Pakete
- ✓ Installiert fehlende Abhängigkeiten
- ✓ Verifiziert die Installation

### Option 2: Manuelle Installation

```bash
conda activate ml_train
pip install -r requirements.txt
```

### Option 3: Einzelne Pakete installieren

```bash
conda run -n ml_train pip install \
  opencv-python==4.9.0.80 \
  tensorflow==2.16.1 \
  keras \
  matplotlib==3.8.4 \
  scikit-learn==1.4.2 \
  numpy==1.26.4 \
  pillow==10.3.0 \
  mediapipe==0.10.11
```

## Nach Installation

Überprüfen Sie die Installation:

```bash
conda run -n ml_train python -c "
import cv2, tensorflow, numpy, matplotlib, sklearn, PIL, mediapipe
print('✓ Alle Pakete OK!')
"
```

## Training starten

```bash
cd "/Users/ervin2/Machine Learning MWI/Neues Projekt ML"
conda run -n ml_train python train_simple.py
```

---

## Erforderliche Pakete

| Paket | Version | Zweck |
|-------|---------|-------|
| tensorflow | 2.16.1 | Deep Learning Framework |
| keras | Latest | Neural Network API |
| opencv-python | 4.9.0.80 | Bildverarbeitung |
| numpy | 1.26.4 | Numerische Berechnungen |
| matplotlib | 3.8.4 | Visualisierung |
| scikit-learn | 1.4.2 | ML-Metriken & Preprocessing |
| pillow | 10.3.0 | Bildbearbeitung |
| mediapipe | 0.10.11 | Hand/Pose Detection (optional) |
