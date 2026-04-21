#!/bin/bash

# Setup Script für ml_train Umgebung
# Installiert alle fehlenden Abhängigkeiten für das Sign Language Recognition Projekt

echo "=========================================="
echo "Sign Language Recognition - Environment Setup"
echo "=========================================="

# Prüfe ob conda vorhanden ist
if ! command -v conda &> /dev/null; then
    echo "❌ Conda nicht gefunden. Bitte installieren Sie Anaconda/Miniconda."
    exit 1
fi

echo "📦 Installiere fehlende Pakete in ml_train Umgebung..."
echo ""

# Installiere fehlende Pakete
conda run -n ml_train pip install -q \
    opencv-python \
    numpy \
    tensorflow \
    keras \
    matplotlib \
    scikit-learn \
    pillow \
    mediapipe

echo ""
echo "✓ Pakete installiert"
echo ""

# Verifiziere Installation
echo "🔍 Verifiziere Installation..."
conda run -n ml_train python -c "
import cv2
import numpy as np
import tensorflow as tf
import matplotlib
import sklearn
import PIL
import mediapipe

print('✓ cv2 (OpenCV):', cv2.__version__)
print('✓ numpy:', np.__version__)
print('✓ TensorFlow:', tf.__version__)
print('✓ matplotlib:', matplotlib.__version__)
print('✓ scikit-learn:', sklearn.__version__)
print('✓ Pillow:', PIL.__version__)
print('✓ MediaPipe: OK')
print('')
print('========================================')
print('✓ Alle Pakete erfolgreich installiert!')
print('========================================')
" 2>&1 | tail -20

echo ""
echo "Sie können jetzt das Training starten:"
echo "  conda run -n ml_train python train_simple.py"
