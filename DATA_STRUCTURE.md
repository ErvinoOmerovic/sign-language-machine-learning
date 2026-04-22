# 📊 Datensatz-Struktur und Evaluation

## 🎯 Endgültige Setup-Struktur (nach Fix)

### Training & Validation
```
data_raw/
├── A/    (mit Bildern von Dataset 1, Dataset 2 mit "dataset2_" Präfix, Dataset 3 mit "dataset3_" Präfix, etc.)
├── B/
├── C/
├── L/
├── O/
├── V/
├── W/
└── Y/
```

**Verwendung:** Training (70%) + Validation (30%)
- Diese Daten werden für das Training des Modells genutzt
- Der Split erfolgt im Training-Skript automatisch

---

### FINALE EVALUATION - Externe Testdatensätze
```
external_test/
├── dataset2/   (Original zweiter Datensatz - Test Only)
│   ├── A/ (75-100 Bilder)
│   ├── B/
│   └── ...
├── dataset3/   (Neuer dritter Datensatz - Test Only)
│   ├── A/ (100 Bilder)
│   ├── B/
│   └── ...
├── dataset4/   (Zukünftiger vierter Datensatz - Test Only)
│   ├── A/
│   └── ...
└── ...
```

**Verwendung:** FINALE TEST (nur für Evaluation!)
- Diese Daten werden NICHT zum Training verwendet
- Sie dienen nur zur finalen Evaluation des trainierten Modells
- Laden alle externen Datensätze zusammen als ein großes Test-Set

---

## 📋 Workflow

### 1️⃣ Trainieren
```bash
python train_simple.py
```
- Lädt Daten aus `data_raw/`
- Splittet automatisch: 70% Training, 30% Validation
- Speichert Modell und interne Validierungsmetriken

**Output:** 
- `models/sign_language_model.h5`
- `logs/classification_report_TIMESTAMP.txt` (interne Validierung)
- `logs/confusion_matrix_TIMESTAMP.png` (interne Validierung)

⚠️ Die interne Validierung ist NUR für Quick-Checks gedacht!

---

### 2️⃣ FINALE Evaluation (mit externen Test-Daten)
```bash
python evaluate.py
```
- Lädt **alle** Bilder aus `external_test/` (dataset2, dataset3, ...)
- Testet das Modell auf komplett externen Daten
- Speichert finale Ergebnisse

**Output:**
- `logs/classification_report_TIMESTAMP.txt` (EXTERNE TEST EVALUATION)
- `logs/confusion_matrix_TIMESTAMP.png` (externe Testdaten)

✅ Dies ist die ECHTE finale Evaluation!

---

## 📊 Datenverteilung Beispiel

### Vor Integration von Dataset 3:
```
data_raw/
├── A: ~986 Bilder (Dataset 1 + Dataset 2 mit dataset2_ Präfix)
├── B: ~984 Bilder
└── ...

external_test/
└── dataset2/
    ├── A: 75 Bilder
    ├── B: 75 Bilder
    └── ...
```

### Nach Integration von Dataset 3:
```
data_raw/
├── A: ~1886 Bilder (Dataset 1 + Dataset 2 + Dataset 3 mit dataset3_ Präfix)
├── B: ~1884 Bilder
└── ...

external_test/
├── dataset2/
│   ├── A: 75 Bilder
│   └── ...
└── dataset3/
    ├── A: 100 Bilder (NEUE externe Testdaten!)
    └── ...
```

---

## 🔄 Neue Datensätze hinzufügen

Wenn du neue Datensätze hast (z.B. Dataset 4):

1. **Vorbereitung** (auf deinem System):
   ```
   /Users/ervin2/Desktop/ML Datensatz 4/
   ├── Train/
   │   ├── A/ (mit Bildern)
   │   ├── B/
   │   └── ...
   └── Test/
       ├── A/ (mit Bildern)
       ├── B/
       └── ...
   ```

2. **Integration** (Skript wird bereitgestellt):
   ```bash
   python integrate_dataset4.py
   ```

3. **Ergebnis:**
   - Train-Bilder mit `dataset4_` Präfix in `data_raw/{CLASS}/`
   - Test-Bilder in `external_test/dataset4/{CLASS}/`

---

## ✅ Checkliste für korrektes Setup

- [ ] `data_raw/` enthält nur Trainingsdaten (inkl. aus mehreren Datasets mit Präfixen)
- [ ] `external_test/` enthält separate, externe Test-Datensätze
- [ ] `train_simple.py` trainiert auf `data_raw/` (70/30 split)
- [ ] `evaluate.py` testet auf `external_test/` (alle Datasets zusammen)
- [ ] Logs mit "EXTERNE TEST EVALUATION" sind die finalen Metriken

---

## 📈 Metriken interpretieren

### Training (internal):
- Anzeigt Overfitting/Underfitting während des Trainings
- Nicht für finale Evaluation relevant

### Externe Evaluation (final):
- Echte Modell-Performance auf unbekannten Daten
- Dies ist das Maß für die echte Generaliserbarkeit
- **Dies ist die Metrik, die zählt!** ✅

---

# 🗂️ Logs-Struktur

```
logs/
├── classification_reports/  # Classification Reports (TXT)
├── confusion_matrices/      # Confusion Matrices (PNG)
├── training_metrics/        # Epoch-Metriken (CSV)
├── training_curves/         # Trainingskurven (PNG)
├── data_analysis/           # Datenanalyse (Plots)
└── cleaning_logs/           # Cleaning-Logs (TXT)
```
