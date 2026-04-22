# 📁 Logs-Struktur - Übersicht

## 🎯 Neue organisierte Logs-Struktur

Alle Metriken und Berichte werden jetzt in thematischen Unterordnern gespeichert:

```
logs/
├── classification_reports/  # Classification Reports (TXT)
├── confusion_matrices/      # Confusion Matrices (PNG)
├── training_metrics/        # Epoch-Metriken (CSV)
├── training_curves/         # Trainingskurven (PNG)
├── data_analysis/           # Datenanalyse (Plots)
└── cleaning_logs/           # Cleaning-Logs (TXT)
```

---

## 📊 Inhalte der einzelnen Ordner

### 1. `classification_reports/`
**Dateien:** `classification_report_TIMESTAMP.txt`
- **Quelle:** `train_simple.py` (interne Validierung) + `evaluate.py` (externe Tests)
- **Inhalt:** Detaillierte Classification Reports mit Precision, Recall, F1-Score pro Klasse
- **Unterschied:**
  - Interne Reports: "Internal Validation Report"
  - Externe Reports: "EXTERNE TEST EVALUATION"

### 2. `confusion_matrices/`
**Dateien:** `confusion_matrix_TIMESTAMP.png`
- **Quelle:** `train_simple.py` (interne Validierung) + `evaluate.py` (externe Tests)
- **Inhalt:** Visuelle Confusion Matrices als PNG-Bilder
- **Unterschied:**
  - Interne: "Confusion Matrix (Internal Validation Set)"
  - Externe: "Confusion Matrix - External Test Data"

### 3. `training_metrics/`
**Dateien:** `epoch_metrics_TIMESTAMP.csv`
- **Quelle:** `train_simple.py` (CSVLogger Callback)
- **Inhalt:** Epoch-by-Epoch Metriken (Loss, Accuracy, Val_Loss, Val_Accuracy)
- **Verwendung:** Für detaillierte Analyse des Trainingsverlaufs

### 4. `training_curves/`
**Dateien:** `training_curves_TIMESTAMP.png`
- **Quelle:** `train_simple.py` (matplotlib Plot)
- **Inhalt:** Loss- und Accuracy-Kurven über alle Epochen
- **Verwendung:** Visuelle Überprüfung von Overfitting/Underfitting

### 5. `data_analysis/`
**Dateien:** `data_distribution.png`
- **Quelle:** `analyze_distribution.py`
- **Inhalt:** Balkendiagramm der Klassenverteilung
- **Verwendung:** Überprüfung der Datenbalance

### 6. `cleaning_logs/`
**Dateien:** `removed_images.txt`, `processing_methods.txt`
- **Quelle:** `clean_dataset.py`
- **Inhalt:**
  - `removed_images.txt`: Welche Bilder warum entfernt wurden
  - `processing_methods.txt`: Welche Verarbeitungsmethode (MediaPipe/Fallback) verwendet wurde

---

## 🔍 Dateinamen-Konvention

Alle Dateien folgen dem Schema: `{typ}_{timestamp}.{extension}`

**Beispiel:**
- `classification_report_2026-04-21_15-30-45.txt`
- `confusion_matrix_2026-04-21_15-30-45.png`
- `epoch_metrics_2026-04-21_15-30-45.csv`

**Timestamp-Format:** `YYYY-MM-DD_HH-MM-SS`

---

## 📋 Workflow & typische Dateien

### Nach Training (`python train_simple.py`):
```
logs/
├── classification_reports/
│   └── classification_report_2026-04-21_15-30-45.txt  # Interne Validierung
├── confusion_matrices/
│   └── confusion_matrix_2026-04-21_15-30-45.png      # Interne Validierung
├── training_metrics/
│   └── epoch_metrics_2026-04-21_15-30-45.csv         # Epoch-Daten
└── training_curves/
    └── training_curves_2026-04-21_15-30-45.png       # Loss/Accuracy Plots
```

### Nach finaler Evaluation (`python evaluate.py`):
```
logs/
├── classification_reports/
│   └── classification_report_2026-04-21_15-45-12.txt  # EXTERNE TEST EVALUATION ⭐
└── confusion_matrices/
    └── confusion_matrix_2026-04-21_15-45-12.png      # Externe Testdaten ⭐
```

### Nach Daten-Cleaning (`python clean_dataset.py`):
```
logs/
├── cleaning_logs/
│   ├── removed_images.txt                              # Entfernte Bilder
│   └── processing_methods.txt                          # Verarbeitungsmethoden
└── data_analysis/
    └── data_distribution.png                           # Klassenverteilung
```

---

## ✅ Automatische Speicherung

Die automatische Speicherung funktioniert weiterhin wie gewohnt:

- ✅ **Training:** Erstellt automatisch alle relevanten Logs
- ✅ **Evaluation:** Erstellt automatisch finale Metriken
- ✅ **Cleaning:** Erstellt automatisch Cleaning-Logs
- ✅ **Analyse:** Erstellt automatisch Datenanalyse-Plots

**Nichts ändert sich an der Benutzung - nur die Ordnerstruktur ist jetzt übersichtlicher!**

---

## 🧹 Aufräumen alter Logs

Wenn du alte Logs aufräumen möchtest:

```bash
# Zeige alle Logs
find logs/ -type f -name "*.txt" -o -name "*.png" -o -name "*.csv" | sort

# Lösche Logs älter als 7 Tage
find logs/ -type f \( -name "*.txt" -o -name "*.png" -o -name "*.csv" \) -mtime +7 -delete

# Lösche alle Logs (Vorsicht!)
rm -rf logs/*/
```

---

## 📈 Empfohlene Analyse

### Für Modell-Performance:
1. Schaue zuerst in `classification_reports/` nach den **externen** Evaluationen
2. Vergleiche mit `confusion_matrices/` für visuelle Analyse
3. Prüfe `training_curves/` auf Overfitting

### Für Datenqualität:
1. Schaue in `cleaning_logs/removed_images.txt` was entfernt wurde
2. Prüfe `data_analysis/data_distribution.png` für Balance
3. Schaue in `cleaning_logs/processing_methods.txt` für Verarbeitungsstatistiken

---

## 🔧 Technische Details

- **Ordner werden automatisch erstellt** wenn nicht vorhanden
- **Dateien werden nie überschrieben** (Timestamp verhindert Konflikte)
- **Relative Pfade** werden verwendet (arbeitet von Projekt-Root)
- **Thread-Safe** (keine Konflikte bei paralleler Ausführung)

---

## 🎯 Zusammenfassung

Die neue Struktur macht es viel einfacher, die verschiedenen Arten von Logs zu finden und zu analysieren:

- **Classification Reports** → `classification_reports/`
- **Confusion Matrices** → `confusion_matrices/`
- **Training-Daten** → `training_metrics/` + `training_curves/`
- **Daten-Analyse** → `data_analysis/`
- **Cleaning-Logs** → `cleaning_logs/`

**Die automatische Speicherung bleibt 100% erhalten!** 🚀
