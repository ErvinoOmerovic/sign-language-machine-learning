# 📋 Cleaning-Logs mit Timestamps

## 🎯 Neue Struktur

Jeder Cleaning-Run bekommt **seinen eigenen Ordner mit Timestamp**:

```
logs/cleaning_logs/
├── 2026-04-20_10-15-30/
│   ├── removed_images.txt
│   ├── processing_methods.txt
│   └── summary.txt
├── 2026-04-21_14-30-45/
│   ├── removed_images.txt
│   ├── processing_methods.txt
│   └── summary.txt
└── 2026-04-21_15-45-12/
    ├── removed_images.txt
    ├── processing_methods.txt
    └── summary.txt
```

**Timestamp-Format:** `YYYY-MM-DD_HH-MM-SS`

---

## 📁 Inhalte der Log-Dateien

### 1. **removed_images.txt**
Detaillierte Liste aller entfernten Bilder mit Grund:

```
2026-04-21T15:45:12	data_raw/A/image_001.jpg	duplikat
2026-04-21T15:45:13	data_raw/A/image_042.jpg	unscharf (threshold=40)
2026-04-21T15:45:14	data_raw/B/image_105.jpg	beschädigt
2026-04-21T15:45:15	data_raw/C/image_067.jpg	extrem niedriger Inhalt
```

### 2. **processing_methods.txt**
Dokumentiert, welche Verarbeitungsmethode für jedes Bild verwendet wurde:

```
2026-04-21T15:45:12	data_raw/A/image_001.jpg	MediaPipe
2026-04-21T15:45:12	data_raw/A/image_002.jpg	Fallback
2026-04-21T15:45:13	data_raw/A/image_003.jpg	MediaPipe
```

### 3. **summary.txt** (NEU!)
Zusammenfassung des gesamten Cleaning-Runs:

```
================================================================================
CLEANING SESSION SUMMARY
================================================================================

Timestamp: 2026-04-21_15-45-12

KONFIGURATION:
----------------------------------------
Source Directory:      /path/to/data_raw
Destination Directory: /path/to/data_cleaned
Min Size:              50 Pixel
Blur Threshold:        40
Normalization:         Nein
Deduplication:         Ja (aktiv)

ERGEBNISSE:
----------------------------------------
Gesamt verarbeitet:    7200
Entfernt:              450
Gespeichert:           6750
Erfolgsquote:          93.8%

LOGS:
----------------------------------------
Removed Images:        removed_images.txt
Processing Methods:    processing_methods.txt

================================================================================
```

---

## 🚀 Verwendung

```bash
python clean_dataset.py
```

**Console Output:**
```
--- Reinigung abgeschlossen ---
Archiviert: 7200 Dateien überprüft
Entfernt: 450 Dateien
Gespeichert: 6750 bereinigte Bilder in data_cleaned

📝 Erstelle Cleaning-Zusammenfassung...
✓ Zusammenfassung gespeichert: logs/cleaning_logs/2026-04-21_15-45-12/summary.txt

✓ Alle Logs gespeichert in: logs/cleaning_logs/2026-04-21_15-45-12
```

---

## ✅ Vorteile der neuen Struktur

### 1. **Nachverfolgbarkeit**
- Jeder Run hat seine eigene Datei
- Keine Vermischung von mehreren Runs
- Vollständige Historie bleibt erhalten

### 2. **Vergleichbarkeit**
- Vergleiche verschiedene Parameter-Kombinationen
- Sehe die Auswirkungen von Änderungen
- Analysiere historische Trends

### 3. **Dokumentation**
- summary.txt speichert alle Parameter
- Reproduzierbarkeit garantiert
- Perfekt für Reporting

### 4. **Automatisch**
- Keine manuelle Dateiverwaltung nötig
- Timestamps verhindern Überschreibung
- Intelligente Ordnerstruktur

---

## 🔍 Typischer Workflow

```bash
# Run 1: Standard-Parameter
python clean_dataset.py
# Logs: logs/cleaning_logs/2026-04-21_15-30-00/

# Run 2: Strengeres Blur-Filtern
python clean_dataset.py --blur-threshold 50
# Logs: logs/cleaning_logs/2026-04-21_15-35-00/

# Run 3: Toleranteres Blur-Filtern
python clean_dataset.py --blur-threshold 30
# Logs: logs/cleaning_logs/2026-04-21_15-40-00/

# Vergleich: Öffne die drei summary.txt Dateien nebeneinander
```

---

## 📊 Beispiel: Historische Analyse

```
logs/cleaning_logs/
├── 2026-04-20_10-00-00/  (blur_threshold=40, deduplicate=True)
│   └── summary.txt: 93.8% Erfolgsquote
├── 2026-04-20_11-00-00/  (blur_threshold=50, deduplicate=True)
│   └── summary.txt: 95.2% Erfolgsquote
└── 2026-04-21_15-45-00/  (blur_threshold=35, deduplicate=True)
    └── summary.txt: 92.1% Erfolgsquote
```

Du kannst jetzt einfach alle summary.txt Dateien öffnen und vergleichen!

---

## 🎯 Wichtige Punkte

- ✅ **Eigener Ordner pro Run** - Keine Vermischung
- ✅ **Timestamps automatisch** - Keine manuellen Namen nötig
- ✅ **data_raw NICHT verändert** - Nur gelesen
- ✅ **data_cleaned bleibt separiert** - Bilder getrennt von Logs
- ✅ **Alle Parameter dokumentiert** - In summary.txt
- ✅ **Volle Nachverfolgbarkeit** - Jedes Bild, jeder Grund geloggt


