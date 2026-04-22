# 📊 Automatische Datenverteilungs-Analyse

## 🎯 Neue Funktion in `clean_dataset.py`

Nach jedem Cleaning wird **automatisch** eine vollständige Datenverteilungs-Analyse erstellt:

### 📁 Gespeicherte Dateien

```
logs/data_analysis/
├── data_distribution_2026-04-21_15-30-45.png  # Balkendiagramm
└── data_distribution_2026-04-21_15-30-45.txt  # Detaillierte Tabelle
```

**Wichtig:** Jede Analyse bekommt einen **Timestamp**, damit vorherige Analysen **nicht überschrieben** werden!

---

## 📈 Was wird erstellt?

### 1. **Balkendiagramm** (PNG)
- **Titel:** "Bildverteilung pro Klasse (nach Cleaning - TIMESTAMP)"
- **X-Achse:** Klassen (A, B, C, ...)
- **Y-Achse:** Anzahl Bilder pro Klasse
- **Features:**
  - Werte über jedem Balken
  - Statistiken als Textbox (Total, Min, Max, Avg)
  - Grid-Linien für bessere Lesbarkeit

### 2. **Text-Tabelle** (TXT)
Detaillierte Analyse mit:
```
================================================================================
DATA DISTRIBUTION ANALYSIS - 2026-04-21_15-30-45
================================================================================

Klasse | Anzahl Bilder
-------|--------------
A      |          850
B      |          845
C      |          820
...
-------|--------------
Total  |         6750

STATISTIKEN:
----------------------------------------
Gesamtanzahl Bilder: 6750
Klassen: 8
Minimum pro Klasse: 755
Maximum pro Klasse: 900
Durchschnitt pro Klasse: 843.8
Balance-Ratio: 83.89%

Balance-Status: GUT: Leichte Ungleichheit
================================================================================
```

---

## 🔍 Balance-Bewertung

Die Analyse bewertet automatisch die Klassenbalance:

| Balance-Ratio | Status | Bedeutung |
|---------------|--------|-----------|
| ≥ 80% | **SEHR GUT** | Ausgeglichene Verteilung |
| ≥ 60% | **GUT** | Leichte Ungleichheit |
| ≥ 40% | **MITTEL** | Moderate Ungleichheit |
| < 40% | **SCHLECHT** | Starke Ungleichheit |
| = 0% | **KRITISCH** | Klasse mit 0 Bildern! |

---

## 🚀 Automatische Ausführung

Die Analyse läuft **automatisch** nach jedem Cleaning:

```bash
python clean_dataset.py
```

**Console Output:**
```
--- Reinigung abgeschlossen ---
Archiviert: 7200 Dateien überprüft
Entfernt: 450 Dateien
Gespeichert: 6750 bereinigte Bilder in data_cleaned

📈 Erstelle Datenverteilungs-Analyse...
✓ Datenverteilungs-Analyse gespeichert:
  Diagramm: logs/data_analysis/data_distribution_2026-04-21_15-30-45.png
  Tabelle:  logs/data_analysis/data_distribution_2026-04-21_15-30-45.txt
```

---

## 📊 Vergleich mit vorheriger Version

### Vorher (nur Console-Output):
```
📊 Klassenbalance nach Reinigung:
  A: 850 Bilder
  B: 845 Bilder
  ...
  Min: 755, Max: 900, Durchschnitt: 843.8
  Balance-Ratio: 83.89%
```

### Jetzt (zusätzlich gespeicherte Analyse):
- ✅ **Balkendiagramm** für visuelle Analyse
- ✅ **Detaillierte Text-Tabelle** mit Statistiken
- ✅ **Balance-Bewertung** (GUT/MITTEL/SCHLECHT)
- ✅ **Timestamps** - keine Überschreibung alter Analysen
- ✅ **Automatisch** nach jedem Cleaning

---

## 📁 Historische Analysen

Da jede Analyse einen Timestamp bekommt, kannst du historische Cleanings vergleichen:

```
logs/data_analysis/
├── data_distribution_2026-04-20_10-15-30.png  # Erstes Cleaning
├── data_distribution_2026-04-20_10-15-30.txt
├── data_distribution_2026-04-21_15-30-45.png  # Zweites Cleaning
├── data_distribution_2026-04-21_15-30-45.txt
└── ...
```

**Perfekt für Experimente:** Vergleiche verschiedene Cleaning-Parameter!

---

## 🎯 Verwendung

### Nach dem Cleaning:
1. **Schaue dir das Diagramm an:** Visuelle Übersicht der Verteilung
2. **Prüfe die Tabelle:** Detaillierte Statistiken und Balance-Status
3. **Vergleiche historische Analysen:** Wie haben sich die Parameter ausgewirkt?

### Für die Dokumentation:
- **Diagramme** für Berichte und Präsentationen
- **Tabellen** für detaillierte Analyse
- **Timestamps** für Nachvollziehbarkeit

---

## ⚙️ Technische Details

- **Import:** Verwendet `matplotlib.pyplot` für Diagramme
- **Timestamp:** Von `utils.get_timestamp()` (konsistent mit anderen Logs)
- **Ordner:** `logs/data_analysis/` wird automatisch erstellt
- **Format:** PNG für Diagramme, TXT für Tabellen
- **Encoding:** UTF-8 für Textdateien

---

## ✅ Zusammenfassung

Die automatische Datenverteilungs-Analyse gibt dir jetzt:

1. **Sofortige visuelle Übersicht** nach jedem Cleaning
2. **Detaillierte Statistiken** für die Bewertung
3. **Historische Vergleichsmöglichkeit** durch Timestamps
4. **Automatische Balance-Bewertung** (GUT/SCHLECHT/etc.)
5. **Keine manuelle Nacharbeit** nötig

**Perfekt für die Qualitätssicherung deiner Datensätze!** 📊✨
