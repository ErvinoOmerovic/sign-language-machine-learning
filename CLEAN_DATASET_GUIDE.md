# 🧹 Clean Dataset Skript - Dokumentation

## Überblick

Das `clean_dataset.py` Skript bereinigt Rohdaten aus `data_raw/` und speichert saubere Bilder in `data_cleaned/`.

**Wichtig:**
- ✅ `data_raw/` wird NICHT verändert (nur gelesen)
- ✅ Alle bereinigten Daten gehen nach `data_cleaned/`
- ✅ Klassenstruktur (A, B, C, ...) bleibt erhalten

---

## 🔧 Cleaning-Logik

### 1. **Beschädigte Bilder filtern**
- Bilder, die nicht geladen werden können
- Falsche Formate oder korrupte Dateien
- **Entfernungsgrund:** `beschädigt`

### 2. **Zu kleine Bilder entfernen**
- Standard: Minimum 50×50 Pixel
- Kann mit `--min-size` angepasst werden
- **Entfernungsgrund:** `zu klein`

### 3. **Hand-Erkennung und Ausschnitt**
- Nutzt **MediaPipe** zur Erkennung
- Schneidet die erkannte Hand aus + Padding
- **Fallback:** Zentraler Ausschnitt (90% des Bildes), falls keine Hand erkannt
- Bewahrt Variation: unterschiedliche Lichtverhältnisse, Hautfarben, Perspektiven

### 4. **Unscharfe Bilder filtern** ⚠️ (nicht zu streng!)
- **Methode:** Laplacian Varianz
- **Default Threshold:** 40 (nicht zu streng)
- Entfernt nur **stark unscharfe** Bilder, behält Variation
- **Entfernungsgrund:** `unscharf`

### 5. **Extrem schlechte Bilder filtern** ⭐ (NEU)
- Erkennt Bilder mit **95% uniformer Farbe**
- Z.B. komplett schwarz, weiß oder einfarbig
- **Entfernungsgrund:** `extrem niedriger Inhalt`

### 6. **Duplikate entfernen** 
- **Methode:** Perceptual Hashing (8×8 Hash)
- Findet identische und sehr ähnliche Bilder
- Behält nur das erste Bild jeder Gruppe
- **Entfernungsgrund:** `duplikat`

### 7. **Größe und Format normalisieren**
- Finales Format: **224×224 Pixel** (mit Padding)
- Format: **PNG**
- Qualität: 95

---

## 📋 Verwendung

### Standard-Cleaning (empfohlen)
```bash
python clean_dataset.py
```

**Das macht:**
- Liest aus `data_raw/`
- Speichert nach `data_cleaned/`
- Duplikat-Prüfung: aktiviert
- Blur-Threshold: 40

---

### Angepasste Parameter

```bash
# Strengeres Blur-Filtern (nur sehr klare Bilder)
python clean_dataset.py --blur-threshold 50

# Toleranteres Blur-Filtern (mehr Variation)
python clean_dataset.py --blur-threshold 30

# Keine Duplikat-Prüfung (schneller, aber mit Duplikaten)
python clean_dataset.py --no-deduplicate

# Normalisierte Arrays zusätzlich speichern (.npy-Dateien)
python clean_dataset.py --normalize

# Mit Zeitstempel (z.B. data_cleaned_2026-04-21_14-30)
python clean_dataset.py --timestamped-folder

# Kombiniert
python clean_dataset.py --blur-threshold 35 --normalize --timestamped-folder
```

---

## 📊 Output und Logs

Nach dem Cleaning findest du:

### 1. **Bereinigte Bilder**
```
data_cleaned/
├── A/ (z.B. 900 Bilder)
├── B/
├── C/
└── ...
```

### 2. **Entfernte Bilder - Log**
```
logs/removed_images.txt
```
Zeigt: Zeitstempel | Dateiname | Grund (z.B. `duplikat`, `unscharf`, `beschädigt`)

### 3. **Verarbeitungsmethode - Log**
```
logs/processing_methods.txt
```
Zeigt: Zeitstempel | Dateiname | Methode (`MediaPipe` oder `Fallback`)

### 4. **Klassenbalance - Konsole-Output**
```
📊 Klassenbalance nach Reinigung:
  A: 850 Bilder
  B: 845 Bilder
  C: 820 Bilder
  ...
  Min: 750, Max: 900, Durchschnitt: 841.2
  Balance-Ratio: 83.33%
```

---

## 🎯 Empfohlene Workflow

```bash
# 1. Erstes Cleaning (mit Standard-Parametern)
python clean_dataset.py

# 2. Prüfe die Logs und die Klassenbalance
# → Schaue: Wie viele Bilder wurden entfernt?
# → Sind die Klassen ausreichend ausgewogen?

# 3. Falls nötig: Mit angepassten Parametern neu bereinigen
python clean_dataset.py --blur-threshold 35

# 4. Wenn zufrieden: Nutze data_cleaned/ zum Training
```

---

## ✅ Was wird BEHALTEN (Variation)

Das Skript behält diese Variation **bewusst**:

- 📸 **Unterschiedliche Lichtverhältnisse** (hell/dunkel)
- 🎨 **Unterschiedliche Hautfarben**
- 🔄 **Leichte Drehungen und Perspektiven**
- 👥 **Unterschiedliche Handhaltungen** (MediaPipe)
- 🌈 **Verschiedene Hintergründe** (bei Fallback)

Das macht das Modell **robuster!**

---

## ⚙️ Technische Details

| Parameter | Default | Beschreibung |
|-----------|---------|-------------|
| `--source-dir` | `data_raw` | Quellordner |
| `--dest-dir` | `data_cleaned` | Zielordner |
| `--min-size` | 50 | Min. Breite/Höhe (Pixel) |
| `--blur-threshold` | 40 | Laplacian Varianz Threshold |
| `--normalize` | False | Speichere auch .npy Arrays |
| `--no-deduplicate` | - | Deaktiviere Duplikat-Check |
| `--timestamped-folder` | False | Ordner mit Zeitstempel |

---

## 🐛 Troubleshooting

### "MediaPipe nicht verfügbar"
```bash
pip install mediapipe
```

### Zu viele Bilder werden entfernt
→ Reduziere `--blur-threshold` (z.B. auf 30)

### Nicht genug Bilder werden entfernt
→ Erhöhe `--blur-threshold` (z.B. auf 50)

### Duplikate trotz `--no-deduplicate` nicht entfernt?
→ Das ist korrekt - der Flag deaktiviert die Duplikat-Prüfung

### `data_cleaned` wird überschrieben
→ Das ist beabsichtigt! Nutze `--timestamped-folder` um alte Versionen zu behalten

---

## 📈 Nächste Schritte

Nach dem Cleaning:

1. **Training mit `data_cleaned/`** (optional)
   ```bash
   python train_simple.py  # (bei Bedarf data_raw → data_cleaned ändern)
   ```

2. **Externe Tests** für finale Evaluation
   ```bash
   python evaluate.py
   ```


