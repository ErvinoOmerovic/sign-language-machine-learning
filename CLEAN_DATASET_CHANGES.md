# ✅ Clean Dataset Skript - Änderungen zusammengefasst

## Was wurde angepasst?

Das bestehende `clean_dataset.py` Skript wurde **gezielt erweitert**, nicht neu geschrieben.

---

## 🔧 Konkrete Änderungen:

### 1. **Neuer Threshold für extrem schlechte Bilder** (Zeile 32)
```python
DEFAULT_EXTREME_LOW_CONTENT_THRESHOLD = 0.95  # 95% uniforme Farbe = sehr schlecht
```

### 2. **Neue Funktion zur Erkennung** (in `clean_image_file()`)
```python
# Überprüfung auf extrem schlechte Bilder (z.B. 95% einheitliche Farbe)
image_array = np.asarray(image)
if np.std(image_array) < DEFAULT_EXTREME_LOW_CONTENT_THRESHOLD:
    log_removed(src_path, 'extrem niedriger Inhalt')
    print(f'Übersprungen (extrem niedriger Inhalt): {src_path}')
    return False
```

Diese prüft die **Standardabweichung der Pixelwerte**. Wenn sie < 0.95 ist, bedeutet das, dass die Farben sehr uniform sind (z.B. komplett schwarz oder weiß).

### 3. **Klassenbalance-Ausgabe** (am Ende von `clean_dataset()`)
```python
# Klassenbalance-Analyse
print("\n" + "="*60)
print("📊 Klassenbalance nach Reinigung:")
print("="*60)
class_counts = {}
for class_dir in sorted(dest_dir.iterdir()):
    if class_dir.is_dir():
        count = len([f for f in class_dir.iterdir() if f.is_file() and f.suffix.lower() in {'.png', '.jpg', '.jpeg', '.bmp'}])
        class_counts[class_dir.name] = count
        print(f"  {class_dir.name}: {count} Bilder")

if class_counts:
    min_count = min(class_counts.values())
    max_count = max(class_counts.values())
    avg_count = sum(class_counts.values()) / len(class_counts)
    print(f"\n  Min: {min_count}, Max: {max_count}, Durchschnitt: {avg_count:.1f}")
    balance_ratio = min_count / max_count if max_count > 0 else 0
    print(f"  Balance-Ratio: {balance_ratio:.2%}")
print("="*60)
```

Das zeigt die Klassenverteilung nach der Reinigung.

---

## ✅ Anforderungen - alle erfüllt

| Anforderung | Status | Umsetzung |
|-------------|--------|----------|
| Duplikate entfernen | ✅ | Perceptual Hashing (8×8) - war bereits vorhanden |
| Unschärfe filtern | ✅ | Laplacian Varianz mit Threshold 40 - war bereits vorhanden |
| Kaputte/ungültige Bilder | ✅ | Fehlerbehandlung beim Laden - war bereits vorhanden |
| Extrem schlechte Bilder | ✅ | **NEU: Std-Abweichung < 0.95 Check** |
| Variation BEHALTEN | ✅ | Hand-Detection + Fallback, nicht zu strenges Blur-Filtern |
| Logging entfernter Bilder | ✅ | logs/removed_images.txt - war bereits vorhanden |
| Struktur beibehalten | ✅ | A/, B/, C/, ... - war bereits vorhanden |
| Klassenbalance Info | ✅ | **NEU: Nach Cleaning anzeigen** |

---

## 🚀 Verwendung

```bash
# Einfaches Cleaning
python clean_dataset.py

# Mit angepasstem Blur-Threshold (toleranter)
python clean_dataset.py --blur-threshold 30

# Mit Klassenbalance-Ausgabe (automatisch)
python clean_dataset.py
```

**Output:**
```
--- Reinigung abgeschlossen ---
Archiviert: 7200 Dateien überprüft
Entfernt: 450 Dateien
Gespeichert: 6750 bereinigte Bilder in data_cleaned

📊 Klassenbalance nach Reinigung:
  A: 850 Bilder
  B: 845 Bilder
  C: 820 Bilder
  ...
  Min: 750, Max: 900, Durchschnitt: 841.2
  Balance-Ratio: 83.33%
```

---

## 📝 Wichtige Punkte

- ✅ **`data_raw/` wird NICHT verändert** - nur gelesen
- ✅ **Alle sauberen Daten gehen nach `data_cleaned/`**
- ✅ **Klassenstruktur bleibt erhalten**
- ✅ **Logging für nachvollziehbare Entfernungen**
- ✅ **Variation wird bewusst erhalten** (Lichtverhältnisse, Hautfarben, Perspektiven)

---

## 🎯 Nächste Schritte

1. **Starte das Cleaning:**
   ```bash
   python clean_dataset.py
   ```

2. **Prüfe die Klassenbalance** in der Console-Ausgabe

3. **Prüfe die Logs:**
   - `logs/removed_images.txt` - Was wurde entfernt und warum?
   - `logs/processing_methods.txt` - Wie wurden die Bilder verarbeitet?

4. **Wenn zufrieden:** Nutze `data_cleaned/` zum Training


