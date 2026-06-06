"""
Datenbereinigungsmodul für Gebärdensprachen-Trainingsdaten

Dieses Modul bereinigt rohe Trainingsbilder aus data_raw/ und speichert
die gefilterten Bilder in data_cleaned/. Die Bereinigung umfasst:

- Duplikat-Erkennung mittels Perceptual Hashing (pHash-ähnlich)
- Unschärfe-Filterung mittels Laplacian Varianz (Threshold=40)
- Entfernung beschädigter oder nicht ladbarer Dateien
- Filterung extrem schlechter Bilder (nahezu monochrom)
- Hand-Erkennung und -Cropping (optional mit MediaPipe)
- Normalisierung und konsistente Größe (224x224)

Pro Cleaning-Run werden zeitgestempelte Logs erstellt:
- logs/cleaning_logs/<TIMESTAMP>/removed_images.txt
- logs/cleaning_logs/<TIMESTAMP>/processing_methods.txt
- logs/cleaning_logs/<TIMESTAMP>/summary.txt
- logs/data_analysis/data_distribution_<TIMESTAMP>.png
- logs/data_analysis/data_distribution_<TIMESTAMP>.txt

Worflow:
1. Lade raw_data/
2. Für jedes Bild: Validierung, Duplikat-Check, Blur-Check, Crop
3. Speichere bereinigte Bilder nach data_cleaned/
4. Erstelle Statistiken und Logs
"""

import argparse
import shutil
from datetime import datetime
from pathlib import Path
from collections import Counter

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError

import matplotlib.pyplot as plt

# ============================================================================
# MediaPipe Setup (optional für Hand-Erkennung + Cropping)
# ============================================================================
try:
    import mediapipe as mp
    mp_hands_module = mp.solutions.hands
    mp_hands = mp_hands_module.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.5,
    )
    mediapipe_available = True
except Exception as e:
    print(f'MediaPipe nicht verfügbar: {e}')
    mp = None
    mp_hands = None
    mediapipe_available = False

# ============================================================================
# KONFIGURATION
# ============================================================================
TARGET_SIZE = (224, 224)                         # Standard-Bildgrösse für Modell
ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}  # Erlaubte Dateitypen
DEFAULT_BLUR_THRESHOLD = 40.0                    # Laplacian Varianz Threshold (< Threshold = unscharf)
DEFAULT_HAND_PADDING = 0.2                       # 20% Padding um erkannte Hand
DEFAULT_FALLBACK_CROP_RATIO = 0.9                # 90% Crop wenn kein Hand erkannt
DEFAULT_EXTREME_LOW_CONTENT_THRESHOLD = 0.95     # 95% uniforme Farbe = extrem schlecht

# Globale Variablen für aktuellen Cleaning-Run (werden initialisiert bei clean_dataset)
RUN_TIMESTAMP = None
REMOVED_LOG_PATH = None
METHOD_LOG_PATH = None
SUMMARY_LOG_PATH = None

# ============================================================================
# HILFSFUNKTIONEN
# ============================================================================
def is_image_file(path: Path) -> bool:
    """Prüft ob Datei ein unterstütztes Bildformat ist."""
    return path.suffix.lower() in ALLOWED_EXTENSIONS


def load_image(path: Path) -> Image.Image:
    """Lädt Bild von Pfad und konvertiert zu RGB."""
    return Image.open(path).convert('RGB')


def resize_with_padding(image: Image.Image, target_size: tuple[int, int]) -> Image.Image:
    """
    Resized Bild mit Letterboxing (schwarzes Padding erhalten Aspect Ratio).

    Dies verhindert, dass Bilder verzerrt werden wenn sie ein anderes Seitenverhältnis haben.

    Args:
        image (Image.Image): Input-Bild
        target_size (tuple): (width, height) der Zielgrösse

    Returns:
        Image.Image: Resized Bild mit Padding, genau target_size Grösse
    """
    target_width, target_height = target_size
    original_width, original_height = image.size
    if original_width == 0 or original_height == 0:
        raise ValueError('Bild hat ungültige Abmessungen.')

    # Berechne Skalierungsfaktor um Aspect Ratio zu erhalten
    ratio = min(target_width / original_width, target_height / original_height)
    new_size = (int(original_width * ratio), int(original_height * ratio))

    # Skaliere mit high-quality Resampling
    resized = image.resize(new_size, Image.LANCZOS)

    # Erstelle schwarzen Background
    background = Image.new('RGB', target_size, (0, 0, 0))
    # Zentriere resized Bild
    offset_x = (target_width - new_size[0]) // 2
    offset_y = (target_height - new_size[1]) // 2
    background.paste(resized, (offset_x, offset_y))
    return background


def normalize_image(image: Image.Image) -> np.ndarray:
    """Konvertiert Bild zu normalisiertem Array (0-1 Pixel-Bereich)."""
    array = np.asarray(image, dtype=np.float32) / 255.0
    return array


def is_blurry(image: Image.Image, threshold: float) -> bool:
    """
    Prüft ob Bild zu unscharf ist mittels Laplacian Varianz.

    Laplacian misst die Schärfe: niedriger Wert = unscharf.
    Standard-Threshold ist 40.0; niedriger = aggressiveres Filtern.

    Args:
        image (Image.Image): Zu prüfendes Bild
        threshold (float): Varianz-Schwellenwert (unter diesem Wert = unscharf)

    Returns:
        bool: True wenn Bild unscharf ist, sonst False
    """
    gray = np.asarray(image.convert('L'))
    if gray.size == 0:
        return True
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return variance < threshold


def compute_image_hash(image: Image.Image, hash_size: int = 8) -> str:
    """
    Berechnet Perceptual Hash (pHash-ähnlich) eines Bildes.

    Dies ermöglicht Duplikat-Erkennung: ähnliche/identische Bilder
    haben ähnliche Hashes.

    Methode: 8x8 DCT-ähnlich (einfache Detektion von Gradienten)

    Args:
        image (Image.Image): Input-Bild
        hash_size (int): Hash-Grösse (Standard 8x8)

    Returns:
        str: Hex-String Hash (z.B. 'a1b2c3d4...')
    """
    # Resize zu 8x9 (9 für Gradient-Berechnung)
    small = image.convert('L').resize((hash_size + 1, hash_size), Image.LANCZOS)
    pixels = np.asarray(small, dtype=np.uint8)

    # Berechne Gradienten (Pixel[i+1] > Pixel[i])
    diff = pixels[:, 1:] > pixels[:, :-1]

    # Konvertiere zu Bit-String und dann zu Hex
    bit_string = ''.join('1' if v else '0' for v in diff.flatten())
    return f'{int(bit_string, 2):0{hash_size * hash_size // 4}x}'


def fallback_crop(image: Image.Image) -> Image.Image:
    """
    Fallback-Cropping wenn MediaPipe Hand nicht erkennt.

    Cropt zentriert einen 90% quadrate des minimums der Dimensionen.
    Dies entfernt typischerweise Ecken mit Hintergrund.

    Args:
        image (Image.Image): Input-Bild

    Returns:
        Image.Image: Gecropptes Bild (quadratisch)
    """
    width, height = image.size
    crop_size = int(min(width, height) * DEFAULT_FALLBACK_CROP_RATIO)
    left = (width - crop_size) // 2
    top = (height - crop_size) // 2
    return image.crop((left, top, left + crop_size, top + crop_size))


def log_method(src_path: Path, method: str) -> None:
    """Loggt welche Crop-Methode für ein Bild verwendet wurde."""
    global METHOD_LOG_PATH
    if METHOD_LOG_PATH is None:
        return
    METHOD_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().isoformat(timespec='seconds')
    with METHOD_LOG_PATH.open('a', encoding='utf-8') as file:
        file.write(f'{timestamp}\t{src_path}\t{method}\n')


def detect_hand_bbox(image: Image.Image, padding: float = DEFAULT_HAND_PADDING) -> tuple[int, int, int, int] | None:
    """
    Erkennt Hand-Position mittels MediaPipe und gibt Bounding-Box zurück.

    Dies ermöglicht intelligentes Cropping auf die Hand, was Hintergrund-Rauschen
    und Lichtvariationen reduziert.

    Args:
        image (Image.Image): Input-Bild
        padding (float): Relative Padding um Hand (0.2 = 20%)

    Returns:
        tuple: (x_min, y_min, x_max, y_max) oder None wenn keine Hand erkannt
    """
    if not mediapipe_available or mp_hands is None:
        return None

    # Konvertiere zu numpy Array für MediaPipe
    rgb = np.asarray(image.convert('RGB'))
    results = mp_hands.process(rgb)
    if not results.multi_hand_landmarks:
        return None

    # Extrahiere alle Hand-Landmarks und berechne Bounding-Box
    h, w = rgb.shape[:2]
    xs = [lm.x * w for lm in results.multi_hand_landmarks[0].landmark]
    ys = [lm.y * h for lm in results.multi_hand_landmarks[0].landmark]
    x_min = int(max(0, min(xs)))
    x_max = int(min(w, max(xs)))
    y_min = int(max(0, min(ys)))
    y_max = int(min(h, max(ys)))

    # Ignoriere zu kleine Hände (wahrscheinlich False Positives)
    if x_max - x_min < 20 or y_max - y_min < 20:
        return None

    # Füge Padding hinzu für Kontext
    side = max(x_max - x_min, y_max - y_min)
    margin = int(side * padding)
    x_min = max(0, x_min - margin)
    y_min = max(0, y_min - margin)
    x_max = min(w, x_max + margin)
    y_max = min(h, y_max + margin)

    return x_min, y_min, x_max, y_max


def crop_hand_region(image: Image.Image, bbox: tuple[int, int, int, int]) -> Image.Image:
    """Cropt Hand-Region basierend auf Bounding-Box."""
    x_min, y_min, x_max, y_max = bbox
    cropped = image.crop((x_min, y_min, x_max, y_max))
    return cropped


def log_removed(src_path: Path, reason: str) -> None:
    """Loggt warum ein Bild entfernt wurde."""
    global REMOVED_LOG_PATH
    if REMOVED_LOG_PATH is None:
        return
    REMOVED_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().isoformat(timespec='seconds')
    with REMOVED_LOG_PATH.open('a', encoding='utf-8') as file:
        file.write(f'{timestamp}\t{src_path}\t{reason}\n')

# ============================================================================
# HAUPTBEREINIGUNGSLOGIK
# ============================================================================
def clean_image_file(
    src_path: Path,
    dst_path: Path,
    min_size: int,
    blur_threshold: float,
    normalize: bool,
    duplicate_hashes: set[str] | None,
) -> bool:
    """
    Bereinigt ein einzelnes Bild durch Validierung, Dedup, Blur-Check, Cropping, Resize.

    Workflow:
    1. Lade Bild (werfe weg wenn beschädigt)
    2. Prüfe Mindestgrösse
    3. Erkenne Hand oder fallback Crop
    4. Prüfe Blur (Laplacian Varianz >= threshold)
    5. Prüfe auf extrem niedrigen Inhalt (monochrom)
    6. Duplikat-Check (falls aktiv)
    7. Resize mit Padding auf TARGET_SIZE
    8. Speichere

    Args:
        src_path (Path): Quelle-Bilddatei
        dst_path (Path): Ziel-Speicherpfad
        min_size (int): Minimale Breite/Höhe
        blur_threshold (float): Laplacian Varianz Threshold
        normalize (bool): Speichere auch normalisierte .npy
        duplicate_hashes(set): Hashes bereits gesehener Bilder für Duplikat-Check

    Returns:
        bool: True wenn Bild erfolgreich bereiningt, sonst False
    """
    # ========== Lade Bild ==========
    try:
        image = load_image(src_path)
    except (UnidentifiedImageError, OSError, ValueError):
        log_removed(src_path, 'beschädigt')
        print(f'Übersprungen (beschädigt): {src_path}')
        return False

    # ========== Prüfe Mindestgrösse ==========
    width, height = image.size
    if width < min_size or height < min_size:
        log_removed(src_path, 'zu klein')
        print(f'Übersprungen (zu klein): {src_path} ({width}x{height})')
        return False

    # ========== Hand-Erkennung und Cropping ==========
    bbox = detect_hand_bbox(image)
    if bbox is not None:
        # MediaPipe erkannte Hand
        image = crop_hand_region(image, bbox)
        log_method(src_path, 'MediaPipe')
    else:
        # Kein Hand erkannt: Fallback Crop
        image = fallback_crop(image)
        log_method(src_path, 'Fallback')

    # ========== Blur-Check ==========
    if is_blurry(image, blur_threshold):
        log_removed(src_path, f'unscharf (threshold={blur_threshold})')
        print(f'Übersprungen (unscharf): {src_path}')
        return False

    # ========== Extreme-Content-Check ==========
    # Prüfe ob Bild zu viel uniforme Farbe hat (schwarz, weiß, monochrom)
    image_array = np.asarray(image)
    if np.std(image_array) < DEFAULT_EXTREME_LOW_CONTENT_THRESHOLD:
        log_removed(src_path, 'extrem niedriger Inhalt')
        print(f'Übersprungen (extrem niedriger Inhalt): {src_path}')
        return False

    # ========== Duplikat-Check ==========
    if duplicate_hashes is not None:
        image_hash = compute_image_hash(image)
        if image_hash in duplicate_hashes:
            log_removed(src_path, 'duplikat')
            print(f'Übersprungen (Duplikat): {src_path}')
            return False
        duplicate_hashes.add(image_hash)

    # ========== Resize und Speichern ==========
    cleaned_image = resize_with_padding(image, TARGET_SIZE)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned_image.save(dst_path, format='PNG', quality=95)

    # ========== Optional: Speichere auch normalisierte Array ==========
    if normalize:
        normalized_array = normalize_image(cleaned_image)
        normalized_path = dst_path.with_suffix('.npy')
        np.save(normalized_path, normalized_array)

    return True

# ============================================================================
# HAUPTBEREINIGUNG (Orchestrierung)
# ============================================================================
def clean_dataset(
    source_dir: Path,
    dest_dir: Path,
    min_size: int,
    blur_threshold: float,
    normalize: bool,
    deduplicate: bool,
) -> None:
    """
    Bereinigt einen kompletten Datensatz.

    Liest alle Bilder aus source_dir (data_raw/), bereinigt und speichert in dest_dir.
    Erstellt auch Cleanup-Logs und Datenverteilungsanalyse.

    wichtig:
    - source_dir muss 'data_raw' sein (Sicherheitsprüfung)
    - dest_dir wird komplett gelöscht und neu erstellt (Achtung!)
    - Pro Run wird ein zeitgestempelter Ordner in logs/cleaning_logs/ erstellt

    Args:
        source_dir (Path): Quelle-Ordner mit Klassen-Unterordnern
        dest_dir (Path): Ziel-Ordner
        min_size (int): Minimale Bildgrösse
        blur_threshold (float): Laplacian Varianz Threshold
        normalize (bool): Speichere normalisierte .npy Dateien
        deduplicate (bool): Aktiviere Duplikat-Filterung
    """
    global RUN_TIMESTAMP, REMOVED_LOG_PATH, METHOD_LOG_PATH, SUMMARY_LOG_PATH

    # ========== Generiere Timestamp für diesen Run ==========
    RUN_TIMESTAMP = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    run_log_dir = Path(f'logs/cleaning_logs/{RUN_TIMESTAMP}')
    REMOVED_LOG_PATH = run_log_dir / 'removed_images.txt'
    METHOD_LOG_PATH = run_log_dir / 'processing_methods.txt'
    SUMMARY_LOG_PATH = run_log_dir / 'summary.txt'

    # ========== Validierung ==========
    if not source_dir.exists():
        raise FileNotFoundError(f'Quelle nicht gefunden: {source_dir}')

    if source_dir.resolve().name != 'data_raw':
        raise ValueError('Als Input soll nur data_raw verwendet werden.')

    if any(part == 'data_cleaned' for part in source_dir.resolve().parts) or source_dir.resolve().name == 'data_cleaned':
        raise ValueError('Der Input muss der ursprüngliche Datensatz sein, nicht data_cleaned.')

    # ========== Initialisierung ==========
    if dest_dir.exists():
        shutil.rmtree(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    run_log_dir.mkdir(parents=True, exist_ok=True)
    class_dirs = [entry for entry in source_dir.iterdir() if entry.is_dir()]

    total_processed = 0
    total_removed = 0
    total_saved = 0

    # ========== Hauptschleife: Prozessiere jede Klasse ==========
    for class_dir in sorted(class_dirs):
        relative_class = class_dir.name
        target_class_dir = dest_dir / relative_class
        target_class_dir.mkdir(parents=True, exist_ok=True)
        duplicate_hashes: set[str] | None = set() if deduplicate else None

        # Prozessiere alle Bilder in dieser Klasse
        for item in sorted(class_dir.iterdir()):
            if not item.is_file() or not is_image_file(item):
                continue

            total_processed += 1
            dst_path = target_class_dir / f'{item.stem}.png'
            success = clean_image_file(
                item,
                dst_path,
                min_size,
                blur_threshold,
                normalize,
                duplicate_hashes if deduplicate else None,
            )
            if success:
                total_saved += 1
            else:
                total_removed += 1

    # ========== Zusammenfassung ausgeben ==========
    print('--- Reinigung abgeschlossen ---')
    print(f'Archiviert: {total_processed} Dateien überprüft')
    print(f'Entfernt: {total_removed} Dateien')
    print(f'Gespeichert: {total_saved} bereinigte Bilder in {dest_dir}')
    if normalize:
        print('Hinweis: Normalisierte Arrays wurden zusätzlich als .npy-Dateien gespeichert.')
    print(f'Entfernte Dateien protokolliert in {REMOVED_LOG_PATH}')
    print(f'Crop-Methode pro Bild protokolliert in {METHOD_LOG_PATH}')

    # ========== Klassenbalance-Analyse ==========
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

    # ========== Erstellung von Datenverteilungsanalyse ==========
    print("\n📈 Erstelle Datenverteilungs-Analyse...")
    create_data_distribution_analysis(dest_dir)

    # ========== Erstelle Summary ==========
    print("\n📝 Erstelle Cleaning-Zusammenfassung...")
    create_cleaning_summary(
        source_dir, dest_dir, min_size, blur_threshold, normalize, deduplicate,
        total_processed, total_removed, total_saved
    )
    print(f"✓ Zusammenfassung gespeichert: {SUMMARY_LOG_PATH}")

    # ========== Finale Info ==========
    print(f"\n✓ Alle Logs gespeichert in: {run_log_dir}")

# ============================================================================
# LOGGING UND ANALYSE
# ============================================================================
def create_cleaning_summary(
    source_dir: Path, dest_dir: Path, min_size: int, blur_threshold: float,
    normalize: bool, deduplicate: bool,
    total_processed: int, total_removed: int, total_saved: int
) -> None:
    """
    Erstellt eine detaillierte Zusammenfassung des Cleaning-Runs:
    - Konfiguration (Parameter)
    - Ergebnisse (Statistiken)
    - Log-Pfade
    """
    global SUMMARY_LOG_PATH
    if SUMMARY_LOG_PATH is None:
        return

    with SUMMARY_LOG_PATH.open('w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("CLEANING SESSION SUMMARY\n")
        f.write("="*80 + "\n\n")

        f.write(f"Timestamp: {RUN_TIMESTAMP}\n\n")

        f.write("KONFIGURATION:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Source Directory:      {source_dir.resolve()}\n")
        f.write(f"Destination Directory: {dest_dir.resolve()}\n")
        f.write(f"Min Size:              {min_size} Pixel\n")
        f.write(f"Blur Threshold:        {blur_threshold}\n")
        f.write(f"Normalization:         {'Ja' if normalize else 'Nein'}\n")
        f.write(f"Deduplication:         {'Ja (aktiv)' if deduplicate else 'Nein (deaktiviert)'}\n\n")

        f.write("ERGEBNISSE:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Gesamt verarbeitet:    {total_processed}\n")
        f.write(f"Entfernt:              {total_removed}\n")
        f.write(f"Gespeichert:           {total_saved}\n")
        f.write(f"Erfolgsquote:          {(total_saved/total_processed*100):.1f}%\n\n")

        f.write("LOGS:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Removed Images:        removed_images.txt\n")
        f.write(f"Processing Methods:    processing_methods.txt\n\n")

        f.write("="*80 + "\n")
        f.write("Ende des Summaries\n")
        f.write("="*80 + "\n")


def create_data_distribution_analysis(data_dir: Path) -> None:
    """
    Erstellt vollständige Datenverteilungs-Analyse mit Diagramm und Tabelle.
    """
    from utils import get_timestamp
    timestamp = get_timestamp()

    # ========== Zähle Bilder pro Klasse ==========
    counts = Counter()
    for class_dir in sorted(data_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        total = 0
        for path in class_dir.iterdir():
            if path.is_file() and is_image_file(path):
                total += 1
        counts[class_dir.name] = total

    if not counts:
        print("Keine Bilder gefunden für Analyse.")
        return

    # ========== Erstelle Visualisierung ==========
    plot_data_distribution(counts, timestamp)

    # ========== Erstelle Text-Report ==========
    save_data_distribution_table(counts, timestamp)

    print(f"✓ Datenverteilungs-Analyse gespeichert:")
    print(f"  Diagramm: logs/data_analysis/data_distribution_{timestamp}.png")
    print(f"  Tabelle:  logs/data_analysis/data_distribution_{timestamp}.txt")


def plot_data_distribution(counts: Counter[str], timestamp: str) -> None:
    """Erstellt und speichert Balkendiagramm der Bildverteilung pro Klasse."""
    output_path = Path(f'logs/data_analysis/data_distribution_{timestamp}.png')
    output_path.parent.mkdir(parents=True, exist_ok=True)

    classes = list(counts.keys())
    values = [counts[c] for c in classes]

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(classes, values, color='tab:blue', alpha=0.8)

    ax.set_title(f'Bildverteilung pro Klasse (nach Cleaning - {timestamp})', fontsize=14, fontweight='bold')
    ax.set_xlabel('Klasse', fontsize=12)
    ax.set_ylabel('Anzahl Bilder', fontsize=12)
    ax.set_ylim(0, max(values) * 1.15 if values else 1)
    ax.grid(True, axis='y', linestyle='--', alpha=0.5)

    # Werte über den Balken anzeigen
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(values) * 0.01,
                f'{value}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Statistiken als Text hinzufügen
    if values:
        min_count = min(values)
        max_count = max(values)
        avg_count = sum(values) / len(values)
        total_count = sum(values)

        stats_text = f'Total: {total_count} | Min: {min_count} | Max: {max_count} | Avg: {avg_count:.1f}'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def save_data_distribution_table(counts: Counter[str], timestamp: str) -> None:
    """Speichert detaillierte Text-Tabelle der Datenverteilung mit Statistiken."""
    output_path = Path(f'logs/data_analysis/data_distribution_{timestamp}.txt')
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open('w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write(f"DATA DISTRIBUTION ANALYSIS - {timestamp}\n")
        f.write("="*80 + "\n\n")

        f.write("Klasse | Anzahl Bilder\n")
        f.write("-------|--------------\n")

        total_images = 0
        for cls, count in sorted(counts.items()):
            f.write(f"{cls:6} | {count:12}\n")
            total_images += count

        f.write("-------|--------------\n")
        f.write(f"Total  | {total_images:12}\n\n")

        # ========== Statistiken ==========
        if counts:
            values = list(counts.values())
            min_count = min(values)
            max_count = max(values)
            avg_count = sum(values) / len(values)
            balance_ratio = min_count / max_count if max_count > 0 else 0

            f.write("STATISTIKEN:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Gesamtanzahl Bilder: {total_images}\n")
            f.write(f"Klassen: {len(counts)}\n")
            f.write(f"Minimum pro Klasse: {min_count}\n")
            f.write(f"Maximum pro Klasse: {max_count}\n")
            f.write(f"Durchschnitt pro Klasse: {avg_count:.1f}\n")
            f.write(f"Balance-Ratio: {balance_ratio:.2%}\n\n")

            # ========== Balance-Bewertung ==========
            if min_count == 0:
                balance_status = "KRITISCH: Mindestens eine Klasse hat 0 Bilder!"
            elif balance_ratio >= 0.8:
                balance_status = "SEHR GUT: Ausgeglichene Verteilung"
            elif balance_ratio >= 0.6:
                balance_status = "GUT: Leichte Ungleichheit"
            elif balance_ratio >= 0.4:
                balance_status = "MITTEL: Moderate Ungleichheit"
            else:
                balance_status = "SCHLECHT: Starke Ungleichheit"

            f.write(f"Balance-Status: {balance_status}\n")

        f.write("\n" + "="*80 + "\n")
        f.write("Erstellt durch clean_dataset.py\n")
        f.write("="*80 + "\n")

# ============================================================================
# CLI UND EINSTIEG
# ============================================================================
def parse_args() -> argparse.Namespace:
    """Parsiere Command-Line-Argumente."""
    parser = argparse.ArgumentParser(description='Bereinigt einen Bilddatensatz für Sign Language Training.')
    parser.add_argument('--source-dir', type=Path, default=Path('data_raw'), help='Quellordner mit Klassen-Unterordnern')
    parser.add_argument('--dest-dir', type=Path, default=Path('data_cleaned'), help='Zielordner für bereinigte Bilder')
    parser.add_argument('--min-size', type=int, default=50, help='Minimale Breite oder Höhe in Pixeln')
    parser.add_argument('--blur-threshold', type=float, default=DEFAULT_BLUR_THRESHOLD, help='Schärfe-Schwellenwert für Laplacian-Varianz')
    parser.add_argument('--normalize', action='store_true', help='Optional: Speichere zusätzlich normalisierte Arrays als .npy')
    parser.add_argument('--no-deduplicate', action='store_false', dest='deduplicate', help='Optional: Deaktiviert die Duplikat-Prüfung')
    parser.add_argument('--timestamped-folder', action='store_true', help='Erzeuge einen neuen Zielordner mit Zeitstempel')
    return parser.parse_args()


def resolve_output_dir(dest_dir: Path, timestamped: bool) -> Path:
    """Bestimmt Zielordner (mit optionalem Zeitstempel)."""
    if not timestamped:
        return dest_dir

    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M')
    return dest_dir.with_name(f'{dest_dir.name}_{timestamp}')


if __name__ == '__main__':
    args = parse_args()
    if not mediapipe_available:
        raise RuntimeError('MediaPipe muss installiert und importierbar sein. Bitte installiere mediapipe oder aktiviere die korrekte Umgebung.')

    output_dir = resolve_output_dir(args.dest_dir, args.timestamped_folder)
    clean_dataset(
        args.source_dir,
        output_dir,
        args.min_size,
        args.blur_threshold,
        args.normalize,
        args.deduplicate,
    )
