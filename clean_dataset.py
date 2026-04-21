import argparse
import shutil
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError

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

TARGET_SIZE = (224, 224)
ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
REMOVED_LOG_PATH = Path('logs/removed_images.txt')
METHOD_LOG_PATH = Path('logs/processing_methods.txt')
DEFAULT_BLUR_THRESHOLD = 40.0
DEFAULT_HAND_PADDING = 0.2
DEFAULT_FALLBACK_CROP_RATIO = 0.9


def is_image_file(path: Path) -> bool:
    return path.suffix.lower() in ALLOWED_EXTENSIONS


def load_image(path: Path) -> Image.Image:
    return Image.open(path).convert('RGB')


def resize_with_padding(image: Image.Image, target_size: tuple[int, int]) -> Image.Image:
    target_width, target_height = target_size
    original_width, original_height = image.size
    if original_width == 0 or original_height == 0:
        raise ValueError('Bild hat ungültige Abmessungen.')

    ratio = min(target_width / original_width, target_height / original_height)
    new_size = (int(original_width * ratio), int(original_height * ratio))
    resized = image.resize(new_size, Image.LANCZOS)

    background = Image.new('RGB', target_size, (0, 0, 0))
    offset_x = (target_width - new_size[0]) // 2
    offset_y = (target_height - new_size[1]) // 2
    background.paste(resized, (offset_x, offset_y))
    return background


def normalize_image(image: Image.Image) -> np.ndarray:
    array = np.asarray(image, dtype=np.float32) / 255.0
    return array


def is_blurry(image: Image.Image, threshold: float) -> bool:
    gray = np.asarray(image.convert('L'))
    if gray.size == 0:
        return True
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return variance < threshold


def compute_image_hash(image: Image.Image, hash_size: int = 8) -> str:
    small = image.convert('L').resize((hash_size + 1, hash_size), Image.LANCZOS)
    pixels = np.asarray(small, dtype=np.uint8)
    diff = pixels[:, 1:] > pixels[:, :-1]
    bit_string = ''.join('1' if v else '0' for v in diff.flatten())
    return f'{int(bit_string, 2):0{hash_size * hash_size // 4}x}'


def fallback_crop(image: Image.Image) -> Image.Image:
    width, height = image.size
    crop_size = int(min(width, height) * DEFAULT_FALLBACK_CROP_RATIO)
    left = (width - crop_size) // 2
    top = (height - crop_size) // 2
    return image.crop((left, top, left + crop_size, top + crop_size))


def log_method(src_path: Path, method: str) -> None:
    METHOD_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().isoformat(timespec='seconds')
    with METHOD_LOG_PATH.open('a', encoding='utf-8') as file:
        file.write(f'{timestamp}\t{src_path}\t{method}\n')


def detect_hand_bbox(image: Image.Image, padding: float = DEFAULT_HAND_PADDING) -> tuple[int, int, int, int] | None:
    if not mediapipe_available or mp_hands is None:
        return None

    rgb = np.asarray(image.convert('RGB'))
    results = mp_hands.process(rgb)
    if not results.multi_hand_landmarks:
        return None

    h, w = rgb.shape[:2]
    xs = [lm.x * w for lm in results.multi_hand_landmarks[0].landmark]
    ys = [lm.y * h for lm in results.multi_hand_landmarks[0].landmark]
    x_min = int(max(0, min(xs)))
    x_max = int(min(w, max(xs)))
    y_min = int(max(0, min(ys)))
    y_max = int(min(h, max(ys)))

    if x_max - x_min < 20 or y_max - y_min < 20:
        return None

    side = max(x_max - x_min, y_max - y_min)
    margin = int(side * padding)
    x_min = max(0, x_min - margin)
    y_min = max(0, y_min - margin)
    x_max = min(w, x_max + margin)
    y_max = min(h, y_max + margin)

    return x_min, y_min, x_max, y_max


def crop_hand_region(image: Image.Image, bbox: tuple[int, int, int, int]) -> Image.Image:
    x_min, y_min, x_max, y_max = bbox
    cropped = image.crop((x_min, y_min, x_max, y_max))
    return cropped


def log_removed(src_path: Path, reason: str) -> None:
    REMOVED_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().isoformat(timespec='seconds')
    with REMOVED_LOG_PATH.open('a', encoding='utf-8') as file:
        file.write(f'{timestamp}\t{src_path}\t{reason}\n')


def clean_image_file(
    src_path: Path,
    dst_path: Path,
    min_size: int,
    blur_threshold: float,
    normalize: bool,
    duplicate_hashes: set[str] | None,
) -> bool:
    try:
        image = load_image(src_path)
    except (UnidentifiedImageError, OSError, ValueError):
        log_removed(src_path, 'beschädigt')
        print(f'Übersprungen (beschädigt): {src_path}')
        return False

    width, height = image.size
    if width < min_size or height < min_size:
        log_removed(src_path, 'zu klein')
        print(f'Übersprungen (zu klein): {src_path} ({width}x{height})')
        return False

    bbox = detect_hand_bbox(image)
    if bbox is not None:
        image = crop_hand_region(image, bbox)
        log_method(src_path, 'MediaPipe')
    else:
        image = fallback_crop(image)
        log_method(src_path, 'Fallback')

    if is_blurry(image, blur_threshold):
        log_removed(src_path, f'unscharf (threshold={blur_threshold})')
        print(f'Übersprungen (unscharf): {src_path}')
        return False

    if duplicate_hashes is not None:
        image_hash = compute_image_hash(image)
        if image_hash in duplicate_hashes:
            log_removed(src_path, 'duplikat')
            print(f'Übersprungen (Duplikat): {src_path}')
            return False
        duplicate_hashes.add(image_hash)

    cleaned_image = resize_with_padding(image, TARGET_SIZE)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned_image.save(dst_path, format='PNG', quality=95)

    if normalize:
        normalized_array = normalize_image(cleaned_image)
        normalized_path = dst_path.with_suffix('.npy')
        np.save(normalized_path, normalized_array)

    return True


def clean_dataset(
    source_dir: Path,
    dest_dir: Path,
    min_size: int,
    blur_threshold: float,
    normalize: bool,
    deduplicate: bool,
) -> None:
    if not source_dir.exists():
        raise FileNotFoundError(f'Quelle nicht gefunden: {source_dir}')

    if source_dir.resolve().name != 'data_raw':
        raise ValueError('Als Input soll nur data_raw verwendet werden.')

    if any(part == 'data_cleaned' for part in source_dir.resolve().parts) or source_dir.resolve().name == 'data_cleaned':
        raise ValueError('Der Input muss der ursprüngliche Datensatz sein, nicht data_cleaned.')

    if dest_dir.exists():
        shutil.rmtree(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    REMOVED_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    METHOD_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    class_dirs = [entry for entry in source_dir.iterdir() if entry.is_dir()]

    total_processed = 0
    total_removed = 0
    total_saved = 0

    for class_dir in sorted(class_dirs):
        relative_class = class_dir.name
        target_class_dir = dest_dir / relative_class
        target_class_dir.mkdir(parents=True, exist_ok=True)
        duplicate_hashes: set[str] | None = set() if deduplicate else None

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

    print('--- Reinigung abgeschlossen ---')
    print(f'Archiviert: {total_processed} Dateien überprüft')
    print(f'Entfernt: {total_removed} Dateien')
    print(f'Gespeichert: {total_saved} bereinigte Bilder in {dest_dir}')
    if normalize:
        print('Hinweis: Normalisierte Arrays wurden zusätzlich als .npy-Dateien gespeichert.')
    print(f'Entfernte Dateien protokolliert in {REMOVED_LOG_PATH}')
    print(f'Crop-Methode pro Bild protokolliert in {METHOD_LOG_PATH}')


def parse_args() -> argparse.Namespace:
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
