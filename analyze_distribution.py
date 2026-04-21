import argparse
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt

ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
DEFAULT_OUTPUT_IMAGE = Path('logs/data_distribution.png')


def is_image_file(path: Path) -> bool:
    return path.suffix.lower() in ALLOWED_EXTENSIONS


def count_images_per_class(source_dir: Path) -> Counter[str]:
    counts = Counter()
    if not source_dir.exists():
        raise FileNotFoundError(f'Quelle nicht gefunden: {source_dir}')

    for class_dir in sorted(source_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        total = 0
        for path in class_dir.iterdir():
            if path.is_file() and is_image_file(path):
                total += 1
        counts[class_dir.name] = total
    return counts


def plot_distribution(counts: Counter[str], output_path: Path) -> None:
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    classes = list(counts.keys())
    values = [counts[c] for c in classes]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(classes, values, color='tab:blue')
    ax.set_title('Bildverteilung pro Klasse')
    ax.set_xlabel('Klasse')
    ax.set_ylabel('Anzahl Bilder')
    ax.set_ylim(0, max(values) * 1.1 if values else 1)
    ax.grid(True, axis='y', linestyle='--', alpha=0.5)

    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(values) * 0.01, str(value), ha='center', va='bottom')

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def print_table(counts: Counter[str]) -> None:
    print('Klasse | Anzahl')
    print('------|--------')
    for cls, count in counts.items():
        print(f'{cls:5} | {count:6}')
    print(f'Gesamt: {sum(counts.values())} Bilder')


def warn_if_imbalanced(counts: Counter[str]) -> None:
    if not counts:
        return
    values = list(counts.values())
    max_count = max(values)
    min_count = min(values)
    if min_count == 0:
        print('WARNUNG: Mindestens eine Klasse enthält 0 Bilder!')
        return
    ratio = max_count / min_count
    if ratio >= 2.0:
        print(f'WARNUNG: Starke Klassen-Ungleichheit erkannt (größte/kleinste Klasse = {ratio:.2f}).')
    elif ratio >= 1.5:
        print(f'Achtung: Mittlere Klassen-Ungleichheit (größte/kleinste Klasse = {ratio:.2f}).')
    else:
        print('Die Klassenverteilung ist relativ ausgeglichen.')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Analysiert die Verteilung von Bildern pro Klasse.')
    parser.add_argument('--source-dir', type=Path, default=Path('data'), help='Quelle mit Klassenordnern')
    parser.add_argument('--output-image', type=Path, default=DEFAULT_OUTPUT_IMAGE, help='Pfad für den Bar-Plot')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    counts = count_images_per_class(args.source_dir)
    if not counts:
        print(f'Keine Klassenverzeichnisse in {args.source_dir} gefunden.')
        raise SystemExit(1)

    print_table(counts)
    warn_if_imbalanced(counts)
    plot_distribution(counts, args.output_image)
    print(f'Plot gespeichert: {Path(args.output_image).resolve()}')
