#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import logging
import shutil
import subprocess
import sys
import tarfile
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

try:
    import requests
except ImportError as exc:  # pragma: no cover - handled at runtime
    raise SystemExit(
        "Das Paket 'requests' wird für Zenodo-Downloads benötigt. "
        "Bitte installiere die Abhängigkeiten mit 'pip install -r requirements.txt'."
    ) from exc


LETTERS = ("A", "B", "C", "L", "O", "V", "W", "Y")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff", ".webp"}
TRAIN_MARKERS = {"train", "training", "trainset", "train_set"}
TEST_MARKERS = {"test", "testing", "eval", "evaluation", "external_test", "holdout"}
SKIP_DIR_MARKERS = {
    "__macosx",
    ".git",
    ".ipynb_checkpoints",
    "sample",
    "samples",
    "preview",
    "thumbnails",
}
LOG_PATH = Path("logs/setup_data.log")
WORK_ROOT = Path("data_downloads")
KAGGLE_CONFIG_PATH = Path.home() / ".kaggle" / "kaggle.json"
ZENODO_API_URL = "https://zenodo.org/api/records/{record_id}"


@dataclass(frozen=True)
class DatasetConfig:
    key: str
    label: str
    source_type: str
    kaggle_slug: str | None = None
    zenodo_record_id: str | None = None
    train_target: Path | None = None
    test_target: Path | None = None
    prefix: str = ""
    allow_train_root_fallback: bool = True
    allow_test_root_fallback: bool = False


@dataclass
class DatasetResult:
    config: DatasetConfig
    skipped: bool = False
    dry_run: bool = False
    downloaded_files: list[Path] = field(default_factory=list)
    extracted_paths: list[Path] = field(default_factory=list)
    train_counts: dict[str, int] = field(default_factory=dict)
    test_counts: dict[str, int] = field(default_factory=dict)
    validation_messages: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def success(self) -> bool:
        return not self.errors


DATASETS: dict[str, DatasetConfig] = {
    "kaggle_asl_1": DatasetConfig(
        key="kaggle_asl_1",
        label="Kaggle ASL Dataset 1",
        source_type="kaggle",
        kaggle_slug="debashishsau/aslamerican-sign-language-aplhabet-dataset",
        train_target=Path("data_raw"),
        prefix="kaggle_asl_1",
        allow_train_root_fallback=True,
    ),
    "kaggle_asl_2": DatasetConfig(
        key="kaggle_asl_2",
        label="Kaggle ASL Dataset 2",
        source_type="kaggle",
        kaggle_slug="lexset/synthetic-asl-alphabet",
        train_target=Path("data_raw"),
        test_target=Path("external_test/dataset2"),
        prefix="kaggle_asl_2",
        allow_train_root_fallback=True,
        allow_test_root_fallback=False,
    ),
    "zenodo_asl": DatasetConfig(
        key="zenodo_asl",
        label="Zenodo ASL Dataset",
        source_type="zenodo",
        zenodo_record_id="14635573",
        test_target=Path("external_test/dataset3"),
        prefix="zenodo_asl",
        allow_train_root_fallback=False,
        allow_test_root_fallback=True,
    ),
}


def setup_logging() -> logging.Logger:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("setup_data")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    file_handler = logging.FileHandler(LOG_PATH, encoding="utf-8")
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


LOGGER = setup_logging()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Lädt ASL-Datensätze manuell herunter, entpackt sie und sortiert nur die "
            "Buchstaben A, B, C, L, O, V, W, Y in die Projektstruktur ein."
        )
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Bereits importierte Daten dieses Skripts löschen und neu herunterladen.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Zeigt nur an, was passieren würde, ohne Dateien zu laden oder zu kopieren.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=sorted(DATASETS.keys()),
        default=list(DATASETS.keys()),
        help="Welche Datensätze verarbeitet werden sollen.",
    )
    return parser.parse_args()


def check_kaggle_auth() -> None:
    kaggle_bin = shutil.which("kaggle")
    if kaggle_bin is None:
        raise RuntimeError(
            "Kaggle CLI wurde nicht gefunden. Installiere sie mit "
            "'pip install kaggle' und stelle sicher, dass 'kaggle' im PATH liegt."
        )
    if not KAGGLE_CONFIG_PATH.exists():
        raise RuntimeError(
            f"Kaggle Authentifizierung fehlt: '{KAGGLE_CONFIG_PATH}' wurde nicht gefunden. "
            "Lege dort deine kaggle.json ab."
        )
    LOGGER.info("Kaggle CLI und Authentifizierung gefunden: %s", KAGGLE_CONFIG_PATH)


def run_subprocess(command: list[str]) -> subprocess.CompletedProcess[str]:
    LOGGER.info("Starte Kommando: %s", " ".join(command))
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.stdout.strip():
        LOGGER.info("stdout: %s", result.stdout.strip())
    if result.stderr.strip():
        LOGGER.info("stderr: %s", result.stderr.strip())
    return result


def download_kaggle_dataset(
    dataset_slug: str,
    download_dir: Path,
    dry_run: bool = False,
    force: bool = False,
) -> list[Path]:
    download_dir.mkdir(parents=True, exist_ok=True)
    archive_name = f"{dataset_slug.split('/')[-1]}.zip"
    target_archive = download_dir / archive_name

    if dry_run:
        LOGGER.info("[Dry-Run] Würde Kaggle-Dataset herunterladen: %s", dataset_slug)
        return [target_archive]

    if target_archive.exists() and not force:
        LOGGER.info("Kaggle-Archiv bereits vorhanden, nutze vorhandene Datei: %s", target_archive)
        return [target_archive]

    command = ["kaggle", "datasets", "download", "-d", dataset_slug, "-p", str(download_dir)]
    if force:
        command.append("--force")

    result = run_subprocess(command)
    if result.returncode != 0:
        stderr = result.stderr.lower()
        if "401" in stderr or "unauthorized" in stderr or "credentials" in stderr:
            raise RuntimeError(
                "Kaggle-Download fehlgeschlagen. Bitte prüfe deine Authentifizierung in "
                f"'{KAGGLE_CONFIG_PATH}'."
            )
        raise RuntimeError(f"Kaggle-Download fehlgeschlagen für '{dataset_slug}'.")

    archives = sorted(download_dir.glob("*.zip"))
    if not archives:
        raise RuntimeError(f"Kein Archiv nach Kaggle-Download gefunden in '{download_dir}'.")
    return archives


def fetch_zenodo_record(record_id: str, dry_run: bool = False) -> dict:
    url = ZENODO_API_URL.format(record_id=record_id)
    if dry_run:
        LOGGER.info("[Dry-Run] Würde Zenodo-Record abrufen: %s", url)
        return {"id": record_id, "files": []}

    LOGGER.info("Rufe Zenodo-Record ab: %s", url)
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    record = response.json()
    LOGGER.info("Zenodo-Record %s mit %s Dateien geladen.", record_id, len(record.get("files", [])))
    return record


def download_zenodo_files(
    record: dict,
    download_dir: Path,
    dry_run: bool = False,
    force: bool = False,
) -> list[Path]:
    download_dir.mkdir(parents=True, exist_ok=True)
    files = record.get("files", [])
    if not files and not dry_run:
        raise RuntimeError("Zenodo-Record enthält keine Dateien.")

    downloaded: list[Path] = []
    for file_entry in files:
        file_name = file_entry.get("key") or file_entry.get("filename")
        if not file_name:
            continue
        destination = download_dir / file_name
        download_url = (
            file_entry.get("links", {}).get("self")
            or file_entry.get("links", {}).get("download")
            or file_entry.get("href")
        )
        if dry_run:
            LOGGER.info("[Dry-Run] Würde Zenodo-Datei herunterladen: %s", file_name)
            downloaded.append(destination)
            continue

        if destination.exists() and not force:
            LOGGER.info("Zenodo-Datei bereits vorhanden, überspringe Download: %s", destination)
            downloaded.append(destination)
            continue

        if not download_url:
            raise RuntimeError(f"Kein Download-Link für Zenodo-Datei '{file_name}' gefunden.")

        destination.parent.mkdir(parents=True, exist_ok=True)
        LOGGER.info("Lade Zenodo-Datei herunter: %s", file_name)
        with requests.get(download_url, stream=True, timeout=120) as response:
            response.raise_for_status()
            with destination.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        handle.write(chunk)
        downloaded.append(destination)
    return downloaded


def extract_archive(archive_path: Path, extract_root: Path, dry_run: bool = False) -> Path:
    destination = extract_root / archive_path.stem
    if dry_run:
        LOGGER.info("[Dry-Run] Würde Archiv entpacken: %s -> %s", archive_path, destination)
        return destination

    destination.mkdir(parents=True, exist_ok=True)

    suffixes = [suffix.lower() for suffix in archive_path.suffixes]
    LOGGER.info("Entpacke Archiv: %s", archive_path)

    if zipfile.is_zipfile(archive_path):
        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(destination)
        return destination

    if tarfile.is_tarfile(archive_path):
        with tarfile.open(archive_path) as archive:
            archive.extractall(destination)
        return destination

    if suffixes and suffixes[-1] == ".gz" and not archive_path.name.endswith(".tar.gz"):
        output_file = destination / archive_path.stem
        with gzip.open(archive_path, "rb") as source, output_file.open("wb") as target:
            shutil.copyfileobj(source, target)
        return destination

    raise RuntimeError(f"Archivformat wird nicht unterstützt: {archive_path.name}")


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def contains_images(directory: Path) -> bool:
    try:
        return any(is_image_file(path) for path in directory.iterdir())
    except PermissionError:
        return False


def path_tokens(path: Path) -> set[str]:
    return {part.lower().replace("-", "_").replace(" ", "_") for part in path.parts}


def iter_letter_dirs(root: Path) -> list[Path]:
    candidates: list[Path] = []
    for directory in root.rglob("*"):
        if not directory.is_dir():
            continue
        if directory.name in LETTERS and contains_images(directory):
            tokens = path_tokens(directory.relative_to(root))
            if tokens & SKIP_DIR_MARKERS:
                continue
            candidates.append(directory)
    return sorted(set(candidates))


def classify_candidate_paths(
    source_root: Path,
    *,
    prefer_test: bool,
    allow_root_fallback: bool,
) -> dict[str, Path]:
    letter_dirs = iter_letter_dirs(source_root)
    selected: dict[str, Path] = {}

    for letter_dir in letter_dirs:
        relative = letter_dir.relative_to(source_root)
        tokens = path_tokens(relative.parent)
        if prefer_test and tokens & TEST_MARKERS:
            selected.setdefault(letter_dir.name, letter_dir)
        if not prefer_test and tokens & TEST_MARKERS:
            continue
        if not prefer_test and tokens & TRAIN_MARKERS:
            selected.setdefault(letter_dir.name, letter_dir)

    if len(selected) == len(LETTERS):
        return selected

    if allow_root_fallback:
        for letter_dir in letter_dirs:
            relative = letter_dir.relative_to(source_root)
            parent_tokens = path_tokens(relative.parent)
            if prefer_test or not (parent_tokens & TEST_MARKERS):
                selected.setdefault(letter_dir.name, letter_dir)

    return selected


def remove_prefixed_files(target_root: Path, prefix: str, dry_run: bool = False) -> int:
    removed = 0
    for letter in LETTERS:
        letter_dir = target_root / letter
        if not letter_dir.exists():
            continue
        for file_path in letter_dir.glob(f"{prefix}__*"):
            if file_path.is_file():
                if dry_run:
                    LOGGER.info("[Dry-Run] Würde Datei löschen: %s", file_path)
                else:
                    file_path.unlink()
                removed += 1
    return removed


def build_target_name(prefix: str, letter: str, source_file: Path) -> str:
    digest = hashlib.sha1(str(source_file).encode("utf-8")).hexdigest()[:10]
    return f"{prefix}__{letter}__{digest}__{source_file.name}"


def copy_letter_folders(
    letter_sources: dict[str, Path],
    destination_root: Path,
    *,
    prefix: str,
    dry_run: bool = False,
) -> dict[str, int]:
    copied_counts = {letter: 0 for letter in LETTERS}
    for letter in LETTERS:
        source_dir = letter_sources.get(letter)
        target_dir = destination_root / letter
        if not source_dir:
            continue

        if dry_run:
            LOGGER.info("[Dry-Run] Würde Zielordner anlegen: %s", target_dir)
        else:
            target_dir.mkdir(parents=True, exist_ok=True)

        for source_file in sorted(source_dir.iterdir()):
            if not is_image_file(source_file):
                continue
            target_name = build_target_name(prefix, letter, source_file)
            target_file = target_dir / target_name
            if dry_run:
                LOGGER.info("[Dry-Run] Würde kopieren: %s -> %s", source_file, target_file)
                copied_counts[letter] += 1
                continue

            shutil.copy2(source_file, target_file)
            copied_counts[letter] += 1
    return copied_counts


def copy_test_data(
    letter_sources: dict[str, Path],
    destination_root: Path,
    *,
    prefix: str,
    dry_run: bool = False,
) -> dict[str, int]:
    return copy_letter_folders(letter_sources, destination_root, prefix=prefix, dry_run=dry_run)


def target_has_dataset_files(target_root: Path | None, prefix: str) -> bool:
    if target_root is None:
        return False
    for letter in LETTERS:
        letter_dir = target_root / letter
        if letter_dir.exists() and any(letter_dir.glob(f"{prefix}__*")):
            return True
    return False


def validate_dataset(
    result: DatasetResult,
    *,
    download_expected: bool,
    extract_expected: bool,
) -> None:
    config = result.config

    if result.dry_run:
        result.validation_messages.append("Dry-Run: Validierung nur simuliert.")
        return

    if download_expected and not result.downloaded_files:
        result.errors.append("Download fehlgeschlagen: keine Dateien vorhanden.")

    if extract_expected and not result.extracted_paths:
        result.errors.append("Entpacken fehlgeschlagen: keine Verzeichnisse vorhanden.")

    for target_root, counts, label in (
        (config.train_target, result.train_counts, "Training"),
        (config.test_target, result.test_counts, "Test"),
    ):
        if target_root is None or not counts:
            continue

        for letter in LETTERS:
            target_dir = target_root / letter
            if not target_dir.exists():
                result.errors.append(f"{label}: Zielordner fehlt: {target_dir}")
                continue
            files = [path for path in target_dir.iterdir() if is_image_file(path)]
            if not files:
                result.errors.append(f"{label}: Keine Dateien in {target_dir}")

        extra_dirs = [path.name for path in target_root.iterdir() if path.is_dir() and path.name not in LETTERS]
        if extra_dirs:
            result.errors.append(
                f"{label}: Unerwartete Unterordner in {target_root}: {', '.join(sorted(extra_dirs))}"
            )

        copied_letters = {letter for letter, count in counts.items() if count > 0}
        invalid_letters = copied_letters - set(LETTERS)
        if invalid_letters:
            result.errors.append(f"{label}: Unerwartete Buchstaben übernommen: {sorted(invalid_letters)}")

        result.validation_messages.append(
            f"{label} validiert: {sum(counts.values())} Dateien in {target_root}"
        )


def print_summary(results: Iterable[DatasetResult]) -> None:
    print("\n" + "=" * 72)
    print("SETUP DATA SUMMARY")
    print("=" * 72)

    for result in results:
        status = "ERFOLG"
        if result.dry_run:
            status = "DRY-RUN"
        elif result.skipped:
            status = "ÜBERSPRUNGEN"
        elif result.errors:
            status = "FEHLER"

        print(f"{result.config.key}: {status}")
        print(f"  Quelle: {result.config.label}")

        if result.train_counts:
            train_total = sum(result.train_counts.values())
            print(f"  Training nach data_raw/: {train_total} Dateien")
            print(f"  Verteilung: {json.dumps(result.train_counts, ensure_ascii=False)}")

        if result.test_counts:
            test_total = sum(result.test_counts.values())
            print(f"  Test nach {result.config.test_target}/: {test_total} Dateien")
            print(f"  Verteilung: {json.dumps(result.test_counts, ensure_ascii=False)}")

        for message in result.validation_messages:
            print(f"  Validierung: {message}")
        for warning in result.warnings:
            print(f"  Warnung: {warning}")
        for error in result.errors:
            print(f"  Fehler: {error}")
        print()

    print(f"Logdatei: {LOG_PATH}")


def gather_extracted_roots(downloaded_files: list[Path], extract_dir: Path, dry_run: bool) -> list[Path]:
    extracted_roots: list[Path] = []
    for file_path in downloaded_files:
        if file_path.suffix.lower() in {".zip", ".tar", ".gz", ".tgz", ".bz2", ".xz"} or tarfile.is_tarfile(file_path):
            extracted_roots.append(extract_archive(file_path, extract_dir, dry_run=dry_run))
        elif file_path.is_dir():
            extracted_roots.append(file_path)
        else:
            LOGGER.info("Datei ist kein Archiv, wird für Struktursuche ignoriert: %s", file_path)
    return extracted_roots


def detect_sources(config: DatasetConfig, extracted_roots: list[Path], result: DatasetResult) -> tuple[dict[str, Path], dict[str, Path]]:
    train_sources: dict[str, Path] = {}
    test_sources: dict[str, Path] = {}
    explicit_test_split_found = False

    for root in extracted_roots:
        if config.train_target and len(train_sources) < len(LETTERS):
            discovered_train = classify_candidate_paths(
                root,
                prefer_test=False,
                allow_root_fallback=config.allow_train_root_fallback,
            )
            for letter, path in discovered_train.items():
                train_sources.setdefault(letter, path)

        if config.test_target and len(test_sources) < len(LETTERS):
            discovered_test = classify_candidate_paths(
                root,
                prefer_test=True,
                allow_root_fallback=config.allow_test_root_fallback,
            )
            for letter, path in discovered_test.items():
                test_sources.setdefault(letter, path)
                if path_tokens(path.parent) & TEST_MARKERS:
                    explicit_test_split_found = True

    if config.train_target and len(train_sources) < len(LETTERS):
        missing = sorted(set(LETTERS) - set(train_sources))
        result.errors.append(f"Trainingsdaten unvollständig. Fehlende Buchstaben: {', '.join(missing)}")

    if config.test_target and len(test_sources) < len(LETTERS):
        missing = sorted(set(LETTERS) - set(test_sources))
        result.errors.append(f"Testdaten unvollständig. Fehlende Buchstaben: {', '.join(missing)}")

    if config.key == "zenodo_asl" and config.test_target and not explicit_test_split_found:
        result.warnings.append(
            "Für Dataset 3 wurde keine klare interne Train/Test-Trennung erkannt. "
            "Es wurden die gefundenen Buchstabenordner als externer Test unter "
            f"'{config.test_target}' einsortiert."
        )

    return train_sources, test_sources


def ensure_base_directories() -> None:
    for base in (Path("data_raw"), Path("external_test/dataset2"), Path("external_test/dataset3")):
        for letter in LETTERS:
            (base / letter).mkdir(parents=True, exist_ok=True)


def process_dataset(config: DatasetConfig, *, force: bool, dry_run: bool) -> DatasetResult:
    result = DatasetResult(config=config, dry_run=dry_run)

    if not force and (
        target_has_dataset_files(config.train_target, config.prefix)
        or target_has_dataset_files(config.test_target, config.prefix)
    ):
        result.skipped = True
        result.validation_messages.append(
            "Bereits importierte Dateien mit passendem Präfix gefunden. Nutze --force zum Neuaufbau."
        )
        return result

    work_dir = WORK_ROOT / config.key
    download_dir = work_dir / "downloads"
    extract_dir = work_dir / "extracted"

    if force:
        if config.train_target:
            removed = remove_prefixed_files(config.train_target, config.prefix, dry_run=dry_run)
            LOGGER.info("%s vorhandene Trainingsdateien für %s entfernt.", removed, config.key)
        if config.test_target:
            removed = remove_prefixed_files(config.test_target, config.prefix, dry_run=dry_run)
            LOGGER.info("%s vorhandene Testdateien für %s entfernt.", removed, config.key)

    if dry_run:
        LOGGER.info("[Dry-Run] Würde Arbeitsordner verwenden: %s", work_dir)
    else:
        download_dir.mkdir(parents=True, exist_ok=True)
        extract_dir.mkdir(parents=True, exist_ok=True)

    try:
        if config.source_type == "kaggle":
            if not dry_run:
                check_kaggle_auth()
            result.downloaded_files = download_kaggle_dataset(
                config.kaggle_slug or "",
                download_dir,
                dry_run=dry_run,
                force=force,
            )
        elif config.source_type == "zenodo":
            record = fetch_zenodo_record(config.zenodo_record_id or "", dry_run=dry_run)
            result.downloaded_files = download_zenodo_files(
                record,
                download_dir,
                dry_run=dry_run,
                force=force,
            )
        else:
            raise RuntimeError(f"Unbekannter Quelltyp: {config.source_type}")

        result.extracted_paths = gather_extracted_roots(result.downloaded_files, extract_dir, dry_run=dry_run)

        if not dry_run:
            train_sources, test_sources = detect_sources(config, result.extracted_paths, result)
            if result.errors:
                return result

            if config.train_target:
                result.train_counts = copy_letter_folders(
                    train_sources,
                    config.train_target,
                    prefix=config.prefix,
                    dry_run=dry_run,
                )
            if config.test_target:
                result.test_counts = copy_test_data(
                    test_sources,
                    config.test_target,
                    prefix=config.prefix,
                    dry_run=dry_run,
                )
        else:
            if config.train_target:
                result.train_counts = {letter: 0 for letter in LETTERS}
            if config.test_target:
                result.test_counts = {letter: 0 for letter in LETTERS}

    except requests.RequestException as exc:
        result.errors.append(f"Netzwerkfehler: {exc}")
    except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
        result.errors.append(str(exc))

    validate_dataset(result, download_expected=True, extract_expected=True)
    return result


def main() -> int:
    args = parse_args()
    ensure_base_directories()

    LOGGER.info("Starte setup_data.py mit Datensätzen: %s", ", ".join(args.datasets))
    if args.dry_run:
        LOGGER.info("Dry-Run aktiv: Es werden keine Dateien verändert.")

    results = [process_dataset(DATASETS[key], force=args.force, dry_run=args.dry_run) for key in args.datasets]
    print_summary(results)

    if any(not result.success for result in results if not result.skipped and not result.dry_run):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
