#!/usr/bin/env python
"""
Conda Environment Setup für Sign Language Recognition
Erstellt/aktualisiert die ml_train Umgebung mit Python 3.11 und allen Abhängigkeiten
"""

import subprocess
import sys

def run_command(cmd, description=""):
    """Führt einen Shell-Befehl aus"""
    if description:
        print(f"\n{description}...")
    print(f"→ {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"❌ Fehler: {result.stderr}")
        return False
    if result.stdout:
        print(result.stdout)
    return True

def main():
    print("=" * 70)
    print("Sign Language Recognition - Conda Environment Setup")
    print("=" * 70)
    print()
    
    print("⚠️  WARNUNG: TensorFlow 2.x ist nicht mit Python 3.13 kompatibel!")
    print("   Erstelle eine neue Umgebung mit Python 3.11...")
    print()
    
    # Option 1: Nur install versuchen
    print("📦 Versuche Installation in ml_train...")
    cmd = [
        'conda', 'run', '-n', 'ml_train', 'pip', 'install',
        'opencv-python',
        'tensorflow==2.13.1',
        'keras',
        'matplotlib',
        'scikit-learn',
        'pillow',
        'mediapipe',
        '-v'
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✓ Installation erfolgreich!")
        return 0
    
    print("⚠️  Installation in ml_train fehlgeschlagen")
    print()
    print("Alternatives: Erstelle neue Umgebung 'ml_env' mit Python 3.11...")
    
    # Option 2: Neue Umgebung erstellen
    if not run_command(
        ['conda', 'create', '-n', 'ml_env', 'python=3.11', '-y'],
        "Erstelle Conda-Umgebung mit Python 3.11"
    ):
        return 1
    
    if not run_command(
        ['conda', 'run', '-n', 'ml_env', 'pip', 'install',
         'opencv-python',
         'tensorflow==2.13.1',
         'keras',
         'matplotlib',
         'scikit-learn',
         'pillow',
         'mediapipe'],
        "Installiere Pakete"
    ):
        return 1
    
    print()
    print("=" * 70)
    print("✓ Umgebung erfolgreich erstellt!")
    print("=" * 70)
    print()
    print("🎯 Sie können jetzt trainieren mit:")
    print("   conda run -n ml_env python train_simple.py")
    print()
    print("Oder aktivieren Sie die Umgebung:")
    print("   conda activate ml_env")
    print("   python train_simple.py")
    print()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
