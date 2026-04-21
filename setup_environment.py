#!/usr/bin/env python
"""
Setup Script für ml_train Umgebung
Installiert alle fehlenden Abhängigkeiten
"""

import subprocess
import sys
import platform

# Python Version prüfen
python_version = sys.version_info

REQUIRED_PACKAGES = {
    'cv2': 'opencv-python',
    'tensorflow': 'tensorflow>=2.13,<2.15' if python_version.minor >= 13 else 'tensorflow>=2.13',
    'keras': 'keras>=2.13',
    'matplotlib': 'matplotlib>=3.7',
    'sklearn': 'scikit-learn>=1.3',
    'numpy': 'numpy>=1.24',
    'PIL': 'pillow>=9.0',
    'mediapipe': 'mediapipe>=0.8'
}

def check_package(package_name):
    """Prüfe ob Paket installiert ist"""
    try:
        __import__(package_name)
        return True
    except ImportError:
        return False

def main():
    print("=" * 60)
    print("Sign Language Recognition - Environment Setup")
    print("=" * 60)
    print(f"Python Version: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    print()
    
    missing_packages = []
    
    print("🔍 Prüfe installierte Pakete...")
    for import_name, pip_package in REQUIRED_PACKAGES.items():
        if check_package(import_name):
            print(f"✓ {import_name}")
        else:
            print(f"✗ {import_name} - FEHLT")
            missing_packages.append(pip_package)
    
    print()
    
    if not missing_packages:
        print("✓ Alle Pakete sind installiert!")
        return 0
    
    print(f"⚠️  {len(missing_packages)} Pakete fehlen.")
    print()
    print("📦 Installiere fehlende Pakete...")
    print()
    
    # Installiere Pakete
    for package in missing_packages:
        print(f"  → {package}")
        result = subprocess.run(
            [sys.executable, '-m', 'pip', 'install', '-q', package],
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            print(f"    ⚠️  Versionskonflikt, versuche ohne Versionsbeschränkung...")
            # Versuche ohne Versionsbeschränkung
            base_package = package.split('==')[0].split('>=')[0].split('<')[0]
            result = subprocess.run(
                [sys.executable, '-m', 'pip', 'install', '-q', base_package],
                capture_output=True,
                text=True
            )
            if result.returncode != 0:
                print(f"    ❌ Fehler: {result.stderr}")
                return 1
            print(f"    ✓ Installiert: {base_package}")
        else:
            print(f"    ✓ Installiert")
    
    print()
    print("✓ Pakete installiert")
    print()
    
    # Verifiziere Installation
    print("🔍 Verifiziere Installation...")
    print()
    
    for import_name in REQUIRED_PACKAGES.keys():
        if check_package(import_name):
            try:
                mod = __import__(import_name)
                version = getattr(mod, '__version__', 'OK')
                print(f"✓ {import_name}: {version}")
            except:
                print(f"✓ {import_name}: OK")
        else:
            print(f"✗ {import_name}: FEHLT")
            return 1
    
    print()
    print("=" * 60)
    print("✓ Umgebung erfolgreich konfiguriert!")
    print("=" * 60)
    print()
    print("Sie können jetzt das Training starten:")
    print("  python train_simple.py")
    print()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
