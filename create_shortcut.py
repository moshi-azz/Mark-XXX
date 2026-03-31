"""
create_shortcut.py — JARVIS Desktop Shortcut Creator
=====================================================
Run this script ONCE from inside the project folder:

    python create_shortcut.py

It will create a "JARVIS.lnk" shortcut on your Desktop that:
  - Opens a CMD window showing live logs  (python.exe main.py)
  - Uses the Arc Reactor icon  (jarvis.ico)
  - Sets the working directory to this project folder
"""

import sys
import subprocess
from pathlib import Path


def find_python_exe() -> str:
    """Return the path to python.exe in the current environment."""
    return sys.executable


def create_jarvis_shortcut():
    project_dir = Path(__file__).resolve().parent
    target      = find_python_exe()
    arguments   = str(project_dir / "main.py")
    icon_path   = str(project_dir / "jarvis.ico")
    shortcut_dst = Path.home() / "Desktop" / "JARVIS.lnk"

    print(f"[SETUP] Project folder : {project_dir}")
    print(f"[SETUP] Python         : {target}")
    print(f"[SETUP] Icon           : {icon_path}")
    print(f"[SETUP] Shortcut       : {shortcut_dst}")

    # Use PowerShell to create the .lnk (no extra dependencies needed)
    ps_script = f"""
$WScriptShell = New-Object -ComObject WScript.Shell
$Shortcut = $WScriptShell.CreateShortcut("{shortcut_dst}")
$Shortcut.TargetPath  = "{target}"
$Shortcut.Arguments   = '"{arguments}"'
$Shortcut.WorkingDirectory = "{project_dir}"
$Shortcut.IconLocation = "{icon_path}"
$Shortcut.Description  = "J.A.R.V.I.S - MARK XXXV"
$Shortcut.WindowStyle  = 1
$Shortcut.Save()
Write-Host "Shortcut created at {shortcut_dst}"
"""

    result = subprocess.run(
        ["powershell", "-NoProfile", "-NonInteractive", "-Command", ps_script],
        capture_output=True, text=True
    )

    if result.returncode == 0:
        print(f"\n✅  Acceso directo creado en: {shortcut_dst}")
        print("    Haz doble clic en 'JARVIS' del escritorio para arrancar.")
    else:
        print(f"\n❌  Error al crear el acceso directo:")
        print(result.stderr)
        print("\n--- Alternativa manual ---")
        print(f"1. Clic derecho en el escritorio → Nuevo → Acceso directo")
        print(f"2. Ubicación: {target} \"{arguments}\"")
        print(f"3. Nombre: JARVIS")
        print(f"4. Clic derecho en el acceso directo → Propiedades → Cambiar icono → {icon_path}")


if __name__ == "__main__":
    if sys.platform != "win32":
        print("⚠️  Este script está pensado para Windows.")
        print("    Ejecútalo en tu PC con: python create_shortcut.py")
        sys.exit(1)

    create_jarvis_shortcut()
    input("\nPresiona Enter para cerrar...")
