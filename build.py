"""Build a release: the Python engine sidecar (PyInstaller) + the Godot UI export.

    python -m pip install -r requirements-build.txt
    python build.py --godot "C:/Tools/Godot_v4.4.1-stable_win64.exe"      # on Windows
    python build.py --godot /Applications/Godot.app/Contents/MacOS/Godot  # on macOS

Godot export templates for the same Godot version must be installed
(Editor > Manage Export Templates). Output goes to dist/<platform>/.
The UI looks for the engine at <exe dir>/engine/meeting-engine[.exe], which on
macOS is MeetingTranscriptions.app/Contents/MacOS/engine/.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DIST = ROOT / "dist"


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=ROOT)


def build_engine() -> Path:
    run([
        sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean", "--onedir",
        "--name", "meeting-engine", "--noconsole" if sys.platform == "win32" else "--console",
        "--distpath", str(DIST / "engine-build"), "--workpath", str(ROOT / "build" / "pyinstaller"),
        "--specpath", str(ROOT / "build"), "--paths", str(ROOT),
        "--collect-submodules", "engine",
        str(ROOT / "packaging" / "engine_main.py"),
    ])
    return DIST / "engine-build" / "meeting-engine"


def export_ui(godot: str, preset: str, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    run([godot, "--headless", "--path", str(ROOT / "ui"), "--import"])
    run([godot, "--headless", "--path", str(ROOT / "ui"), "--export-release", preset, str(out)])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--godot", required=True, help="path to the Godot 4.4+ executable")
    parser.add_argument("--skip-engine", action="store_true")
    args = parser.parse_args()

    engine_dir = DIST / "engine-build" / "meeting-engine"
    if not args.skip_engine:
        engine_dir = build_engine()

    if sys.platform == "win32":
        target = DIST / "windows"
        export_ui(args.godot, "Windows Desktop", target / "MeetingTranscriptions.exe")
        shutil.copytree(engine_dir, target / "engine", dirs_exist_ok=True)
        print(f"Done: {target}")
    elif sys.platform == "darwin":
        target = DIST / "macos"
        archive = target / "MeetingTranscriptions.zip"
        export_ui(args.godot, "macOS", archive)
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(target)
        app = next(target.glob("*.app"))
        # zipfile drops exec bits, so restore them on the app binary.
        for exe in (app / "Contents" / "MacOS").iterdir():
            exe.chmod(0o755)
        shutil.copytree(engine_dir, app / "Contents" / "MacOS" / "engine", dirs_exist_ok=True)
        # Re-sign ad hoc after adding the sidecar so Gatekeeper accepts the bundle locally.
        subprocess.run(["codesign", "--force", "--deep", "--sign", "-", str(app)], check=False)
        print(f"Done: {app}")
    else:
        print("Release builds are for Windows and macOS. On Linux use ./run.sh.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
