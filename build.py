"""Build a release: the Python engine sidecar (PyInstaller) + the Godot UI export.

    python -m pip install -r requirements-build.txt
    python build.py --godot "C:/Tools/Godot_v4.4.1-stable_win64.exe"      # on Windows
    python build.py --godot /Applications/Godot.app/Contents/MacOS/Godot  # on macOS
    add --zip to also pack dist/MeetingTranscriptions-<platform>.zip (what the
    release workflow uploads and install.ps1 / install.sh download)

Godot export templates for the same Godot version must be installed
(Editor > Manage Export Templates). Output goes to dist/<platform>/.
The UI looks for the engine at <exe dir>/engine/meeting-engine[.exe], which on
macOS is MeetingTranscriptions.app/Contents/MacOS/engine/.
"""
from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
import urllib.request
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


RCEDIT_URL = "https://github.com/electron/rcedit/releases/download/v2.0.0/rcedit-x64.exe"


def ensure_rcedit(explicit: str | None) -> str:
    """Godot needs rcedit to put the app icon and version info into the Windows exe."""
    found = explicit or os.environ.get("RCEDIT") or shutil.which("rcedit") or shutil.which("rcedit-x64")
    if found:
        return found
    dest = ROOT / "build" / "rcedit-x64.exe"
    if not dest.exists():
        print(f"+ download {RCEDIT_URL}", flush=True)
        dest.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(RCEDIT_URL, timeout=60) as resp:
            dest.write_bytes(resp.read())
    return str(dest)


def point_godot_at_rcedit(rcedit: str) -> None:
    """Set Export > Windows > rcedit in Godot's editor settings (kept otherwise as they are)."""
    godot_dir = Path(os.environ["APPDATA"]) / "Godot"
    files = [f for f in (godot_dir / "editor_settings-4.4.tres", godot_dir / "editor_settings-4.tres") if f.exists()]
    if not files:
        godot_dir.mkdir(parents=True, exist_ok=True)
        files = [godot_dir / "editor_settings-4.4.tres", godot_dir / "editor_settings-4.tres"]
        for f in files:
            f.write_text('[gd_resource type="EditorSettings" format=3]\n\n[resource]\n', encoding="utf-8")
    line = 'export/windows/rcedit = "%s"' % Path(rcedit).resolve().as_posix()
    for f in files:
        lines = [l for l in f.read_text(encoding="utf-8").splitlines() if not l.startswith("export/windows/rcedit ")]
        at = lines.index("[resource]") + 1 if "[resource]" in lines else len(lines)
        lines.insert(at, line)
        f.write_text("\n".join(lines) + "\n", encoding="utf-8")


def export_ui(godot: str, preset: str, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    run([godot, "--headless", "--path", str(ROOT / "ui"), "--import"])
    cmd = [godot, "--headless", "--path", str(ROOT / "ui"), "--export-release", preset, str(out)]
    print("+", " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace")
    print(proc.stdout, flush=True)
    if proc.returncode != 0:
        raise SystemExit(f"Godot export failed ({proc.returncode})")
    if "Could not start rcedit" in proc.stdout or "rcedit failed" in proc.stdout:
        raise SystemExit("Godot couldn't set the Windows icon (rcedit problem, see above)")


def check_exe_icon(exe: Path) -> None:
    """Fail the build if the exe still carries Godot's icon instead of ui/icon.ico."""
    import pefile  # comes with PyInstaller on Windows

    want = {e[1] for e in _ico_entries(ROOT / "ui" / "icon.ico")}
    pe = pefile.PE(str(exe), fast_load=True)
    pe.parse_data_directories([pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_RESOURCE"]])
    got = set()
    root = getattr(pe, "DIRECTORY_ENTRY_RESOURCE", None)
    for kind in root.entries if root else []:
        if kind.id != pefile.RESOURCE_TYPE["RT_ICON"]:
            continue
        for res in kind.directory.entries:
            for lang in res.directory.entries:
                got.add(pe.get_data(lang.data.struct.OffsetToData, lang.data.struct.Size))
    pe.close()
    # Godot rewrites the .ico before rcedit gets it, so a size or two may be
    # re-encoded; Godot's own icon would match none of them.
    found = len(want & got)
    if found * 2 <= len(want):
        raise SystemExit(f"{exe.name} doesn't carry ui/icon.ico ({found}/{len(want)} images found)")
    print(f"Icon check: {exe.name} carries {found}/{len(want)} images of ui/icon.ico", flush=True)


def _ico_entries(path: Path) -> list[tuple[int, bytes]]:
    import struct
    data = path.read_bytes()
    count = struct.unpack_from("<H", data, 4)[0]
    out = []
    for i in range(count):
        w, _h, _c, _r, _p, _b, size, off = struct.unpack_from("<BBBBHHII", data, 6 + 16 * i)
        out.append((w or 256, data[off:off + size]))
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--godot", required=True, help="path to the Godot 4.4+ executable")
    parser.add_argument("--skip-engine", action="store_true")
    parser.add_argument("--rcedit", help="path to rcedit.exe (Windows; downloaded when missing)")
    parser.add_argument("--zip", action="store_true", help="also pack a release zip into dist/")
    args = parser.parse_args()

    engine_dir = DIST / "engine-build" / "meeting-engine"
    if not args.skip_engine:
        engine_dir = build_engine()

    if sys.platform == "win32":
        target = DIST / "windows"
        point_godot_at_rcedit(ensure_rcedit(args.rcedit))
        export_ui(args.godot, "Windows Desktop", target / "MeetingTranscriptions.exe")
        check_exe_icon(target / "MeetingTranscriptions.exe")
        shutil.copytree(engine_dir, target / "engine", dirs_exist_ok=True)
        if args.zip:
            archive = shutil.make_archive(str(DIST / "MeetingTranscriptions-windows-x64"), "zip", target)
            print(f"Release zip: {archive}")
        print(f"Done: {target}")
    elif sys.platform == "darwin":
        target = DIST / "macos"
        archive = target / "MeetingTranscriptions.zip"
        export_ui(args.godot, "macOS", archive)
        for old_app in target.glob("*.app"):
            shutil.rmtree(old_app)
        run(["ditto", "-x", "-k", str(archive), str(target)])
        app = next(target.glob("*.app"))
        shutil.copytree(engine_dir, app / "Contents" / "MacOS" / "engine", dirs_exist_ok=True)
        # Re-sign ad hoc after adding the sidecar so Gatekeeper accepts the bundle locally.
        subprocess.run(["codesign", "--force", "--deep", "--sign", "-", str(app)], check=False)
        if args.zip:
            arch = "arm64" if platform.machine() == "arm64" else "x86_64"
            release = DIST / f"MeetingTranscriptions-macos-{arch}.zip"
            release.unlink(missing_ok=True)
            # ditto keeps the bundle's symlinks and exec bits, zipfile would not.
            run(["ditto", "-c", "-k", "--keepParent", str(app), str(release)])
            print(f"Release zip: {release}")
        print(f"Done: {app}")
    else:
        print("Release builds are for Windows and macOS. On Linux use ./run.sh.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
