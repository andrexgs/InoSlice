"""Gera o executavel standalone do InoSlice com PyInstaller.

Uso (no Windows, para obter ``dist/InoSlice.exe``)::

    python build_exe.py            # arquivo unico (--onefile), sem console
    python build_exe.py --onedir   # pasta com dependencias (inicializacao mais rapida)
    python build_exe.py --console  # mantem o console (util para depuracao)

O PyInstaller nao faz compilacao cruzada: o ``.exe`` deve ser gerado em uma
maquina Windows. Em Linux/macOS o mesmo script gera o binario nativo.
"""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
APP_NAME = "InoSlice"
ENTRY_POINT = ROOT / "main.py"
ICON_CANDIDATES = (ROOT / "assets" / "icon.ico", ROOT / "assets" / "icon.icns", ROOT / "assets" / "icon.png")

#: Modulos pesados nao utilizados (reduzem drasticamente o tamanho do executavel).
EXCLUDED_MODULES: tuple[str, ...] = (
    "tkinter",
    "matplotlib",
    "pytest",
    "IPython",
    "PySide6.QtWebEngineCore",
    "PySide6.QtWebEngineWidgets",
    "PySide6.QtWebEngineQuick",
    "PySide6.QtWebChannel",
    "PySide6.QtWebSockets",
    "PySide6.QtQuick",
    "PySide6.QtQuick3D",
    "PySide6.QtQuickWidgets",
    "PySide6.QtQml",
    "PySide6.Qt3DCore",
    "PySide6.Qt3DRender",
    "PySide6.Qt3DExtras",
    "PySide6.QtMultimedia",
    "PySide6.QtMultimediaWidgets",
    "PySide6.QtCharts",
    "PySide6.QtDataVisualization",
    "PySide6.QtGraphs",
    "PySide6.QtPdf",
    "PySide6.QtPdfWidgets",
    "PySide6.QtBluetooth",
    "PySide6.QtNfc",
    "PySide6.QtPositioning",
    "PySide6.QtLocation",
    "PySide6.QtSensors",
    "PySide6.QtSerialPort",
    "PySide6.QtSql",
    "PySide6.QtTest",
    "PySide6.QtDesigner",
    "PySide6.QtHelp",
    "PySide6.QtSvg",
)


def build_arguments(onefile: bool, windowed: bool, clean: bool) -> list[str]:
    """Monta a lista de argumentos para ``PyInstaller.__main__.run``."""
    args: list[str] = [
        str(ENTRY_POINT),
        "--name", APP_NAME,
        "--noconfirm",
        "--distpath", str(ROOT / "dist"),
        "--workpath", str(ROOT / "build"),
        "--specpath", str(ROOT / "build"),
        "--paths", str(ROOT),
        "--hidden-import", "yaml",
        "--hidden-import", "cv2",
        "--add-data", f"{ROOT / 'assets'}{os.pathsep}assets",  # logo usada na janela
        "--add-data", f"{ROOT / 'LICENSE'}{os.pathsep}.",  # licenca e atribuicao junto ao executavel
        "--add-data", f"{ROOT / 'NOTICE'}{os.pathsep}.",
        "--onefile" if onefile else "--onedir",
    ]
    if windowed:
        args.append("--windowed")  # equivalente a --noconsole
    if clean:
        args.append("--clean")
    for mod in EXCLUDED_MODULES:
        args += ["--exclude-module", mod]
    icon = next((p for p in ICON_CANDIDATES if p.is_file()), None)
    if icon is not None:
        args += ["--icon", str(icon)]
    return args


def main(argv: list[str] | None = None) -> int:
    """Valida o ambiente, executa o PyInstaller e informa o caminho do artefato."""
    parser = argparse.ArgumentParser(description="Empacota o InoSlice com PyInstaller.")
    parser.add_argument("--onedir", action="store_true", help="Gera uma pasta em vez de arquivo unico.")
    parser.add_argument("--console", action="store_true", help="Mantem a janela de console (depuracao).")
    parser.add_argument("--no-clean", action="store_true", help="Reaproveita o cache de build anterior.")
    opts = parser.parse_args(argv)

    try:
        import PyInstaller.__main__ as pyinstaller
    except ImportError:
        print("PyInstaller nao encontrado. Instale com: pip install -r requirements.txt", file=sys.stderr)
        return 1

    if platform.system() != "Windows":
        print(
            f"[aviso] Sistema atual: {platform.system()}. O PyInstaller gera binarios nativos; "
            "para obter o .exe execute este script no Windows."
        )

    if not opts.no_clean:
        shutil.rmtree(ROOT / "build", ignore_errors=True)

    args = build_arguments(onefile=not opts.onedir, windowed=not opts.console, clean=not opts.no_clean)
    print("PyInstaller", " ".join(args))
    os.chdir(ROOT)
    pyinstaller.run(args)

    suffix = ".exe" if platform.system() == "Windows" else ""
    artifact = ROOT / "dist" / (f"{APP_NAME}{suffix}" if not opts.onedir else APP_NAME)
    if artifact.exists():
        print(f"\nBuild concluido: {artifact}")
        return 0
    print("\nBuild finalizado, mas o artefato esperado nao foi encontrado.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
