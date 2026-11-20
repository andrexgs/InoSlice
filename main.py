"""Ponto de entrada do InoSlice.

Sem argumentos, abre a interface grafica. Com ``--images``/``--labels``/``--output``
executa em modo linha de comando (util para servidores e pipelines)::

    python main.py --images data/images --labels data/labels --output out \
        --padding 15 --mode letterbox --color gray --target-size 224
"""

from __future__ import annotations

import argparse
import multiprocessing
import sys
from pathlib import Path

from src import __app_name__, __author__, __version__


def build_parser() -> argparse.ArgumentParser:
    """Cria o parser de argumentos do modo CLI."""
    p = argparse.ArgumentParser(
        prog="InoSlice",
        description=f"{__app_name__} v{__version__} - recortes de deteccao para classificacao sem data leakage.",
        epilog=f"Desenvolvido por {__author__}. Licenca Apache-2.0.",
    )
    p.add_argument("--images", help="Diretorio das imagens originais.")
    p.add_argument("--labels", help="Diretorio de .txt YOLO ou arquivo COCO .json.")
    p.add_argument("--classes", default=None, help="classes.txt / data.yaml opcional.")
    p.add_argument("--output", help="Diretorio de saida.")
    p.add_argument("--padding", type=float, default=10.0, help="Margem de contexto por lado em %% (padrao 10).")
    p.add_argument("--min-width", type=int, default=32, help="Largura minima da caixa original (px).")
    p.add_argument("--min-height", type=int, default=32, help="Altura minima da caixa original (px).")
    p.add_argument("--min-confidence", type=float, default=0.0, help="Confianca minima (0-1).")
    p.add_argument("--mode", choices=["rectangular", "letterbox"], default="letterbox")
    p.add_argument("--color", choices=["gray", "black"], default="gray", help="Cor do letterbox.")
    p.add_argument("--target-size", type=int, default=None, help="Lado alvo (ex.: 224, 384).")
    p.add_argument("--format", choices=[".jpg", ".png"], default=".jpg", help="Formato dos recortes.")
    p.add_argument("--quality", type=int, default=95, help="Qualidade JPEG (1-100).")
    p.add_argument("--workers", type=int, default=None, help="Numero de threads.")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return p


def run_cli(args: argparse.Namespace) -> int:
    """Executa a colheita sem interface grafica."""
    from src.core.exporter import CropHarvester, HarvestProgress
    from src.core.parsers import load_annotations
    from src.core.processor import ProcessingConfig
    from src.utils.logger import get_logger

    log = get_logger()
    if not (args.images and args.labels and args.output):
        log.error("Modo CLI requer --images, --labels e --output.")
        return 2

    dataset = load_annotations(args.images, args.labels, args.classes)
    config = ProcessingConfig(
        padding_pct=args.padding,
        min_width=args.min_width,
        min_height=args.min_height,
        min_confidence=args.min_confidence,
        aspect_mode=args.mode,
        letterbox_color=args.color,
        target_size=args.target_size,
        jpeg_quality=args.quality,
    )
    step = max(1, len(dataset.items) // 20)

    def on_progress(p: HarvestProgress) -> None:
        if p.images_done % step == 0 or p.images_done == p.images_total:
            print(
                f"\r[{p.fraction * 100:5.1f}%] {p.images_done}/{p.images_total} imagens | "
                f"{p.crops_done} crops | {p.crops_per_second:.1f} crops/s",
                end="",
                flush=True,
            )

    result = CropHarvester(
        dataset,
        config,
        Path(args.output),
        workers=args.workers,
        image_ext=args.format,
        source_info={"labels_path": args.labels, "class_file": args.classes},
    ).run(progress_cb=on_progress)
    print()
    log.info("Manifesto: %s", result.manifest_path)
    log.info("Resumo:    %s", result.summary_path)
    return 0


def main(argv: list[str] | None = None) -> int:
    """Despacha para CLI (quando ha argumentos de entrada) ou GUI."""
    argv = sys.argv[1:] if argv is None else argv
    args = build_parser().parse_args(argv)
    if args.images or args.labels or args.output:
        return run_cli(args)
    from src.gui.main_window import run_gui

    return run_gui([sys.argv[0]])


if __name__ == "__main__":
    multiprocessing.freeze_support()  # necessario para executaveis PyInstaller no Windows
    sys.exit(main())
