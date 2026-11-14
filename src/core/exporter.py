"""Execucao em lote e exportacao rastreavel dos recortes.

Gera, dentro do diretorio de saida::

    <saida>/
    ├── dataset_crops/
    │   ├── <classe_a>/<imagem>_crop_<id>.jpg
    │   └── <classe_b>/...
    ├── metadata_manifest.csv   # rastreabilidade crop -> imagem de origem
    └── harvest_summary.json    # estatisticas e parametros da execucao

O manifesto contem a coluna ``group_id`` (caminho relativo da imagem de
origem): use-a como ``groups`` em ``GroupKFold``/``StratifiedGroupKFold`` para
que recortes de uma mesma imagem nunca fiquem simultaneamente em treino e
teste (prevencao de *data leakage*).
"""

from __future__ import annotations

import csv
import json
import os
import re
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from src import __app_name__, __version__
from src.core.parsers import AnnotationDataset, ImageAnnotations
from src.core.processor import (
    DISCARD_INVALID,
    DISCARD_LOW_CONFIDENCE,
    DISCARD_MIN_SIZE,
    ProcessingConfig,
    extract_crops,
    read_image,
    write_image,
)
from src.utils.logger import get_logger

log = get_logger()

CROPS_DIRNAME = "dataset_crops"
MANIFEST_FILENAME = "metadata_manifest.csv"
SUMMARY_FILENAME = "harvest_summary.json"

#: Colunas do manifesto (as 21 obrigatorias primeiro, seguidas de colunas extras de rastreio).
MANIFEST_COLUMNS: list[str] = [
    "crop_filename",
    "crop_relative_path",
    "source_image_filename",
    "source_image_width",
    "source_image_height",
    "class_id",
    "class_name",
    "confidence",
    "orig_xmin",
    "orig_ymin",
    "orig_xmax",
    "orig_ymax",
    "padded_xmin",
    "padded_ymin",
    "padded_xmax",
    "padded_ymax",
    "crop_width",
    "crop_height",
    "padding_pct",
    "aspect_mode",
    "letterbox_color",
    # extras
    "group_id",
    "source_image_relpath",
    "detection_index",
    "output_width",
    "output_height",
    "target_size",
    "annotation_format",
]

_INVALID_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_WINDOWS_RESERVED = {
    "CON", "PRN", "AUX", "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}


def sanitize_name(name: str, fallback: str = "unnamed") -> str:
    """Torna ``name`` seguro como nome de arquivo/pasta em Windows, Linux e macOS."""
    cleaned = _INVALID_CHARS.sub("_", name.strip())
    cleaned = re.sub(r"\s+", "_", cleaned).rstrip(". ")
    if not cleaned:
        cleaned = fallback
    if cleaned.upper().split(".")[0] in _WINDOWS_RESERVED:
        cleaned = f"{cleaned}_"
    return cleaned[:150]


def image_key(relative_path: Path) -> str:
    """Identificador de arquivo unico por imagem (subpastas unidas por ``__``)."""
    parts = [sanitize_name(p) for p in relative_path.with_suffix("").parts]
    return "__".join(parts)


def build_class_dirs(class_ids: set[int], dataset: AnnotationDataset) -> dict[int, str]:
    """Mapeia cada ``class_id`` a um nome de pasta unico e seguro."""
    dirs: dict[int, str] = {}
    used: set[str] = set()
    for cid in sorted(class_ids):
        base = sanitize_name(dataset.class_name(cid), fallback=f"class_{cid}")
        name = base
        if name.lower() in used:
            name = f"{base}_{cid}"
        used.add(name.lower())
        dirs[cid] = name
    return dirs


# --------------------------------------------------------------------------- #
# Estruturas de progresso e resultado
# --------------------------------------------------------------------------- #
@dataclass
class HarvestProgress:
    """Instantaneo do progresso, enviado ao callback a cada imagem concluida."""

    images_done: int
    images_total: int
    crops_done: int
    elapsed: float
    current_image: str

    @property
    def crops_per_second(self) -> float:
        """Velocidade media de extracao."""
        return self.crops_done / self.elapsed if self.elapsed > 0 else 0.0

    @property
    def fraction(self) -> float:
        """Fracao concluida em ``[0, 1]``."""
        return self.images_done / self.images_total if self.images_total else 1.0


@dataclass
class _ImageOutcome:
    """Resultado interno do processamento de uma imagem."""

    item: ImageAnnotations
    rows: list[dict[str, object]] = field(default_factory=list)
    discarded: dict[str, int] = field(default_factory=dict)
    discarded_by_class: dict[int, int] = field(default_factory=dict)
    processed: bool = False
    read_failed: bool = False
    write_failures: int = 0
    warnings: list[str] = field(default_factory=list)


@dataclass
class HarvestResult:
    """Resultado final de uma execucao."""

    summary: dict[str, object]
    crops_dir: Path
    manifest_path: Path
    summary_path: Path
    cancelled: bool


ProgressCallback = Callable[[HarvestProgress], None]


# --------------------------------------------------------------------------- #
# Motor de colheita
# --------------------------------------------------------------------------- #
class CropHarvester:
    """Orquestra o processamento paralelo e a exportacao dos recortes.

    Args:
        dataset: anotacoes carregadas pelos parsers.
        config: parametros geometricos.
        output_dir: diretorio de saida (criado se nao existir).
        workers: numero de threads (OpenCV libera o GIL); padrao ``min(8, cpus)``.
        image_ext: extensao dos recortes (``.jpg`` ou ``.png``).
        source_info: informacoes de entrada registradas no resumo (caminhos etc.).
    """

    def __init__(
        self,
        dataset: AnnotationDataset,
        config: ProcessingConfig,
        output_dir: str | Path,
        workers: int | None = None,
        image_ext: str = ".jpg",
        source_info: dict[str, object] | None = None,
    ) -> None:
        config.validate()
        if image_ext.lower() not in {".jpg", ".jpeg", ".png"}:
            raise ValueError("image_ext deve ser .jpg, .jpeg ou .png")
        self.dataset = dataset
        self.config = config
        self.output_dir = Path(output_dir)
        self.crops_dir = self.output_dir / CROPS_DIRNAME
        self.workers = max(1, workers or min(8, os.cpu_count() or 1))
        self.image_ext = image_ext.lower()
        self.source_info = source_info or {}
        all_ids = {b.class_id for it in dataset.items for b in it.boxes}
        self.class_dirs = build_class_dirs(all_ids, dataset)

    # ------------------------------------------------------------------ worker
    def _process_item(self, item: ImageAnnotations, cancel: threading.Event) -> _ImageOutcome:
        """Processa uma imagem: le, extrai recortes, salva e monta as linhas do manifesto."""
        outcome = _ImageOutcome(item=item)
        if cancel.is_set():
            return outcome
        image = read_image(item.image_path)
        if image is None:
            outcome.read_failed = True
            outcome.warnings.append(f"Falha ao ler imagem: {item.image_path}")
            return outcome
        outcome.processed = True

        img_h, img_w = image.shape[:2]
        if item.declared_size and item.declared_size != (img_w, img_h):
            outcome.warnings.append(
                f"{item.relative_path}: tamanho declarado {item.declared_size} difere do real {(img_w, img_h)}."
            )

        report = extract_crops(image, item.boxes, self.config)
        outcome.discarded = dict(report.discarded)
        outcome.discarded_by_class = dict(report.discarded_by_class)

        key = image_key(item.relative_path)
        rel_posix = item.relative_path.as_posix()
        cfg = self.config
        lb_color = cfg.letterbox_color if cfg.aspect_mode == "letterbox" else "none"

        for crop in report.crops:
            class_dir = self.class_dirs[crop.class_id]
            filename = f"{key}_crop_{crop.detection_index:03d}{self.image_ext}"
            dest = self.crops_dir / class_dir / filename
            if not write_image(dest, crop.image, cfg.jpeg_quality):
                outcome.write_failures += 1
                outcome.warnings.append(f"Falha ao salvar recorte: {dest}")
                continue
            ox1, oy1, ox2, oy2 = crop.orig_box
            px1, py1, px2, py2 = crop.padded_box
            outcome.rows.append(
                {
                    "crop_filename": filename,
                    "crop_relative_path": f"{CROPS_DIRNAME}/{class_dir}/{filename}",
                    "source_image_filename": item.image_path.name,
                    "source_image_width": img_w,
                    "source_image_height": img_h,
                    "class_id": crop.class_id,
                    "class_name": self.dataset.class_name(crop.class_id),
                    "confidence": "" if crop.confidence is None else f"{crop.confidence:.6f}",
                    "orig_xmin": ox1,
                    "orig_ymin": oy1,
                    "orig_xmax": ox2,
                    "orig_ymax": oy2,
                    "padded_xmin": px1,
                    "padded_ymin": py1,
                    "padded_xmax": px2,
                    "padded_ymax": py2,
                    "crop_width": crop.crop_width,
                    "crop_height": crop.crop_height,
                    "padding_pct": cfg.padding_pct,
                    "aspect_mode": cfg.aspect_mode,
                    "letterbox_color": lb_color,
                    "group_id": rel_posix,
                    "source_image_relpath": rel_posix,
                    "detection_index": crop.detection_index,
                    "output_width": crop.output_width,
                    "output_height": crop.output_height,
                    "target_size": cfg.target_size or "",
                    "annotation_format": self.dataset.format,
                }
            )
        return outcome

    # ------------------------------------------------------------------ run
    def run(
        self,
        progress_cb: ProgressCallback | None = None,
        cancel_event: threading.Event | None = None,
    ) -> HarvestResult:
        """Executa a colheita completa e grava manifesto e resumo.

        Mesmo se cancelada, a execucao grava o manifesto e o resumo parciais,
        de modo que todos os arquivos em disco permanecem rastreaveis.
        """
        cancel = cancel_event or threading.Event()
        self.crops_dir.mkdir(parents=True, exist_ok=True)
        items = self.dataset.items
        total = len(items)
        log.info(
            "Iniciando colheita: %d imagens, %d caixas, %d threads -> %s",
            total, self.dataset.total_boxes, self.workers, self.output_dir,
        )

        rows: list[dict[str, object]] = []
        discarded = {DISCARD_MIN_SIZE: 0, DISCARD_LOW_CONFIDENCE: 0, DISCARD_INVALID: 0}
        discarded_by_class: dict[int, int] = {}
        images_processed = 0
        images_failed = 0
        write_failures = 0
        warnings: list[str] = []
        start = time.perf_counter()

        with ThreadPoolExecutor(max_workers=self.workers, thread_name_prefix="harvest") as pool:
            futures: list[Future[_ImageOutcome]] = [pool.submit(self._process_item, it, cancel) for it in items]
            done = 0
            for fut in as_completed(futures):
                if fut.cancelled():
                    continue
                try:
                    out = fut.result()
                except Exception as exc:  # erro inesperado em uma imagem nao derruba o lote
                    images_failed += 1
                    warnings.append(f"Erro inesperado: {exc!r}")
                    log.exception("Erro ao processar imagem")
                    done += 1
                    continue
                done += 1
                if out.read_failed:
                    images_failed += 1
                elif out.processed:
                    images_processed += 1
                rows.extend(out.rows)
                write_failures += out.write_failures
                for k, v in out.discarded.items():
                    discarded[k] = discarded.get(k, 0) + v
                for k, v in out.discarded_by_class.items():
                    discarded_by_class[k] = discarded_by_class.get(k, 0) + v
                for w in out.warnings:
                    warnings.append(w)
                    log.warning(w)
                if progress_cb is not None:
                    progress_cb(
                        HarvestProgress(
                            images_done=done,
                            images_total=total,
                            crops_done=len(rows),
                            elapsed=time.perf_counter() - start,
                            current_image=out.item.relative_path.as_posix(),
                        )
                    )
                if cancel.is_set():
                    for f in futures:
                        f.cancel()

        elapsed = time.perf_counter() - start
        rows.sort(key=lambda r: (str(r["source_image_relpath"]), int(r["detection_index"])))  # type: ignore[arg-type]

        manifest_path = self.output_dir / MANIFEST_FILENAME
        self._write_manifest(manifest_path, rows)

        crops_per_class: dict[str, int] = {}
        for r in rows:
            name = str(r["class_name"])
            crops_per_class[name] = crops_per_class.get(name, 0) + 1

        summary: dict[str, object] = {
            "app": __app_name__,
            "version": __version__,
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "cancelled": cancel.is_set(),
            "elapsed_seconds": round(elapsed, 3),
            "crops_per_second": round(len(rows) / elapsed, 2) if elapsed > 0 else 0.0,
            "inputs": {
                "images_dir": str(self.dataset.images_dir),
                "annotation_format": self.dataset.format,
                **{k: (str(v) if v is not None else None) for k, v in self.source_info.items()},
            },
            "parameters": {**self.config.to_dict(), "output_image_ext": self.image_ext, "workers": self.workers},
            "totals": {
                "images_annotated": total,
                "images_processed": images_processed,
                "images_failed_to_read": images_failed,
                "images_without_labels": self.dataset.images_without_labels,
                "annotations_without_image": self.dataset.annotations_without_image,
                "boxes_total": self.dataset.total_boxes,
                "crops_extracted": len(rows),
                "discarded_min_size": discarded[DISCARD_MIN_SIZE],
                "discarded_low_confidence": discarded[DISCARD_LOW_CONFIDENCE],
                "discarded_invalid_box": discarded[DISCARD_INVALID],
                "write_failures": write_failures,
                "unique_source_groups": len({r["group_id"] for r in rows}),
            },
            "crops_per_class": dict(sorted(crops_per_class.items())),
            "discarded_per_class": {
                self.dataset.class_name(k): v for k, v in sorted(discarded_by_class.items())
            },
            "class_map": {str(k): self.dataset.class_name(k) for k in sorted(self.class_dirs)},
            "class_folders": {self.dataset.class_name(k): v for k, v in sorted(self.class_dirs.items())},
            "outputs": {
                "crops_dir": str(self.crops_dir),
                "manifest": str(manifest_path),
                "summary": str(self.output_dir / SUMMARY_FILENAME),
            },
            "parser_warnings": len(self.dataset.warnings),
            "runtime_warnings": warnings[:200],
        }
        summary_path = self.output_dir / SUMMARY_FILENAME
        with summary_path.open("w", encoding="utf-8") as fh:
            json.dump(summary, fh, ensure_ascii=False, indent=2)

        log.info(
            "Colheita %s: %d recortes em %.2fs (%.1f crops/s). Descartes: %d min_size, %d confianca, %d invalidos.",
            "CANCELADA" if cancel.is_set() else "concluida",
            len(rows), elapsed, summary["crops_per_second"],
            discarded[DISCARD_MIN_SIZE], discarded[DISCARD_LOW_CONFIDENCE], discarded[DISCARD_INVALID],
        )
        return HarvestResult(
            summary=summary,
            crops_dir=self.crops_dir,
            manifest_path=manifest_path,
            summary_path=summary_path,
            cancelled=cancel.is_set(),
        )

    @staticmethod
    def _write_manifest(path: Path, rows: list[dict[str, object]]) -> None:
        """Grava o manifesto CSV (UTF-8 com BOM para abrir corretamente no Excel)."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8-sig", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=MANIFEST_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
