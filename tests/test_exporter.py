"""Testes de integracao do exportador: arvore por classe, manifesto CSV e resumo JSON."""

from __future__ import annotations

import csv
import json
import threading
from pathlib import Path

import numpy as np

from src.core.exporter import (
    MANIFEST_COLUMNS,
    CropHarvester,
    image_key,
    sanitize_name,
)
from src.core.parsers import load_annotations
from src.core.processor import ProcessingConfig, write_image

REQUIRED = [
    "crop_filename", "crop_relative_path", "source_image_filename", "source_image_width",
    "source_image_height", "class_id", "class_name", "confidence", "orig_xmin", "orig_ymin",
    "orig_xmax", "orig_ymax", "padded_xmin", "padded_ymin", "padded_xmax", "padded_ymax",
    "crop_width", "crop_height", "padding_pct", "aspect_mode", "letterbox_color",
]


def _make_yolo_dataset(root: Path) -> tuple[Path, Path]:
    images, labels = root / "images", root / "labels"
    labels.mkdir(parents=True)
    rng = np.random.default_rng(0)
    for name in ("img_a", "img_b"):
        write_image(images / f"{name}.jpg", rng.integers(0, 255, (240, 320, 3), dtype=np.uint8))
    write_image(images / "sub" / "img_a.png", np.zeros((100, 100, 3), dtype=np.uint8))
    (labels / "sub").mkdir()
    (labels / "img_a.txt").write_text(
        "0 0.5 0.5 0.25 0.25 0.95\n1 0.2 0.2 0.3 0.3 0.80\n0 0.9 0.9 0.05 0.05\n", encoding="utf-8"
    )
    (labels / "img_b.txt").write_text("1 0.5 0.5 0.5 0.5\n", encoding="utf-8")
    (labels / "sub" / "img_a.txt").write_text("2 0.5 0.5 0.5 0.5\n", encoding="utf-8")
    (labels / "classes.txt").write_text("formiga\nabelha\nvespa/besouro\n", encoding="utf-8")
    return images, labels


def test_sanitize_and_keys() -> None:
    assert sanitize_name("vespa/besouro") == "vespa_besouro"
    assert sanitize_name("CON") == "CON_"
    assert sanitize_name("  ") == "unnamed"
    assert image_key(Path("sub/img a.png")) == "sub__img_a"


def test_full_harvest(tmp_path: Path) -> None:
    images, labels = _make_yolo_dataset(tmp_path)
    ds = load_annotations(images, labels)
    out = tmp_path / "out"
    cfg = ProcessingConfig(padding_pct=10, min_width=32, min_height=32, aspect_mode="letterbox", target_size=64)
    progress: list[float] = []
    result = CropHarvester(ds, cfg, out, workers=2).run(progress_cb=lambda p: progress.append(p.fraction))

    assert not result.cancelled
    assert progress and progress[-1] == 1.0
    # arvore ImageFolder
    assert sorted(p.name for p in result.crops_dir.iterdir()) == ["abelha", "formiga", "vespa_besouro"]
    assert (result.crops_dir / "formiga" / "img_a_crop_000.jpg").is_file()
    assert (result.crops_dir / "vespa_besouro" / "sub__img_a_crop_000.jpg").is_file()

    with result.manifest_path.open(encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        assert reader.fieldnames == MANIFEST_COLUMNS
        rows = list(reader)
    assert set(REQUIRED).issubset(MANIFEST_COLUMNS)
    assert len(rows) == 4  # caixa 16x12 descartada pelo filtro minimo
    for r in rows:
        assert (out / r["crop_relative_path"]).is_file()
        assert r["output_width"] == r["output_height"] == "64"
        assert r["letterbox_color"] == "gray"
    groups = {r["group_id"] for r in rows}
    assert groups == {"img_a.jpg", "img_b.jpg", "sub/img_a.png"}

    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    t = summary["totals"]
    assert t["images_processed"] == 3
    assert t["crops_extracted"] == 4
    assert t["discarded_min_size"] == 1
    assert summary["crops_per_class"] == {"abelha": 2, "formiga": 1, "vespa/besouro": 1}
    assert summary["parameters"]["padding_pct"] == 10


def test_rectangular_mode_and_cancel(tmp_path: Path) -> None:
    images, labels = _make_yolo_dataset(tmp_path)
    ds = load_annotations(images, labels)
    cfg = ProcessingConfig(padding_pct=0, min_width=1, min_height=1, aspect_mode="rectangular")
    result = CropHarvester(ds, cfg, tmp_path / "rect", workers=1).run()
    with result.manifest_path.open(encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    assert all(r["letterbox_color"] == "none" for r in rows)
    assert all(r["crop_width"] == r["output_width"] for r in rows)

    ev = threading.Event()
    ev.set()
    cancelled = CropHarvester(ds, cfg, tmp_path / "cancel", workers=1).run(cancel_event=ev)
    assert cancelled.cancelled
    assert cancelled.manifest_path.is_file()
