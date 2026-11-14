"""Testes dos parsers YOLO/COCO e do mapeamento de classes."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.core.parsers import (
    detect_format,
    load_annotations,
    load_class_names,
    parse_yolo_line,
)


def _touch_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\x00")  # o parser nao decodifica pixels


def test_parse_yolo_line_bbox_and_confidence() -> None:
    box = parse_yolo_line("2 0.5 0.5 0.2 0.4 0.87")
    assert box is not None
    assert box.class_id == 2
    assert box.confidence == pytest.approx(0.87)
    assert box.coords == pytest.approx((0.4, 0.3, 0.6, 0.7))
    assert box.to_absolute(100, 200) == pytest.approx((40.0, 60.0, 60.0, 140.0))


def test_parse_yolo_line_polygon_and_invalid() -> None:
    poly = parse_yolo_line("1 0.1 0.2 0.5 0.2 0.3 0.9")
    assert poly is not None and poly.coords == pytest.approx((0.1, 0.2, 0.5, 0.9))
    assert parse_yolo_line("   ") is None
    with pytest.raises(ValueError):
        parse_yolo_line("1 0.5 0.5")


def test_load_class_names_txt_and_yaml(tmp_path: Path) -> None:
    txt = tmp_path / "classes.txt"
    txt.write_text("gato\ncachorro\n\n", encoding="utf-8")
    assert load_class_names(txt) == {0: "gato", 1: "cachorro"}

    yml = tmp_path / "data.yaml"
    yml.write_text("names:\n  0: abelha\n  3: vespa\n", encoding="utf-8")
    assert load_class_names(yml) == {0: "abelha", 3: "vespa"}

    yml2 = tmp_path / "data2.yaml"
    yml2.write_text("names: [a, b, c]\n", encoding="utf-8")
    assert load_class_names(yml2) == {0: "a", 1: "b", 2: "c"}


def test_yolo_dataset_with_subfolders(tmp_path: Path) -> None:
    images = tmp_path / "images"
    labels = tmp_path / "labels"
    _touch_image(images / "a.jpg")
    _touch_image(images / "sub" / "b.png")
    _touch_image(images / "sem_label.jpg")
    (labels / "sub").mkdir(parents=True)
    (labels / "a.txt").write_text("0 0.5 0.5 0.1 0.1\nlinha ruim\n", encoding="utf-8")
    (labels / "sub" / "b.txt").write_text("1 0.5 0.5 0.2 0.2 0.9\n", encoding="utf-8")
    (labels / "orfao.txt").write_text("0 0.5 0.5 0.1 0.1\n", encoding="utf-8")
    (labels / "classes.txt").write_text("gato\ncachorro\n", encoding="utf-8")

    assert detect_format(labels) == "yolo"
    ds = load_annotations(images, labels)
    assert ds.format == "yolo"
    assert len(ds.items) == 2
    assert ds.total_boxes == 2
    assert ds.images_without_labels == 1
    assert ds.annotations_without_image == 1
    assert ds.class_name(1) == "cachorro"
    assert ds.class_name(9) == "class_9"
    assert any("a.txt:2" in w for w in ds.warnings)


def test_coco_dataset(tmp_path: Path) -> None:
    images = tmp_path / "imgs"
    _touch_image(images / "x.jpg")
    _touch_image(images / "y.jpg")
    coco = {
        "images": [
            {"id": 1, "file_name": "x.jpg", "width": 640, "height": 480},
            {"id": 2, "file_name": "nested/y.jpg", "width": 100, "height": 100},
            {"id": 3, "file_name": "ausente.jpg"},
        ],
        "annotations": [
            {"id": 10, "image_id": 1, "category_id": 7, "bbox": [10, 20, 30, 40], "score": 0.5},
            {"id": 11, "image_id": 2, "category_id": 7, "bbox": [0, 0, 50, 50], "iscrowd": 1},
            {"id": 12, "image_id": 2, "category_id": 8, "bbox": [5, 5, 10, 10]},
            {"id": 13, "image_id": 99, "category_id": 8, "bbox": [5, 5, 10, 10]},
        ],
        "categories": [{"id": 7, "name": "formiga"}, {"id": 8, "name": "besouro"}],
    }
    jp = tmp_path / "ann.json"
    jp.write_text(json.dumps(coco), encoding="utf-8")

    assert detect_format(jp) == "coco"
    ds = load_annotations(images, jp)
    assert ds.format == "coco"
    assert len(ds.items) == 2
    assert ds.total_boxes == 2  # crowd ignorado
    assert ds.annotations_without_image == 1
    first = next(it for it in ds.items if it.image_path.name == "x.jpg")
    assert first.boxes[0].to_absolute(640, 480) == (10.0, 20.0, 40.0, 60.0)
    assert first.boxes[0].confidence == 0.5
    assert ds.class_name(8) == "besouro"
