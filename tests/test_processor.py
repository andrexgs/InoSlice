"""Testes unitarios da geometria de recorte (padding, clamping e filtro de tamanho)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.core.parsers import AnnotationBox
from src.core.processor import (
    DISCARD_INVALID,
    DISCARD_LOW_CONFIDENCE,
    DISCARD_MIN_SIZE,
    ProcessingConfig,
    clamp_box,
    expand_box,
    extract_crops,
    passes_min_size,
    read_image,
    write_image,
)


def _abs_box(x1: float, y1: float, x2: float, y2: float, cls: int = 0, conf: float | None = None) -> AnnotationBox:
    return AnnotationBox(class_id=cls, coords=(x1, y1, x2, y2), normalized=False, confidence=conf)


# ------------------------------------------------------------------ padding
def test_expand_box_zero_padding_is_identity() -> None:
    assert expand_box((10, 20, 110, 70), 0.0, 500, 500) == (10, 20, 110, 70)


def test_expand_box_padding_proportional_per_side() -> None:
    # caixa 100x50, 10% => +10 px em x por lado, +5 px em y por lado
    assert expand_box((100, 100, 200, 150), 10.0, 1000, 1000) == (90, 95, 210, 155)


def test_expand_box_padding_30_percent() -> None:
    # caixa 40x80, 30% => +12 px em x, +24 px em y
    assert expand_box((100, 100, 140, 180), 30.0, 1000, 1000) == (88, 76, 152, 204)


def test_expand_box_clamps_to_image_borders() -> None:
    # caixa colada no canto superior esquerdo e inferior direito
    assert expand_box((0, 0, 50, 50), 20.0, 100, 80) == (0, 0, 60, 60)
    assert expand_box((60, 40, 100, 80), 25.0, 100, 80) == (50, 30, 100, 80)
    x1, y1, x2, y2 = expand_box((-30, -30, 500, 500), 50.0, 320, 240)
    assert (x1, y1, x2, y2) == (0, 0, 320, 240)


def test_clamp_box_rounds_outward_and_orders() -> None:
    assert clamp_box((10.4, 20.6, 30.2, 40.9), 100, 100) == (10, 20, 31, 41)
    assert clamp_box((30, 40, 10, 20), 100, 100) == (10, 20, 30, 40)


def test_passes_min_size() -> None:
    assert passes_min_size((0, 0, 32, 32), 32, 32)
    assert not passes_min_size((0, 0, 31, 64), 32, 32)
    assert not passes_min_size((0, 0, 64, 31), 32, 32)


# ------------------------------------------------------------------ pipeline
def test_extract_crops_filters_and_geometry() -> None:
    image = np.zeros((200, 300, 3), dtype=np.uint8)
    image[50:100, 100:200] = 255
    boxes = [
        _abs_box(100, 50, 200, 100, cls=1, conf=0.9),  # aceito
        _abs_box(10, 10, 20, 20, cls=1),  # muito pequeno
        _abs_box(100, 50, 200, 100, cls=2, conf=0.1),  # baixa confianca
        _abs_box(400, 400, 500, 500, cls=3),  # fora da imagem
    ]
    cfg = ProcessingConfig(padding_pct=10.0, min_width=32, min_height=32, min_confidence=0.25)
    report = extract_crops(image, boxes, cfg)

    assert len(report.crops) == 1
    crop = report.crops[0]
    assert crop.orig_box == (100, 50, 200, 100)
    assert crop.padded_box == (90, 45, 210, 105)
    assert (crop.crop_width, crop.crop_height) == (120, 60)
    assert crop.image.shape[:2] == (60, 120)
    assert crop.detection_index == 0
    assert report.discarded[DISCARD_MIN_SIZE] == 1
    assert report.discarded[DISCARD_LOW_CONFIDENCE] == 1
    assert report.discarded[DISCARD_INVALID] == 1
    assert report.discarded_by_class == {1: 1, 2: 1, 3: 1}


def test_extract_crops_normalized_yolo_box() -> None:
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    box = AnnotationBox(class_id=0, coords=(0.25, 0.25, 0.75, 0.75), normalized=True)
    report = extract_crops(image, [box], ProcessingConfig(padding_pct=0, min_width=1, min_height=1))
    assert report.crops[0].orig_box == (50, 25, 150, 75)


def test_config_validation() -> None:
    with pytest.raises(ValueError):
        ProcessingConfig(padding_pct=-1).validate()
    with pytest.raises(ValueError):
        ProcessingConfig(min_confidence=1.5).validate()
    ProcessingConfig().validate()


def test_read_write_roundtrip_unicode_path(tmp_path: Path) -> None:
    image = np.full((10, 12, 3), 127, dtype=np.uint8)
    target = tmp_path / "pasta_acentuação" / "abelha_ç.png"
    assert write_image(target, image)
    loaded = read_image(target)
    assert loaded is not None and loaded.shape == (10, 12, 3)
    assert read_image(tmp_path / "nao_existe.jpg") is None
