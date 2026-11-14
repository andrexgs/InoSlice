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
    cfg = ProcessingConfig(
        padding_pct=10.0, min_width=32, min_height=32, min_confidence=0.25, aspect_mode="rectangular"
    )
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


# ------------------------------------------------------------------ letterbox
from src.core.processor import (  # noqa: E402
    LETTERBOX_COLORS_RGB,
    finalize_crop,
    letterbox,
    letterbox_geometry,
    resize_longest_side,
)


def test_letterbox_geometry_native_resolution() -> None:
    # 100x40 -> quadrado 100x100, conteudo centralizado verticalmente
    assert letterbox_geometry(100, 40) == (100, 100, 40, 0, 30)
    assert letterbox_geometry(30, 90) == (90, 30, 90, 30, 0)
    assert letterbox_geometry(50, 50) == (50, 50, 50, 0, 0)


def test_letterbox_geometry_with_target_preserves_aspect_ratio() -> None:
    side, new_w, new_h, ox, oy = letterbox_geometry(400, 200, 224)
    assert (side, new_w, new_h) == (224, 224, 112)
    assert (ox, oy) == (0, 56)
    side, new_w, new_h, ox, oy = letterbox_geometry(60, 180, 384)  # ampliacao
    assert (side, new_w, new_h, ox, oy) == (384, 128, 384, 128, 0)
    assert new_w / new_h == pytest.approx(60 / 180, rel=1e-2)


@pytest.mark.parametrize("color", ["gray", "black"])
def test_letterbox_fills_borders_and_keeps_content(color: str) -> None:
    crop = np.full((40, 100, 3), 200, dtype=np.uint8)
    out = letterbox(crop, LETTERBOX_COLORS_RGB[color])
    assert out.shape == (100, 100, 3)
    expected = LETTERBOX_COLORS_RGB[color][0]
    # faixas superior/inferior preenchidas
    assert np.all(out[:30] == expected)
    assert np.all(out[70:] == expected)
    # conteudo intacto (sem reamostragem quando nao ha target)
    assert np.all(out[30:70] == 200)


def test_letterbox_gray_is_rgb_114() -> None:
    crop = np.zeros((10, 20, 3), dtype=np.uint8)
    out = letterbox(crop, LETTERBOX_COLORS_RGB["gray"])
    assert tuple(int(v) for v in out[0, 0]) == (114, 114, 114)


def test_letterbox_with_target_size() -> None:
    crop = np.full((150, 300, 3), 50, dtype=np.uint8)
    out = letterbox(crop, (0, 0, 0), 224)
    assert out.shape == (224, 224, 3)
    assert np.all(out[:56] == 0) and np.all(out[168:] == 0)
    assert np.all(out[56:168] == 50)


def test_resize_longest_side_keeps_ratio() -> None:
    crop = np.zeros((100, 300, 3), dtype=np.uint8)
    out = resize_longest_side(crop, 150)
    assert out.shape[:2] == (50, 150)


def test_finalize_crop_modes() -> None:
    crop = np.zeros((30, 90, 3), dtype=np.uint8)
    rect = finalize_crop(crop, ProcessingConfig(aspect_mode="rectangular"))
    assert rect.shape[:2] == (30, 90)
    rect_resized = finalize_crop(crop, ProcessingConfig(aspect_mode="rectangular", target_size=180))
    assert rect_resized.shape[:2] == (60, 180)
    square = finalize_crop(crop, ProcessingConfig(aspect_mode="letterbox", letterbox_color="black"))
    assert square.shape[:2] == (90, 90)
    square_t = finalize_crop(crop, ProcessingConfig(aspect_mode="letterbox", target_size=384))
    assert square_t.shape[:2] == (384, 384)


def test_end_to_end_letterbox_with_padding_near_border() -> None:
    image = np.full((100, 100, 3), 255, dtype=np.uint8)
    box = _abs_box(0, 40, 80, 60)  # 80x20 colada na borda esquerda
    cfg = ProcessingConfig(padding_pct=25, min_width=1, min_height=1, aspect_mode="letterbox", target_size=64)
    crop = extract_crops(image, [box], cfg).crops[0]
    assert crop.padded_box == (0, 35, 100, 65)  # 20 px em x (clamp 0/100), 5 px em y
    assert crop.image.shape == (64, 64, 3)
    assert (crop.output_width, crop.output_height) == (64, 64)


def test_render_preview_accepted_and_rejected() -> None:
    from src.core.processor import render_preview

    image = np.zeros((100, 100, 3), dtype=np.uint8)
    cfg = ProcessingConfig(padding_pct=10, min_width=20, min_height=20, aspect_mode="letterbox", target_size=32)
    ok = render_preview(image, _abs_box(20, 20, 60, 40), cfg)
    assert ok.accepted and ok.crop is not None and ok.crop.shape == (32, 32, 3)
    assert ok.padded_box == (16, 18, 64, 42)
    assert ok.overlay.shape == image.shape and ok.overlay.any()
    bad = render_preview(image, _abs_box(20, 20, 30, 30), cfg)
    assert not bad.accepted and bad.crop is None and "menor" in bad.reason
