"""Motor geometrico de recorte do CropHarvester Studio.

Responsabilidades
-----------------
1. **Filtro de resolucao minima** - descarta caixas originais menores que
   ``min_width x min_height`` pixels (recortes borrados, sem informacao
   discriminativa).
2. **Margem de contexto (context padding)** - expande a caixa em ``p%`` da sua
   *propria* largura/altura **em cada lado**, com *clamping* seguro nas bordas
   da imagem (``0 <= x <= W_img``, ``0 <= y <= H_img``).
3. **Leitura/escrita robusta** - usa ``np.fromfile``/``cv2.imdecode`` para
   suportar caminhos com acentos no Windows.

Convencao geometrica: caixas sao ``(x_min, y_min, x_max, y_max)`` com
``x_max``/``y_max`` *exclusivos*, de modo que o recorte NumPy e
``img[y_min:y_max, x_min:x_max]`` e a largura e ``x_max - x_min``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from src.core.parsers import AnnotationBox

IntBox = tuple[int, int, int, int]
FloatBox = tuple[float, float, float, float]


# --------------------------------------------------------------------------- #
# Configuracao
# --------------------------------------------------------------------------- #
@dataclass
class ProcessingConfig:
    """Parametros geometricos aplicados a cada deteccao.

    Attributes:
        padding_pct: margem de contexto em % da largura/altura da caixa, aplicada
            em cada lado (0 a 100).
        min_width: largura minima (px) da caixa original para ser aceita.
        min_height: altura minima (px) da caixa original para ser aceita.
        min_confidence: confianca minima (aplicada apenas quando a caixa a possui).
        jpeg_quality: qualidade JPEG dos recortes salvos (1-100).
    """

    padding_pct: float = 10.0
    min_width: int = 32
    min_height: int = 32
    min_confidence: float = 0.0
    jpeg_quality: int = 95

    def validate(self) -> None:
        """Valida os limites dos parametros, levantando ``ValueError`` se invalidos."""
        if not 0.0 <= self.padding_pct <= 100.0:
            raise ValueError("padding_pct deve estar entre 0 e 100.")
        if self.min_width < 0 or self.min_height < 0:
            raise ValueError("Tamanho minimo nao pode ser negativo.")
        if not 0.0 <= self.min_confidence <= 1.0:
            raise ValueError("min_confidence deve estar entre 0 e 1.")
        if not 1 <= self.jpeg_quality <= 100:
            raise ValueError("jpeg_quality deve estar entre 1 e 100.")

    def to_dict(self) -> dict[str, object]:
        """Representacao serializavel (para o resumo JSON)."""
        return {
            "padding_pct": self.padding_pct,
            "min_width": self.min_width,
            "min_height": self.min_height,
            "min_confidence": self.min_confidence,
            "jpeg_quality": self.jpeg_quality,
        }


# --------------------------------------------------------------------------- #
# Resultados
# --------------------------------------------------------------------------- #
DISCARD_MIN_SIZE = "min_size"
DISCARD_LOW_CONFIDENCE = "low_confidence"
DISCARD_INVALID = "invalid_box"


@dataclass
class CropResult:
    """Um recorte pronto para ser salvo, com toda a sua geometria rastreavel."""

    detection_index: int
    class_id: int
    confidence: float | None
    orig_box: IntBox
    padded_box: IntBox
    crop_width: int
    crop_height: int
    image: np.ndarray

    @property
    def output_width(self) -> int:
        """Largura final da imagem salva."""
        return int(self.image.shape[1])

    @property
    def output_height(self) -> int:
        """Altura final da imagem salva."""
        return int(self.image.shape[0])


@dataclass
class ExtractionReport:
    """Recortes aceitos e contagem de descartes de uma imagem."""

    crops: list[CropResult] = field(default_factory=list)
    discarded: dict[str, int] = field(
        default_factory=lambda: {DISCARD_MIN_SIZE: 0, DISCARD_LOW_CONFIDENCE: 0, DISCARD_INVALID: 0}
    )
    discarded_by_class: dict[int, int] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Geometria
# --------------------------------------------------------------------------- #
def clamp_box(box: FloatBox, img_width: int, img_height: int) -> IntBox:
    """Arredonda a caixa para pixels inteiros (piso/teto) e limita aos limites da imagem.

    ``x_min``/``y_min`` usam ``floor`` e ``x_max``/``y_max`` usam ``ceil`` para
    nunca cortar parte do objeto; em seguida os valores sao restritos a
    ``[0, W]`` e ``[0, H]``.
    """
    x1, y1, x2, y2 = box
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    xi1 = min(max(int(math.floor(x1)), 0), img_width)
    yi1 = min(max(int(math.floor(y1)), 0), img_height)
    xi2 = min(max(int(math.ceil(x2)), 0), img_width)
    yi2 = min(max(int(math.ceil(y2)), 0), img_height)
    return (xi1, yi1, xi2, yi2)


def expand_box(box: FloatBox, padding_pct: float, img_width: int, img_height: int) -> IntBox:
    """Expande ``box`` em ``padding_pct``% da sua largura/altura em cada lado, com clamping.

    Exemplo: caixa 100x50 com ``padding_pct=10`` cresce 10 px a esquerda e a
    direita e 5 px acima e abaixo (120x60), desde que caiba na imagem.
    """
    x1, y1, x2, y2 = box
    w = abs(x2 - x1)
    h = abs(y2 - y1)
    pad_x = w * padding_pct / 100.0
    pad_y = h * padding_pct / 100.0
    return clamp_box(
        (min(x1, x2) - pad_x, min(y1, y2) - pad_y, max(x1, x2) + pad_x, max(y1, y2) + pad_y),
        img_width,
        img_height,
    )


def passes_min_size(box: IntBox, min_width: int, min_height: int) -> bool:
    """Verdadeiro se a caixa (ja em pixels) atende a resolucao minima."""
    x1, y1, x2, y2 = box
    return (x2 - x1) >= min_width and (y2 - y1) >= min_height


def crop_image(image: np.ndarray, box: IntBox) -> np.ndarray:
    """Recorta ``image`` na caixa inteira (copia contigua, independente do original)."""
    x1, y1, x2, y2 = box
    return np.ascontiguousarray(image[y1:y2, x1:x2]).copy()


# --------------------------------------------------------------------------- #
# Pipeline por imagem
# --------------------------------------------------------------------------- #
def finalize_crop(crop: np.ndarray, config: ProcessingConfig) -> np.ndarray:
    """Pos-processamento do recorte (identidade nesta versao)."""
    return crop


def extract_crops(image: np.ndarray, boxes: list[AnnotationBox], config: ProcessingConfig) -> ExtractionReport:
    """Aplica filtros, padding e transformacao final a todas as caixas de uma imagem.

    Args:
        image: imagem BGR ``(H, W, 3)``.
        boxes: caixas da imagem (normalizadas ou absolutas).
        config: parametros de processamento.
    """
    img_h, img_w = image.shape[:2]
    report = ExtractionReport()

    def discard(reason: str, class_id: int) -> None:
        report.discarded[reason] += 1
        report.discarded_by_class[class_id] = report.discarded_by_class.get(class_id, 0) + 1

    for idx, box in enumerate(boxes):
        if box.confidence is not None and box.confidence < config.min_confidence:
            discard(DISCARD_LOW_CONFIDENCE, box.class_id)
            continue

        abs_box = box.to_absolute(img_w, img_h)
        orig = clamp_box(abs_box, img_w, img_h)
        if orig[2] - orig[0] <= 0 or orig[3] - orig[1] <= 0:
            discard(DISCARD_INVALID, box.class_id)
            continue
        if not passes_min_size(orig, config.min_width, config.min_height):
            discard(DISCARD_MIN_SIZE, box.class_id)
            continue

        padded = expand_box(abs_box, config.padding_pct, img_w, img_h)
        raw = crop_image(image, padded)
        report.crops.append(
            CropResult(
                detection_index=idx,
                class_id=box.class_id,
                confidence=box.confidence,
                orig_box=orig,
                padded_box=padded,
                crop_width=padded[2] - padded[0],
                crop_height=padded[3] - padded[1],
                image=finalize_crop(raw, config),
            )
        )
    return report


# --------------------------------------------------------------------------- #
# E/S de imagens (compativel com caminhos Unicode no Windows)
# --------------------------------------------------------------------------- #
def read_image(path: str | Path) -> np.ndarray | None:
    """Le uma imagem como BGR de 8 bits; retorna ``None`` se nao puder ser decodificada."""
    try:
        buffer = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if buffer.size == 0:
        return None
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    return image


def write_image(path: str | Path, image: np.ndarray, jpeg_quality: int = 95) -> bool:
    """Salva ``image`` em ``path`` (formato pela extensao). Retorna ``True`` em caso de sucesso."""
    p = Path(path)
    ext = p.suffix.lower() or ".jpg"
    params: list[int] = []
    if ext in {".jpg", ".jpeg"}:
        params = [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)]
    ok, encoded = cv2.imencode(ext, image, params)
    if not ok:
        return False
    p.parent.mkdir(parents=True, exist_ok=True)
    encoded.tofile(str(p))
    return True
