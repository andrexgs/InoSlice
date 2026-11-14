"""Motor geometrico de recorte do CropHarvester Studio.

Responsabilidades
-----------------
1. **Filtro de resolucao minima** - descarta caixas originais menores que
   ``min_width x min_height`` pixels (recortes borrados, sem informacao
   discriminativa).
2. **Margem de contexto (context padding)** - expande a caixa em ``p%`` da sua
   *propria* largura/altura **em cada lado**, com *clamping* seguro nas bordas
   da imagem (``0 <= x <= W_img``, ``0 <= y <= H_img``).
3. **Preservacao morfologica** - modo ``rectangular`` (mantem a proporcao exata
   do recorte) ou ``letterbox`` (quadrado ``L x L`` com
   ``L = max(W_crop, H_crop)``, ou redimensionado a uma resolucao alvo como
   224/384, preenchendo as bordas com cinza neutro ``RGB(114,114,114)`` ou
   preto). Nenhum dos modos distorce o objeto.
4. **Leitura/escrita robusta** - usa ``np.fromfile``/``cv2.imdecode`` para
   suportar caminhos com acentos no Windows.

Convencao geometrica: caixas sao ``(x_min, y_min, x_max, y_max)`` com
``x_max``/``y_max`` *exclusivos*, de modo que o recorte NumPy e
``img[y_min:y_max, x_min:x_max]`` e a largura e ``x_max - x_min``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import cv2
import numpy as np

from src.core.parsers import AnnotationBox

IntBox = tuple[int, int, int, int]
FloatBox = tuple[float, float, float, float]

AspectMode = Literal["rectangular", "letterbox"]
LetterboxColor = Literal["gray", "black"]

#: Cores de preenchimento em RGB (convertidas para BGR ao desenhar com OpenCV).
LETTERBOX_COLORS_RGB: dict[str, tuple[int, int, int]] = {
    "gray": (114, 114, 114),
    "black": (0, 0, 0),
}
ASPECT_MODES: tuple[str, ...] = ("rectangular", "letterbox")


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
        aspect_mode: ``"rectangular"`` (proporcao original) ou ``"letterbox"``
            (quadrado com preenchimento).
        letterbox_color: ``"gray"`` (RGB 114,114,114) ou ``"black"`` (RGB 0,0,0).
        target_size: lado alvo opcional (ex.: 224, 384). No modo letterbox gera
            ``target x target``; no modo retangular redimensiona o maior lado
            para ``target`` mantendo a proporcao. ``None`` mantem a resolucao nativa.
        jpeg_quality: qualidade JPEG dos recortes salvos (1-100).
    """

    padding_pct: float = 10.0
    min_width: int = 32
    min_height: int = 32
    min_confidence: float = 0.0
    aspect_mode: AspectMode = "letterbox"
    letterbox_color: LetterboxColor = "gray"
    target_size: int | None = None
    jpeg_quality: int = 95

    @property
    def fill_rgb(self) -> tuple[int, int, int]:
        """Cor de preenchimento em RGB."""
        return LETTERBOX_COLORS_RGB[self.letterbox_color]

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
        if self.aspect_mode not in ASPECT_MODES:
            raise ValueError(f"aspect_mode invalido: {self.aspect_mode}")
        if self.letterbox_color not in LETTERBOX_COLORS_RGB:
            raise ValueError(f"letterbox_color invalido: {self.letterbox_color}")
        if self.target_size is not None and self.target_size <= 0:
            raise ValueError("target_size deve ser positivo.")

    def to_dict(self) -> dict[str, object]:
        """Representacao serializavel (para o resumo JSON)."""
        return {
            "padding_pct": self.padding_pct,
            "min_width": self.min_width,
            "min_height": self.min_height,
            "min_confidence": self.min_confidence,
            "aspect_mode": self.aspect_mode,
            "letterbox_color": self.letterbox_color,
            "letterbox_rgb": list(self.fill_rgb),
            "target_size": self.target_size,
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
# Preservacao morfologica (resize sem distorcao e letterbox)
# --------------------------------------------------------------------------- #
def _interpolation(src_size: int, dst_size: int) -> int:
    """``INTER_AREA`` para reducao (anti-aliasing) e ``INTER_CUBIC`` para ampliacao."""
    return cv2.INTER_AREA if dst_size < src_size else cv2.INTER_CUBIC


def resize_longest_side(image: np.ndarray, target: int) -> np.ndarray:
    """Redimensiona mantendo a proporcao, de modo que o maior lado seja ``target``."""
    h, w = image.shape[:2]
    longest = max(h, w)
    if longest == target:
        return image
    scale = target / float(longest)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(image, (new_w, new_h), interpolation=_interpolation(longest, target))


def letterbox_geometry(width: int, height: int, target_size: int | None = None) -> tuple[int, int, int, int, int]:
    """Calcula a geometria do letterbox sem tocar em pixels.

    Returns:
        ``(side, new_w, new_h, offset_x, offset_y)``: lado do quadrado final,
        dimensoes do conteudo apos o redimensionamento e deslocamento
        (centralizado) do conteudo dentro do quadrado.
    """
    if width <= 0 or height <= 0:
        raise ValueError("Dimensoes do recorte devem ser positivas.")
    longest = max(width, height)
    side = int(target_size) if target_size else longest
    if side == longest:
        new_w, new_h = width, height
    else:
        scale = side / float(longest)
        new_w = min(side, max(1, int(round(width * scale))))
        new_h = min(side, max(1, int(round(height * scale))))
    offset_x = (side - new_w) // 2
    offset_y = (side - new_h) // 2
    return side, new_w, new_h, offset_x, offset_y


def letterbox(
    image: np.ndarray,
    fill_rgb: tuple[int, int, int] = LETTERBOX_COLORS_RGB["gray"],
    target_size: int | None = None,
) -> np.ndarray:
    """Converte ``image`` (BGR) em um quadrado sem distorcao, centralizando o conteudo.

    O conteudo e redimensionado uniformemente (mesmo fator em x e y) e as faixas
    restantes sao preenchidas com ``fill_rgb``.
    """
    h, w = image.shape[:2]
    side, new_w, new_h, off_x, off_y = letterbox_geometry(w, h, target_size)
    if (new_w, new_h) != (w, h):
        content = cv2.resize(image, (new_w, new_h), interpolation=_interpolation(max(w, h), side))
    else:
        content = image
    channels = image.shape[2] if image.ndim == 3 else 1
    fill_bgr = (fill_rgb[2], fill_rgb[1], fill_rgb[0])
    if channels == 1:
        canvas = np.full((side, side), fill_rgb[0], dtype=image.dtype)
    else:
        canvas = np.empty((side, side, channels), dtype=image.dtype)
        canvas[:] = np.array(fill_bgr[:channels] if channels <= 3 else (*fill_bgr, 255), dtype=image.dtype)
    canvas[off_y : off_y + new_h, off_x : off_x + new_w] = content
    return canvas


# --------------------------------------------------------------------------- #
# Pipeline por imagem
# --------------------------------------------------------------------------- #
def finalize_crop(crop: np.ndarray, config: ProcessingConfig) -> np.ndarray:
    """Aplica o modo de proporcao (retangular/letterbox) e o redimensionamento opcional."""
    if config.aspect_mode == "letterbox":
        return letterbox(crop, config.fill_rgb, config.target_size)
    if config.target_size:
        return resize_longest_side(crop, config.target_size)
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


# --------------------------------------------------------------------------- #
# Pre-visualizacao
# --------------------------------------------------------------------------- #
@dataclass
class PreviewResult:
    """Dados para o painel de pre-visualizacao de uma unica deteccao."""

    overlay: np.ndarray
    crop: np.ndarray | None
    orig_box: IntBox
    padded_box: IntBox
    accepted: bool
    reason: str


def render_preview(image: np.ndarray, box: AnnotationBox, config: ProcessingConfig) -> PreviewResult:
    """Desenha caixa original (verde) e caixa com padding (laranja) e gera o recorte final.

    Caixas rejeitadas pelos filtros sao desenhadas em vermelho e ``crop`` fica ``None``.
    """
    img_h, img_w = image.shape[:2]
    abs_box = box.to_absolute(img_w, img_h)
    orig = clamp_box(abs_box, img_w, img_h)
    padded = expand_box(abs_box, config.padding_pct, img_w, img_h)

    reason = ""
    if box.confidence is not None and box.confidence < config.min_confidence:
        reason = f"confianca {box.confidence:.2f} < {config.min_confidence:.2f}"
    elif orig[2] - orig[0] <= 0 or orig[3] - orig[1] <= 0:
        reason = "caixa invalida/fora da imagem"
    elif not passes_min_size(orig, config.min_width, config.min_height):
        reason = (
            f"{orig[2] - orig[0]}x{orig[3] - orig[1]} px menor que "
            f"{config.min_width}x{config.min_height} px"
        )
    accepted = reason == ""

    overlay = image.copy()
    thickness = max(2, int(round(max(img_w, img_h) / 400)))
    if accepted:
        cv2.rectangle(overlay, padded[:2], (padded[2] - 1, padded[3] - 1), (0, 140, 255), thickness)
        cv2.rectangle(overlay, orig[:2], (orig[2] - 1, orig[3] - 1), (60, 220, 60), thickness)
        crop = finalize_crop(crop_image(image, padded), config)
    else:
        cv2.rectangle(overlay, orig[:2], (max(orig[2] - 1, orig[0]), max(orig[3] - 1, orig[1])), (40, 40, 230), thickness)
        crop = None
    return PreviewResult(overlay=overlay, crop=crop, orig_box=orig, padded_box=padded, accepted=accepted, reason=reason)
