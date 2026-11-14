"""Painel de pre-visualizacao interativa dos recortes.

Mostra, lado a lado, a imagem de origem com a caixa original (verde) e a
caixa expandida pelo padding (laranja), e o recorte final exatamente como
sera salvo (retangular ou quadrado com letterbox), permitindo navegar entre
deteccoes antes da extracao em lote.
"""

from __future__ import annotations

import random

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap, QResizeEvent
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from src.core.parsers import AnnotationDataset
from src.core.processor import ProcessingConfig, read_image, render_preview


def ndarray_to_pixmap(image: np.ndarray) -> QPixmap:
    """Converte uma imagem BGR (ou cinza) ``np.ndarray`` em ``QPixmap``."""
    img = np.ascontiguousarray(image)
    h, w = img.shape[:2]
    if img.ndim == 2:
        qimg = QImage(img.data, w, h, img.strides[0], QImage.Format.Format_Grayscale8)
    else:
        qimg = QImage(img.data, w, h, img.strides[0], QImage.Format.Format_BGR888)
    return QPixmap.fromImage(qimg.copy())


class ScaledImageLabel(QLabel):
    """``QLabel`` que redimensiona o pixmap mantendo a proporcao ao mudar de tamanho."""

    def __init__(self, placeholder: str, parent: QWidget | None = None) -> None:
        super().__init__(placeholder, parent)
        self._pixmap: QPixmap | None = None
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(200, 180)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setObjectName("previewCanvas")

    def set_image(self, pixmap: QPixmap | None, placeholder: str = "") -> None:
        """Define a imagem exibida (ou um texto quando ``pixmap`` e ``None``)."""
        self._pixmap = pixmap
        if pixmap is None:
            self.clear()
            self.setText(placeholder)
        else:
            self._rescale()

    def _rescale(self) -> None:
        if self._pixmap is None:
            return
        target = self.size()
        scaled = self._pixmap.scaled(
            target,
            Qt.AspectRatioMode.KeepAspectRatio,
            # amplia sem suavizar para que pixels pequenos permanecam nitidos
            Qt.TransformationMode.SmoothTransformation
            if self._pixmap.width() > target.width()
            else Qt.TransformationMode.FastTransformation,
        )
        self.setPixmap(scaled)

    def resizeEvent(self, event: QResizeEvent) -> None:  # noqa: N802 - API Qt
        super().resizeEvent(event)
        self._rescale()


class PreviewWidget(QWidget):
    """Visualizador de amostras com navegacao entre deteccoes."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._dataset: AnnotationDataset | None = None
        self._config = ProcessingConfig()
        self._index: list[tuple[int, int]] = []  # (item, caixa)
        self._pos = 0
        self._cache_path: str | None = None
        self._cache_image: np.ndarray | None = None

        self.source_view = ScaledImageLabel("Carregue as anotacoes para visualizar")
        self.crop_view = ScaledImageLabel("Recorte final")
        self.source_caption = QLabel("Origem: caixa original (verde) | com padding (laranja)")
        self.crop_caption = QLabel("Recorte final")
        for cap in (self.source_caption, self.crop_caption):
            cap.setObjectName("caption")
            cap.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.info_label = QLabel("-")
        self.info_label.setObjectName("previewInfo")
        self.info_label.setWordWrap(True)

        self.prev_btn = QPushButton("◀ Anterior")
        self.next_btn = QPushButton("Proxima ▶")
        self.rand_btn = QPushButton("🎲 Aleatoria")
        self.prev_btn.clicked.connect(lambda: self._step(-1))
        self.next_btn.clicked.connect(lambda: self._step(+1))
        self.rand_btn.clicked.connect(self._random)

        src_col = QVBoxLayout()
        src_col.addWidget(self.source_view, 3)
        src_col.addWidget(self.source_caption)
        crop_col = QVBoxLayout()
        crop_col.addWidget(self.crop_view, 3)
        crop_col.addWidget(self.crop_caption)
        views = QHBoxLayout()
        views.addLayout(src_col, 3)
        views.addLayout(crop_col, 2)

        nav = QHBoxLayout()
        nav.addWidget(self.prev_btn)
        nav.addWidget(self.rand_btn)
        nav.addWidget(self.next_btn)
        nav.addStretch(1)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(views, 1)
        layout.addWidget(self.info_label)
        layout.addLayout(nav)
        self._update_buttons()

    # ------------------------------------------------------------------ API
    def set_dataset(self, dataset: AnnotationDataset | None) -> None:
        """Define o dataset a ser navegado e exibe a primeira deteccao."""
        self._dataset = dataset
        self._index = []
        if dataset is not None:
            self._index = [(i, j) for i, it in enumerate(dataset.items) for j in range(len(it.boxes))]
        self._pos = 0
        self._cache_path = None
        self._cache_image = None
        self._update_buttons()
        self.refresh()

    def set_config(self, config: ProcessingConfig) -> None:
        """Atualiza os parametros e redesenha a amostra atual."""
        self._config = config
        self.refresh()

    @property
    def sample_count(self) -> int:
        """Quantidade de deteccoes navegaveis."""
        return len(self._index)

    def refresh(self) -> None:
        """Recalcula a pre-visualizacao da deteccao atual com a configuracao corrente."""
        if not self._index or self._dataset is None:
            self.source_view.set_image(None, "Carregue as anotacoes para visualizar")
            self.crop_view.set_image(None, "Recorte final")
            self.info_label.setText("-")
            return

        item_idx, box_idx = self._index[self._pos]
        item = self._dataset.items[item_idx]
        box = item.boxes[box_idx]
        image = self._load(str(item.image_path))
        if image is None:
            self.source_view.set_image(None, f"Falha ao ler {item.image_path.name}")
            self.crop_view.set_image(None, "-")
            self.info_label.setText(str(item.image_path))
            return

        result = render_preview(image, box, self._config)
        self.source_view.set_image(ndarray_to_pixmap(result.overlay))
        if result.crop is not None:
            self.crop_view.set_image(ndarray_to_pixmap(result.crop))
            self.crop_caption.setText(f"Recorte final {result.crop.shape[1]}x{result.crop.shape[0]} px")
        else:
            self.crop_view.set_image(None, f"DESCARTADA\n{result.reason}")
            self.crop_caption.setText("Recorte final")

        ox1, oy1, ox2, oy2 = result.orig_box
        px1, py1, px2, py2 = result.padded_box
        conf = "-" if box.confidence is None else f"{box.confidence:.3f}"
        status = "✔ aceita" if result.accepted else f"✖ descartada ({result.reason})"
        self.info_label.setText(
            f"[{self._pos + 1}/{len(self._index)}] {item.relative_path.as_posix()} "
            f"({image.shape[1]}x{image.shape[0]}) | classe: {self._dataset.class_name(box.class_id)} "
            f"(id {box.class_id}) | conf: {conf}\n"
            f"Original: ({ox1},{oy1})-({ox2},{oy2}) {ox2 - ox1}x{oy2 - oy1} px | "
            f"Com padding: ({px1},{py1})-({px2},{py2}) {px2 - px1}x{py2 - py1} px | {status}"
        )

    # ------------------------------------------------------------------ internos
    def _load(self, path: str) -> np.ndarray | None:
        if path != self._cache_path:
            self._cache_image = read_image(path)
            self._cache_path = path
        return self._cache_image

    def _step(self, delta: int) -> None:
        if self._index:
            self._pos = (self._pos + delta) % len(self._index)
            self.refresh()

    def _random(self) -> None:
        if self._index:
            self._pos = random.randrange(len(self._index))
            self.refresh()

    def _update_buttons(self) -> None:
        enabled = bool(self._index)
        for btn in (self.prev_btn, self.next_btn, self.rand_btn):
            btn.setEnabled(enabled)
