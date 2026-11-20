"""Painel de pre-visualizacao interativa dos recortes.

Mostra a imagem de origem com a caixa original (verde) e a caixa expandida
pelo padding (laranja) e, ao lado, o recorte final exatamente como sera salvo.
No modo "Tamanho real" o recorte e exibido em escala 1:1 (reduzido apenas se
nao couber), de modo que o efeito do redimensionamento final fica visivel.
"""

from __future__ import annotations

import random

import numpy as np
from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QImage, QPixmap, QResizeEvent
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from src.core.parsers import AnnotationDataset
from src.core.processor import ProcessingConfig, read_image, render_preview
from src.gui.widgets import chip


def ndarray_to_pixmap(image: np.ndarray) -> QPixmap:
    """Converte uma imagem BGR (ou cinza) ``np.ndarray`` em ``QPixmap``."""
    img = np.ascontiguousarray(image)
    h, w = img.shape[:2]
    if img.ndim == 2:
        qimg = QImage(img.data, w, h, img.strides[0], QImage.Format.Format_Grayscale8)
    else:
        qimg = QImage(img.data, w, h, img.strides[0], QImage.Format.Format_BGR888)
    return QPixmap.fromImage(qimg.copy())


class ImageCanvas(QLabel):
    """Exibe um pixmap centralizado, reescalando conforme o tamanho do widget.

    Usa ``QSizePolicy.Ignored`` para que o pixmap nunca force o tamanho da
    janela: o canvas encolhe e cresce livremente junto com o layout.
    """

    def __init__(self, placeholder: str, parent: QWidget | None = None) -> None:
        super().__init__(placeholder, parent)
        self._pixmap: QPixmap | None = None
        self._real_size = False
        self.scale = 1.0
        self.setObjectName("canvas")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setWordWrap(True)
        self.setMinimumSize(140, 110)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)

    def sizeHint(self) -> QSize:  # noqa: N802 - API Qt
        return QSize(480, 360)

    def set_real_size(self, on: bool) -> None:
        """``True``: nunca amplia alem de 100% (mostra o tamanho real de saida)."""
        self._real_size = on
        self._rescale()

    def set_image(self, pixmap: QPixmap | None, placeholder: str = "") -> None:
        """Define a imagem exibida (ou um texto quando ``pixmap`` e ``None``)."""
        self._pixmap = pixmap
        if pixmap is None:
            self.scale = 1.0
            self.clear()
            self.setText(placeholder)
        else:
            self._rescale()

    def _rescale(self) -> None:
        if self._pixmap is None or self._pixmap.isNull():
            return
        pw, ph = self._pixmap.width(), self._pixmap.height()
        avail_w, avail_h = max(1, self.width() - 16), max(1, self.height() - 16)
        scale = min(avail_w / pw, avail_h / ph)
        if self._real_size:
            scale = min(scale, 1.0)
        self.scale = scale
        dpr = self.devicePixelRatioF()
        w, h = max(1, round(pw * scale * dpr)), max(1, round(ph * scale * dpr))
        mode = Qt.TransformationMode.SmoothTransformation if scale < 1 else Qt.TransformationMode.FastTransformation
        scaled = self._pixmap.scaled(w, h, Qt.AspectRatioMode.IgnoreAspectRatio, mode)
        scaled.setDevicePixelRatio(dpr)
        self.setPixmap(scaled)

    def resizeEvent(self, event: QResizeEvent) -> None:  # noqa: N802 - API Qt
        super().resizeEvent(event)
        self._rescale()


class PreviewWidget(QWidget):
    """Visualizador de amostras com navegacao e filtro por classe."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._dataset: AnnotationDataset | None = None
        self._config = ProcessingConfig()
        self._all: list[tuple[int, int]] = []  # (item, caixa)
        self._index: list[tuple[int, int]] = []
        self._pos = 0
        self._cache_path: str | None = None
        self._cache_image: np.ndarray | None = None

        # ---- canvases
        self.source_view = ImageCanvas("Selecione as imagens e as anotacoes\npara visualizar os recortes aqui")
        self.crop_view = ImageCanvas("Recorte final")

        legend = QHBoxLayout()
        legend.setSpacing(14)
        g = QLabel("■ original")
        g.setObjectName("legendGreen")
        o = QLabel("■ com margem")
        o.setObjectName("legendOrange")
        legend.addWidget(g)
        legend.addWidget(o)
        legend.addStretch(1)
        self.source_dims = QLabel("")
        self.source_dims.setObjectName("faint")
        legend.addWidget(self.source_dims)

        self.crop_dims = QLabel("")
        self.crop_dims.setObjectName("chipAccent")
        self.crop_dims.setToolTip("Tamanho da imagem salva  ·  zoom da visualizacao")
        self.real_btn = QPushButton("Tamanho real")
        self.real_btn.setObjectName("ghost")
        self.real_btn.setCheckable(True)
        self.real_btn.setChecked(True)
        self.real_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.real_btn.setToolTip(
            "Tamanho real: exibe o recorte em escala 1:1 (como sera salvo).\n"
            "Desmarque para ampliar o recorte e ver detalhes."
        )
        self.real_btn.toggled.connect(self._on_real_toggled)
        crop_head = QHBoxLayout()
        crop_head.addWidget(self.crop_dims)
        crop_head.addStretch(1)
        crop_head.addWidget(self.real_btn)

        src_col = QVBoxLayout()
        src_col.setSpacing(6)
        src_col.addWidget(self.source_view, 1)
        src_col.addLayout(legend)
        crop_col = QVBoxLayout()
        crop_col.setSpacing(6)
        crop_col.addWidget(self.crop_view, 1)
        crop_col.addLayout(crop_head)
        views = QHBoxLayout()
        views.setSpacing(12)
        views.addLayout(src_col, 3)
        views.addLayout(crop_col, 2)

        # ---- detalhes da deteccao
        self.class_chip = chip("-")
        self.status_chip = chip("-")
        self.conf_chip = chip("conf -")
        self.file_label = QLabel("")
        self.file_label.setObjectName("faint")
        self.file_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        details = QHBoxLayout()
        details.setSpacing(6)
        details.addWidget(self.status_chip)
        details.addWidget(self.class_chip)
        details.addWidget(self.conf_chip)
        details.addSpacing(6)
        details.addWidget(self.file_label, 1)
        self.geom_label = QLabel("")
        self.geom_label.setObjectName("mono")
        self.geom_label.setWordWrap(True)

        # ---- navegacao
        self.prev_btn = QPushButton("◀")
        self.next_btn = QPushButton("▶")
        self.rand_btn = QPushButton("Aleatoria")
        self.prev_btn.setToolTip("Deteccao anterior (←)")
        self.next_btn.setToolTip("Proxima deteccao (→)")
        self.rand_btn.setToolTip("Deteccao aleatoria (R)")
        for b in (self.prev_btn, self.next_btn, self.rand_btn):
            b.setCursor(Qt.CursorShape.PointingHandCursor)
        self.prev_btn.setFixedWidth(40)
        self.next_btn.setFixedWidth(40)
        self.prev_btn.clicked.connect(lambda: self.step(-1))
        self.next_btn.clicked.connect(lambda: self.step(+1))
        self.rand_btn.clicked.connect(self.random_sample)
        self.counter = QLabel("0 / 0")
        self.counter.setObjectName("muted")
        self.counter.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.counter.setMinimumWidth(80)
        self.class_filter = QComboBox()
        self.class_filter.setToolTip("Navegar somente pelas deteccoes de uma classe")
        self.class_filter.setMinimumContentsLength(12)
        self.class_filter.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.class_filter.currentIndexChanged.connect(self._apply_filter)

        nav = QHBoxLayout()
        nav.setSpacing(6)
        nav.addWidget(self.prev_btn)
        nav.addWidget(self.counter)
        nav.addWidget(self.next_btn)
        nav.addWidget(self.rand_btn)
        nav.addStretch(1)
        nav.addWidget(self.class_filter)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        layout.addLayout(views, 1)
        layout.addLayout(details)
        layout.addWidget(self.geom_label)
        layout.addLayout(nav)
        self.crop_view.set_real_size(True)
        self._update_buttons()

    # ------------------------------------------------------------------ API
    def set_dataset(self, dataset: AnnotationDataset | None) -> None:
        """Define o dataset a ser navegado e exibe a primeira deteccao."""
        self._dataset = dataset
        self._all = []
        self._cache_path = None
        self._cache_image = None
        self.class_filter.blockSignals(True)
        self.class_filter.clear()
        self.class_filter.addItem("Todas as classes", None)
        if dataset is not None:
            self._all = [(i, j) for i, it in enumerate(dataset.items) for j in range(len(it.boxes))]
            ids = sorted({b.class_id for it in dataset.items for b in it.boxes})
            for cid in ids:
                self.class_filter.addItem(dataset.class_name(cid), cid)
        self.class_filter.blockSignals(False)
        self._apply_filter()

    def set_config(self, config: ProcessingConfig) -> None:
        """Atualiza os parametros e redesenha a amostra atual."""
        self._config = config
        self.refresh()

    @property
    def sample_count(self) -> int:
        """Quantidade de deteccoes navegaveis (apos o filtro de classe)."""
        return len(self._index)

    def step(self, delta: int) -> None:
        """Avanca ``delta`` deteccoes (com volta circular)."""
        if self._index:
            self._pos = (self._pos + delta) % len(self._index)
            self.refresh()

    def random_sample(self) -> None:
        """Salta para uma deteccao aleatoria."""
        if self._index:
            self._pos = random.randrange(len(self._index))
            self.refresh()

    def refresh(self) -> None:
        """Recalcula a pre-visualizacao da deteccao atual com a configuracao corrente."""
        if not self._index or self._dataset is None:
            self.source_view.set_image(None, "Selecione as imagens e as anotacoes\npara visualizar os recortes aqui")
            self.crop_view.set_image(None, "Recorte final")
            self._set_chip(self.status_chip, "-", "chip")
            self.class_chip.setText("-")
            self.conf_chip.setText("conf -")
            self.crop_dims.setText("")
            self.crop_dims.setVisible(False)
            self.file_label.setText("")
            self.geom_label.setText("")
            self.source_dims.setText("")
            self.counter.setText("0 / 0")
            return

        item_idx, box_idx = self._index[self._pos]
        item = self._dataset.items[item_idx]
        box = item.boxes[box_idx]
        self.counter.setText(f"{self._pos + 1} / {len(self._index)}")
        self.file_label.setText(item.relative_path.as_posix())
        self.file_label.setToolTip(str(item.image_path))
        self.class_chip.setText(self._dataset.class_name(box.class_id))
        self.conf_chip.setText("conf -" if box.confidence is None else f"conf {box.confidence:.2f}")

        image = self._load(str(item.image_path))
        if image is None:
            self.source_view.set_image(None, f"Falha ao ler {item.image_path.name}")
            self.crop_view.set_image(None, "-")
            self._set_chip(self.status_chip, "Erro de leitura", "chipBad")
            self.crop_dims.setVisible(False)
            self.geom_label.setText("")
            return

        result = render_preview(image, box, self._config)
        self.source_view.set_image(ndarray_to_pixmap(result.overlay))
        self.source_dims.setText(f"{image.shape[1]}×{image.shape[0]} px")
        if result.crop is not None:
            self.crop_view.set_image(ndarray_to_pixmap(result.crop))
            self._set_chip(self.status_chip, "✓ Aceita", "chipOk")
            self.crop_dims.setVisible(True)
            self._update_crop_dims()
        else:
            self.crop_view.set_image(None, f"Descartada\n{result.reason}")
            self._set_chip(self.status_chip, "✕ Descartada", "chipBad")
            self.crop_dims.setVisible(False)

        ox1, oy1, ox2, oy2 = result.orig_box
        px1, py1, px2, py2 = result.padded_box
        self.geom_label.setText(
            f"original {ox2 - ox1}×{oy2 - oy1} px @ ({ox1},{oy1})   →   "
            f"com margem {px2 - px1}×{py2 - py1} px @ ({px1},{py1})"
            + ("" if result.accepted else f"   ·   {result.reason}")
        )

    # ------------------------------------------------------------------ internos
    @staticmethod
    def _set_chip(label: QLabel, text: str, kind: str) -> None:
        label.setText(text)
        if label.objectName() != kind:
            label.setObjectName(kind)
            label.style().unpolish(label)
            label.style().polish(label)

    def _update_crop_dims(self) -> None:
        pm = self.crop_view._pixmap  # noqa: SLF001 - leitura interna do proprio modulo
        if pm is None:
            return
        self.crop_dims.setText(f"{pm.width()}×{pm.height()} px  ·  {self.crop_view.scale * 100:.0f}%")

    def _on_real_toggled(self, on: bool) -> None:
        self.crop_view.set_real_size(on)
        self._update_crop_dims()

    def resizeEvent(self, event: QResizeEvent) -> None:  # noqa: N802 - API Qt
        super().resizeEvent(event)
        self._update_crop_dims()

    def _apply_filter(self, *_: object) -> None:
        cid = self.class_filter.currentData()
        if cid is None or self._dataset is None:
            self._index = list(self._all)
        else:
            self._index = [(i, j) for i, j in self._all if self._dataset.items[i].boxes[j].class_id == cid]
        self._pos = 0
        self._update_buttons()
        self.refresh()

    def _load(self, path: str) -> np.ndarray | None:
        if path != self._cache_path:
            self._cache_image = read_image(path)
            self._cache_path = path
        return self._cache_image

    def _update_buttons(self) -> None:
        enabled = bool(self._index)
        for btn in (self.prev_btn, self.next_btn, self.rand_btn):
            btn.setEnabled(enabled)
        self.class_filter.setEnabled(bool(self._all))
