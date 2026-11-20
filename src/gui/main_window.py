"""Janela principal do InoSlice (PySide6).

A extracao em lote e o carregamento das anotacoes rodam em ``QThread`` para
que a interface permaneca responsiva; o motor interno ainda paraleliza o
processamento das imagens com ``ThreadPoolExecutor``.
"""

from __future__ import annotations

import logging
import os
import sys
import threading
from pathlib import Path

from PySide6.QtCore import QByteArray, QObject, QSettings, Qt, QThread, QTimer, QUrl, Signal
from PySide6.QtGui import QCloseEvent, QColor, QDesktopServices, QFont, QIcon, QKeySequence, QPixmap, QShortcut, QTextCharFormat
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QDoubleSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
    QFileDialog,
)

from src import __app_name__, __author__, __url__, __version__
from src.core.exporter import CropHarvester, HarvestProgress, HarvestResult
from src.core.parsers import AnnotationDataset, load_annotations
from src.core.processor import ProcessingConfig
from src.gui import theme
from src.gui.preview_widget import PreviewWidget
from src.gui.widgets import Card, PathField, Segmented, chip
from src.utils.logger import ValidationError, attach_callback, detach_handler, get_logger, validate_directory

log = get_logger()


def asset_path(name: str) -> Path:
    """Caminho de um arquivo em ``assets/`` (funciona no codigo-fonte e no executavel PyInstaller)."""
    base = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parents[2]))
    return base / "assets" / name


#: Opcoes de tamanho de saida: (rotulo, lado em px | None = original | -1 = personalizado, dica)
TARGET_OPTIONS: list[tuple[str, int | None, str]] = [
    ("Original", None, "Mantem a resolucao nativa do recorte"),
    ("224", 224, "224 px · ResNet / EfficientNet / ViT-B"),
    ("256", 256, "256 px"),
    ("299", 299, "299 px · Inception"),
    ("384", 384, "384 px · SwinV2 / MaxViT"),
    ("448", 448, "448 px"),
    ("512", 512, "512 px"),
    ("Outro", -1, "Tamanho personalizado"),
]
MODE_OPTIONS = [
    ("Retangular", "Mantem a proporcao exata da caixa recortada (com margem)."),
    ("Quadrado cinza", "Quadrado sem distorcao, bordas preenchidas com cinza neutro RGB(114,114,114)."),
    ("Quadrado preto", "Quadrado sem distorcao, bordas preenchidas com preto RGB(0,0,0)."),
]
GENERIC_DIR_NAMES = {"images", "image", "imgs", "img", "train", "val", "valid", "test", "jpegimages"}


# --------------------------------------------------------------------------- #
# Workers (QThread)
# --------------------------------------------------------------------------- #
class LogBridge(QObject):
    """Encaminha mensagens de log de qualquer thread para a GUI via sinal Qt."""

    message = Signal(str, int)


class LoadWorker(QThread):
    """Carrega as anotacoes fora da thread da GUI."""

    loaded = Signal(object)
    failed = Signal(str)

    def __init__(self, images_dir: str, labels_path: str, class_file: str | None) -> None:
        super().__init__()
        self.images_dir = images_dir
        self.labels_path = labels_path
        self.class_file = class_file

    def run(self) -> None:  # noqa: D102
        try:
            self.loaded.emit(load_annotations(self.images_dir, self.labels_path, self.class_file))
        except Exception as exc:  # erros de parsing sao reportados ao usuario
            self.failed.emit(str(exc))


class HarvestWorker(QThread):
    """Executa o :class:`CropHarvester` em segundo plano."""

    progress = Signal(object)
    finished_ok = Signal(object)
    failed = Signal(str)

    def __init__(self, harvester: CropHarvester) -> None:
        super().__init__()
        self.harvester = harvester
        self.cancel_event = threading.Event()

    def run(self) -> None:  # noqa: D102
        try:
            result = self.harvester.run(progress_cb=self.progress.emit, cancel_event=self.cancel_event)
            self.finished_ok.emit(result)
        except Exception as exc:
            log.exception("Falha na colheita")
            self.failed.emit(str(exc))


# --------------------------------------------------------------------------- #
# Deteccao automatica de caminhos
# --------------------------------------------------------------------------- #
def guess_labels_path(images_dir: Path) -> Path | None:
    """Procura a pasta de labels YOLO ou o JSON COCO correspondente a uma pasta de imagens."""
    candidates = [
        images_dir.parent.parent / "labels" / images_dir.name,  # dataset/images/train -> dataset/labels/train
        images_dir.parent / "labels",
        images_dir / "labels",
    ]
    for c in candidates:
        if c.is_dir() and c != images_dir and any(c.glob("*.txt")):
            return c
    for folder in (images_dir, images_dir.parent):
        jsons = sorted(folder.glob("*.json"))
        preferred = [j for j in jsons if "coco" in j.name.lower() or "annot" in j.name.lower() or "instances" in j.name.lower()]
        if preferred or len(jsons) == 1:
            return (preferred or jsons)[0]
    return None


def guess_output_dir(images_dir: Path) -> Path:
    """Sugere ``<dataset>_crops`` ao lado do dataset (ignorando nomes genericos como ``images``)."""
    base = images_dir
    while base.name.lower() in GENERIC_DIR_NAMES and base.parent != base:
        base = base.parent
    return base.parent / f"{base.name}_crops"


# --------------------------------------------------------------------------- #
# Janela principal
# --------------------------------------------------------------------------- #
class MainWindow(QMainWindow):
    """Interface principal: dados, parametros, preview, execucao e logs."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(f"{__app_name__} v{__version__}")
        self.setMinimumSize(920, 600)
        self.resize(1320, 840)
        self.settings = QSettings("InoSlice", "InoSlice")

        self.dataset: AnnotationDataset | None = None
        self.load_worker: LoadWorker | None = None
        self.harvest_worker: HarvestWorker | None = None
        self.last_result: HarvestResult | None = None
        self._warn_count = 0
        self._restoring = False

        self.log_bridge = LogBridge()
        self.log_bridge.message.connect(self._append_log)
        self._log_handler = attach_callback(lambda msg, lvl: self.log_bridge.message.emit(msg, lvl))

        self._preview_timer = QTimer(self, singleShot=True, interval=120)
        self._preview_timer.timeout.connect(self._push_config_to_preview)
        self._autoload_timer = QTimer(self, singleShot=True, interval=500)
        self._autoload_timer.timeout.connect(lambda: self._load_annotations(silent=True))

        self._build_ui()
        self._install_shortcuts()
        self._restore_settings()
        self._update_state()
        log.info("%s v%s pronto. Selecione (ou arraste) a pasta de imagens para comecar.", __app_name__, __version__)
        if self.images_field.text() and self.labels_field.text():
            self._autoload_timer.start()

    # ================================================================== UI
    def _build_ui(self) -> None:
        central = QWidget()
        central.setObjectName("central")
        self.setCentralWidget(central)

        body = QSplitter(Qt.Orientation.Horizontal)
        body.setChildrenCollapsible(False)
        body.setHandleWidth(8)
        body.addWidget(self._build_sidebar())
        body.addWidget(self._build_workspace())
        body.setStretchFactor(0, 0)
        body.setStretchFactor(1, 1)
        body.setSizes([400, 920])
        self.body_split = body

        wrap = QVBoxLayout()
        wrap.setContentsMargins(14, 14, 14, 10)
        wrap.addWidget(body, 1)

        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        root.addWidget(self._build_header())
        root.addLayout(wrap, 1)

        self.status_msg = QLabel("Pronto")
        hint = QLabel("Ctrl+Enter extrai  ·  ← → navegam na pre-visualizacao")
        self.statusBar().addWidget(self.status_msg, 1)
        credit = QLabel(f'Desenvolvido por <a href="{__url__}" style="color:{theme.ORANGE_400};'
                        f'text-decoration:none">{__author__}</a>')
        credit.setOpenExternalLinks(True)
        credit.setToolTip(f"{__app_name__} · Apache License 2.0 · {__url__}")
        self.statusBar().addPermanentWidget(hint)
        self.statusBar().addPermanentWidget(credit)
        self.statusBar().setSizeGripEnabled(True)

    def _build_header(self) -> QFrame:
        header = QFrame()
        header.setObjectName("header")
        logo = QLabel()
        pix = QPixmap(str(asset_path("logo.png")))
        if not pix.isNull():
            dpr = self.devicePixelRatioF()
            pix = pix.scaled(int(38 * dpr), int(38 * dpr), Qt.AspectRatioMode.KeepAspectRatio,
                             Qt.TransformationMode.SmoothTransformation)
            pix.setDevicePixelRatio(dpr)
            logo.setPixmap(pix)
        name = QLabel(__app_name__)
        name.setObjectName("appName")
        tag = QLabel("Recortes de deteccao prontos para classificacao — sem distorcao e sem data leakage")
        tag.setObjectName("appTag")
        version = QLabel(f"v{__version__}")
        version.setObjectName("versionPill")
        titles = QVBoxLayout()
        titles.setSpacing(0)
        row = QHBoxLayout()
        row.setSpacing(8)
        row.addWidget(name)
        row.addWidget(version, 0, Qt.AlignmentFlag.AlignVCenter)
        row.addStretch(1)
        titles.addLayout(row)
        titles.addWidget(tag)

        self.logs_btn = QPushButton("Logs")
        self.logs_btn.setObjectName("ghost")
        self.logs_btn.setCheckable(True)
        self.logs_btn.setToolTip("Mostrar/ocultar o console de logs (Ctrl+L)")
        self.logs_btn.toggled.connect(self._toggle_logs)
        self.open_out_btn = QPushButton("Abrir pasta de saida")
        self.open_out_btn.setObjectName("ghost")
        self.open_out_btn.clicked.connect(self._open_output)
        for b in (self.logs_btn, self.open_out_btn):
            b.setCursor(Qt.CursorShape.PointingHandCursor)

        lay = QHBoxLayout(header)
        lay.setContentsMargins(18, 10, 18, 10)
        lay.setSpacing(12)
        lay.addWidget(logo)
        lay.addLayout(titles, 1)
        lay.addWidget(self.open_out_btn)
        lay.addWidget(self.logs_btn)
        return header

    # ------------------------------------------------------------------ sidebar
    def _build_sidebar(self) -> QScrollArea:
        side = QWidget()
        side.setObjectName("sidebar")
        lay = QVBoxLayout(side)
        lay.setContentsMargins(0, 0, 8, 0)
        lay.setSpacing(12)
        self.data_card = self._build_data_card()
        self.crop_card = self._build_crop_card()
        self.filter_card = self._build_filter_card()
        self.output_card = self._build_output_card()
        for c in (self.data_card, self.crop_card, self.filter_card, self.output_card):
            lay.addWidget(c)
        lay.addStretch(1)

        scroll = QScrollArea()
        scroll.setObjectName("sidebarScroll")
        scroll.setWidgetResizable(True)
        scroll.setWidget(side)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        # largura minima = conteudo + barra de rolagem, para nada ser cortado ao encolher a janela
        scroll.setMinimumWidth(max(340, side.minimumSizeHint().width() + 16))
        scroll.setMaximumWidth(600)
        return scroll

    def _build_data_card(self) -> Card:
        card = Card("1", "Dados", "Escolha ou arraste as pastas. As anotacoes sao carregadas automaticamente.")
        self.images_field = PathField("Pasta com as imagens (.jpg, .png, .tif...)", [("Procurar", "Selecionar pasta de imagens")])
        self.images_field.buttons[0].clicked.connect(
            lambda: self._pick_dir(self.images_field, "Selecione a pasta de imagens"))
        self.labels_field = PathField(
            "Pasta .txt (YOLO) ou arquivo .json (COCO)",
            [("Pasta", "Selecionar pasta de labels YOLO"), ("JSON", "Selecionar arquivo COCO .json")],
        )
        self.labels_field.buttons[0].clicked.connect(
            lambda: self._pick_dir(self.labels_field, "Selecione a pasta de labels YOLO"))
        self.labels_field.buttons[1].clicked.connect(
            lambda: self._pick_file(self.labels_field, "Selecione o arquivo COCO", "COCO JSON (*.json)"))
        self.classes_field = PathField("Opcional · detectado automaticamente", [("Procurar", "classes.txt / data.yaml")])
        self.classes_field.buttons[0].clicked.connect(
            lambda: self._pick_file(self.classes_field, "Selecione o arquivo de classes",
                                    "Classes (*.txt *.yaml *.yml *.names *.json)"))

        card.add_field("IMAGENS", self.images_field)
        card.add_field("ANOTACOES", self.labels_field)
        card.add_field("CLASSES", self.classes_field)

        self.images_field.changed.connect(self._on_images_changed)
        for f in (self.labels_field, self.classes_field):
            f.changed.connect(self._on_inputs_changed)

        self.dataset_chips = QHBoxLayout()
        self.dataset_chips.setSpacing(6)
        self.dataset_info = QLabel("Nenhuma anotacao carregada.")
        self.dataset_info.setObjectName("faint")
        self.dataset_info.setWordWrap(True)
        self.load_btn = QPushButton("Recarregar")
        self.load_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.load_btn.setToolTip("Recarregar as anotacoes do disco")
        self.load_btn.clicked.connect(lambda: self._load_annotations(silent=False))
        chips_row = QHBoxLayout()
        chips_row.addLayout(self.dataset_chips)
        chips_row.addStretch(1)
        chips_row.addWidget(self.load_btn)
        card.body.addSpacing(4)
        card.body.addLayout(chips_row)
        card.body.addWidget(self.dataset_info)
        return card

    def _build_crop_card(self) -> Card:
        card = Card("2", "Recorte", "Defina quanto contexto incluir e o formato das imagens geradas.")

        # margem de contexto
        pad_head = QHBoxLayout()
        lbl = QLabel("MARGEM DE CONTEXTO")
        lbl.setObjectName("fieldLabel")
        self.padding_badge = QLabel("10%")
        self.padding_badge.setObjectName("valueBadge")
        pad_head.addWidget(lbl)
        pad_head.addStretch(1)
        pad_head.addWidget(self.padding_badge)
        self.padding_slider = QSlider(Qt.Orientation.Horizontal)
        self.padding_slider.setRange(0, 100)
        self.padding_slider.setPageStep(5)
        self.padding_slider.setValue(10)
        self.padding_slider.setToolTip("Margem adicionada em cada lado, proporcional ao tamanho da caixa.")
        self.padding_slider.valueChanged.connect(self._on_padding)
        card.body.addLayout(pad_head)
        card.body.addWidget(self.padding_slider)

        # formato
        self.mode_seg = Segmented(MODE_OPTIONS)
        self.mode_seg.set_index(1)
        self.mode_seg.changed.connect(self._on_geometry_changed)
        card.add_field("FORMATO", self.mode_seg)

        # tamanho de saida
        self.target_seg = Segmented([(t, tip) for t, _, tip in TARGET_OPTIONS], columns=4)
        self.target_seg.set_index(0)
        self.target_seg.changed.connect(self._on_geometry_changed)
        self.custom_size_spin = QSpinBox()
        self.custom_size_spin.setRange(16, 4096)
        self.custom_size_spin.setValue(320)
        self.custom_size_spin.setSuffix(" px")
        self.custom_size_spin.setPrefix("Lado: ")
        self.custom_size_spin.valueChanged.connect(self._on_geometry_changed)
        card.add_field("TAMANHO DE SAIDA", self.target_seg)
        card.body.addWidget(self.custom_size_spin)
        self.output_hint = QLabel("")
        self.output_hint.setObjectName("cardHint")
        self.output_hint.setWordWrap(True)
        card.body.addWidget(self.output_hint)
        return card

    def _build_filter_card(self) -> Card:
        card = Card("3", "Filtros", "Descarta deteccoes pequenas demais ou com baixa confianca.")
        self.min_w_spin = QSpinBox()
        self.min_h_spin = QSpinBox()
        for s, pre in ((self.min_w_spin, "L "), (self.min_h_spin, "A ")):
            s.setRange(0, 10000)
            s.setValue(32)
            s.setPrefix(pre)
            s.setSuffix(" px")
            s.valueChanged.connect(self._schedule_preview)
        self.conf_spin = QDoubleSpinBox()
        self.conf_spin.setRange(0.0, 1.0)
        self.conf_spin.setSingleStep(0.05)
        self.conf_spin.setDecimals(2)
        self.conf_spin.setValue(0.0)
        self.conf_spin.setToolTip("Aplicado apenas a predicoes com confianca (YOLO 6 colunas / COCO score).")
        self.conf_spin.valueChanged.connect(self._schedule_preview)

        grid = QGridLayout()
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(6)
        for col, text in enumerate(("TAMANHO MINIMO", "", "CONFIANCA MIN.")):
            l = QLabel(text)
            l.setObjectName("fieldLabel")
            grid.addWidget(l, 0, col)
        grid.addWidget(self.min_w_spin, 1, 0)
        grid.addWidget(self.min_h_spin, 1, 1)
        grid.addWidget(self.conf_spin, 1, 2)
        card.body.addLayout(grid)
        return card

    def _build_output_card(self) -> Card:
        card = Card("4", "Saida", "Gera dataset_crops/<classe>/, o manifesto CSV e o resumo JSON.")
        self.output_field = PathField("Pasta de destino", [("Procurar", "Selecionar pasta de saida")])
        self.output_field.buttons[0].clicked.connect(
            lambda: self._pick_dir(self.output_field, "Selecione a pasta de saida"))
        self.output_field.changed.connect(self._update_state)
        card.add_field("PASTA DE SAIDA", self.output_field)

        self.format_seg = Segmented([("JPEG", "Arquivos menores (com perdas)"), ("PNG", "Sem perdas")])
        self.format_seg.set_index(0)
        self.format_seg.changed.connect(lambda i: self.quality_spin.setEnabled(i == 0))
        self.quality_spin = QSpinBox()
        self.quality_spin.setRange(50, 100)
        self.quality_spin.setValue(95)
        self.quality_spin.setToolTip("Qualidade JPEG (50-100)")
        self.workers_spin = QSpinBox()
        self.workers_spin.setRange(1, max(1, (os.cpu_count() or 1) * 2))
        self.workers_spin.setValue(min(8, os.cpu_count() or 1))
        self.workers_spin.setToolTip("Numero de threads de processamento")

        grid = QGridLayout()
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(6)
        for col, text in enumerate(("ARQUIVO", "QUALIDADE", "THREADS")):
            l = QLabel(text)
            l.setObjectName("fieldLabel")
            grid.addWidget(l, 0, col)
        grid.addWidget(self.format_seg, 1, 0)
        grid.addWidget(self.quality_spin, 1, 1)
        grid.addWidget(self.workers_spin, 1, 2)
        grid.setColumnStretch(0, 2)
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(2, 1)
        card.body.addLayout(grid)
        return card

    # ------------------------------------------------------------------ area de trabalho
    def _build_workspace(self) -> QWidget:
        # pre-visualizacao
        self.preview = PreviewWidget()
        self.preview_card = QFrame()
        self.preview_card.setObjectName("card")
        self.preview_card.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
        pv_title = QLabel("Pre-visualizacao")
        pv_title.setObjectName("cardTitle")
        pv_hint = QLabel("Ajustes na barra lateral sao refletidos aqui em tempo real")
        pv_hint.setObjectName("faint")
        pv_head = QHBoxLayout()
        pv_head.addWidget(pv_title)
        pv_head.addSpacing(8)
        pv_head.addWidget(pv_hint, 1)
        pv = QVBoxLayout(self.preview_card)
        pv.setContentsMargins(16, 14, 16, 14)
        pv.setSpacing(10)
        pv.addLayout(pv_head)
        pv.addWidget(self.preview, 1)

        # console
        self.log_card = QFrame()
        self.log_card.setObjectName("card")
        log_title = QLabel("Console")
        log_title.setObjectName("cardTitle")
        clear_btn = QPushButton("Limpar")
        clear_btn.setObjectName("ghost")
        clear_btn.clicked.connect(self._clear_logs)
        log_head = QHBoxLayout()
        log_head.addWidget(log_title)
        log_head.addStretch(1)
        log_head.addWidget(clear_btn)
        self.console = QPlainTextEdit()
        self.console.setObjectName("console")
        self.console.setReadOnly(True)
        self.console.setMaximumBlockCount(5000)
        self.console.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        lg = QVBoxLayout(self.log_card)
        lg.setContentsMargins(16, 12, 16, 14)
        lg.addLayout(log_head)
        lg.addWidget(self.console, 1)
        self.log_card.setMinimumHeight(120)
        self.log_card.setVisible(False)

        split = QSplitter(Qt.Orientation.Vertical)
        split.setChildrenCollapsible(False)
        split.setHandleWidth(8)
        split.addWidget(self.preview_card)
        split.addWidget(self.log_card)
        split.setStretchFactor(0, 4)
        split.setStretchFactor(1, 1)
        split.setSizes([620, 180])
        self.right_split = split

        # barra de execucao
        run_card = QFrame()
        run_card.setObjectName("card")
        self.run_btn = QPushButton("Extrair recortes")
        self.run_btn.setObjectName("primary")
        self.run_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.run_btn.setMinimumWidth(190)
        self.run_btn.setToolTip("Iniciar a extracao (Ctrl+Enter)")
        self.run_btn.clicked.connect(self._start_harvest)
        self.cancel_btn = QPushButton("Cancelar")
        self.cancel_btn.setObjectName("danger")
        self.cancel_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.cancel_btn.clicked.connect(self._cancel_harvest)
        self.run_status = QLabel("Preencha as etapas 1 e 4 para comecar.")
        self.run_status.setObjectName("cardTitle")
        self.stats_label = QLabel("")
        self.stats_label.setObjectName("muted")
        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        self.progress.setValue(0)
        self.progress.setTextVisible(False)
        info = QVBoxLayout()
        info.setSpacing(6)
        top = QHBoxLayout()
        top.addWidget(self.run_status, 1)
        top.addWidget(self.stats_label)
        info.addLayout(top)
        info.addWidget(self.progress)
        rl = QHBoxLayout(run_card)
        rl.setContentsMargins(16, 12, 12, 12)
        rl.setSpacing(14)
        rl.addLayout(info, 1)
        rl.addWidget(self.cancel_btn)
        rl.addWidget(self.run_btn)
        run_card.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)

        ws = QWidget()
        wl = QVBoxLayout(ws)
        wl.setContentsMargins(4, 0, 0, 0)
        wl.setSpacing(12)
        wl.addWidget(split, 1)
        wl.addWidget(run_card)
        return ws

    def _install_shortcuts(self) -> None:
        QShortcut(QKeySequence("Ctrl+Return"), self, activated=self._start_harvest)
        QShortcut(QKeySequence("Ctrl+Enter"), self, activated=self._start_harvest)
        QShortcut(QKeySequence("Ctrl+L"), self, activated=self.logs_btn.toggle)
        QShortcut(QKeySequence("Ctrl+O"), self, activated=lambda: self._pick_dir(self.images_field, "Selecione a pasta de imagens"))
        ctx = Qt.ShortcutContext.WidgetWithChildrenShortcut
        QShortcut(QKeySequence(Qt.Key.Key_Left), self.preview_card, activated=lambda: self.preview.step(-1), context=ctx)
        QShortcut(QKeySequence(Qt.Key.Key_Right), self.preview_card, activated=lambda: self.preview.step(+1), context=ctx)
        QShortcut(QKeySequence(Qt.Key.Key_R), self.preview_card, activated=self.preview.random_sample, context=ctx)

    # ================================================================== helpers
    def _pick_dir(self, field: PathField, title: str) -> None:
        start = field.text() or str(self.settings.value("last_dir", str(Path.home())))
        path = QFileDialog.getExistingDirectory(self, title, start)
        if path:
            field.setText(path)
            self.settings.setValue("last_dir", path)

    def _pick_file(self, field: PathField, title: str, flt: str) -> None:
        start = field.text() or str(self.settings.value("last_dir", str(Path.home())))
        path, _ = QFileDialog.getOpenFileName(self, title, start, flt)
        if path:
            field.setText(path)
            self.settings.setValue("last_dir", str(Path(path).parent))

    def _set_status(self, text: str) -> None:
        self.status_msg.setText(text)

    def _on_images_changed(self, text: str) -> None:
        path = Path(text.strip()) if text.strip() else None
        if path is not None and path.is_dir() and not self._restoring:
            if not self.labels_field.text():
                guess = guess_labels_path(path)
                if guess is not None:
                    self.labels_field.setText(str(guess))
                    log.info("Anotacoes detectadas automaticamente: %s", guess)
            if not self.output_field.text():
                self.output_field.setText(str(guess_output_dir(path)))
        self._on_inputs_changed()

    def _on_inputs_changed(self, *_: object) -> None:
        img, lbl, cls = self.images_field.text(), self.labels_field.text(), self.classes_field.text()
        self.images_field.set_valid(None if not img else Path(img).is_dir())
        self.labels_field.set_valid(None if not lbl else Path(lbl).exists())
        self.classes_field.set_valid(None if not cls else Path(cls).is_file())
        if self.dataset is not None:  # entradas mudaram: o dataset atual deixa de valer
            self.dataset = None
            self.preview.set_dataset(None)
            self._render_dataset_summary()
        if img and lbl and Path(img).is_dir() and Path(lbl).exists() and not self._restoring:
            self._autoload_timer.start()
        self._update_state()

    def _on_padding(self, value: int) -> None:
        self.padding_badge.setText(f"{value}%")
        self._schedule_preview()

    def _on_geometry_changed(self, *_: object) -> None:
        custom = TARGET_OPTIONS[max(0, self.target_seg.index())][1] == -1
        self.custom_size_spin.setVisible(custom)
        size = self._target_size()
        mode = self.mode_seg.index()
        if mode == 0:
            desc = (f"Retangular, maior lado com {size} px (proporcao mantida)." if size
                    else "Retangular, na resolucao nativa do recorte.")
        else:
            color = "cinza" if mode == 1 else "preto"
            desc = (f"Quadrado {size}×{size} px, conteudo centralizado sem distorcao, bordas em {color}." if size
                    else f"Quadrado com o lado do maior eixo do recorte, bordas em {color}.")
        self.output_hint.setText(desc)
        self._schedule_preview()

    def _target_size(self) -> int | None:
        value = TARGET_OPTIONS[max(0, self.target_seg.index())][1]
        if value == -1:
            return int(self.custom_size_spin.value())
        return value

    def current_config(self) -> ProcessingConfig:
        """Monta a :class:`ProcessingConfig` a partir dos controles da interface."""
        mode_id = self.mode_seg.index()
        return ProcessingConfig(
            padding_pct=float(self.padding_slider.value()),
            min_width=int(self.min_w_spin.value()),
            min_height=int(self.min_h_spin.value()),
            min_confidence=float(self.conf_spin.value()),
            aspect_mode="rectangular" if mode_id == 0 else "letterbox",
            letterbox_color="black" if mode_id == 2 else "gray",
            target_size=self._target_size(),
            jpeg_quality=int(self.quality_spin.value()),
        )

    def _schedule_preview(self, *_: object) -> None:
        self._preview_timer.start()

    def _push_config_to_preview(self) -> None:
        self.preview.set_config(self.current_config())

    def _append_log(self, message: str, level: int) -> None:
        color = {
            logging.DEBUG: theme.TEXT_FAINT,
            logging.INFO: theme.TEXT_MUTED,
            logging.WARNING: theme.WARNING,
            logging.ERROR: theme.DANGER,
            logging.CRITICAL: theme.DANGER,
        }.get(level, theme.TEXT_MUTED)
        fmt = QTextCharFormat()
        fmt.setForeground(QColor(color))
        if level >= logging.ERROR:
            fmt.setFontWeight(QFont.Weight.Bold)
        cursor = self.console.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        cursor.insertText(message + "\n", fmt)
        self.console.setTextCursor(cursor)
        self.console.ensureCursorVisible()
        if level >= logging.WARNING:
            self._warn_count += 1
            self._update_logs_btn()
        if level >= logging.ERROR and not self.logs_btn.isChecked():
            self.logs_btn.setChecked(True)

    def _update_logs_btn(self) -> None:
        self.logs_btn.setText(f"Logs  ⚠ {self._warn_count}" if self._warn_count else "Logs")

    def _clear_logs(self) -> None:
        self.console.clear()
        self._warn_count = 0
        self._update_logs_btn()

    def _toggle_logs(self, on: bool) -> None:
        self.log_card.setVisible(on)

    def _busy(self) -> bool:
        return bool(
            (self.harvest_worker and self.harvest_worker.isRunning())
            or (self.load_worker and self.load_worker.isRunning())
        )

    def _update_state(self, *_: object) -> None:
        harvesting = bool(self.harvest_worker and self.harvest_worker.isRunning())
        busy = self._busy()
        out = self.output_field.text()
        self.run_btn.setEnabled(not busy and self.dataset is not None and bool(out))
        self.cancel_btn.setVisible(harvesting)
        self.cancel_btn.setEnabled(harvesting)
        self.load_btn.setEnabled(not busy and bool(self.images_field.text()) and bool(self.labels_field.text()))
        self.open_out_btn.setEnabled(bool(out) and Path(out).is_dir())
        self.data_card.set_done(self.dataset is not None)
        self.output_card.set_done(bool(out))
        for w in (self.data_card, self.crop_card, self.filter_card, self.output_card):
            w.setEnabled(not harvesting)
        if not harvesting:
            if self.dataset is None:
                self.run_status.setText("Selecione as imagens e as anotacoes para comecar.")
            elif not out:
                self.run_status.setText("Escolha a pasta de saida.")
            elif self.last_result is None:
                self.run_status.setText(f"Pronto para extrair {self.dataset.total_boxes} deteccoes.")

    def _render_dataset_summary(self) -> None:
        while self.dataset_chips.count():
            item = self.dataset_chips.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        ds = self.dataset
        if ds is None:
            self.dataset_info.setText("Nenhuma anotacao carregada.")
            return
        counts: dict[str, int] = {}
        for it in ds.items:
            for b in it.boxes:
                name = ds.class_name(b.class_id)
                counts[name] = counts.get(name, 0) + 1
        self.dataset_chips.addWidget(chip(ds.format.upper(), "chipAccent"))
        self.dataset_chips.addWidget(chip(f"{len(ds.items)} imagens"))
        self.dataset_chips.addWidget(chip(f"{ds.total_boxes} caixas"))
        top = sorted(counts.items(), key=lambda kv: -kv[1])
        text = " · ".join(f"{k} {v}" for k, v in top[:8])
        if len(top) > 8:
            text += f" · +{len(top) - 8} classes"
        self.dataset_info.setText(f"{len(counts)} classes: {text}")

    # ================================================================== acoes
    def _load_annotations(self, silent: bool = False) -> None:
        if self._busy():
            return
        try:
            images = validate_directory(self.images_field.text(), "Imagens")
            labels = self.labels_field.text()
            if not labels or not Path(labels).exists():
                raise ValidationError("Anotacoes: selecione uma pasta YOLO ou um arquivo COCO .json existente.")
        except ValidationError as exc:
            if not silent:
                QMessageBox.warning(self, "Entradas invalidas", str(exc))
            return
        class_file = self.classes_field.text() or None
        log.info("Carregando anotacoes de %s ...", labels)
        self._set_status("Carregando anotacoes...")
        self.dataset_info.setText("Carregando anotacoes...")
        self.load_worker = LoadWorker(str(images), labels, class_file)
        self.load_worker.loaded.connect(self._on_loaded)
        self.load_worker.failed.connect(self._on_load_failed)
        self.load_worker.finished.connect(self._update_state)
        self.load_worker.start()
        self._update_state()

    def _on_loaded(self, dataset: AnnotationDataset) -> None:
        self.dataset = dataset
        self.last_result = None
        for w in dataset.warnings[:50]:
            log.warning(w)
        if len(dataset.warnings) > 50:
            log.warning("... e mais %d avisos de parsing.", len(dataset.warnings) - 50)
        if not dataset.class_names:
            log.warning("Nenhum mapeamento de classes encontrado: pastas usarao 'class_<id>'.")
        self._render_dataset_summary()
        self.preview.set_config(self.current_config())
        self.preview.set_dataset(dataset)
        self._set_status(f"{dataset.total_boxes} caixas carregadas de {len(dataset.items)} imagens")
        self._save_settings()
        self._update_state()

    def _on_load_failed(self, message: str) -> None:
        log.error("Falha ao carregar anotacoes: %s", message)
        self._set_status("Falha ao carregar anotacoes")
        self.dataset_info.setText("Falha ao carregar as anotacoes — veja os logs.")

    def _start_harvest(self) -> None:
        if self.dataset is None or self._busy() or not self.run_btn.isEnabled():
            return
        try:
            out_dir = validate_directory(self.output_field.text(), "Saida", must_exist=False)
            config = self.current_config()
            config.validate()
        except (ValidationError, ValueError) as exc:
            QMessageBox.warning(self, "Parametros invalidos", str(exc))
            return

        crops_dir = out_dir / "dataset_crops"
        if crops_dir.is_dir() and any(crops_dir.iterdir()):
            answer = QMessageBox.question(
                self,
                "Saida existente",
                f"'{crops_dir}' ja contem arquivos.\nRecortes com o mesmo nome serao sobrescritos e o "
                "manifesto sera regravado. Deseja continuar?",
            )
            if answer != QMessageBox.StandardButton.Yes:
                return

        harvester = CropHarvester(
            self.dataset,
            config,
            out_dir,
            workers=int(self.workers_spin.value()),
            image_ext=".jpg" if self.format_seg.index() == 0 else ".png",
            source_info={
                "labels_path": self.labels_field.text(),
                "class_file": self.classes_field.text() or None,
            },
        )
        self.harvest_worker = HarvestWorker(harvester)
        self.harvest_worker.progress.connect(self._on_progress)
        self.harvest_worker.finished_ok.connect(self._on_harvest_done)
        self.harvest_worker.failed.connect(self._on_harvest_failed)
        self.harvest_worker.finished.connect(self._update_state)
        self.progress.setValue(0)
        self.stats_label.setText("")
        self.run_status.setText("Extraindo recortes...")
        self._save_settings()
        self.harvest_worker.start()
        self._set_status("Extraindo recortes...")
        self._update_state()

    def _cancel_harvest(self) -> None:
        if self.harvest_worker and self.harvest_worker.isRunning():
            log.warning("Cancelamento solicitado; aguardando imagens em andamento...")
            self.harvest_worker.cancel_event.set()
            self.cancel_btn.setEnabled(False)
            self.run_status.setText("Cancelando...")

    def _on_progress(self, p: HarvestProgress) -> None:
        self.progress.setValue(int(p.fraction * 1000))
        self.run_status.setText(f"Extraindo... {p.fraction * 100:.0f}%")
        self.stats_label.setText(
            f"{p.images_done}/{p.images_total} imagens  ·  {p.crops_done} recortes  ·  "
            f"{p.crops_per_second:.1f}/s  ·  {p.elapsed:.1f} s"
        )

    def _on_harvest_done(self, result: HarvestResult) -> None:
        self.last_result = result
        totals = result.summary["totals"]
        assert isinstance(totals, dict)
        discarded = (totals["discarded_min_size"] + totals["discarded_low_confidence"]
                     + totals["discarded_invalid_box"])
        headline = f"{totals['crops_extracted']} recortes salvos de {totals['images_processed']} imagens"
        msg = (
            f"{headline}.\n"
            f"Descartados: {totals['discarded_min_size']} (tamanho minimo), "
            f"{totals['discarded_low_confidence']} (confianca), {totals['discarded_invalid_box']} (invalidos).\n\n"
            f"Manifesto: {result.manifest_path}\nResumo: {result.summary_path}"
        )
        self._update_state()
        if result.cancelled:
            self.run_status.setText("Extracao cancelada — resultados parciais salvos.")
            self._set_status("Extracao cancelada")
            QMessageBox.warning(self, "Extracao cancelada", "Execucao cancelada (resultados parciais salvos).\n\n" + msg)
            return
        self.progress.setValue(1000)
        self.run_status.setText(f"✓ {headline}" + (f"  ·  {discarded} descartados" if discarded else ""))
        self._set_status("Extracao concluida")
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Information)
        box.setWindowTitle("Extracao concluida")
        box.setText(msg)
        open_btn = box.addButton("Abrir pasta", QMessageBox.ButtonRole.AcceptRole)
        box.addButton("Fechar", QMessageBox.ButtonRole.RejectRole)
        box.exec()
        if box.clickedButton() is open_btn:
            self._open_output()

    def _on_harvest_failed(self, message: str) -> None:
        self.run_status.setText("Falha na extracao — veja os logs.")
        self._set_status("Falha na extracao")
        QMessageBox.critical(self, "Erro na extracao", message)

    def _open_output(self) -> None:
        out = self.output_field.text()
        if out and Path(out).is_dir():
            QDesktopServices.openUrl(QUrl.fromLocalFile(out))

    # ================================================================== persistencia
    def _save_settings(self) -> None:
        s = self.settings
        s.setValue("images", self.images_field.text())
        s.setValue("labels", self.labels_field.text())
        s.setValue("classes", self.classes_field.text())
        s.setValue("output", self.output_field.text())
        s.setValue("padding", self.padding_slider.value())
        s.setValue("mode", self.mode_seg.index())
        s.setValue("min_w", self.min_w_spin.value())
        s.setValue("min_h", self.min_h_spin.value())
        s.setValue("target_opt", self.target_seg.index())
        s.setValue("custom_size", self.custom_size_spin.value())
        s.setValue("conf", self.conf_spin.value())
        s.setValue("format_idx", self.format_seg.index())
        s.setValue("quality", self.quality_spin.value())
        s.setValue("workers", self.workers_spin.value())
        s.setValue("logs_visible", self.logs_btn.isChecked())
        s.setValue("geometry", self.saveGeometry())
        s.setValue("body_split", self.body_split.saveState())
        s.setValue("right_split", self.right_split.saveState())

    def _restore_settings(self) -> None:
        s = self.settings

        def get(key: str, default: object, typ: type) -> object:
            try:
                return s.value(key, default, type=typ)
            except (TypeError, ValueError):
                return default

        self._restoring = True
        try:
            self.images_field.setText(str(get("images", "", str)))
            self.labels_field.setText(str(get("labels", "", str)))
            self.classes_field.setText(str(get("classes", "", str)))
            self.output_field.setText(str(get("output", "", str)))
            self.padding_slider.setValue(int(round(float(get("padding", 10.0, float)))))  # type: ignore[arg-type]
            self.mode_seg.set_index(int(get("mode", 1, int)))  # type: ignore[arg-type]
            self.min_w_spin.setValue(int(get("min_w", 32, int)))  # type: ignore[arg-type]
            self.min_h_spin.setValue(int(get("min_h", 32, int)))  # type: ignore[arg-type]
            self.target_seg.set_index(int(get("target_opt", 0, int)))  # type: ignore[arg-type]
            self.custom_size_spin.setValue(int(get("custom_size", 320, int)))  # type: ignore[arg-type]
            self.conf_spin.setValue(float(get("conf", 0.0, float)))  # type: ignore[arg-type]
            self.format_seg.set_index(int(get("format_idx", 0, int)))  # type: ignore[arg-type]
            self.quality_spin.setEnabled(self.format_seg.index() == 0)
            self.quality_spin.setValue(int(get("quality", 95, int)))  # type: ignore[arg-type]
            self.workers_spin.setValue(int(get("workers", self.workers_spin.value(), int)))  # type: ignore[arg-type]
            self.logs_btn.setChecked(bool(get("logs_visible", False, bool)))
            for key, restore in (("geometry", self.restoreGeometry), ("body_split", self.body_split.restoreState),
                                 ("right_split", self.right_split.restoreState)):
                value = s.value(key)
                if isinstance(value, QByteArray) and not value.isEmpty():
                    restore(value)
        finally:
            self._restoring = False
        self._on_inputs_changed()
        self._on_geometry_changed()
        self._on_padding(self.padding_slider.value())

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802 - API Qt
        """Solicita cancelamento e aguarda os workers antes de fechar."""
        if self.harvest_worker and self.harvest_worker.isRunning():
            answer = QMessageBox.question(self, "Extracao em andamento", "Cancelar a extracao e sair?")
            if answer != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
            self.harvest_worker.cancel_event.set()
            self.harvest_worker.wait(30000)
        if self.load_worker and self.load_worker.isRunning():
            self.load_worker.wait(30000)
        self._save_settings()
        detach_handler(self._log_handler)
        super().closeEvent(event)


def run_gui(argv: list[str] | None = None) -> int:
    """Cria a ``QApplication``, aplica o tema e exibe a janela principal."""

    if sys.platform == "win32":  # faz a barra de tarefas exibir a logo do app, nao a do Python
        try:
            import ctypes

            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("InoSlice.InoSlice")
        except (AttributeError, OSError):
            pass
    app = QApplication.instance() or QApplication(argv if argv is not None else sys.argv)
    app.setApplicationName(__app_name__)
    app.setApplicationVersion(__version__)
    icon = QIcon(str(asset_path("logo.png")))
    if not icon.isNull():
        app.setWindowIcon(icon)
    theme.apply_theme(app)  # type: ignore[arg-type]
    window = MainWindow()
    window.show()
    return int(app.exec())
