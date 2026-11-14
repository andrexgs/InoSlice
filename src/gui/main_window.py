"""Janela principal do InoSlice (PySide6, tema escuro).

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

from PySide6.QtCore import QObject, QSettings, Qt, QThread, QTimer, QUrl, Signal
from PySide6.QtGui import QCloseEvent, QColor, QDesktopServices, QFont, QIcon, QPalette, QPixmap, QTextCharFormat
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from src import __app_name__, __version__
from src.core.exporter import CropHarvester, HarvestProgress, HarvestResult
from src.core.parsers import AnnotationDataset, load_annotations
from src.core.processor import ProcessingConfig
from src.gui.preview_widget import PreviewWidget
from src.utils.logger import ValidationError, attach_callback, detach_handler, get_logger, validate_directory

log = get_logger()


def asset_path(name: str) -> Path:
    """Caminho de um arquivo em ``assets/`` (funciona no codigo-fonte e no executavel PyInstaller)."""
    base = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parents[2]))
    return base / "assets" / name

TARGET_SIZES: list[tuple[str, int | None]] = [
    ("Nativo (sem redimensionar)", None),
    ("224 px (ResNet / EfficientNet)", 224),
    ("256 px", 256),
    ("299 px (Inception)", 299),
    ("384 px (SwinV2 / MaxViT)", 384),
    ("448 px", 448),
    ("512 px", 512),
]

DARK_QSS = """
QWidget { font-size: 10pt; }
QMainWindow, QWidget#central { background: #1b1d23; }
QGroupBox {
    border: 1px solid #343844; border-radius: 8px; margin-top: 14px; padding: 10px 8px 8px 8px;
    background: #22252d; font-weight: 600; color: #cfd6e4;
}
QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 4px; color: #7fb4ff; }
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QPlainTextEdit {
    background: #15171c; border: 1px solid #3a3f4b; border-radius: 5px; padding: 4px 6px; color: #e6e9ef;
    selection-background-color: #2f6fdb;
}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus { border-color: #4c8dff; }
QPushButton {
    background: #2d323d; border: 1px solid #3f4553; border-radius: 6px; padding: 6px 12px; color: #e6e9ef;
}
QPushButton:hover { background: #363c49; }
QPushButton:pressed { background: #262a33; }
QPushButton:disabled { color: #6b7180; background: #24272e; }
QPushButton#primary { background: #2f6fdb; border-color: #3d7dea; font-weight: 700; padding: 9px 18px; }
QPushButton#primary:hover { background: #3d7dea; }
QPushButton#primary:disabled { background: #2a3550; color: #8792a8; }
QPushButton#danger { background: #8b2c2c; border-color: #a33a3a; }
QPushButton#danger:hover { background: #a33a3a; }
QProgressBar {
    border: 1px solid #3a3f4b; border-radius: 6px; background: #15171c; text-align: center; color: #e6e9ef; height: 20px;
}
QProgressBar::chunk { background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #2f6fdb, stop:1 #39c0a0); border-radius: 5px; }
QSlider::groove:horizontal { height: 6px; background: #3a3f4b; border-radius: 3px; }
QSlider::handle:horizontal { background: #4c8dff; width: 16px; margin: -6px 0; border-radius: 8px; }
QSlider::sub-page:horizontal { background: #2f6fdb; border-radius: 3px; }
QLabel#title { font-size: 17pt; font-weight: 800; color: #e6e9ef; }
QLabel#subtitle { color: #8a93a6; }
QLabel#caption { color: #8a93a6; font-size: 9pt; }
QLabel#previewInfo { color: #b9c2d3; font-family: Consolas, 'DejaVu Sans Mono', monospace; font-size: 9pt; }
QLabel#previewCanvas { background: #111317; border: 1px dashed #3a3f4b; border-radius: 6px; color: #6b7180; }
QLabel#stat { color: #39c0a0; font-weight: 700; }
QPlainTextEdit#console { font-family: Consolas, 'DejaVu Sans Mono', monospace; font-size: 9pt; background: #0f1115; }
QScrollArea { border: none; }
QSplitter::handle { background: #2a2e37; }
"""


def apply_dark_theme(app: QApplication) -> None:
    """Aplica o estilo Fusion com paleta escura e folha de estilos da aplicacao."""
    app.setStyle("Fusion")
    pal = QPalette()
    base = QColor(27, 29, 35)
    pal.setColor(QPalette.ColorRole.Window, base)
    pal.setColor(QPalette.ColorRole.WindowText, QColor(230, 233, 239))
    pal.setColor(QPalette.ColorRole.Base, QColor(21, 23, 28))
    pal.setColor(QPalette.ColorRole.AlternateBase, QColor(34, 37, 45))
    pal.setColor(QPalette.ColorRole.ToolTipBase, QColor(34, 37, 45))
    pal.setColor(QPalette.ColorRole.ToolTipText, QColor(230, 233, 239))
    pal.setColor(QPalette.ColorRole.Text, QColor(230, 233, 239))
    pal.setColor(QPalette.ColorRole.Button, QColor(45, 50, 61))
    pal.setColor(QPalette.ColorRole.ButtonText, QColor(230, 233, 239))
    pal.setColor(QPalette.ColorRole.Highlight, QColor(47, 111, 219))
    pal.setColor(QPalette.ColorRole.HighlightedText, QColor(255, 255, 255))
    pal.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text, QColor(107, 113, 128))
    pal.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.ButtonText, QColor(107, 113, 128))
    app.setPalette(pal)
    app.setStyleSheet(DARK_QSS)


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
# Janela principal
# --------------------------------------------------------------------------- #
class MainWindow(QMainWindow):
    """Interface principal: entradas, parametros, preview, progresso e logs."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(f"{__app_name__} v{__version__}")
        self.resize(1360, 860)
        self.settings = QSettings("InoSlice", "InoSlice")

        self.dataset: AnnotationDataset | None = None
        self.load_worker: LoadWorker | None = None
        self.harvest_worker: HarvestWorker | None = None
        self.last_result: HarvestResult | None = None

        self.log_bridge = LogBridge()
        self.log_bridge.message.connect(self._append_log)
        self._log_handler = attach_callback(lambda msg, lvl: self.log_bridge.message.emit(msg, lvl))

        self._preview_timer = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(120)
        self._preview_timer.timeout.connect(self._push_config_to_preview)

        self._build_ui()
        self._restore_settings()
        self._update_state()
        log.info("%s v%s pronto. Selecione as pastas e clique em 'Carregar anotacoes'.", __app_name__, __version__)

    # ================================================================== UI
    def _build_ui(self) -> None:
        central = QWidget()
        central.setObjectName("central")
        self.setCentralWidget(central)

        logo = QLabel()
        logo_pix = QPixmap(str(asset_path("logo.png")))
        if not logo_pix.isNull():
            dpr = self.devicePixelRatioF()
            logo_pix = logo_pix.scaled(
                int(48 * dpr), int(48 * dpr), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
            )
            logo_pix.setDevicePixelRatio(dpr)
            logo.setPixmap(logo_pix)
        title = QLabel(__app_name__)
        title.setObjectName("title")
        subtitle = QLabel(
            "Deteccao → Classificacao: recortes higienizados, letterbox sem distorcao e manifesto anti-leakage"
        )
        subtitle.setObjectName("subtitle")
        header_text = QVBoxLayout()
        header_text.setSpacing(0)
        header_text.addWidget(title)
        header_text.addWidget(subtitle)
        header = QHBoxLayout()
        header.setSpacing(12)
        header.addWidget(logo)
        header.addLayout(header_text, 1)

        # ---------------- painel esquerdo (controles) ----------------
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 6, 0)
        left_layout.addWidget(self._build_inputs_group())
        left_layout.addWidget(self._build_params_group())
        left_layout.addWidget(self._build_output_group())
        left_layout.addStretch(1)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(left)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setMinimumWidth(left.sizeHint().width() + 24)

        # ---------------- painel direito (preview + progresso + logs) ----------------
        self.preview = PreviewWidget()
        preview_box = QGroupBox("Pre-visualizacao interativa")
        pv = QVBoxLayout(preview_box)
        pv.addWidget(self.preview)

        run_box = QGroupBox("Execucao")
        rb = QVBoxLayout(run_box)
        self.run_btn = QPushButton("▶  Extrair recortes")
        self.run_btn.setObjectName("primary")
        self.run_btn.clicked.connect(self._start_harvest)
        self.cancel_btn = QPushButton("■  Cancelar")
        self.cancel_btn.setObjectName("danger")
        self.cancel_btn.clicked.connect(self._cancel_harvest)
        self.open_out_btn = QPushButton("📂 Abrir saida")
        self.open_out_btn.clicked.connect(self._open_output)
        btns = QHBoxLayout()
        btns.addWidget(self.run_btn, 2)
        btns.addWidget(self.cancel_btn, 1)
        btns.addWidget(self.open_out_btn, 1)
        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        self.progress.setValue(0)
        self.progress.setFormat("%p%")
        self.stats_label = QLabel("Imagens: 0/0  |  Crops: 0  |  Velocidade: 0.0 crops/s  |  Tempo: 0.0 s")
        self.stats_label.setObjectName("stat")
        rb.addLayout(btns)
        rb.addWidget(self.progress)
        rb.addWidget(self.stats_label)

        log_box = QGroupBox("Console de logs")
        lb = QVBoxLayout(log_box)
        self.console = QPlainTextEdit()
        self.console.setObjectName("console")
        self.console.setReadOnly(True)
        self.console.setMaximumBlockCount(5000)
        clear_btn = QPushButton("Limpar")
        clear_btn.clicked.connect(self.console.clear)
        lb.addWidget(self.console)
        lb.addWidget(clear_btn, 0, Qt.AlignmentFlag.AlignRight)

        right_split = QSplitter(Qt.Orientation.Vertical)
        right_split.addWidget(preview_box)
        right_split.addWidget(run_box)
        right_split.addWidget(log_box)
        right_split.setStretchFactor(0, 5)
        right_split.setStretchFactor(1, 0)
        right_split.setStretchFactor(2, 2)

        main_split = QSplitter(Qt.Orientation.Horizontal)
        main_split.addWidget(scroll)
        main_split.addWidget(right_split)
        main_split.setStretchFactor(0, 0)
        main_split.setStretchFactor(1, 1)
        main_split.setSizes([scroll.minimumWidth(), 900])

        root = QVBoxLayout(central)
        root.setContentsMargins(14, 10, 14, 12)
        root.addLayout(header)
        root.addWidget(main_split, 1)
        self.statusBar().showMessage("Pronto")

    def _path_row(self, edit: QLineEdit, *buttons: QPushButton) -> QWidget:
        w = QWidget()
        h = QHBoxLayout(w)
        h.setContentsMargins(0, 0, 0, 0)
        h.addWidget(edit, 1)
        for b in buttons:
            b.setMinimumWidth(64)
            h.addWidget(b)
        return w

    def _build_inputs_group(self) -> QGroupBox:
        box = QGroupBox("1. Entradas")
        form = QFormLayout(box)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

        self.images_edit = QLineEdit()
        self.images_edit.setPlaceholderText("Pasta com .jpg/.png/.bmp/.tif")
        b_img = QPushButton("Pasta")
        b_img.clicked.connect(lambda: self._pick_dir(self.images_edit, "Selecione a pasta de imagens"))

        self.labels_edit = QLineEdit()
        self.labels_edit.setPlaceholderText("Pasta de .txt (YOLO) ou arquivo .json (COCO)")
        b_lbl_dir = QPushButton("Pasta")
        b_lbl_dir.clicked.connect(lambda: self._pick_dir(self.labels_edit, "Selecione a pasta de labels YOLO"))
        b_lbl_json = QPushButton("JSON")
        b_lbl_json.clicked.connect(
            lambda: self._pick_file(self.labels_edit, "Selecione o arquivo COCO", "COCO JSON (*.json)")
        )

        self.classes_edit = QLineEdit()
        self.classes_edit.setPlaceholderText("Opcional: classes.txt / data.yaml (auto-detectado)")
        b_cls = QPushButton("Arquivo")
        b_cls.clicked.connect(
            lambda: self._pick_file(
                self.classes_edit, "Selecione o arquivo de classes", "Classes (*.txt *.yaml *.yml *.names *.json)"
            )
        )

        self.load_btn = QPushButton("⟳  Carregar anotacoes")
        self.load_btn.clicked.connect(self._load_annotations)
        self.dataset_label = QLabel("Nenhuma anotacao carregada.")
        self.dataset_label.setObjectName("caption")
        self.dataset_label.setWordWrap(True)

        form.addRow("Imagens:", self._path_row(self.images_edit, b_img))
        form.addRow("Labels:", self._path_row(self.labels_edit, b_lbl_dir, b_lbl_json))
        form.addRow("Classes:", self._path_row(self.classes_edit, b_cls))
        form.addRow(self.load_btn)
        form.addRow(self.dataset_label)
        return box

    def _build_params_group(self) -> QGroupBox:
        box = QGroupBox("2. Parametros geometricos")
        form = QFormLayout(box)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

        # padding
        self.padding_slider = QSlider(Qt.Orientation.Horizontal)
        self.padding_slider.setRange(0, 30)
        self.padding_slider.setValue(10)
        self.padding_spin = QDoubleSpinBox()
        self.padding_spin.setRange(0.0, 100.0)
        self.padding_spin.setDecimals(1)
        self.padding_spin.setSuffix(" %")
        self.padding_spin.setValue(10.0)
        self.padding_spin.setToolTip("Margem por lado, proporcional a largura/altura da caixa (aceita > 30%).")
        self.padding_slider.valueChanged.connect(lambda v: self.padding_spin.setValue(float(v)))
        self.padding_spin.valueChanged.connect(self._on_padding_spin)
        pad_row = QWidget()
        ph = QHBoxLayout(pad_row)
        ph.setContentsMargins(0, 0, 0, 0)
        ph.addWidget(self.padding_slider, 1)
        ph.addWidget(self.padding_spin)
        form.addRow("Margem de contexto:", pad_row)

        # modo
        self.mode_rect = QRadioButton("Retangular original")
        self.mode_lb_gray = QRadioButton("Quadrado letterbox · cinza (114)")
        self.mode_lb_black = QRadioButton("Quadrado letterbox · preto (0)")
        self.mode_rect.setToolTip("Mantem a proporcao exata da caixa recortada (com padding).")
        self.mode_lb_gray.setToolTip("Quadrado L x L preenchido com cinza neutro RGB(114,114,114).")
        self.mode_lb_black.setToolTip("Quadrado L x L preenchido com preto RGB(0,0,0).")
        self.mode_lb_gray.setChecked(True)
        self.mode_group = QButtonGroup(self)
        mode_col = QWidget()
        mv = QVBoxLayout(mode_col)
        mv.setContentsMargins(0, 0, 0, 0)
        for i, rb in enumerate((self.mode_rect, self.mode_lb_gray, self.mode_lb_black)):
            self.mode_group.addButton(rb, i)
            mv.addWidget(rb)
        self.mode_group.idToggled.connect(lambda *_: self._schedule_preview())
        form.addRow("Modo de recorte:", mode_col)

        # tamanho minimo
        self.min_w_spin = QSpinBox()
        self.min_h_spin = QSpinBox()
        for s in (self.min_w_spin, self.min_h_spin):
            s.setRange(0, 10000)
            s.setValue(32)
            s.setSuffix(" px")
            s.valueChanged.connect(self._schedule_preview)
        min_row = QWidget()
        mh = QHBoxLayout(min_row)
        mh.setContentsMargins(0, 0, 0, 0)
        mh.addWidget(self.min_w_spin)
        mh.addWidget(QLabel("×"))
        mh.addWidget(self.min_h_spin)
        form.addRow("Tamanho minimo (L×A):", min_row)

        # resize
        self.target_combo = QComboBox()
        self.target_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.target_combo.setMinimumContentsLength(18)
        for label, size in TARGET_SIZES:
            self.target_combo.addItem(label, size)
        self.custom_size_check = QCheckBox("Personalizado:")
        self.custom_size_spin = QSpinBox()
        self.custom_size_spin.setRange(16, 4096)
        self.custom_size_spin.setValue(320)
        self.custom_size_spin.setSuffix(" px")
        self.custom_size_spin.setEnabled(False)
        self.custom_size_check.toggled.connect(self.custom_size_spin.setEnabled)
        self.custom_size_check.toggled.connect(lambda c: self.target_combo.setEnabled(not c))
        self.custom_size_check.toggled.connect(self._schedule_preview)
        self.custom_size_spin.valueChanged.connect(self._schedule_preview)
        self.target_combo.currentIndexChanged.connect(self._schedule_preview)
        custom_row = QWidget()
        ch = QHBoxLayout(custom_row)
        ch.setContentsMargins(0, 0, 0, 0)
        ch.addWidget(self.custom_size_check)
        ch.addWidget(self.custom_size_spin, 1)
        form.addRow("Redimensionamento final:", self.target_combo)
        form.addRow("", custom_row)

        # confianca
        self.conf_spin = QDoubleSpinBox()
        self.conf_spin.setRange(0.0, 1.0)
        self.conf_spin.setSingleStep(0.05)
        self.conf_spin.setDecimals(2)
        self.conf_spin.setValue(0.0)
        self.conf_spin.setToolTip("Aplicado apenas a predicoes que possuem confianca (YOLO 6 colunas / COCO score).")
        self.conf_spin.valueChanged.connect(self._schedule_preview)
        form.addRow("Confianca minima:", self.conf_spin)
        return box

    def _build_output_group(self) -> QGroupBox:
        box = QGroupBox("3. Saida")
        form = QFormLayout(box)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        self.output_edit = QLineEdit()
        self.output_edit.setPlaceholderText("Pasta onde serao criados dataset_crops/, CSV e JSON")
        b_out = QPushButton("Pasta")
        b_out.clicked.connect(lambda: self._pick_dir(self.output_edit, "Selecione a pasta de saida"))
        self.output_edit.textChanged.connect(self._update_state)
        self.format_combo = QComboBox()
        self.format_combo.addItem("JPEG (.jpg)", ".jpg")
        self.format_combo.addItem("PNG (.png, sem perdas)", ".png")
        self.quality_spin = QSpinBox()
        self.quality_spin.setRange(50, 100)
        self.quality_spin.setValue(95)
        self.format_combo.currentIndexChanged.connect(
            lambda _: self.quality_spin.setEnabled(self.format_combo.currentData() == ".jpg")
        )
        self.workers_spin = QSpinBox()
        self.workers_spin.setRange(1, max(1, (os.cpu_count() or 1) * 2))
        self.workers_spin.setValue(min(8, os.cpu_count() or 1))
        form.addRow("Pasta de saida:", self._path_row(self.output_edit, b_out))
        form.addRow("Formato:", self.format_combo)
        form.addRow("Qualidade JPEG:", self.quality_spin)
        form.addRow("Threads:", self.workers_spin)
        return box

    # ================================================================== helpers
    def _pick_dir(self, edit: QLineEdit, title: str) -> None:
        start = edit.text() or str(self.settings.value("last_dir", str(Path.home())))
        path = QFileDialog.getExistingDirectory(self, title, start)
        if path:
            edit.setText(path)
            self.settings.setValue("last_dir", path)

    def _pick_file(self, edit: QLineEdit, title: str, flt: str) -> None:
        start = edit.text() or str(self.settings.value("last_dir", str(Path.home())))
        path, _ = QFileDialog.getOpenFileName(self, title, start, flt)
        if path:
            edit.setText(path)
            self.settings.setValue("last_dir", str(Path(path).parent))

    def _on_padding_spin(self, value: float) -> None:
        self.padding_slider.blockSignals(True)
        self.padding_slider.setValue(int(round(min(value, 30.0))))
        self.padding_slider.blockSignals(False)
        self._schedule_preview()

    def _target_size(self) -> int | None:
        if self.custom_size_check.isChecked():
            return int(self.custom_size_spin.value())
        data = self.target_combo.currentData()
        return int(data) if data else None

    def current_config(self) -> ProcessingConfig:
        """Monta a :class:`ProcessingConfig` a partir dos controles da interface."""
        mode_id = self.mode_group.checkedId()
        return ProcessingConfig(
            padding_pct=float(self.padding_spin.value()),
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
            logging.DEBUG: "#7c8597",
            logging.INFO: "#c9d1e0",
            logging.WARNING: "#f0b85a",
            logging.ERROR: "#ff6b6b",
            logging.CRITICAL: "#ff6b6b",
        }.get(level, "#c9d1e0")
        fmt = QTextCharFormat()
        fmt.setForeground(QColor(color))
        if level >= logging.ERROR:
            fmt.setFontWeight(QFont.Weight.Bold)
        cursor = self.console.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        cursor.insertText(message + "\n", fmt)
        self.console.setTextCursor(cursor)
        self.console.ensureCursorVisible()

    def _busy(self) -> bool:
        return bool(
            (self.harvest_worker and self.harvest_worker.isRunning())
            or (self.load_worker and self.load_worker.isRunning())
        )

    def _update_state(self) -> None:
        harvesting = bool(self.harvest_worker and self.harvest_worker.isRunning())
        busy = self._busy()
        self.run_btn.setEnabled(not busy and self.dataset is not None and bool(self.output_edit.text().strip()))
        self.cancel_btn.setEnabled(harvesting)
        self.load_btn.setEnabled(not busy)
        out = self.output_edit.text().strip()
        self.open_out_btn.setEnabled(bool(out) and Path(out).is_dir())

    # ================================================================== acoes
    def _load_annotations(self) -> None:
        try:
            images = validate_directory(self.images_edit.text().strip(), "Imagens")
            labels = self.labels_edit.text().strip()
            if not labels or not Path(labels).exists():
                raise ValidationError("Labels: selecione uma pasta YOLO ou um arquivo COCO .json existente.")
        except ValidationError as exc:
            QMessageBox.warning(self, "Entradas invalidas", str(exc))
            return
        class_file = self.classes_edit.text().strip() or None
        log.info("Carregando anotacoes de %s ...", labels)
        self.statusBar().showMessage("Carregando anotacoes...")
        self.load_worker = LoadWorker(str(images), labels, class_file)
        self.load_worker.loaded.connect(self._on_loaded)
        self.load_worker.failed.connect(self._on_load_failed)
        self.load_worker.finished.connect(self._update_state)
        self.load_worker.start()
        self._update_state()

    def _on_loaded(self, dataset: AnnotationDataset) -> None:
        self.dataset = dataset
        counts: dict[str, int] = {}
        for it in dataset.items:
            for b in it.boxes:
                name = dataset.class_name(b.class_id)
                counts[name] = counts.get(name, 0) + 1
        top = ", ".join(f"{k}: {v}" for k, v in sorted(counts.items(), key=lambda kv: -kv[1])[:6])
        self.dataset_label.setText(
            f"Formato {dataset.format.upper()} • {len(dataset.items)} imagens • {dataset.total_boxes} caixas • "
            f"{len(counts)} classes\n{top}{' ...' if len(counts) > 6 else ''}"
        )
        for w in dataset.warnings[:50]:
            log.warning(w)
        if len(dataset.warnings) > 50:
            log.warning("... e mais %d avisos de parsing.", len(dataset.warnings) - 50)
        if not dataset.class_names:
            log.warning("Nenhum mapeamento de classes encontrado: pastas usarao 'class_<id>'.")
        self.preview.set_config(self.current_config())
        self.preview.set_dataset(dataset)
        self.statusBar().showMessage(f"{dataset.total_boxes} caixas carregadas")
        self._save_settings()
        self._update_state()

    def _on_load_failed(self, message: str) -> None:
        log.error("Falha ao carregar anotacoes: %s", message)
        self.statusBar().showMessage("Falha ao carregar anotacoes")
        QMessageBox.critical(self, "Erro ao carregar anotacoes", message)

    def _start_harvest(self) -> None:
        if self.dataset is None:
            return
        try:
            out_dir = validate_directory(self.output_edit.text().strip(), "Saida", must_exist=False)
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
            image_ext=str(self.format_combo.currentData()),
            source_info={
                "labels_path": self.labels_edit.text().strip(),
                "class_file": self.classes_edit.text().strip() or None,
            },
        )
        self.harvest_worker = HarvestWorker(harvester)
        self.harvest_worker.progress.connect(self._on_progress)
        self.harvest_worker.finished_ok.connect(self._on_harvest_done)
        self.harvest_worker.failed.connect(self._on_harvest_failed)
        self.harvest_worker.finished.connect(self._update_state)
        self.progress.setValue(0)
        self._save_settings()
        self.harvest_worker.start()
        self.statusBar().showMessage("Extraindo recortes...")
        self._update_state()

    def _cancel_harvest(self) -> None:
        if self.harvest_worker and self.harvest_worker.isRunning():
            log.warning("Cancelamento solicitado; aguardando imagens em andamento...")
            self.harvest_worker.cancel_event.set()
            self.cancel_btn.setEnabled(False)

    def _on_progress(self, p: HarvestProgress) -> None:
        self.progress.setValue(int(p.fraction * 1000))
        self.stats_label.setText(
            f"Imagens: {p.images_done}/{p.images_total}  |  Crops: {p.crops_done}  |  "
            f"Velocidade: {p.crops_per_second:.1f} crops/s  |  Tempo: {p.elapsed:.1f} s"
        )

    def _on_harvest_done(self, result: HarvestResult) -> None:
        self.last_result = result
        totals = result.summary["totals"]
        assert isinstance(totals, dict)
        msg = (
            f"{totals['crops_extracted']} recortes extraidos de {totals['images_processed']} imagens.\n"
            f"Descartados: {totals['discarded_min_size']} (tamanho minimo), "
            f"{totals['discarded_low_confidence']} (confianca), {totals['discarded_invalid_box']} (invalidos).\n\n"
            f"Manifesto: {result.manifest_path}\nResumo: {result.summary_path}"
        )
        self.statusBar().showMessage("Colheita cancelada" if result.cancelled else "Colheita concluida")
        if result.cancelled:
            QMessageBox.warning(self, "Colheita cancelada", "Execucao cancelada (resultados parciais salvos).\n\n" + msg)
        else:
            self.progress.setValue(1000)
            QMessageBox.information(self, "Colheita concluida", msg)
        self._update_state()

    def _on_harvest_failed(self, message: str) -> None:
        self.statusBar().showMessage("Falha na colheita")
        QMessageBox.critical(self, "Erro na colheita", message)

    def _open_output(self) -> None:
        out = self.output_edit.text().strip()
        if out and Path(out).is_dir():
            QDesktopServices.openUrl(QUrl.fromLocalFile(out))

    # ================================================================== persistencia
    def _save_settings(self) -> None:
        s = self.settings
        s.setValue("images", self.images_edit.text())
        s.setValue("labels", self.labels_edit.text())
        s.setValue("classes", self.classes_edit.text())
        s.setValue("output", self.output_edit.text())
        s.setValue("padding", self.padding_spin.value())
        s.setValue("mode", self.mode_group.checkedId())
        s.setValue("min_w", self.min_w_spin.value())
        s.setValue("min_h", self.min_h_spin.value())
        s.setValue("target_idx", self.target_combo.currentIndex())
        s.setValue("custom_on", self.custom_size_check.isChecked())
        s.setValue("custom_size", self.custom_size_spin.value())
        s.setValue("conf", self.conf_spin.value())
        s.setValue("format_idx", self.format_combo.currentIndex())
        s.setValue("quality", self.quality_spin.value())

    def _restore_settings(self) -> None:
        s = self.settings

        def get(key: str, default: object, typ: type) -> object:
            try:
                return s.value(key, default, type=typ)
            except (TypeError, ValueError):
                return default

        self.images_edit.setText(str(get("images", "", str)))
        self.labels_edit.setText(str(get("labels", "", str)))
        self.classes_edit.setText(str(get("classes", "", str)))
        self.output_edit.setText(str(get("output", "", str)))
        self.padding_spin.setValue(float(get("padding", 10.0, float)))  # type: ignore[arg-type]
        btn = self.mode_group.button(int(get("mode", 1, int)))  # type: ignore[arg-type]
        if btn is not None:
            btn.setChecked(True)
        self.min_w_spin.setValue(int(get("min_w", 32, int)))  # type: ignore[arg-type]
        self.min_h_spin.setValue(int(get("min_h", 32, int)))  # type: ignore[arg-type]
        self.target_combo.setCurrentIndex(int(get("target_idx", 0, int)))  # type: ignore[arg-type]
        self.custom_size_check.setChecked(bool(get("custom_on", False, bool)))
        self.custom_size_spin.setValue(int(get("custom_size", 320, int)))  # type: ignore[arg-type]
        self.conf_spin.setValue(float(get("conf", 0.0, float)))  # type: ignore[arg-type]
        self.format_combo.setCurrentIndex(int(get("format_idx", 0, int)))  # type: ignore[arg-type]
        self.quality_spin.setValue(int(get("quality", 95, int)))  # type: ignore[arg-type]

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
    """Cria a ``QApplication``, aplica o tema escuro e exibe a janela principal."""

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
    apply_dark_theme(app)  # type: ignore[arg-type]
    window = MainWindow()
    window.show()
    return int(app.exec())
