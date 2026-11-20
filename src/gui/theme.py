"""Tema visual do InoSlice, derivado da paleta da logo (azul-marinho, azul e laranja)."""

from __future__ import annotations

from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication

# Paleta extraida da logo -------------------------------------------------- #
NAVY_950 = "#0a1422"  # fundo da janela
NAVY_900 = "#0f1d31"  # barra lateral / cabecalho
NAVY_850 = "#13243c"  # cartoes
NAVY_800 = "#183052"  # campos de entrada
NAVY_700 = "#1f3d66"  # bordas e trilhos
NAVY_600 = "#2a4f80"  # bordas em hover
BLUE_500 = "#2e7cd6"  # azul da logo (destaques secundarios)
BLUE_400 = "#4d95e6"
ORANGE_500 = "#f7931e"  # laranja da logo (acao principal)
ORANGE_400 = "#ffa940"
ORANGE_600 = "#d97a0b"
TEXT = "#e8eef7"
TEXT_MUTED = "#8ea3c0"
TEXT_FAINT = "#5d7392"
SUCCESS = "#3ccf91"
WARNING = "#f5b942"
DANGER = "#f0605d"

QSS = f"""
* {{ font-family: "Segoe UI", "Inter", "Helvetica Neue", Arial, sans-serif; font-size: 10pt; color: {TEXT}; }}
QMainWindow, QWidget#central {{ background: {NAVY_950}; }}
QToolTip {{ background: {NAVY_800}; color: {TEXT}; border: 1px solid {NAVY_600}; padding: 6px; border-radius: 6px; }}

/* ---------- cabecalho ---------- */
QFrame#header {{ background: {NAVY_900}; border-bottom: 1px solid {NAVY_700}; }}
QLabel#appName {{ font-size: 16pt; font-weight: 700; letter-spacing: 0.5px; }}
QLabel#appTag {{ color: {TEXT_MUTED}; font-size: 9pt; }}
QLabel#versionPill {{
    background: {NAVY_800}; color: {BLUE_400}; border: 1px solid {NAVY_700};
    border-radius: 9px; padding: 2px 9px; font-size: 8.5pt; font-weight: 600;
}}

/* ---------- barra lateral e cartoes ---------- */
QWidget#sidebar {{ background: transparent; }}
QScrollArea#sidebarScroll {{ background: transparent; border: none; }}
QScrollArea#sidebarScroll > QWidget > QWidget {{ background: transparent; }}
QFrame#card {{ background: {NAVY_850}; border: 1px solid {NAVY_700}; border-radius: 12px; }}
QLabel#cardStep {{
    background: {ORANGE_500}; color: {NAVY_950}; border-radius: 11px; font-weight: 800; font-size: 9pt;
    min-width: 22px; max-width: 22px; min-height: 22px; max-height: 22px;
}}
QLabel#cardStep[done="true"] {{ background: {SUCCESS}; }}
QLabel#cardTitle {{ font-size: 11pt; font-weight: 700; }}
QLabel#cardHint {{ color: {TEXT_MUTED}; font-size: 9pt; }}
QLabel#fieldLabel {{ color: {TEXT_MUTED}; font-size: 9pt; font-weight: 600; }}
QLabel#valueBadge {{
    background: {NAVY_800}; color: {ORANGE_400}; border-radius: 6px; padding: 2px 8px; font-weight: 700;
}}
QLabel#muted {{ color: {TEXT_MUTED}; }}
QLabel#faint {{ color: {TEXT_FAINT}; font-size: 9pt; }}

/* ---------- chips de resumo ---------- */
QLabel#chip {{
    background: {NAVY_800}; border: 1px solid {NAVY_700}; border-radius: 10px; padding: 3px 10px;
    color: {TEXT}; font-size: 9pt;
}}
QLabel#chipAccent {{
    background: rgba(247,147,30,0.14); border: 1px solid rgba(247,147,30,0.45); border-radius: 10px;
    padding: 3px 10px; color: {ORANGE_400}; font-size: 9pt; font-weight: 600;
}}
QLabel#chipOk {{
    background: rgba(60,207,145,0.12); border: 1px solid rgba(60,207,145,0.45); border-radius: 10px;
    padding: 3px 10px; color: {SUCCESS}; font-size: 9pt; font-weight: 600;
}}
QLabel#chipBad {{
    background: rgba(240,96,93,0.12); border: 1px solid rgba(240,96,93,0.45); border-radius: 10px;
    padding: 3px 10px; color: {DANGER}; font-size: 9pt; font-weight: 600;
}}

/* ---------- campos ---------- */
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
    background: {NAVY_800}; border: 1px solid {NAVY_700}; border-radius: 8px; padding: 6px 9px;
    selection-background-color: {BLUE_500}; min-height: 18px;
}}
QLineEdit:hover, QSpinBox:hover, QDoubleSpinBox:hover, QComboBox:hover {{ border-color: {NAVY_600}; }}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{ border-color: {ORANGE_500}; }}
QLineEdit:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled {{ color: {TEXT_FAINT}; background: {NAVY_850}; }}
QLineEdit[valid="true"] {{ border-color: rgba(60,207,145,0.55); }}
QLineEdit[valid="false"] {{ border-color: rgba(240,96,93,0.7); }}
QLineEdit[dragOver="true"] {{ border: 2px dashed {ORANGE_500}; background: rgba(247,147,30,0.08); }}
QSpinBox::up-button, QSpinBox::down-button, QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {{
    width: 0px; border: none;
}}
QComboBox::drop-down {{ border: none; width: 22px; }}
QComboBox QAbstractItemView {{
    background: {NAVY_800}; border: 1px solid {NAVY_600}; border-radius: 6px;
    selection-background-color: {BLUE_500}; outline: none; padding: 4px;
}}

/* ---------- botoes ---------- */
QPushButton, QToolButton {{
    background: {NAVY_800}; border: 1px solid {NAVY_700}; border-radius: 8px; padding: 7px 12px; font-weight: 600;
}}
QPushButton:hover, QToolButton:hover {{ background: {NAVY_700}; border-color: {NAVY_600}; }}
QPushButton:pressed, QToolButton:pressed {{ background: {NAVY_850}; }}
QPushButton:disabled, QToolButton:disabled {{ color: {TEXT_FAINT}; background: {NAVY_850}; border-color: {NAVY_800}; }}
QPushButton#primary {{
    background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 {ORANGE_500}, stop:1 {ORANGE_400});
    color: {NAVY_950}; border: none; border-radius: 10px; padding: 11px 22px; font-size: 11pt; font-weight: 800;
}}
QPushButton#primary:hover {{ background: {ORANGE_400}; }}
QPushButton#primary:pressed {{ background: {ORANGE_600}; }}
QPushButton#primary:disabled {{ background: {NAVY_800}; color: {TEXT_FAINT}; }}
QPushButton#ghost {{ background: transparent; border: 1px solid transparent; color: {TEXT_MUTED}; }}
QPushButton#ghost:hover {{ background: {NAVY_800}; color: {TEXT}; }}
QPushButton#ghost:checked {{ color: {ORANGE_400}; }}
QPushButton#danger {{ background: transparent; border: 1px solid rgba(240,96,93,0.55); color: {DANGER}; }}
QPushButton#danger:hover {{ background: rgba(240,96,93,0.12); }}
QPushButton#danger:disabled {{ border-color: {NAVY_800}; color: {TEXT_FAINT}; }}
QPushButton#browse {{ padding: 6px 10px; min-width: 34px; }}

/* controle segmentado */
QWidget#segmented {{ background: {NAVY_800}; border: 1px solid {NAVY_700}; border-radius: 9px; }}
QPushButton#segment {{
    background: transparent; border: none; border-radius: 7px; padding: 6px 8px; color: {TEXT_MUTED}; font-weight: 600;
}}
QPushButton#segment:hover {{ color: {TEXT}; background: {NAVY_700}; }}
QPushButton#segment:checked {{ background: {BLUE_500}; color: white; }}
QPushButton#segment:disabled {{ color: {TEXT_FAINT}; }}

/* ---------- slider ---------- */
QSlider::groove:horizontal {{ height: 6px; background: {NAVY_700}; border-radius: 3px; }}
QSlider::sub-page:horizontal {{
    background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 {BLUE_500}, stop:1 {ORANGE_500}); border-radius: 3px;
}}
QSlider::handle:horizontal {{
    background: {TEXT}; border: 3px solid {ORANGE_500}; width: 12px; height: 12px; margin: -6px 0; border-radius: 9px;
}}
QSlider::handle:horizontal:hover {{ border-color: {ORANGE_400}; }}

/* ---------- checkbox ---------- */
QCheckBox {{ spacing: 8px; color: {TEXT_MUTED}; }}
QCheckBox::indicator {{ width: 16px; height: 16px; border-radius: 4px; border: 1px solid {NAVY_600}; background: {NAVY_800}; }}
QCheckBox::indicator:checked {{ background: {ORANGE_500}; border-color: {ORANGE_500}; }}

/* ---------- progresso ---------- */
QProgressBar {{
    background: {NAVY_800}; border: none; border-radius: 5px; height: 10px; max-height: 10px; text-align: center;
    color: transparent;
}}
QProgressBar::chunk {{
    background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 {BLUE_500}, stop:1 {ORANGE_500}); border-radius: 5px;
}}

/* ---------- preview ---------- */
QLabel#canvas {{ background: {NAVY_950}; border: 1px solid {NAVY_700}; border-radius: 10px; color: {TEXT_FAINT}; }}
QLabel#overlayBadge {{
    background: rgba(10,20,34,0.82); border: 1px solid {NAVY_700}; border-radius: 7px; padding: 3px 8px;
    color: {TEXT}; font-size: 9pt; font-weight: 600;
}}
QLabel#legendGreen {{ color: {SUCCESS}; font-size: 9pt; font-weight: 600; }}
QLabel#legendOrange {{ color: {ORANGE_400}; font-size: 9pt; font-weight: 600; }}
QLabel#mono {{ font-family: "Cascadia Mono", Consolas, monospace; font-size: 9pt; color: {TEXT_MUTED}; }}

/* ---------- console ---------- */
QPlainTextEdit#console {{
    background: {NAVY_950}; border: 1px solid {NAVY_700}; border-radius: 10px; padding: 6px;
    font-family: "Cascadia Mono", Consolas, monospace; font-size: 9pt; selection-background-color: {BLUE_500};
}}

/* ---------- barras de rolagem / divisores / status ---------- */
QScrollBar:vertical {{ background: transparent; width: 10px; margin: 2px; }}
QScrollBar::handle:vertical {{ background: {NAVY_700}; border-radius: 4px; min-height: 30px; }}
QScrollBar::handle:vertical:hover {{ background: {NAVY_600}; }}
QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 2px; }}
QScrollBar::handle:horizontal {{ background: {NAVY_700}; border-radius: 4px; min-width: 30px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}
QSplitter::handle {{ background: transparent; }}
QSplitter::handle:hover {{ background: {NAVY_700}; }}
QStatusBar {{ background: {NAVY_900}; border-top: 1px solid {NAVY_700}; color: {TEXT_MUTED}; font-size: 9pt; }}
QStatusBar QLabel {{ color: {TEXT_MUTED}; font-size: 9pt; }}
QMessageBox {{ background: {NAVY_850}; }}
QMessageBox QPushButton {{ min-width: 80px; }}
"""


def apply_theme(app: QApplication) -> None:
    """Aplica o estilo Fusion com a paleta escura da marca e a folha de estilos."""
    app.setStyle("Fusion")
    pal = QPalette()
    pal.setColor(QPalette.ColorRole.Window, QColor(NAVY_950))
    pal.setColor(QPalette.ColorRole.WindowText, QColor(TEXT))
    pal.setColor(QPalette.ColorRole.Base, QColor(NAVY_800))
    pal.setColor(QPalette.ColorRole.AlternateBase, QColor(NAVY_850))
    pal.setColor(QPalette.ColorRole.ToolTipBase, QColor(NAVY_800))
    pal.setColor(QPalette.ColorRole.ToolTipText, QColor(TEXT))
    pal.setColor(QPalette.ColorRole.Text, QColor(TEXT))
    pal.setColor(QPalette.ColorRole.PlaceholderText, QColor(TEXT_FAINT))
    pal.setColor(QPalette.ColorRole.Button, QColor(NAVY_800))
    pal.setColor(QPalette.ColorRole.ButtonText, QColor(TEXT))
    pal.setColor(QPalette.ColorRole.Highlight, QColor(BLUE_500))
    pal.setColor(QPalette.ColorRole.HighlightedText, QColor("#ffffff"))
    pal.setColor(QPalette.ColorRole.Link, QColor(ORANGE_400))
    for role in (QPalette.ColorRole.Text, QPalette.ColorRole.ButtonText, QPalette.ColorRole.WindowText):
        pal.setColor(QPalette.ColorGroup.Disabled, role, QColor(TEXT_FAINT))
    app.setPalette(pal)
    app.setStyleSheet(QSS)
