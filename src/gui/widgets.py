"""Componentes visuais reutilizaveis da interface do InoSlice."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QDragEnterEvent, QDragLeaveEvent, QDropEvent
from PySide6.QtWidgets import (
    QButtonGroup,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)


def repolish(widget: QWidget) -> None:
    """Reaplica a folha de estilos apos mudar uma propriedade dinamica."""
    style = widget.style()
    style.unpolish(widget)
    style.polish(widget)
    widget.update()


class Card(QFrame):
    """Cartao com numero da etapa, titulo, dica e area de conteudo."""

    def __init__(self, step: str, title: str, hint: str = "", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("card")
        self._step = step
        self.step_label = QLabel(step)
        self.step_label.setObjectName("cardStep")
        self.step_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label = QLabel(title)
        title_label.setObjectName("cardTitle")

        head = QHBoxLayout()
        head.setSpacing(10)
        head.addWidget(self.step_label)
        head.addWidget(title_label, 1)
        self.head = head

        self.body = QVBoxLayout()
        self.body.setSpacing(8)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(16, 14, 16, 16)
        outer.setSpacing(10)
        outer.addLayout(head)
        if hint:
            hint_label = QLabel(hint)
            hint_label.setObjectName("cardHint")
            hint_label.setWordWrap(True)
            outer.addWidget(hint_label)
        outer.addLayout(self.body)

    def set_done(self, done: bool) -> None:
        """Marca a etapa como concluida (circulo verde com check)."""
        self.step_label.setProperty("done", "true" if done else "false")
        self.step_label.setText("✓" if done else self._step)
        repolish(self.step_label)

    def add_field(self, label: str, widget: QWidget) -> None:
        """Adiciona um campo com rotulo pequeno acima."""
        lbl = QLabel(label)
        lbl.setObjectName("fieldLabel")
        self.body.addWidget(lbl)
        self.body.addWidget(widget)


class PathField(QWidget):
    """Campo de caminho com botoes de busca e suporte a arrastar-e-soltar."""

    changed = Signal(str)

    def __init__(self, placeholder: str, buttons: list[tuple[str, str]], parent: QWidget | None = None) -> None:
        """``buttons``: lista de ``(texto, dica)``; o sinal ``browse`` emite o indice clicado."""
        super().__init__(parent)
        self.edit = _DropLineEdit()
        self.edit.setPlaceholderText(placeholder)
        self.edit.setClearButtonEnabled(True)
        self.edit.textChanged.connect(self.changed.emit)
        self.buttons: list[QPushButton] = []
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)
        lay.addWidget(self.edit, 1)
        for text, tip in buttons:
            b = QPushButton(text)
            b.setObjectName("browse")
            b.setToolTip(tip)
            b.setCursor(Qt.CursorShape.PointingHandCursor)
            self.buttons.append(b)
            lay.addWidget(b)

    def text(self) -> str:
        """Caminho atual sem espacos nas pontas."""
        return self.edit.text().strip()

    def setText(self, text: str) -> None:  # noqa: N802 - padrao Qt
        """Define o caminho exibido."""
        self.edit.setText(text)

    def set_valid(self, valid: bool | None) -> None:
        """Colore a borda: verde (valido), vermelho (invalido) ou neutra (``None``)."""
        self.edit.setProperty("valid", "" if valid is None else ("true" if valid else "false"))
        repolish(self.edit)


class _DropLineEdit(QLineEdit):
    """``QLineEdit`` que aceita arquivos/pastas arrastados do Explorer."""

    def __init__(self) -> None:
        super().__init__()
        self.setAcceptDrops(True)

    def _set_drag(self, on: bool) -> None:
        self.setProperty("dragOver", "true" if on else "false")
        repolish(self)

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:  # noqa: N802 - API Qt
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            self._set_drag(True)
        else:
            super().dragEnterEvent(event)

    def dragLeaveEvent(self, event: QDragLeaveEvent) -> None:  # noqa: N802 - API Qt
        self._set_drag(False)
        super().dragLeaveEvent(event)

    def dropEvent(self, event: QDropEvent) -> None:  # noqa: N802 - API Qt
        self._set_drag(False)
        urls = event.mimeData().urls()
        if urls and urls[0].isLocalFile():
            self.setText(str(Path(urls[0].toLocalFile())))
            event.acceptProposedAction()
        else:
            super().dropEvent(event)


class Segmented(QWidget):
    """Controle segmentado (botoes exclusivos) em uma ou mais linhas."""

    changed = Signal(int)

    def __init__(self, options: list[tuple[str, str]], columns: int = 0, parent: QWidget | None = None) -> None:
        """``options``: lista de ``(texto, dica)``. ``columns=0`` coloca tudo em uma linha."""
        super().__init__(parent)
        self.setObjectName("segmented")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.group = QButtonGroup(self)
        self.group.setExclusive(True)
        grid = QGridLayout(self)
        grid.setContentsMargins(3, 3, 3, 3)
        grid.setSpacing(3)
        cols = columns or len(options)
        for i, (text, tip) in enumerate(options):
            b = QPushButton(text)
            b.setObjectName("segment")
            b.setCheckable(True)
            b.setToolTip(tip)
            b.setCursor(Qt.CursorShape.PointingHandCursor)
            b.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            self.group.addButton(b, i)
            grid.addWidget(b, i // cols, i % cols)
        self.group.idToggled.connect(lambda i, on: on and self.changed.emit(i))

    def index(self) -> int:
        """Indice selecionado (``-1`` se nenhum)."""
        return self.group.checkedId()

    def set_index(self, idx: int) -> None:
        """Seleciona o botao ``idx`` (ignora indices invalidos)."""
        btn = self.group.button(idx)
        if btn is not None:
            btn.setChecked(True)


def chip(text: str, kind: str = "chip") -> QLabel:
    """Etiqueta arredondada (``chip``, ``chipAccent``, ``chipOk`` ou ``chipBad``)."""
    lbl = QLabel(text)
    lbl.setObjectName(kind)
    return lbl
