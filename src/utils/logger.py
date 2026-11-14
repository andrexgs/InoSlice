"""Sistema de logs e validacoes de entrada do CropHarvester Studio.

O logger padrao escreve no console e, opcionalmente, encaminha cada registro
para callbacks registrados (por exemplo, o console integrado da GUI).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Iterable

LOGGER_NAME = "crop_harvester"
_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(message)s"
_DATE_FORMAT = "%H:%M:%S"

IMAGE_EXTENSIONS: frozenset[str] = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"})


class CallbackHandler(logging.Handler):
    """Handler de logging que repassa mensagens formatadas a um callback."""

    def __init__(self, callback: Callable[[str, int], None]) -> None:
        super().__init__()
        self._callback = callback
        self.setFormatter(logging.Formatter(_LOG_FORMAT, _DATE_FORMAT))

    def emit(self, record: logging.LogRecord) -> None:  # noqa: D102
        try:
            self._callback(self.format(record), record.levelno)
        except Exception:  # pragma: no cover - nunca derrubar o app por causa de log
            self.handleError(record)


def get_logger(level: int = logging.INFO) -> logging.Logger:
    """Retorna o logger da aplicacao, configurando o handler de console uma unica vez."""
    logger = logging.getLogger(LOGGER_NAME)
    if not any(getattr(h, "_ch_console", False) for h in logger.handlers):
        console = logging.StreamHandler()
        console.setFormatter(logging.Formatter(_LOG_FORMAT, _DATE_FORMAT))
        console._ch_console = True  # type: ignore[attr-defined]
        logger.addHandler(console)
    logger.setLevel(level)
    logger.propagate = False
    return logger


def attach_callback(callback: Callable[[str, int], None]) -> CallbackHandler:
    """Registra um callback que recebera todas as mensagens do logger da aplicacao."""
    handler = CallbackHandler(callback)
    get_logger().addHandler(handler)
    return handler


def detach_handler(handler: logging.Handler) -> None:
    """Remove um handler previamente registrado."""
    get_logger().removeHandler(handler)


class ValidationError(ValueError):
    """Erro de validacao de parametros ou caminhos informados pelo usuario."""


def validate_directory(path: str | Path | None, label: str, must_exist: bool = True) -> Path:
    """Valida que ``path`` e um diretorio (existente, se ``must_exist``)."""
    if path is None or str(path).strip() == "":
        raise ValidationError(f"{label}: nenhum caminho informado.")
    p = Path(path).expanduser()
    if must_exist and not p.is_dir():
        raise ValidationError(f"{label}: diretorio inexistente -> {p}")
    if not must_exist and p.exists() and not p.is_dir():
        raise ValidationError(f"{label}: o caminho existe mas nao e um diretorio -> {p}")
    return p


def validate_file(path: str | Path | None, label: str, suffixes: Iterable[str] | None = None) -> Path:
    """Valida que ``path`` e um arquivo existente com uma das extensoes permitidas."""
    if path is None or str(path).strip() == "":
        raise ValidationError(f"{label}: nenhum arquivo informado.")
    p = Path(path).expanduser()
    if not p.is_file():
        raise ValidationError(f"{label}: arquivo inexistente -> {p}")
    if suffixes is not None:
        allowed = {s.lower() for s in suffixes}
        if p.suffix.lower() not in allowed:
            raise ValidationError(f"{label}: extensao '{p.suffix}' nao suportada (esperado: {sorted(allowed)}).")
    return p


def validate_range(value: float, low: float, high: float, label: str) -> float:
    """Garante que ``low <= value <= high``."""
    if not (low <= value <= high):
        raise ValidationError(f"{label}: valor {value} fora do intervalo [{low}, {high}].")
    return value


def is_image_file(path: Path) -> bool:
    """Indica se o arquivo possui extensao de imagem suportada."""
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
