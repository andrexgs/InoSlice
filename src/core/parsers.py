"""Leitura de anotacoes/predicoes de deteccao de objetos.

Formatos suportados
-------------------
* **YOLO (.txt)** - uma linha por objeto::

      <class_id> <x_center> <y_center> <width> <height> [confidence]

  com coordenadas normalizadas em ``[0, 1]``. Linhas de segmentacao YOLO
  (``<class_id> x1 y1 x2 y2 ... xn yn``) tambem sao aceitas e convertidas
  para a caixa envolvente do poligono.
* **Mapeamento de classes** - ``classes.txt`` / ``*.names`` (um nome por
  linha, indice = ``class_id``) ou ``data.yaml`` (chave ``names`` como lista
  ou dicionario).
* **COCO (.json)** - ``images``, ``annotations`` (``bbox`` absoluto em
  ``[x_min, y_min, width, height]``, ``score`` opcional) e ``categories``.

Todas as caixas sao entregues em coordenadas que podem ser convertidas para
pixels absolutos ``(x_min, y_min, x_max, y_max)`` via
:meth:`AnnotationBox.to_absolute`, pois o formato YOLO depende do tamanho real
da imagem, conhecido apenas no momento da leitura dos pixels.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Literal

import yaml

from src.utils.logger import IMAGE_EXTENSIONS, ValidationError, get_logger

AnnotationFormat = Literal["yolo", "coco"]

log = get_logger()


# --------------------------------------------------------------------------- #
# Estruturas de dados
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class AnnotationBox:
    """Uma caixa delimitadora associada a uma imagem.

    Attributes:
        class_id: identificador numerico da classe.
        coords: ``(x_min, y_min, x_max, y_max)``; normalizadas em ``[0, 1]`` se
            ``normalized`` for verdadeiro, caso contrario em pixels absolutos.
        normalized: indica se ``coords`` estao normalizadas (YOLO).
        confidence: confianca da predicao, quando disponivel.
    """

    class_id: int
    coords: tuple[float, float, float, float]
    normalized: bool
    confidence: float | None = None

    def to_absolute(self, img_width: int, img_height: int) -> tuple[float, float, float, float]:
        """Converte a caixa para pixels absolutos ``(x_min, y_min, x_max, y_max)``."""
        x1, y1, x2, y2 = self.coords
        if self.normalized:
            return (x1 * img_width, y1 * img_height, x2 * img_width, y2 * img_height)
        return (x1, y1, x2, y2)


@dataclass
class ImageAnnotations:
    """Conjunto de caixas de uma unica imagem de origem."""

    image_path: Path
    relative_path: Path
    boxes: list[AnnotationBox] = field(default_factory=list)
    declared_size: tuple[int, int] | None = None  # (largura, altura) informada pelo COCO


@dataclass
class AnnotationDataset:
    """Resultado do parsing: imagens, caixas e mapeamento de classes."""

    format: AnnotationFormat
    images_dir: Path
    items: list[ImageAnnotations]
    class_names: dict[int, str]
    warnings: list[str] = field(default_factory=list)
    images_without_labels: int = 0
    annotations_without_image: int = 0

    @property
    def total_boxes(self) -> int:
        """Numero total de caixas em todas as imagens."""
        return sum(len(item.boxes) for item in self.items)

    def class_name(self, class_id: int) -> str:
        """Nome legivel da classe; usa ``class_<id>`` quando nao mapeada."""
        return self.class_names.get(class_id, f"class_{class_id}")


# --------------------------------------------------------------------------- #
# Indexacao de imagens
# --------------------------------------------------------------------------- #
def list_images(images_dir: Path) -> list[Path]:
    """Lista recursivamente as imagens suportadas em ``images_dir`` (ordem deterministica)."""
    return sorted(
        p for p in images_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


# --------------------------------------------------------------------------- #
# Mapeamento de classes
# --------------------------------------------------------------------------- #
def load_class_names(path: str | Path) -> dict[int, str]:
    """Carrega o mapeamento ``class_id -> nome`` de ``classes.txt``, ``.names`` ou ``data.yaml``.

    Raises:
        ValidationError: se o arquivo nao existir ou tiver formato invalido.
    """
    p = Path(path)
    if not p.is_file():
        raise ValidationError(f"Arquivo de classes inexistente: {p}")

    suffix = p.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        with p.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        names = data.get("names") if isinstance(data, dict) else None
        if isinstance(names, list):
            return {i: str(n) for i, n in enumerate(names)}
        if isinstance(names, dict):
            return {int(k): str(v) for k, v in names.items()}
        raise ValidationError(f"'{p.name}' nao possui a chave 'names' como lista ou dicionario.")

    if suffix in {".txt", ".names", ""}:
        with p.open("r", encoding="utf-8-sig") as fh:
            lines = [ln.strip() for ln in fh.readlines()]
        # remove linhas vazias apenas do final, preservando os indices
        while lines and not lines[-1]:
            lines.pop()
        return {i: name for i, name in enumerate(lines) if name}

    if suffix == ".json":
        with p.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, dict) and "categories" in data:
            return {int(c["id"]): str(c["name"]) for c in data["categories"]}
        if isinstance(data, list):
            return {i: str(n) for i, n in enumerate(data)}
        if isinstance(data, dict):
            return {int(k): str(v) for k, v in data.items()}
        raise ValidationError(f"JSON de classes com estrutura nao reconhecida: {p}")

    raise ValidationError(f"Extensao de arquivo de classes nao suportada: {p.suffix}")


def find_class_file(*search_dirs: Path) -> Path | None:
    """Procura automaticamente ``classes.txt``, ``data.yaml`` ou ``*.names`` nos diretorios dados."""
    candidates = ("classes.txt", "data.yaml", "data.yml", "obj.names", "classes.names")
    for d in search_dirs:
        if d is None or not d.is_dir():
            continue
        for name in candidates:
            if (d / name).is_file():
                return d / name
    return None


# --------------------------------------------------------------------------- #
# YOLO
# --------------------------------------------------------------------------- #
def parse_yolo_line(line: str) -> AnnotationBox | None:
    """Converte uma linha YOLO em :class:`AnnotationBox` (coordenadas normalizadas).

    Retorna ``None`` para linhas vazias ou comentarios. Levanta ``ValueError``
    para linhas malformadas.
    """
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None
    tokens = stripped.split()
    n = len(tokens)
    class_id = int(float(tokens[0]))
    values = [float(t) for t in tokens[1:]]

    if n in (5, 6):
        xc, yc, w, h = values[:4]
        if w < 0 or h < 0:
            raise ValueError("largura/altura negativas")
        conf = values[4] if n == 6 else None
        return AnnotationBox(
            class_id=class_id,
            coords=(xc - w / 2.0, yc - h / 2.0, xc + w / 2.0, yc + h / 2.0),
            normalized=True,
            confidence=conf,
        )

    if n >= 7 and (n - 1) % 2 == 0:  # poligono de segmentacao YOLO
        xs = values[0::2]
        ys = values[1::2]
        return AnnotationBox(
            class_id=class_id,
            coords=(min(xs), min(ys), max(xs), max(ys)),
            normalized=True,
            confidence=None,
        )

    raise ValueError(f"numero de campos invalido ({n})")


def parse_yolo_file(path: Path) -> tuple[list[AnnotationBox], list[str]]:
    """Le um arquivo ``.txt`` YOLO, retornando caixas validas e avisos de linhas invalidas."""
    boxes: list[AnnotationBox] = []
    warnings: list[str] = []
    with path.open("r", encoding="utf-8-sig") as fh:
        for lineno, line in enumerate(fh, start=1):
            try:
                box = parse_yolo_line(line)
            except (ValueError, IndexError) as exc:
                warnings.append(f"{path.name}:{lineno} ignorada ({exc})")
                continue
            if box is not None:
                boxes.append(box)
    return boxes, warnings


def _resolve_yolo_label(image_rel: Path, labels_dir: Path) -> Path | None:
    """Localiza o ``.txt`` correspondente a uma imagem (espelhando subpastas ou pelo nome)."""
    mirrored = labels_dir / image_rel.with_suffix(".txt")
    if mirrored.is_file():
        return mirrored
    flat = labels_dir / f"{image_rel.stem}.txt"
    if flat.is_file():
        return flat
    return None


def parse_yolo_dataset(
    images_dir: str | Path,
    labels_dir: str | Path,
    class_file: str | Path | None = None,
) -> AnnotationDataset:
    """Associa cada imagem de ``images_dir`` ao seu arquivo YOLO em ``labels_dir``.

    Args:
        images_dir: diretorio das imagens originais (busca recursiva).
        labels_dir: diretorio com os ``.txt`` (mesmo nome-base das imagens).
        class_file: ``classes.txt``/``data.yaml`` opcional. Se omitido, tenta
            localizar automaticamente em ``labels_dir`` e no diretorio pai.
    """
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    if not images_dir.is_dir():
        raise ValidationError(f"Diretorio de imagens inexistente: {images_dir}")
    if not labels_dir.is_dir():
        raise ValidationError(f"Diretorio de labels inexistente: {labels_dir}")

    if class_file is None or str(class_file).strip() == "":
        found = find_class_file(labels_dir, labels_dir.parent, images_dir.parent)
        class_names = load_class_names(found) if found else {}
        if found:
            log.info("Arquivo de classes detectado automaticamente: %s", found)
    else:
        class_names = load_class_names(class_file)

    dataset = AnnotationDataset(format="yolo", images_dir=images_dir, items=[], class_names=class_names)
    used_labels: set[Path] = set()

    for img_path in list_images(images_dir):
        rel = img_path.relative_to(images_dir)
        label_path = _resolve_yolo_label(rel, labels_dir)
        if label_path is None:
            dataset.images_without_labels += 1
            continue
        used_labels.add(label_path.resolve())
        boxes, warns = parse_yolo_file(label_path)
        dataset.warnings.extend(warns)
        dataset.items.append(ImageAnnotations(image_path=img_path, relative_path=rel, boxes=boxes))

    ignored = {"classes.txt"}
    for txt in labels_dir.rglob("*.txt"):
        if txt.name in ignored:
            continue
        if txt.resolve() not in used_labels:
            dataset.annotations_without_image += 1

    log.info(
        "YOLO: %d imagens anotadas, %d caixas, %d imagens sem label, %d labels sem imagem.",
        len(dataset.items),
        dataset.total_boxes,
        dataset.images_without_labels,
        dataset.annotations_without_image,
    )
    return dataset


# --------------------------------------------------------------------------- #
# COCO
# --------------------------------------------------------------------------- #
def _resolve_coco_image(file_name: str, images_dir: Path, by_name: dict[str, Path]) -> Path | None:
    """Resolve o caminho de uma imagem COCO tentando caminho relativo e depois so o nome."""
    direct = images_dir / file_name
    if direct.is_file():
        return direct
    base = Path(file_name.replace("\\", "/")).name
    return by_name.get(base.lower())


def parse_coco_dataset(
    images_dir: str | Path,
    json_path: str | Path,
    class_file: str | Path | None = None,
    skip_crowd: bool = True,
) -> AnnotationDataset:
    """Le um arquivo COCO e associa cada anotacao a sua imagem em ``images_dir``.

    Args:
        images_dir: diretorio das imagens originais.
        json_path: arquivo COCO ``.json``.
        class_file: arquivo de classes opcional que sobrescreve ``categories``.
        skip_crowd: ignora anotacoes com ``iscrowd=1`` (regioes com multiplos objetos).
    """
    images_dir = Path(images_dir)
    json_path = Path(json_path)
    if not images_dir.is_dir():
        raise ValidationError(f"Diretorio de imagens inexistente: {images_dir}")
    if not json_path.is_file():
        raise ValidationError(f"Arquivo COCO inexistente: {json_path}")

    with json_path.open("r", encoding="utf-8") as fh:
        data: Any = json.load(fh)
    if not isinstance(data, dict) or "images" not in data or "annotations" not in data:
        raise ValidationError("JSON COCO invalido: chaves 'images' e 'annotations' sao obrigatorias.")

    class_names: dict[int, str] = {
        int(c["id"]): str(c.get("name", f"class_{c['id']}")) for c in data.get("categories", [])
    }
    if class_file is not None and str(class_file).strip() != "":
        class_names.update(load_class_names(class_file))

    by_name: dict[str, Path] = {}
    for p in list_images(images_dir):
        by_name.setdefault(p.name.lower(), p)

    dataset = AnnotationDataset(format="coco", images_dir=images_dir, items=[], class_names=class_names)
    items_by_id: dict[Any, ImageAnnotations] = {}
    for img in data["images"]:
        file_name = str(img.get("file_name", ""))
        resolved = _resolve_coco_image(file_name, images_dir, by_name)
        if resolved is None:
            dataset.warnings.append(f"Imagem COCO nao encontrada em disco: {file_name}")
            continue
        size = None
        if img.get("width") and img.get("height"):
            size = (int(img["width"]), int(img["height"]))
        items_by_id[img["id"]] = ImageAnnotations(
            image_path=resolved,
            relative_path=resolved.relative_to(images_dir),
            declared_size=size,
        )

    skipped_crowd = 0
    for ann in data["annotations"]:
        item = items_by_id.get(ann.get("image_id"))
        if item is None:
            dataset.annotations_without_image += 1
            continue
        if skip_crowd and int(ann.get("iscrowd", 0)) == 1:
            skipped_crowd += 1
            continue
        bbox = ann.get("bbox")
        if not bbox or len(bbox) != 4:
            dataset.warnings.append(f"Anotacao {ann.get('id')} sem bbox valido.")
            continue
        x, y, w, h = (float(v) for v in bbox)
        score = ann.get("score")
        item.boxes.append(
            AnnotationBox(
                class_id=int(ann["category_id"]),
                coords=(x, y, x + w, y + h),
                normalized=False,
                confidence=float(score) if score is not None else None,
            )
        )

    dataset.items = [items_by_id[k] for k in items_by_id]
    dataset.images_without_labels = sum(1 for it in dataset.items if not it.boxes)
    dataset.items = [it for it in dataset.items if it.boxes]
    if skipped_crowd:
        dataset.warnings.append(f"{skipped_crowd} anotacoes 'iscrowd' ignoradas.")

    log.info(
        "COCO: %d imagens anotadas, %d caixas, %d categorias.",
        len(dataset.items),
        dataset.total_boxes,
        len(class_names),
    )
    return dataset


# --------------------------------------------------------------------------- #
# Deteccao automatica do formato
# --------------------------------------------------------------------------- #
def detect_format(labels_path: str | Path) -> AnnotationFormat:
    """Infere o formato a partir do caminho de labels (arquivo .json => COCO, pasta com .txt => YOLO)."""
    p = Path(labels_path)
    if p.is_file():
        if p.suffix.lower() == ".json":
            return "coco"
        raise ValidationError(f"Arquivo de anotacao nao suportado: {p.name} (use .json ou uma pasta de .txt)")
    if p.is_dir():
        has_txt = any(t.name != "classes.txt" for t in p.rglob("*.txt"))
        if has_txt:
            return "yolo"
        jsons = list(p.glob("*.json"))
        if len(jsons) == 1:
            return "coco"
        raise ValidationError(f"Nao foi possivel inferir o formato das anotacoes em: {p}")
    raise ValidationError(f"Caminho de labels inexistente: {p}")


def load_annotations(
    images_dir: str | Path,
    labels_path: str | Path,
    class_file: str | Path | None = None,
    fmt: AnnotationFormat | None = None,
) -> AnnotationDataset:
    """Ponto unico de entrada: detecta o formato (se necessario) e executa o parser adequado."""
    fmt = fmt or detect_format(labels_path)
    if fmt == "yolo":
        return parse_yolo_dataset(images_dir, labels_path, class_file)
    lp = Path(labels_path)
    if lp.is_dir():
        jsons = sorted(lp.glob("*.json"))
        if len(jsons) != 1:
            raise ValidationError("Selecione um unico arquivo COCO .json.")
        lp = jsons[0]
    return parse_coco_dataset(images_dir, lp, class_file)


def iter_class_ids(dataset: AnnotationDataset) -> Iterable[int]:
    """Itera sobre todos os ``class_id`` presentes nas caixas (com repeticao)."""
    for item in dataset.items:
        for box in item.boxes:
            yield box.class_id
