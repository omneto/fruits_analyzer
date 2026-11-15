"""Baixa datasets públicos e unifica as classes para o treino.

Cada dataset público usa nomes de classes próprios ("ripe", "b_green",
"unripe_orange"...). Aqui eles são convertidos para o padrão do projeto,
``<fruta>_<estágio>``, por exemplo ``banana_maduro`` ou ``tomate_verde``.

Usado pelo notebook ``notebooks/treino_frutas.ipynb`` e testado em
``training/test_datasets.py``.
"""

from __future__ import annotations

import random
import re
import shutil
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import yaml

FRUITS = ["banana", "laranja", "tomate"]
STAGES = ["verde", "amadurecendo", "maduro", "passado"]
CLASS_NAMES = [f"{f}_{s}" for f in FRUITS for s in STAGES]
CLASS_INDEX = {name: i for i, name in enumerate(CLASS_NAMES)}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLIT_ALIASES = {"train": "train", "valid": "val", "val": "val", "test": "test"}


@dataclass
class DatasetSource:
    """Um dataset de origem e como as classes dele viram estágios.

    ``class_map`` leva o nome da classe original ao estágio (``verde``,
    ``amadurecendo``, ``maduro``, ``passado``) ou a ``None`` para descartar.
    Classes que não estão no mapa passam por ``guess_stage``.
    """

    name: str
    fruit: str
    class_map: dict[str, str | None] = field(default_factory=dict)
    roboflow_workspace: str | None = None
    roboflow_project: str | None = None
    roboflow_version: int | None = None  # None = versão mais recente
    local_path: str | None = None
    url: str = ""
    license: str = ""


# Datasets públicos do Roboflow Universe usados por padrão. Todos CC BY 4.0
# conforme as páginas consultadas em 15/11/2026. Troque ou acrescente à vontade.
DEFAULT_SOURCES = [
    DatasetSource(
        name="banana",
        fruit="banana",
        roboflow_workspace="fruit-ripening",
        roboflow_project="banana-ripening-process",
        url="https://universe.roboflow.com/fruit-ripening/banana-ripening-process",
        license="CC BY 4.0",
        class_map={
            "freshunripe": "verde",
            "unripe": "amadurecendo",
            "freshripe": "maduro",
            "ripe": "maduro",
            "overripe": "passado",
            "rotten": "passado",
        },
    ),
    DatasetSource(
        name="laranja",
        fruit="laranja",
        roboflow_workspace="yashas",
        roboflow_project="orange_ripeness",
        url="https://universe.roboflow.com/yashas/orange_ripeness",
        license="CC BY 4.0",
        class_map={"unripe_orange": "verde", "ripe_orange": "maduro"},
    ),
    DatasetSource(
        name="tomate",
        fruit="tomate",
        roboflow_workspace="tomatoes-dectection-riped",
        roboflow_project="tomatoes-growth-by-laboro",
        url="https://universe.roboflow.com/tomatoes-dectection-riped/tomatoes-growth-by-laboro",
        license="CC BY 4.0 (derivado do Laboro Tomato)",
        class_map={
            "b_green": "verde",
            "l_green": "verde",
            "b_half_ripened": "amadurecendo",
            "l_half_ripened": "amadurecendo",
            "b_fully_ripened": "maduro",
            "l_fully_ripened": "maduro",
        },
    ),
]

_GUESS_RULES = [
    (r"rot|overripe|spoil|bad|passad|podre", "passado"),
    (r"half|semi|turn|breaker|partial|amadurecend|mid", "amadurecendo"),
    (r"unripe|green|raw|immature|verde", "verde"),
    (r"ripe|mature|madur|red|yellow", "maduro"),
]


def guess_stage(class_name: str) -> str | None:
    """Tenta deduzir o estágio pelo nome da classe (para datasets novos)."""
    name = class_name.lower()
    for pattern, stage in _GUESS_RULES:
        if re.search(pattern, name):
            return stage
    return None


def target_class(source: DatasetSource, original: str) -> str | None:
    if original in source.class_map:
        stage = source.class_map[original]
    else:
        stage = guess_stage(original)
    if stage is None:
        return None
    if stage not in STAGES:
        raise ValueError(f"Estágio inválido '{stage}' para a classe '{original}' em {source.name}")
    return f"{source.fruit}_{stage}"


def download_roboflow(source: DatasetSource, api_key: str, dest: Path) -> Path:
    """Baixa o dataset no formato YOLO e retorna a pasta."""
    from roboflow import Roboflow  # import local: só é preciso no Colab

    project = Roboflow(api_key=api_key).workspace(source.roboflow_workspace).project(source.roboflow_project)
    version = source.roboflow_version
    if version is None:
        version = max(int(str(v.version).split("/")[-1]) for v in project.versions())
    target = dest / source.name
    project.version(version).download("yolov11", location=str(target), overwrite=True)
    return target


def read_class_names(dataset_dir: Path) -> list[str]:
    data = yaml.safe_load((dataset_dir / "data.yaml").read_text())
    names = data["names"]
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names, key=int)]
    return [str(n) for n in names]


def find_splits(dataset_dir: Path) -> dict[str, Path]:
    """Mapeia ``train``/``val``/``test`` para as pastas que têm ``images`` e ``labels``."""
    splits = {}
    for child in sorted(dataset_dir.iterdir()):
        alias = SPLIT_ALIASES.get(child.name.lower())
        if alias and (child / "images").is_dir():
            splits.setdefault(alias, child)
    if "train" not in splits:
        raise FileNotFoundError(f"Pasta train/images não encontrada em {dataset_dir}")
    return splits


def convert_label_line(line: str) -> tuple[int, list[float]] | None:
    """Lê uma linha YOLO. Polígonos (segmentação) viram caixa."""
    parts = line.split()
    if len(parts) < 5:
        return None
    cls = int(float(parts[0]))
    values = [float(v) for v in parts[1:]]
    if len(values) == 4:
        return cls, values
    xs, ys = values[0::2], values[1::2]
    x1, x2, y1, y2 = min(xs), max(xs), min(ys), max(ys)
    return cls, [(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1]


def _image_label_pairs(split_dir: Path):
    for img in sorted((split_dir / "images").iterdir()):
        if img.suffix.lower() in IMAGE_EXTS:
            yield img, split_dir / "labels" / f"{img.stem}.txt"


def merge_datasets(
    datasets: list[tuple[DatasetSource, Path]],
    out_dir: Path,
    val_fraction: float = 0.15,
    seed: int = 42,
) -> dict:
    """Une os datasets em ``out_dir`` com as classes do projeto.

    Retorna estatísticas por classe e por divisão (train/val/test).
    """
    out_dir = Path(out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    rng = random.Random(seed)
    counts: dict[str, Counter] = {"train": Counter(), "val": Counter(), "test": Counter()}
    images: Counter = Counter()
    skipped: Counter = Counter()

    for source, path in datasets:
        names = read_class_names(path)
        mapping = {i: target_class(source, n) for i, n in enumerate(names)}
        splits = find_splits(path)
        for split_name, split_dir in splits.items():
            for img, label in _image_label_pairs(split_dir):
                dest_split = split_name
                if split_name == "train" and "val" not in splits and rng.random() < val_fraction:
                    dest_split = "val"
                lines = []
                if label.exists():
                    for raw in label.read_text().splitlines():
                        parsed = convert_label_line(raw)
                        if parsed is None:
                            continue
                        cls, box = parsed
                        target = mapping.get(cls)
                        if target is None:
                            skipped[f"{source.name}:{names[cls] if cls < len(names) else cls}"] += 1
                            continue
                        lines.append(f"{CLASS_INDEX[target]} " + " ".join(f"{v:.6f}" for v in box))
                        counts[dest_split][target] += 1
                stem = f"{source.name}_{img.stem}"
                (out_dir / dest_split / "images").mkdir(parents=True, exist_ok=True)
                (out_dir / dest_split / "labels").mkdir(parents=True, exist_ok=True)
                shutil.copy2(img, out_dir / dest_split / "images" / f"{stem}{img.suffix.lower()}")
                (out_dir / dest_split / "labels" / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
                images[dest_split] += 1

    data = {
        "path": str(out_dir.resolve()),
        "train": "train/images",
        "val": "val/images",
        "names": {i: n for i, n in enumerate(CLASS_NAMES)},
    }
    if images["test"]:
        data["test"] = "test/images"
    (out_dir / "data.yaml").write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False))
    return {
        "images": dict(images),
        "boxes": {split: dict(c) for split, c in counts.items()},
        "skipped": dict(skipped),
        "data_yaml": str(out_dir / "data.yaml"),
    }


def stats_table(stats: dict) -> str:
    """Tabela em texto com caixas por classe em cada divisão."""
    header = f"{'classe':<22}{'train':>8}{'val':>8}{'test':>8}"
    rows = [header, "-" * len(header)]
    for name in CLASS_NAMES:
        b = stats["boxes"]
        rows.append(
            f"{name:<22}{b['train'].get(name, 0):>8}{b['val'].get(name, 0):>8}{b['test'].get(name, 0):>8}"
        )
    rows.append(f"imagens: {stats['images']}")
    if stats["skipped"]:
        rows.append(f"classes descartadas: {stats['skipped']}")
    return "\n".join(rows)
