"""Datasets sintéticos minúsculos para testar o notebook sem internet nem GPU.

Gera imagens com "frutas" desenhadas (elipses coloridas) no mesmo formato que
o Roboflow exporta, com os nomes de classe dos datasets reais. Não serve para
treinar um modelo útil: só confirma que download, união, treino e exportação
funcionam de ponta a ponta.
"""

from __future__ import annotations

import random
from pathlib import Path

import yaml
from PIL import Image, ImageDraw

from .datasets import DEFAULT_SOURCES, DatasetSource

COLORS = {
    "verde": (70, 150, 40),
    "amadurecendo": (180, 190, 40),
    "maduro": (235, 200, 30),
    "passado": (100, 65, 30),
}


def _color_for(source: DatasetSource, cls: str) -> tuple[int, int, int]:
    stage = source.class_map.get(cls) or "maduro"
    color = COLORS[stage]
    if source.fruit != "banana" and stage == "maduro":
        color = (230, 120, 20) if source.fruit == "laranja" else (200, 30, 25)
    return color


def make_synthetic_dataset(source: DatasetSource, dest: Path, n_images: int = 12, size: int = 320, seed: int = 0) -> Path:
    rng = random.Random(seed)
    names = list(source.class_map)
    root = dest / source.name
    (root).mkdir(parents=True, exist_ok=True)
    (root / "data.yaml").write_text(yaml.safe_dump({"names": names, "nc": len(names)}))
    for split, count in (("train", n_images), ("valid", max(2, n_images // 4))):
        (root / split / "images").mkdir(parents=True, exist_ok=True)
        (root / split / "labels").mkdir(parents=True, exist_ok=True)
        for i in range(count):
            bg = tuple(rng.randint(150, 230) for _ in range(3))
            img = Image.new("RGB", (size, size), bg)
            draw = ImageDraw.Draw(img)
            lines = []
            for _ in range(rng.randint(1, 4)):
                cls = rng.randrange(len(names))
                w = rng.randint(size // 8, size // 3)
                h = w // 2 if source.fruit == "banana" else w
                x = rng.randint(0, size - w)
                y = rng.randint(0, size - h)
                draw.ellipse((x, y, x + w, y + h), fill=_color_for(source, names[cls]))
                lines.append(f"{cls} {(x + w / 2) / size:.5f} {(y + h / 2) / size:.5f} {w / size:.5f} {h / size:.5f}")
            img.save(root / split / "images" / f"{i:03d}.jpg")
            (root / split / "labels" / f"{i:03d}.txt").write_text("\n".join(lines) + "\n")
    return root


def make_synthetic_sources(dest: Path) -> list[tuple[DatasetSource, Path]]:
    return [(s, make_synthetic_dataset(s, dest, seed=k)) for k, s in enumerate(DEFAULT_SOURCES)]
