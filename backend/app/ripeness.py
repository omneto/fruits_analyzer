"""Classificação de maturação por regras de cor (HSV).

É a versão simples da etapa 1. Cada pixel do centro da caixa da fruta é
colocado em uma faixa de cor (verde, amarelo, laranja, vermelho, marrom).
Cada fruta define o índice de maturação de cada cor (``fruits.py``) e a média
ponderada desses índices decide o estágio:

    índice < 0,6  -> verde
    índice < 1,6  -> amadurecendo
    índice < 2,45 -> maduro
    caso contrário -> passado

Limitação conhecida: em fotos da árvore, folhas dentro da caixa puxam o
resultado para "verde". O modelo treinado na etapa 2 classifica a maturação
diretamente e resolve isso.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image

from .fruits import Fruit

STAGES = ["verde", "amadurecendo", "maduro", "passado"]
STAGE_THRESHOLDS = [0.6, 1.6, 2.45]
COLORS = ["green", "yellow", "orange", "red", "brown"]


@dataclass
class RipenessResult:
    stage: str
    index: float | None
    color_fractions: dict[str, float]


def color_fractions(crop: Image.Image) -> dict[str, float]:
    """Fração de pixels "de fruta" em cada faixa de cor."""
    crop = crop.convert("RGB")
    if max(crop.size) > 128:
        crop.thumbnail((128, 128))
    hsv = np.asarray(crop.convert("HSV"), dtype=np.float32)
    h = hsv[..., 0] * (360.0 / 255.0)
    s = hsv[..., 1] / 255.0
    v = hsv[..., 2] / 255.0

    valid = v > 0.12  # ignora sombras muito escuras
    # Marrom: tons alaranjados escuros. Vermelho escuro (tomate maduro) não conta.
    brown = valid & (h >= 12) & (h < 50) & (s > 0.2) & (v < 0.45)
    chromatic = valid & ~brown & (s > 0.25) & (v > 0.2)
    red = chromatic & ((h < 12) | (h >= 340))
    orange = chromatic & (h >= 12) & (h < 38)
    yellow = chromatic & (h >= 38) & (h < 68)
    green = chromatic & (h >= 68) & (h < 170)

    counts = {
        "green": int(green.sum()),
        "yellow": int(yellow.sum()),
        "orange": int(orange.sum()),
        "red": int(red.sum()),
        "brown": int(brown.sum()),
    }
    total = sum(counts.values())
    if total == 0:
        return {c: 0.0 for c in COLORS}
    return {c: counts[c] / total for c in COLORS}


def center_crop(image: Image.Image, box: tuple[float, float, float, float], keep: float = 0.6) -> Image.Image:
    """Recorta os ``keep`` centrais da caixa, para reduzir o fundo."""
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    hw, hh = (x2 - x1) * keep / 2, (y2 - y1) * keep / 2
    left, top = int(max(cx - hw, 0)), int(max(cy - hh, 0))
    right, bottom = int(max(cx + hw, left + 1)), int(max(cy + hh, top + 1))
    return image.crop((left, top, right, bottom))


def stage_from_index(index: float) -> str:
    for stage, limit in zip(STAGES, STAGE_THRESHOLDS):
        if index < limit:
            return stage
    return STAGES[-1]


def classify_ripeness(crop: Image.Image, fruit: Fruit) -> RipenessResult:
    fractions = color_fractions(crop)
    weight = sum(fractions[c] for c in fruit.color_stage)
    if weight < 0.15:
        # Quase nenhum pixel com cor de fruta reconhecível.
        return RipenessResult("indefinido", None, fractions)
    index = sum(fractions[c] * fruit.color_stage[c] for c in fruit.color_stage) / weight
    return RipenessResult(stage_from_index(index), round(index, 2), fractions)
