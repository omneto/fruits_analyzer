"""Classificação de tamanho relativa às demais frutas da mesma foto.

Uma foto não informa o tamanho real em centímetros. Comparamos o tamanho de
cada caixa (raiz da área) com a mediana da foto:

    < 0,8 × mediana  -> pequeno
    > 1,25 × mediana -> grande
    caso contrário   -> médio

Com menos de 3 frutas não há base de comparação e todas ficam "médio".
"""

from __future__ import annotations

import numpy as np

SIZES = ["pequeno", "médio", "grande"]
SMALL_RATIO = 0.8
LARGE_RATIO = 1.25
MIN_FRUITS = 3


def classify_sizes(boxes: list[tuple[float, float, float, float]]) -> list[str]:
    if not boxes:
        return []
    if len(boxes) < MIN_FRUITS:
        return ["médio"] * len(boxes)
    dims = np.array([np.sqrt(max(x2 - x1, 0) * max(y2 - y1, 0)) for x1, y1, x2, y2 in boxes])
    median = float(np.median(dims)) or 1.0
    result = []
    for d in dims:
        ratio = d / median
        if ratio < SMALL_RATIO:
            result.append("pequeno")
        elif ratio > LARGE_RATIO:
            result.append("grande")
        else:
            result.append("médio")
    return result
