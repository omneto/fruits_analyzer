"""Catálogo de frutas suportadas.

Na etapa 1 usamos o modelo COCO pré-treinado, que conhece "banana" (id 46) e
"orange" (id 49). Tomate não existe no COCO e fica disponível após o treino
da etapa 2, quando basta trocar o modelo e o ``class_id`` aqui.

``color_stage`` mapeia cada faixa de cor (ver ``ripeness.py``) para um índice
de maturação: 0 = verde, 1 = amadurecendo, 2 = maduro, 3 = passado.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Fruit:
    key: str
    label: str
    class_id: int | None
    color_stage: dict[str, float] = field(default_factory=dict)

    @property
    def available(self) -> bool:
        return self.class_id is not None


FRUITS: dict[str, Fruit] = {
    "banana": Fruit(
        key="banana",
        label="Banana",
        class_id=46,
        color_stage={"green": 0, "yellow": 2, "orange": 2, "brown": 3},
    ),
    "laranja": Fruit(
        key="laranja",
        label="Laranja",
        class_id=49,
        color_stage={"green": 0, "yellow": 1, "orange": 2, "red": 2, "brown": 3},
    ),
    "tomate": Fruit(
        key="tomate",
        label="Tomate",
        class_id=None,  # disponível após o treino da etapa 2
        color_stage={"green": 0, "yellow": 1, "orange": 1, "red": 2, "brown": 3},
    ),
}
