"""Catálogo de frutas suportadas, montado a partir das classes do modelo.

Dois tipos de modelo são aceitos:

* **Modelo treinado do projeto** (etapa 2, ``models/frutas_yolo11n.onnx``):
  classes no formato ``<fruta>_<estágio>``, como ``banana_maduro``. A
  maturação vem direto do modelo.
* **Modelo COCO pré-treinado** (etapa 1): conhece "banana" e "orange". A
  maturação é estimada pelas regras de cor de ``ripeness.py``. Tomate fica
  indisponível.

``color_stage`` mapeia cada faixa de cor para um índice de maturação
(0 = verde, 1 = amadurecendo, 2 = maduro, 3 = passado) e só é usado nas
regras de cor.
"""

from __future__ import annotations

from dataclasses import dataclass, field

STAGES = ["verde", "amadurecendo", "maduro", "passado"]

# Ids das classes no COCO.
COCO_CLASS_IDS = {"banana": 46, "laranja": 49}


@dataclass(frozen=True)
class Fruit:
    key: str
    label: str
    color_stage: dict[str, float]
    class_ids: frozenset[int] = frozenset()
    # Preenchido quando o modelo já classifica a maturação (id -> estágio).
    stage_by_class: dict[int, str] = field(default_factory=dict)

    @property
    def available(self) -> bool:
        return bool(self.class_ids)

    @property
    def ripeness_source(self) -> str:
        return "modelo" if self.stage_by_class else "cor"


FRUIT_INFO = {
    "banana": ("Banana", {"green": 0, "yellow": 2, "orange": 2, "brown": 3}),
    # Laranja madura pode ser amarelada (ex.: pera, valência): amarelo puro já conta como maduro.
    "laranja": ("Laranja", {"green": 0, "yellow": 1.7, "orange": 2, "red": 2, "brown": 3}),
    "tomate": ("Tomate", {"green": 0, "yellow": 1, "orange": 1, "red": 2, "brown": 3}),
}


def build_catalog(model_names: dict[int, str] | None = None) -> dict[str, Fruit]:
    """Monta o catálogo conforme as classes do modelo carregado."""
    names = model_names or {}
    trained: dict[str, dict[int, str]] = {}
    for class_id, name in names.items():
        fruit, _, stage = str(name).rpartition("_")
        if fruit in FRUIT_INFO and stage in STAGES:
            trained.setdefault(fruit, {})[int(class_id)] = stage

    catalog = {}
    for key, (label, color_stage) in FRUIT_INFO.items():
        if trained:
            stages = trained.get(key, {})
            catalog[key] = Fruit(key, label, color_stage, frozenset(stages), stages)
        else:
            coco_id = COCO_CLASS_IDS.get(key)
            ids = frozenset({coco_id}) if coco_id is not None else frozenset()
            catalog[key] = Fruit(key, label, color_stage, ids)
    return catalog


# Catálogo padrão (modelo COCO). Usado pelas regras de cor e nos testes.
FRUITS = build_catalog()
