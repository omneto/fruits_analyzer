from pathlib import Path

import yaml
from PIL import Image

from training.datasets import (
    CLASS_INDEX,
    DatasetSource,
    convert_label_line,
    guess_stage,
    merge_datasets,
)


def make_dataset(root: Path, names, splits):
    """Cria um dataset no formato exportado pelo Roboflow."""
    root.mkdir(parents=True)
    (root / "data.yaml").write_text(yaml.safe_dump({"names": names, "nc": len(names)}))
    for split, items in splits.items():
        (root / split / "images").mkdir(parents=True)
        (root / split / "labels").mkdir(parents=True)
        for stem, lines in items.items():
            Image.new("RGB", (32, 32)).save(root / split / "images" / f"{stem}.jpg")
            (root / split / "labels" / f"{stem}.txt").write_text("\n".join(lines))
    return root


def read_labels(path: Path):
    return [line.split()[0] for line in path.read_text().splitlines()]


def test_guess_stage():
    assert guess_stage("unripe_orange") == "verde"
    assert guess_stage("Overripe") == "passado"
    assert guess_stage("b_half_ripened") == "amadurecendo"
    assert guess_stage("ripe") == "maduro"
    assert guess_stage("leaf") is None


def test_polygon_becomes_box():
    cls, box = convert_label_line("2 0.1 0.2 0.3 0.2 0.3 0.6 0.1 0.6")
    assert cls == 2
    assert [round(v, 3) for v in box] == [0.2, 0.4, 0.2, 0.4]


def test_merge_remaps_classes_and_splits(tmp_path):
    banana = make_dataset(
        tmp_path / "src_banana",
        ["freshunripe", "ripe", "leaf"],
        {
            "train": {"a": ["0 0.5 0.5 0.2 0.2", "2 0.1 0.1 0.1 0.1"], "b": ["1 0.5 0.5 0.3 0.3"]},
            "valid": {"c": ["1 0.4 0.4 0.2 0.2"]},
        },
    )
    tomato = make_dataset(
        tmp_path / "src_tomate",
        ["b_green", "b_fully_ripened"],
        {"train": {f"t{i}": ["1 0.5 0.5 0.2 0.2"] for i in range(20)}},
    )
    sources = [
        (DatasetSource("banana", "banana", {"freshunripe": "verde", "ripe": "maduro", "leaf": None}), banana),
        (DatasetSource("tomate", "tomate"), tomato),
    ]
    out = tmp_path / "merged"
    stats = merge_datasets(sources, out, val_fraction=0.3, seed=1)

    labels = read_labels(out / "train" / "labels" / "banana_a.txt")
    assert labels == [str(CLASS_INDEX["banana_verde"])]  # "leaf" descartada
    assert stats["skipped"] == {"banana:leaf": 1}
    assert (out / "val" / "images" / "banana_c.jpg").exists()
    # o tomate não tinha validação: parte do train vai para val
    assert 0 < stats["boxes"]["val"].get("tomate_maduro", 0) < 20
    data = yaml.safe_load((out / "data.yaml").read_text())
    assert data["names"][CLASS_INDEX["tomate_maduro"]] == "tomate_maduro"
    assert len(data["names"]) == 12
