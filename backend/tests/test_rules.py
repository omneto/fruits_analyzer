import numpy as np
from PIL import Image

from app.detector import letterbox, nms
from app.fruits import FRUITS
from app.ripeness import classify_ripeness, stage_from_index
from app.size import classify_sizes


def solid(rgb, size=(60, 60)):
    return Image.new("RGB", size, rgb)


def mixed(colors):
    """Imagem com faixas verticais de cores iguais."""
    w = 30
    img = Image.new("RGB", (w * len(colors), 40))
    for i, c in enumerate(colors):
        img.paste(c, (i * w, 0, (i + 1) * w, 40))
    return img


GREEN = (70, 160, 40)
YELLOW = (235, 210, 40)
BROWN = (95, 60, 25)
RED = (200, 25, 20)
ORANGE = (240, 140, 20)


def test_banana_stages():
    banana = FRUITS["banana"]
    assert classify_ripeness(solid(GREEN), banana).stage == "verde"
    assert classify_ripeness(mixed([GREEN, YELLOW]), banana).stage == "amadurecendo"
    assert classify_ripeness(solid(YELLOW), banana).stage == "maduro"
    assert classify_ripeness(mixed([BROWN, BROWN, YELLOW]), banana).stage == "passado"


def test_tomato_and_orange_colors():
    assert classify_ripeness(solid(RED), FRUITS["tomate"]).stage == "maduro"
    assert classify_ripeness(solid(GREEN), FRUITS["tomate"]).stage == "verde"
    assert classify_ripeness(solid(ORANGE), FRUITS["laranja"]).stage == "maduro"


def test_gray_is_undefined():
    assert classify_ripeness(solid((128, 128, 128)), FRUITS["banana"]).stage == "indefinido"


def test_stage_thresholds():
    assert [stage_from_index(x) for x in (0, 1, 2, 3)] == ["verde", "amadurecendo", "maduro", "passado"]


def test_sizes_relative_to_median():
    boxes = [(0, 0, 10, 10), (0, 0, 20, 20), (0, 0, 21, 21), (0, 0, 19, 19), (0, 0, 40, 40)]
    assert classify_sizes(boxes) == ["pequeno", "médio", "médio", "médio", "grande"]


def test_sizes_need_three_fruits():
    assert classify_sizes([(0, 0, 10, 10), (0, 0, 50, 50)]) == ["médio", "médio"]
    assert classify_sizes([]) == []


def test_nms_removes_overlap():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 10, 10], [50, 50, 60, 60]], dtype=float)
    scores = np.array([0.9, 0.8, 0.7])
    assert sorted(nms(boxes, scores, 0.45)) == [0, 2]


def test_letterbox_shape_and_scale():
    tensor, scale, pad = letterbox(Image.new("RGB", (1280, 640)), 640)
    assert tensor.shape == (1, 3, 640, 640)
    assert scale == 0.5
    assert pad == (0, 160)
