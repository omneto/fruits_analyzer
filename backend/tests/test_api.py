"""Testes da API com o modelo ONNX real."""

import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

from app.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_health_and_fruits(client):
    assert client.get("/api/health").json() == {"status": "ok"}
    fruits = {f["key"]: f for f in client.get("/api/fruits").json()}
    assert fruits["banana"]["available"] is True
    assert fruits["tomate"]["available"] is False


def test_index_served(client):
    res = client.get("/")
    assert res.status_code == 200
    assert "Analisador de Frutas" in res.text


def test_analyze_returns_report(client):
    img = Image.new("RGB", (800, 600), "white")
    ImageDraw.Draw(img).ellipse((200, 200, 600, 400), fill=(235, 210, 40))
    res = client.post(
        "/api/analyze",
        files={"file": ("teste.png", png_bytes(img), "image/png")},
        data={"fruit": "banana", "tiled": "true"},
    )
    assert res.status_code == 200
    body = res.json()
    assert body["fruit"] == "banana"
    assert body["total"] == len(body["detections"])
    assert sum(body["by_size"].values()) == body["total"]
    assert body["annotated_image"].startswith("data:image/jpeg;base64,")


def test_rejects_unavailable_fruit(client):
    img = Image.new("RGB", (64, 64))
    res = client.post("/api/analyze", files={"file": ("a.png", png_bytes(img), "image/png")}, data={"fruit": "tomate"})
    assert res.status_code == 400


def test_rejects_non_image(client):
    res = client.post("/api/analyze", files={"file": ("a.txt", b"nao sou imagem", "text/plain")}, data={"fruit": "banana"})
    assert res.status_code == 400


def test_routes_mix_trained_and_coco(monkeypatch):
    """Com o modelo treinado, a laranja vai para o COCO e o tomate para o treinado."""
    import dataclasses
    from pathlib import Path

    from app import main

    models = Path(__file__).resolve().parents[2] / "models"
    trained = models / "frutas_yolo11n.onnx"
    if not trained.exists():
        pytest.skip("modelo treinado ausente")
    patched = dataclasses.replace(main.settings, model_path=trained, coco_model_path=models / "coco_yolo11n.onnx")
    monkeypatch.setattr(main, "settings", patched)
    for fn in (main.get_detector, main.get_coco_detector, main.get_routes):
        fn.cache_clear()
    try:
        routes = main.get_routes()
        assert routes["laranja"][0].ripeness_source == "cor"
        assert routes["laranja"][1] is main.get_coco_detector()
        assert routes["tomate"][0].ripeness_source == "modelo"
        assert routes["banana"][1] is main.get_detector()
    finally:
        for fn in (main.get_detector, main.get_coco_detector, main.get_routes):
            fn.cache_clear()


def test_report_has_color_comparison(client):
    img = Image.new("RGB", (64, 64))
    body = client.post(
        "/api/analyze", files={"file": ("a.png", png_bytes(img), "image/png")}, data={"fruit": "banana"}
    ).json()
    assert set(body["by_ripeness_color"]) == set(body["by_ripeness"])
