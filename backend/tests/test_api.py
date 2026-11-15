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
