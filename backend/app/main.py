"""API FastAPI e servidor da app web."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from functools import lru_cache

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .analyzer import InvalidImage, analyze, load_image
from .config import settings
from .detector import YoloOnnxDetector
from .fruits import FRUITS

# Uma análise por vez: a VPS tem 1 vCPU e 2 GB de RAM.
_analysis_lock = asyncio.Lock()


@lru_cache(maxsize=1)
def get_detector() -> YoloOnnxDetector:
    return YoloOnnxDetector(settings.model_path, threads=settings.onnx_threads)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    get_detector()  # carrega o modelo uma vez, na inicialização
    yield


app = FastAPI(title="Fruits Analyzer", version="0.1.0", lifespan=lifespan)


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/api/fruits")
def list_fruits() -> list[dict]:
    return [{"key": f.key, "label": f.label, "available": f.available} for f in FRUITS.values()]


@app.post("/api/analyze")
async def analyze_image(
    file: UploadFile = File(...),
    fruit: str = Form("banana"),
    tiled: bool = Form(False),
) -> dict:
    selected = FRUITS.get(fruit)
    if selected is None:
        raise HTTPException(400, f"Fruta desconhecida: {fruit}")
    if not selected.available:
        raise HTTPException(400, f"{selected.label} ainda não está disponível (depende do modelo treinado).")

    limit = settings.max_upload_mb * 1024 * 1024
    data = await file.read(limit + 1)
    if len(data) > limit:
        raise HTTPException(413, f"Imagem maior que {settings.max_upload_mb} MB.")

    async with _analysis_lock:
        try:
            image = await asyncio.to_thread(load_image, data, settings.max_image_side)
        except InvalidImage as exc:
            raise HTTPException(400, str(exc)) from exc
        result = await asyncio.to_thread(
            analyze,
            image,
            selected,
            get_detector(),
            settings.conf_threshold,
            settings.iou_threshold,
            tiled,
        )
    result["filename"] = file.filename
    return result


if settings.frontend_dir.is_dir():
    app.mount("/static", StaticFiles(directory=settings.frontend_dir), name="static")

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(settings.frontend_dir / "index.html")
