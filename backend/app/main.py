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
from .fruits import Fruit, build_catalog

# Uma análise por vez: a VPS tem 1 vCPU e 2 GB de RAM.
_analysis_lock = asyncio.Lock()


@lru_cache(maxsize=1)
def get_detector() -> YoloOnnxDetector:
    return YoloOnnxDetector(settings.model_path, threads=settings.onnx_threads)


@lru_cache(maxsize=1)
def get_coco_detector() -> YoloOnnxDetector | None:
    path = settings.coco_model_path
    if not path.exists() or path.resolve() == settings.model_path.resolve():
        return None
    return YoloOnnxDetector(path, threads=settings.onnx_threads)


@lru_cache(maxsize=1)
def get_routes() -> dict[str, tuple[Fruit, YoloOnnxDetector]]:
    """Para cada fruta, a definição e o modelo que vai analisá-la.

    Usa o modelo principal. Frutas listadas em ``COCO_FRUITS``, ou que o modelo
    principal não conhece, vão para o modelo COCO com regras de cor.
    """
    main_det = get_detector()
    routes = {k: (f, main_det) for k, f in build_catalog(main_det.names).items()}
    coco = get_coco_detector()
    if coco is not None:
        for key, fruit in build_catalog(coco.names).items():
            if fruit.available and (key in settings.coco_fruits or not routes[key][0].available):
                routes[key] = (fruit, coco)
    return routes


def get_catalog() -> dict[str, Fruit]:
    return {k: f for k, (f, _) in get_routes().items()}


@asynccontextmanager
async def lifespan(_app: FastAPI):
    get_routes()  # carrega os modelos uma vez, na inicialização
    yield


app = FastAPI(title="Fruits Analyzer", version="0.1.0", lifespan=lifespan)


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/api/fruits")
def list_fruits() -> list[dict]:
    return [
        {"key": f.key, "label": f.label, "available": f.available, "ripeness_source": f.ripeness_source}
        for f in get_catalog().values()
    ]


@app.post("/api/analyze")
async def analyze_image(
    file: UploadFile = File(...),
    fruit: str = Form("banana"),
    tiled: bool = Form(False),
) -> dict:
    route = get_routes().get(fruit)
    if route is None:
        raise HTTPException(400, f"Fruta desconhecida: {fruit}")
    selected, detector = route
    if not selected.available:
        raise HTTPException(400, f"{selected.label} não está disponível no modelo carregado.")

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
            detector,
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
