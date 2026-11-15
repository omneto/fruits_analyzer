"""Configuração da aplicação, lida de variáveis de ambiente."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]


def _env_float(name: str, default: float) -> float:
    return float(os.getenv(name, default))


def _env_int(name: str, default: int) -> int:
    return int(os.getenv(name, default))


def _default_model() -> Path:
    """Usa o modelo treinado (etapa 2) se existir; senão, o COCO pré-treinado."""
    trained = ROOT_DIR / "models" / "frutas_yolo11n.onnx"
    return trained if trained.exists() else ROOT_DIR / "models" / "coco_yolo11n.onnx"


@dataclass(frozen=True)
class Settings:
    model_path: Path = Path(os.getenv("MODEL_PATH", _default_model()))
    frontend_dir: Path = Path(os.getenv("FRONTEND_DIR", ROOT_DIR / "frontend"))
    # Limiar de confiança mínimo de uma detecção.
    conf_threshold: float = _env_float("CONF_THRESHOLD", 0.25)
    iou_threshold: float = _env_float("IOU_THRESHOLD", 0.45)
    # Lado máximo (px) da imagem antes do processamento. Fotos de drone/celular
    # são reduzidas para caber na memória da VPS.
    max_image_side: int = _env_int("MAX_IMAGE_SIDE", 2048)
    max_upload_mb: int = _env_int("MAX_UPLOAD_MB", 20)
    # Threads do ONNX Runtime. Na VPS de 1 vCPU, 1 é o ideal.
    onnx_threads: int = _env_int("ONNX_THREADS", 1)


settings = Settings()
