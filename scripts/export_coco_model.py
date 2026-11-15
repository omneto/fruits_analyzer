"""Gera models/coco_yolo11n.onnx a partir do YOLO11n pré-treinado no COCO.

Rode em uma máquina com PyTorch (ou no Google Colab), nunca na VPS:

    pip install ultralytics onnx onnxslim
    python scripts/export_coco_model.py
"""

import shutil
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]

model = YOLO("yolo11n.pt")
onnx_path = model.export(format="onnx", imgsz=640, simplify=True, opset=17)
dest = ROOT / "models" / "coco_yolo11n.onnx"
shutil.copy(onnx_path, dest)
print(f"Modelo salvo em {dest}")
