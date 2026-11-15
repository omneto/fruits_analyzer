import os
from pathlib import Path

# Os testes usam sempre o modelo COCO, mesmo que o modelo treinado exista em models/.
os.environ.setdefault("MODEL_PATH", str(Path(__file__).resolve().parents[2] / "models" / "coco_yolo11n.onnx"))
