"""Detector YOLO em ONNX Runtime (somente CPU, sem PyTorch)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import onnxruntime as ort
from PIL import Image


@dataclass
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1


def letterbox(image: Image.Image, size: int) -> tuple[np.ndarray, float, tuple[int, int]]:
    """Redimensiona mantendo a proporção e completa com cinza até ``size``×``size``.

    Retorna o tensor NCHW float32 em [0, 1], a escala e o deslocamento (pad).
    """
    w, h = image.size
    scale = min(size / w, size / h)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    resized = image.resize((nw, nh), Image.BILINEAR)
    canvas = Image.new("RGB", (size, size), (114, 114, 114))
    pad_x, pad_y = (size - nw) // 2, (size - nh) // 2
    canvas.paste(resized, (pad_x, pad_y))
    arr = np.asarray(canvas, dtype=np.float32) / 255.0
    return arr.transpose(2, 0, 1)[None], scale, (pad_x, pad_y)


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> list[int]:
    """Non-maximum suppression simples. ``boxes`` em formato x1, y1, x2, y2."""
    order = scores.argsort()[::-1]
    keep: list[int] = []
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    while order.size:
        i = int(order[0])
        keep.append(i)
        rest = order[1:]
        xx1 = np.maximum(boxes[i, 0], boxes[rest, 0])
        yy1 = np.maximum(boxes[i, 1], boxes[rest, 1])
        xx2 = np.minimum(boxes[i, 2], boxes[rest, 2])
        yy2 = np.minimum(boxes[i, 3], boxes[rest, 3])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        iou = inter / (areas[i] + areas[rest] - inter + 1e-9)
        order = rest[iou <= iou_threshold]
    return keep


class YoloOnnxDetector:
    def __init__(self, model_path: str, threads: int = 1):
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        opts.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(model_path), sess_options=opts, providers=["CPUExecutionProvider"]
        )
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        self.input_size = int(inp.shape[2]) if isinstance(inp.shape[2], int) else 640

    def _infer(self, image: Image.Image, class_ids: set[int], conf: float):
        tensor, scale, (pad_x, pad_y) = letterbox(image, self.input_size)
        out = self.session.run(None, {self.input_name: tensor})[0][0]  # (4 + nc, N)
        out = out.T
        scores_all = out[:, 4:]
        cls = scores_all.argmax(axis=1)
        scores = scores_all[np.arange(len(cls)), cls]
        mask = (scores >= conf) & np.isin(cls, list(class_ids))
        boxes_cxcywh, scores, cls = out[mask, :4], scores[mask], cls[mask]
        boxes = np.empty_like(boxes_cxcywh)
        boxes[:, 0] = boxes_cxcywh[:, 0] - boxes_cxcywh[:, 2] / 2
        boxes[:, 1] = boxes_cxcywh[:, 1] - boxes_cxcywh[:, 3] / 2
        boxes[:, 2] = boxes_cxcywh[:, 0] + boxes_cxcywh[:, 2] / 2
        boxes[:, 3] = boxes_cxcywh[:, 1] + boxes_cxcywh[:, 3] / 2
        boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_x) / scale
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_y) / scale
        w, h = image.size
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h)
        return boxes, scores, cls

    def detect(
        self,
        image: Image.Image,
        class_ids: set[int],
        conf: float = 0.25,
        iou: float = 0.45,
        tiled: bool = False,
    ) -> list[Detection]:
        """Detecta objetos das classes pedidas.

        Com ``tiled=True`` a imagem também é dividida em blocos sobrepostos, o que
        ajuda a encontrar frutas pequenas em fotos de árvores inteiras ou de drone.
        """
        all_boxes, all_scores, all_cls = [], [], []
        regions = [(0, 0, *image.size)]
        if tiled:
            regions += self._tiles(image.size)
        for x0, y0, x1, y1 in regions:
            crop = image if (x0, y0, x1, y1) == (0, 0, *image.size) else image.crop((x0, y0, x1, y1))
            boxes, scores, cls = self._infer(crop, class_ids, conf)
            if crop is not image:
                # Descarta frutas cortadas na borda interna do bloco: elas
                # aparecem inteiras na imagem completa ou no bloco vizinho.
                cw, ch = crop.size
                m = 2
                cut = (
                    ((boxes[:, 0] <= m) & (x0 > 0))
                    | ((boxes[:, 1] <= m) & (y0 > 0))
                    | ((boxes[:, 2] >= cw - m) & (x1 < image.size[0]))
                    | ((boxes[:, 3] >= ch - m) & (y1 < image.size[1]))
                )
                boxes, scores, cls = boxes[~cut], scores[~cut], cls[~cut]
            boxes[:, [0, 2]] += x0
            boxes[:, [1, 3]] += y0
            all_boxes.append(boxes)
            all_scores.append(scores)
            all_cls.append(cls)

        boxes = np.concatenate(all_boxes) if all_boxes else np.zeros((0, 4))
        scores = np.concatenate(all_scores) if all_scores else np.zeros(0)
        cls = np.concatenate(all_cls) if all_cls else np.zeros(0, dtype=int)

        detections: list[Detection] = []
        for c in np.unique(cls):
            idx = np.where(cls == c)[0]
            for k in nms(boxes[idx], scores[idx], iou):
                b = boxes[idx[k]]
                if b[2] - b[0] < 2 or b[3] - b[1] < 2:
                    continue
                detections.append(
                    Detection(float(b[0]), float(b[1]), float(b[2]), float(b[3]), float(scores[idx[k]]), int(c))
                )
        detections.sort(key=lambda d: (d.y1, d.x1))
        return detections

    def _tiles(self, size: tuple[int, int]) -> list[tuple[int, int, int, int]]:
        w, h = size
        tile = self.input_size
        if max(w, h) <= tile * 1.2:
            return []
        step = int(tile * 0.8)  # 20% de sobreposição
        xs = list(range(0, max(w - tile, 0) + 1, step)) or [0]
        ys = list(range(0, max(h - tile, 0) + 1, step)) or [0]
        if xs[-1] + tile < w:
            xs.append(w - tile)
        if ys[-1] + tile < h:
            ys.append(h - tile)
        return [(x, y, min(x + tile, w), min(y + tile, h)) for y in ys for x in xs]
