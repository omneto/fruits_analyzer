"""Orquestra detecção, maturação, tamanho e a imagem anotada."""

from __future__ import annotations

import base64
import io
import time
from collections import Counter

from PIL import Image, ImageDraw, ImageFont, ImageOps

from .detector import YoloOnnxDetector
from .fruits import Fruit
from .ripeness import STAGES, center_crop, classify_ripeness
from .size import SIZES, classify_sizes

# Evita "bombas de descompressão" (imagens gigantes que estouram a memória).
Image.MAX_IMAGE_PIXELS = 80_000_000

STAGE_COLORS = {
    "verde": (46, 160, 67),
    "amadurecendo": (230, 180, 20),
    "maduro": (240, 120, 20),
    "passado": (120, 70, 40),
    "indefinido": (130, 130, 130),
}


class InvalidImage(ValueError):
    pass


def load_image(data: bytes, max_side: int) -> Image.Image:
    try:
        image = Image.open(io.BytesIO(data))
        image = ImageOps.exif_transpose(image)  # corrige a rotação de fotos de celular
        image = image.convert("RGB")
    except Exception as exc:  # noqa: BLE001
        raise InvalidImage("Arquivo não é uma imagem válida.") from exc
    if max(image.size) > max_side:
        image.thumbnail((max_side, max_side), Image.LANCZOS)
    return image


def _font(size: int):
    # Pillow >= 10.1 traz uma fonte escalável embutida, sem depender do sistema.
    return ImageFont.load_default(size=size)


def draw_annotations(image: Image.Image, items: list[dict]) -> Image.Image:
    out = image.copy()
    draw = ImageDraw.Draw(out)
    line = max(2, round(max(image.size) / 400))
    font = _font(max(12, round(max(image.size) / 60)))
    for n, item in enumerate(items, start=1):
        x1, y1, x2, y2 = item["box"]
        color = STAGE_COLORS.get(item["ripeness"], (255, 255, 255))
        draw.rectangle((x1, y1, x2, y2), outline=color, width=line)
        label = str(n)
        tb = draw.textbbox((x1, y1), label, font=font)
        ty = max(y1 - (tb[3] - tb[1]) - 4, 0)
        draw.rectangle((x1, ty, x1 + (tb[2] - tb[0]) + 6, ty + (tb[3] - tb[1]) + 4), fill=color)
        draw.text((x1 + 3, ty), label, fill=(255, 255, 255), font=font)
    return out


def encode_jpeg(image: Image.Image, max_side: int = 1280) -> str:
    preview = image.copy()
    preview.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    preview.save(buf, format="JPEG", quality=82)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def analyze(
    image: Image.Image,
    fruit: Fruit,
    detector: YoloOnnxDetector,
    conf: float,
    iou: float,
    tiled: bool = False,
) -> dict:
    start = time.perf_counter()
    detections = detector.detect(image, set(fruit.class_ids), conf=conf, iou=iou, tiled=tiled)
    boxes = [(d.x1, d.y1, d.x2, d.y2) for d in detections]
    sizes = classify_sizes(boxes)

    items = []
    for n, (det, size) in enumerate(zip(detections, sizes), start=1):
        rip = classify_ripeness(center_crop(image, (det.x1, det.y1, det.x2, det.y2)), fruit)
        color_stage = rip.stage
        if fruit.stage_by_class:
            # O modelo treinado diz o estágio; o estágio pela cor vai junto para comparação.
            rip.stage = fruit.stage_by_class[det.class_id]
            rip.index = None
        items.append(
            {
                "id": n,
                "box": [round(v, 1) for v in (det.x1, det.y1, det.x2, det.y2)],
                "confidence": round(det.confidence, 3),
                "ripeness": rip.stage,
                "ripeness_index": rip.index,
                "ripeness_color": color_stage,
                "size": size,
                "colors": {k: round(v, 3) for k, v in rip.color_fractions.items()},
            }
        )

    by_ripeness = Counter(i["ripeness"] for i in items)
    by_ripeness_color = Counter(i["ripeness_color"] for i in items)
    by_size = Counter(i["size"] for i in items)
    annotated = draw_annotations(image, items)
    return {
        "fruit": fruit.key,
        "fruit_label": fruit.label,
        "total": len(items),
        "by_ripeness": {s: by_ripeness.get(s, 0) for s in [*STAGES, "indefinido"]},
        # Mesma contagem pelas regras de cor (igual a by_ripeness quando a fonte é "cor").
        "by_ripeness_color": {s: by_ripeness_color.get(s, 0) for s in [*STAGES, "indefinido"]},
        "by_size": {s: by_size.get(s, 0) for s in SIZES},
        "detections": items,
        "image": {"width": image.size[0], "height": image.size[1]},
        "tiled": tiled,
        "ripeness_source": fruit.ripeness_source,
        "processing_ms": round((time.perf_counter() - start) * 1000),
        "annotated_image": "data:image/jpeg;base64," + encode_jpeg(annotated),
    }
