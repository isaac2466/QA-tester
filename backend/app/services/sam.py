"""Envoltura de SAM 2.1 (Segment Anything, Meta): silueta exacta de cada elemento de la interfaz.

DINOv2 dice *qué* es cada elemento; SAM dice *qué píxeles* le pertenecen. Con esa máscara
podemos mover, redimensionar o recolorear un botón en la vista corregida sin dejar parches.
"""
from __future__ import annotations

import os
import threading

import numpy as np
from PIL import Image

try:
    import torch
    from transformers import Sam2Model, Sam2Processor
    _import_error = None
except Exception as e:
    _import_error = f"{type(e).__name__}: {e}"

SAM_MODEL = os.getenv("SAM_MODEL", "facebook/sam2.1-hiera-tiny")  # la versión más ligera (corre en CPU)


class SamService:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = Sam2Processor.from_pretrained(SAM_MODEL)
        self.model = Sam2Model.from_pretrained(SAM_MODEL).to(self.device).eval()
        self._lock = threading.Lock()

    def masks(self, rgb: np.ndarray, boxes: list[list[int]], batch: int = 64) -> list[np.ndarray]:
        """Una máscara booleana (alto×ancho de la imagen) por cada caja [x, y, w, h]."""
        if not boxes:
            return []
        img = Image.fromarray(rgb)
        out: list[np.ndarray] = []
        for i in range(0, len(boxes), batch):
            chunk = [[x, y, x + w, y + h] for x, y, w, h in boxes[i:i + batch]]
            inputs = self.processor(images=img, input_boxes=[chunk], return_tensors="pt").to(self.device)
            with self._lock, torch.inference_mode():
                pred = self.model(**inputs, multimask_output=False)
            masks = self.processor.post_process_masks(pred.pred_masks.cpu(), inputs["original_sizes"].cpu())[0]
            out += [m[0].numpy().astype(bool) for m in masks]  # [n_cajas, 1, H, W]
        return out


_service: SamService | None = None
_load_error: str | None = None


def load() -> None:
    global _service, _load_error
    if _import_error:
        _load_error = _import_error
        return
    try:
        _service = SamService()
    except Exception as e:  # sin SAM la vista corregida usa recortes rectangulares
        _load_error = f"{type(e).__name__}: {e}"


def get() -> SamService | None:
    return _service


def status() -> dict:
    if _service:
        return {"loaded": True, "model": SAM_MODEL, "device": _service.device}
    return {"loaded": False, "model": SAM_MODEL, "error": _load_error}


def refine_blocks(rgb: np.ndarray, elems: list, max_blocks: int = 40) -> int:
    """SAM revisa la caja de cada bloque (botón, tarjeta) y la ajusta a la silueta real.

    Corrige cajas que juntan dos botones pegados o que cortan un botón a la mitad. Si la respuesta
    de SAM no es confiable (muy distinta a la caja original), se deja la caja como estaba.
    Devuelve cuántas cajas se corrigieron.
    """
    if _service is None:
        return 0
    H, W = rgb.shape[:2]
    blocks = [e for e in elems if e.kind == "bloque" and e.w * e.h < 0.25 * W * H][:max_blocks]
    if not blocks:
        return 0
    masks = _service.masks(rgb, [e.box for e in blocks])
    fixed = 0
    for e, m in zip(blocks, masks):
        x0, y0 = max(0, e.x - 6), max(0, e.y - 6)
        x1, y1 = min(W, e.x + e.w + 6), min(H, e.y + e.h + 6)
        sub = m[y0:y1, x0:x1]
        ys, xs = np.nonzero(sub)
        if len(xs) < 30:
            continue
        nx, ny = x0 + int(xs.min()), y0 + int(ys.min())
        nw, nh = int(xs.max() - xs.min()) + 1, int(ys.max() - ys.min()) + 1
        ratio = (nw * nh) / max(1, e.w * e.h)
        # sólo cajas claramente mal hechas (≥4px de diferencia): las casi correctas se dejan como estaban,
        # para no meter ruido de 1-2px en las medidas
        if 0.45 <= ratio <= 1.25 and max(abs(nx - e.x), abs(ny - e.y), abs(nx + nw - e.x - e.w), abs(ny + nh - e.y - e.h)) >= 4:
            e.x, e.y, e.w, e.h = nx, ny, nw, nh
            fixed += 1
    return fixed
