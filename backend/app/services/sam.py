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
