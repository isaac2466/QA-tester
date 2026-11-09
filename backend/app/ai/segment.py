"""Paso 1 de la IA: SAM 2.1 encuentra los elementos de la interfaz ("segmentar todo").

Cómo funciona:
1. SAM procesa la imagen UNA vez (el codificador, la parte pesada: ~3 s en CPU).
2. Le "preguntamos" por cientos de puntos en cuadrícula: "¿qué objeto hay aquí?".
   Cada respuesta es una máscara + qué tan seguro está SAM (iou_score).
3. Nos quedamos con las máscaras seguras y quitamos repetidas.
4. Cada elemento se vuelve a pedir a SAM con su caja para tener la silueta exacta (precisión ~1px).
"""
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

SIZE = 1024  # SAM 2 trabaja internamente a 1024×1024


@dataclass
class Element:
    x: int
    y: int
    w: int
    h: int
    score: float  # confianza de SAM
    mask: np.ndarray  # máscara booleana recortada a la caja

    @property
    def box(self):
        return [self.x, self.y, self.w, self.h]

    @property
    def right(self):
        return self.x + self.w

    @property
    def bottom(self):
        return self.y + self.h

    @property
    def cx(self):
        return self.x + self.w / 2

    @property
    def cy(self):
        return self.y + self.h / 2

    @property
    def area(self):
        return self.w * self.h


def _box_iou(a, b) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[0] + a[2], b[0] + b[2]), min(a[1] + a[3], b[1] + b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    return inter / (a[2] * a[3] + b[2] * b[3] - inter + 1e-6)


class Segmenter:
    def __init__(self, sam_service):
        self.model = sam_service.model
        self.proc = sam_service.processor
        self.device = sam_service.device
        self.lock = sam_service._lock

    def _encode(self, img: Image.Image):
        inputs = self.proc(images=img, return_tensors="pt").to(self.device)
        with self.lock, torch.inference_mode():
            return self.model.get_image_embeddings(inputs["pixel_values"])

    def _decode(self, emb, points=None, boxes=None, multimask=True):
        kw = {}
        if points is not None:
            kw["input_points"] = points.to(self.device)
            kw["input_labels"] = torch.ones(points.shape[:3], dtype=torch.long, device=self.device)
        if boxes is not None:
            kw["input_boxes"] = boxes.to(self.device)
        with self.lock, torch.inference_mode():
            out = self.model(image_embeddings=emb, multimask_output=multimask, **kw)
        return out.pred_masks[0].cpu(), out.iou_scores[0].cpu()  # (n, k, 256, 256), (n, k)

    def _candidates(self, emb, W, H, grid=(48, 32), batch=256, min_score=0.75):
        gx, gy = grid
        xs = (torch.arange(gx) + 0.5) * SIZE / gx
        ys = (torch.arange(gy) + 0.5) * SIZE / gy
        pts = torch.stack(torch.meshgrid(xs, ys, indexing="xy"), -1).reshape(-1, 2)
        cands = []
        for i in range(0, len(pts), batch):
            p = pts[i:i + batch][None, :, None, :]  # (1, n, 1 punto, xy)
            masks, scores = self._decode(emb, points=p, multimask=True)
            m = masks > 0
            rows, cols = m.any(-1), m.any(-2)  # (n, k, 256)
            ok = (scores >= min_score) & rows.any(-1)
            for a, b in ok.nonzero().tolist():
                r, c = rows[a, b].nonzero().flatten(), cols[a, b].nonzero().flatten()
                y0, y1 = r[0].item(), r[-1].item() + 1
                x0, x1 = c[0].item(), c[-1].item() + 1
                box = [x0 * W / 256, y0 * H / 256, (x1 - x0) * W / 256, (y1 - y0) * H / 256]
                cands.append((float(scores[a, b]), box))
        return cands

    def _refine(self, emb, boxes, W, H, batch=64):
        """Pide a SAM cada caja: devuelve máscaras a resolución completa (precisión ~1px)."""
        out = []
        for i in range(0, len(boxes), batch):
            chunk = torch.tensor([[x * SIZE / W, y * SIZE / H, (x + w) * SIZE / W, (y + h) * SIZE / H]
                                  for x, y, w, h in boxes[i:i + batch]], dtype=torch.float32)[None]
            masks, scores = self._decode(emb, boxes=chunk, multimask=False)
            for j in range(masks.shape[0]):
                full = F.interpolate(masks[j:j + 1, :1], size=(H, W), mode="bilinear", align_corners=False)[0, 0] > 0
                out.append((full.numpy(), float(scores[j, 0])))
        return out

    def elements(self, rgb: np.ndarray, boxes: list | None = None) -> list[Element]:
        """Elementos de la imagen. Con `boxes` sólo afina esas cajas (lo usa el entrenamiento)."""
        H, W = rgb.shape[:2]
        emb = self._encode(Image.fromarray(rgb))
        if boxes is None:
            cands = sorted(self._candidates(emb, W, H), key=lambda c: -c[0])
            kept = []
            for s, b in cands:
                if b[2] < 10 or b[3] < 8 or b[2] * b[3] > 0.25 * W * H:
                    continue
                if all(_box_iou(b, k) < 0.7 for _, k in kept):
                    kept.append((s, b))
            boxes = [[round(v) for v in b] for _, b in kept[:160]]
        els = []
        for (full, s), b in zip(self._refine(emb, boxes, W, H), boxes):
            ys, xs = np.nonzero(full)
            if len(xs) < 20:
                continue
            x0, x1, y0, y1 = int(xs.min()), int(xs.max()) + 1, int(ys.min()), int(ys.max()) + 1
            els.append(Element(x0, y0, x1 - x0, y1 - y0, s, full[y0:y1, x0:x1]))
        # quitar duplicados que quedaron iguales tras afinar
        out: list[Element] = []
        for e in sorted(els, key=lambda e: -e.score):
            if all(_box_iou(e.box, o.box) < 0.85 for o in out):
                out.append(e)
        return out
