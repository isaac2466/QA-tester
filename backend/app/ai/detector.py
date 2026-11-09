"""Paso 1 de la IA: DETECTOR de elementos entrenado sobre DINOv2 (estilo CenterNet).

DINOv2 (congelado, no se entrena) convierte la pantalla en una cuadrícula de "parches" de ~32px,
cada uno descrito por un vector de 384 números. Encima ponemos una red pequeña (la "cabeza") que SÍ
entrenamos con las pantallas sintéticas. Para cada parche aprende a predecir:

    - ¿aquí está el CENTRO de un elemento?  (2 mapas: bloques y textos)
    - ¿cuánto se corre el centro dentro del parche? (offset)
    - ¿qué tamaño tiene el elemento? (ancho, alto)

Después, SAM afina la silueta de cada bloque detectado (precisión ~1px).
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

GRID_W = 48  # parches a lo ancho (en una captura de 1536px cada parche ≈ 32px)
CLASSES = ["bloque", "texto"]


def dino_grid(dino, img: Image.Image) -> torch.Tensor:
    """Características de DINOv2 por parche: tensor (384, gh, gw)."""
    W, H = img.size
    gw = GRID_W
    gh = max(4, round(H * gw / W))
    x = dino._prep(img, (gh, gw))
    with dino._lock, torch.inference_mode():
        h = dino.model(pixel_values=x).last_hidden_state[0]
    start = 1 + dino.num_registers
    return h[start:].reshape(gh, gw, -1).permute(2, 0, 1).float().cpu()


class DetectorHead(nn.Module):
    def __init__(self, c_in=384, c=160):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(c_in, c, 3, padding=1), nn.ReLU(),
            nn.Conv2d(c, c, 3, padding=1), nn.ReLU(),
        )
        self.heat = nn.Conv2d(c, len(CLASSES), 1)
        self.offset = nn.Conv2d(c, 2, 1)
        self.size = nn.Conv2d(c, 2 * len(CLASSES), 1)  # log(ancho), log(alto) por clase, en parches
        nn.init.constant_(self.heat.bias, -2.2)  # al inicio "casi nada es centro"

    def forward(self, f):
        z = self.body(f)
        return self.heat(z), self.offset(z), self.size(z)


def _cls(t: str) -> int:
    return 1 if t == "texto" else 0


def targets(elements, W, H, gh, gw):
    """Mapas objetivo (lo que el modelo debería responder) para una pantalla etiquetada."""
    sx, sy = gw / W, gh / H
    heat = torch.zeros(len(CLASSES), gh, gw)
    off = torch.zeros(2, gh, gw)
    size = torch.zeros(2 * len(CLASSES), gh, gw)
    mask = torch.zeros(len(CLASSES), gh, gw)
    yy, xx = torch.meshgrid(torch.arange(gh), torch.arange(gw), indexing="ij")
    for e in elements:
        x, y, w, h = e["box"]
        c = _cls(e["type"])
        cx, cy = (x + w / 2) * sx, (y + h / 2) * sy
        ix, iy = min(gw - 1, int(cx)), min(gh - 1, int(cy))
        sigma = max(0.5, min(w * sx, h * sy) / 4)
        g = torch.exp(-((xx - ix) ** 2 + (yy - iy) ** 2) / (2 * sigma ** 2))
        heat[c] = torch.maximum(heat[c], g)
        off[:, iy, ix] = torch.tensor([cx - ix, cy - iy])
        size[2 * c:2 * c + 2, iy, ix] = torch.tensor([math.log(max(w * sx, 0.05)), math.log(max(h * sy, 0.05))])
        mask[c, iy, ix] = 1
    return heat, off, size, mask


def loss_fn(pred, tgt):
    heat_p, off_p, size_p = pred
    heat_t, off_t, size_t, mask = tgt
    p = torch.sigmoid(heat_p).clamp(1e-4, 1 - 1e-4)
    pos = heat_t.eq(1).float()
    # focal loss de CenterNet: castiga más los errores en los centros
    lp = -(torch.log(p) * (1 - p) ** 2 * pos).sum()
    ln = -(torch.log(1 - p) * p ** 2 * (1 - heat_t) ** 4 * (1 - pos)).sum()
    n = pos.sum().clamp(min=1)
    any_pos = mask.amax(0, keepdim=True)
    l_off = (F.l1_loss(off_p, off_t, reduction="none") * any_pos).sum() / n
    m2 = mask.repeat_interleave(2, dim=0)
    l_size = (F.l1_loss(size_p, size_t, reduction="none") * m2).sum() / n
    return (lp + ln) / n + l_off + 0.5 * l_size


def decode(pred, W, H, thr=0.3, max_det=200):
    """De los mapas del modelo a cajas [x, y, w, h] en píxeles."""
    heat_p, off_p, size_p = (t[0] if t.dim() == 4 else t for t in pred)
    _, gh, gw = heat_p.shape
    heat = torch.sigmoid(heat_p)
    peaks = heat * (F.max_pool2d(heat[None], 3, 1, 1)[0] == heat)  # máximos locales
    out = []
    for c in range(len(CLASSES)):
        ys, xs = torch.nonzero(peaks[c] >= thr, as_tuple=True)
        for y, x in zip(ys.tolist(), xs.tolist()):
            cx = (x + off_p[0, y, x].item()) * W / gw
            cy = (y + off_p[1, y, x].item()) * H / gh
            w = math.exp(size_p[2 * c, y, x].item()) * W / gw
            h = math.exp(size_p[2 * c + 1, y, x].item()) * H / gh
            out.append({"box": [round(cx - w / 2), round(cy - h / 2), round(w), round(h)],
                        "kind": CLASSES[c], "score": float(heat[c, y, x])})
    out.sort(key=lambda d: -d["score"])
    return out[:max_det]


class Detector:
    def __init__(self, head: DetectorHead, dino):
        self.head = head.eval()
        self.dino = dino

    def __call__(self, img: Image.Image, thr=0.3):
        f = dino_grid(self.dino, img)
        with torch.inference_mode():
            pred = self.head(f[None])
        return decode(pred, *img.size, thr=thr)


# ---------- uso en la app ----------

WEIGHTS = __import__("pathlib").Path(__file__).parent / "weights" / "detector.pt"
_loaded = None


def trained_elements(dino, rgb, thr=0.35):
    """Elementos según el detector ENTRENADO (o [] si todavía no existe app/ai/weights/detector.pt)."""
    global _loaded
    if not WEIGHTS.exists() or dino is None:
        return []
    if _loaded is None:
        head = DetectorHead()
        head.load_state_dict(torch.load(WEIGHTS, map_location="cpu"))
        _loaded = Detector(head, dino)
    from app.services.layout import Element
    H, W = rgb.shape[:2]
    out = []
    for d in _loaded(Image.fromarray(rgb), thr=thr):
        x, y, w, h = d["box"]
        x, y = max(0, x), max(0, y)
        w, h = min(w, W - x), min(h, H - y)
        # sólo bloques: en la 1ª prueba (60 pantallas) los textos salieron con 65% de precisión y meten ruido
        if d["kind"] == "bloque" and w >= 8 and h >= 6:
            out.append(Element(x, y, w, h, "bloque"))
    return out


def status():
    return {"entrenado": WEIGHTS.exists()}
