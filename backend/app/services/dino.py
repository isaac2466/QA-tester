"""Envoltura de DINOv2: extrae atención y características por parche de una captura."""
from __future__ import annotations
import threading

import cv2
import numpy as np
from PIL import Image

try:
    import torch
    import torch.nn.functional as F
    from transformers import AutoModel
    _import_error = None
except Exception as e:  # p.ej. DLL de torch que no carga en Windows
    _import_error = f"{type(e).__name__}: {e}"

from app.core.config import DINO_INPUT_WIDTH, DINO_MAX_HEIGHT, DINO_MODEL

PATCH = 14
CROP = 224  # tamaño del recorte de cada elemento (16×16 parches)
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _normalize01(x: np.ndarray, lo_pct=1, hi_pct=99) -> np.ndarray:
    lo, hi = np.percentile(x, lo_pct), np.percentile(x, hi_pct)
    if hi - lo < 1e-8:
        return np.zeros_like(x, dtype=np.float32)
    return np.clip((x - lo) / (hi - lo), 0, 1).astype(np.float32)


class DinoService:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # "eager" es necesario para poder pedir los mapas de atención.
        self.model = AutoModel.from_pretrained(DINO_MODEL, attn_implementation="eager")
        self.model.to(self.device).eval()
        self.num_registers = int(getattr(self.model.config, "num_register_tokens", 0) or 0)
        self._lock = threading.Lock()

    def grid_for(self, width: int, height: int) -> tuple[int, int]:
        gw = DINO_INPUT_WIDTH // PATCH
        gh = round(height * DINO_INPUT_WIDTH / width / PATCH)
        gh = int(np.clip(gh, 1, DINO_MAX_HEIGHT // PATCH))
        return gh, gw

    def _prep(self, img: Image.Image, grid: tuple[int, int]) -> torch.Tensor:
        gh, gw = grid
        # No usamos el AutoImageProcessor porque hace center-crop a 224px y
        # recortaría la captura. Aquí conservamos toda la página.
        resized = img.convert("RGB").resize((gw * PATCH, gh * PATCH), Image.BICUBIC)
        arr = (np.asarray(resized, dtype=np.float32) / 255.0 - MEAN) / STD
        return torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0).to(self.device)

    def features(self, img: Image.Image, grid: tuple[int, int] | None = None) -> dict:
        grid = grid or self.grid_for(*img.size)
        gh, gw = grid
        pixel_values = self._prep(img, grid)
        with self._lock, torch.inference_mode():
            out = self.model(pixel_values=pixel_values, output_attentions=True)
        start = 1 + self.num_registers  # [CLS, registros..., parches...]
        tokens = out.last_hidden_state[0]
        cls = tokens[0]
        patches = tokens[start:].reshape(gh, gw, -1)
        # Atención del token CLS hacia cada parche, última capa, promedio de cabezas.
        attn = out.attentions[-1][0, :, 0, start:].mean(0).reshape(gh, gw)
        return {"cls": cls.cpu(), "patches": patches.cpu(), "attn": attn.cpu().numpy(), "grid": grid}

    def saliency(self, img: Image.Image) -> np.ndarray:
        """Mapa 0..1 (tamaño de la imagen original) de qué zonas atraen la mirada.

        Combina la atención del CLS (lo que el modelo considera "el sujeto") con
        lo distintivo de cada parche frente al resto de la página (lo que destaca).
        """
        f = self.features(img)
        attn = _normalize01(f["attn"])
        p = F.normalize(f["patches"], dim=-1)
        mean = F.normalize(p.mean(dim=(0, 1)), dim=0)
        distinct = _normalize01((1 - p @ mean).numpy())
        sal = 0.5 * attn + 0.5 * distinct
        w, h = img.size
        sal = cv2.resize(sal, (w, h), interpolation=cv2.INTER_CUBIC)
        k = max(3, (min(w, h) // 40) | 1)
        sal = cv2.GaussianBlur(sal, (k, k), 0)
        return _normalize01(sal, 0, 100)

    def _crop_tensor(self, rgb: np.ndarray, box: list[int], size: int = CROP, margin: int = 4) -> "torch.Tensor":
        """Recorte de un elemento, centrado en un lienzo cuadrado (sin deformarlo)."""
        H, W = rgb.shape[:2]
        x, y, w, h = (int(round(float(v))) for v in box)
        x0, y0 = max(0, x - margin), max(0, y - margin)
        x1, y1 = min(W, x + w + margin), min(H, y + h + margin)
        crop = rgb[y0:y1, x0:x1]
        border = np.concatenate([crop[0], crop[-1], crop[:, 0], crop[:, -1]])
        canvas = np.empty((size, size, 3), dtype=np.uint8)
        canvas[:] = np.median(border, axis=0).astype(np.uint8)
        s = size / max(crop.shape[:2])
        cw, ch = max(1, round(crop.shape[1] * s)), max(1, round(crop.shape[0] * s))
        small = cv2.resize(crop, (cw, ch), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
        ox, oy = (size - cw) // 2, (size - ch) // 2
        canvas[oy:oy + ch, ox:ox + cw] = small
        arr = (canvas.astype(np.float32) / 255.0 - MEAN) / STD
        return torch.from_numpy(arr.transpose(2, 0, 1))

    def embed_boxes(self, rgb: np.ndarray, boxes: list[list[int]], batch: int = 32) -> np.ndarray:
        """Vector de DINOv2 (L2-normalizado) por cada elemento: CLS + promedio de parches.

        Dos elementos con vectores parecidos son "el mismo tipo de cosa" (p.ej. dos botones
        del mismo estilo) aunque tengan distinto texto, posición o tamaño.
        """
        if not boxes:
            return np.zeros((0, 2 * self.model.config.hidden_size), dtype=np.float32)
        start = 1 + self.num_registers
        out = []
        for i in range(0, len(boxes), batch):
            x = torch.stack([self._crop_tensor(rgb, b) for b in boxes[i:i + batch]]).to(self.device)
            with self._lock, torch.inference_mode():
                h = self.model(pixel_values=x).last_hidden_state
            emb = torch.cat([h[:, 0], h[:, start:].mean(1)], dim=-1)
            out.append(F.normalize(emb, dim=-1).cpu().numpy())
        return np.concatenate(out)

    def patch_similarity(self, a: Image.Image, b: Image.Image) -> tuple[np.ndarray, float]:
        """Similitud coseno por parche entre dos imágenes (mismo grid) y similitud global."""
        grid = self.grid_for(*a.size)
        fa, fb = self.features(a, grid), self.features(b, grid)
        sim = F.cosine_similarity(fa["patches"], fb["patches"], dim=-1).numpy()
        glob = float(F.cosine_similarity(fa["cls"], fb["cls"], dim=0))
        return sim, glob


_service: DinoService | None = None
_load_error: str | None = None


def load() -> None:
    global _service, _load_error
    if _import_error:
        _load_error = _import_error
        return
    try:
        _service = DinoService()
    except Exception as e:  # el servidor sigue funcionando sin la parte de IA
        _load_error = f"{type(e).__name__}: {e}"


def get() -> DinoService | None:
    return _service


def status() -> dict:
    if _service:
        return {"loaded": True, "model": DINO_MODEL, "device": _service.device}
    return {"loaded": False, "model": DINO_MODEL, "error": _load_error}
