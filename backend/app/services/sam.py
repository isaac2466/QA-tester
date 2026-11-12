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

    def masks(self, rgb: np.ndarray, boxes: list[list[int]], batch: int = 8) -> list[np.ndarray]:
        """Una máscara booleana (alto×ancho de la imagen) por cada caja [x, y, w, h].

        La imagen se procesa UNA vez y las cajas se piden de 8 en 8: así el pico de memoria es chico
        (con 8 GB de RAM, pedir 64 cajas de golpe a tamaño completo tumbaba el servidor).
        """
        if not boxes:
            return []
        img = Image.fromarray(rgb)
        chunk = [[x, y, x + w, y + h] for x, y, w, h in boxes]
        inputs = self.processor(images=img, input_boxes=[chunk], return_tensors="pt").to(self.device)
        out: list[np.ndarray] = []
        with self._lock, torch.inference_mode():
            emb = self.model.get_image_embeddings(inputs["pixel_values"])
            for i in range(0, len(boxes), batch):
                pred = self.model(image_embeddings=emb, input_boxes=inputs["input_boxes"][:, i:i + batch],
                                  multimask_output=False)
                masks = self.processor.post_process_masks(pred.pred_masks.cpu(), inputs["original_sizes"].cpu())[0]
                out += [m[0].numpy().astype(bool) for m in masks]
                del pred, masks
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


def refine_blocks(rgb: np.ndarray, elems: list, max_blocks: int = 25) -> int:
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


def _edge_angle(rgb: np.ndarray, box) -> float | None:
    """Ángulo (grados) de los bordes largos casi horizontales del elemento, o None.

    Usa las líneas largas de los bordes (arriba/abajo): una etiqueta que se sale del borde o una
    sombra no las mueven, así que es más robusto que la silueta completa.
    """
    import cv2
    H, W = rgb.shape[:2]
    x, y, w, h = box
    x0, y0, x1, y1 = max(0, x - 12), max(0, y - 12), min(W, x + w + 12), min(H, y + h + 12)
    gray = cv2.cvtColor(np.ascontiguousarray(rgb[y0:y1, x0:x1]), cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 8, 24)  # sensible: bordes oscuros sobre fondo oscuro también cuentan
    lines = cv2.HoughLinesP(edges, 1, np.pi / 720, threshold=int(0.25 * w), minLineLength=int(0.45 * w), maxLineGap=16)
    if lines is None:
        return None
    angs = [float(np.degrees(np.arctan2(y2 - y1, x2 - x1))) for x1, y1, x2, y2 in np.asarray(lines).reshape(-1, 4)]
    angs = [a for a in angs if abs(a) <= 15]
    if len(angs) < 2:  # al menos dos bordes (o dos tramos) que coincidan
        return None
    med = float(np.median(angs))
    if sum(abs(a - med) <= 0.6 for a in angs) < 2:
        return None
    return med


def _filled_rect(rgb: np.ndarray, box):
    """Botón de color sólido: el relleno del mismo color alrededor de la caja → rectángulo girado. (caja, polígono) o None."""
    import cv2
    H, W = rgb.shape[:2]
    x, y, w, h = box
    px, py = int(max(0.6 * w, 4 * h)) + 10, int(1.2 * h) + 10
    x0, y0, x1, y1 = max(0, x - px), max(0, y - py), min(W, x + w + px), min(H, y + h + py)
    win = rgb[y0:y1, x0:x1].astype(np.int16)
    inner = rgb[y + h // 4:y + 3 * h // 4 + 1, x + w // 4:x + 3 * w // 4 + 1].reshape(-1, 3).astype(np.int16)
    border = np.concatenate([win[0], win[-1], win[:, 0], win[:, -1]])
    bg = np.median(border, axis=0)
    # color del relleno: el más frecuente adentro que NO es el fondo (el texto es minoría)
    cand = inner[np.abs(inner - bg).max(axis=1) > 40]
    if len(cand) < 0.3 * len(inner):
        return None
    fillc = np.median(cand, axis=0)
    m = (np.abs(win - fillc).max(axis=2) <= 30).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))  # quita bordes finos (otro botón con contorno)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))  # tapa el texto
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = (x + w / 2 - x0, y + h / 2 - y0)
    inside = [k for k in cnts if cv2.pointPolygonTest(k, c, False) >= 0]
    if not inside:
        return None
    k = max(inside, key=cv2.contourArea)
    rect = cv2.minAreaRect(k)
    area = rect[1][0] * rect[1][1]
    if area < 0.8 * w * h or area > 0.6 * (x1 - x0) * (y1 - y0) or cv2.contourArea(k) < 0.85 * area:
        return None  # no es un rectángulo sólido (o se pegó al fondo)
    pts = cv2.boxPoints(((rect[0][0] + x0, rect[0][1] + y0), (rect[1][0] + 4, rect[1][1] + 4), rect[2]))
    bx0, by0 = np.floor(pts.min(0)).astype(int) - 3
    bx1, by1 = np.ceil(pts.max(0)).astype(int) + 3
    bx0, by0, bx1, by1 = max(0, bx0), max(0, by0), min(W, bx1), min(H, by1)
    return [int(bx0), int(by0), int(bx1 - bx0), int(by1 - by0)], pts.round().astype(int).tolist()


def _tilted_rect(rgb: np.ndarray, box, angle: float):
    """Rectángulo girado COMPLETO del elemento (aunque la caja detectada sólo cubra una parte).

    Busca, alrededor de la caja, los bordes rectos con el mismo ángulo y mide hasta dónde llegan
    (funciona aunque la tarjeta sea casi del color del fondo). Devuelve (caja, polígono) o None.
    """
    import cv2
    H, W = rgb.shape[:2]
    x, y, w, h = box
    px, py = int(0.6 * w) + 10, int(0.6 * h) + 10
    x0, y0, x1, y1 = max(0, x - px), max(0, y - py), min(W, x + w + px), min(H, y + h + py)
    gray = cv2.cvtColor(np.ascontiguousarray(rgb[y0:y1, x0:x1]), cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 8, 24)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 720, threshold=int(0.25 * w), minLineLength=int(0.45 * w), maxLineGap=16)
    if lines is None:
        return None
    t = np.radians(angle)
    u, v = np.array([np.cos(t), np.sin(t)]), np.array([-np.sin(t), np.cos(t)])
    c = np.array([x + w / 2 - x0, y + h / 2 - y0])
    pts = []
    for a, b, c2, d in np.asarray(lines).reshape(-1, 4):
        ang = np.degrees(np.arctan2(d - b, c2 - a))
        if abs(ang - angle) > 1.0:
            continue
        for q in ((a, b), (c2, d)):
            pts.append(np.array(q, dtype=float) - c)
    if len(pts) < 4:
        return None
    pu = np.array([p @ u for p in pts])
    pv = np.array([p @ v for p in pts])
    # a lo alto la caja detectada sí suele estar completa: sólo bordes dentro de esa franja (no la línea de otra sección)
    band = np.abs(pv) <= 0.65 * h
    if band.sum() < 4:
        return None
    pu, pv = pu[band], pv[band]
    top, bot, left, right = pv.min(), pv.max(), pu.min(), pu.max()
    if right - left < 0.8 * w or bot - top < 0.4 * h:
        return None
    mu, mv = 0.04 * (right - left) + 3, 4  # margen: esquinas redondeadas y antialias, para que no queden restos
    top, bot, left, right = top - mv, bot + mv, left - mu, right + mu
    corners = np.array([c + su * u + sv * v + [x0, y0]
                        for su, sv in ((left, top), (right, top), (right, bot), (left, bot))])
    bx0, by0 = np.floor(corners.min(0)).astype(int) - 4
    bx1, by1 = np.ceil(corners.max(0)).astype(int) + 4
    bx0, by0, bx1, by1 = max(0, bx0), max(0, by0), min(W, bx1), min(H, by1)
    return [int(bx0), int(by0), int(bx1 - bx0), int(by1 - by0)], corners.round().astype(int).tolist()


def check_tilt(rgb: np.ndarray, elems: list, max_items: int = 6) -> list:
    """Tarjetas, imágenes y botones inclinados: se mide el ángulo de sus bordes largos (después se enderezan)."""
    from app.services.findings import Finding
    H, W = rgb.shape[:2]
    cands = [e for e in elems if e.kind == "bloque" and e.w >= 70 and e.h >= 24 and e.w * e.h <= 0.4 * W * H]
    cands.sort(key=lambda e: -e.w * e.h)
    out, seen = [], []
    for e in cands[:20]:
        a = _edge_angle(rgb, e.box)
        if a is None or not (0.8 <= abs(a) <= 15):
            continue
        full = _filled_rect(rgb, e.box) or _tilted_rect(rgb, e.box, a)
        if full:
            box, poly = full
        elif e.w >= 120 and e.h >= 80:
            box, poly = list(e.box), None  # tarjeta/imagen grande: se gira su caja
        else:
            continue  # sólo vemos un pedazo: girarlo dejaría el elemento roto
        # varias cajas parciales de la misma tarjeta → un solo hallazgo
        if any(_overlap(box, b) > 0.5 for b in seen):
            continue
        seen.append(box)
        out.append(Finding(
            category="alineacion", severity="baja", source="sam",
            title=f"Elemento inclinado {abs(a):.1f}°",
            detail=f"Los bordes de este elemento están girados {abs(a):.1f}° respecto a la página, "
                   f"mientras el resto de la interfaz está recto.",
            suggestion="Si no es un efecto intencional (algunas landings inclinan imágenes del producto como decoración), "
                       "quita el transform: rotate(...) para enderezarlo. Si es intencional, márcalo como «No es error».",
            box=box, fix={"type": "rotate", "box": box, "angle": a, "poly": poly},
        ))
    return out[:max_items]


def _overlap(a, b) -> float:
    """Qué fracción de la caja más chica queda dentro de la otra."""
    ix = max(0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    return ix * iy / max(1, min(a[2] * a[3], b[2] * b[3]))
