"""Paso 2 de la IA: el "juez" que decide si un elemento tiene un error.

1. DINOv2 agrupa los elementos que son "el mismo componente" (mismo botón, misma tarjeta...).
2. De cada elemento se calcula cómo se compara con sus gemelos (posición, tamaño, color, parecido DINOv2).
3. Una red pequeña ENTRENADA con ejemplos etiquetados decide:
       ninguno | desalineado | tamano | color
   Nadie le programó "si difiere 2px es error": lo aprende de los ejemplos (y de sus errores).
4. Otra red pequeña decide si un texto es poco legible ("contraste").
5. Otra más reconoce qué tipo de componente es (botón, tarjeta, etiqueta...), sólo con DINOv2.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

BLOCK_CLASSES = ["ninguno", "desalineado", "tamano", "color"]
TYPES = ["boton", "tarjeta", "etiqueta", "avatar", "campo", "icono", "pagina"]
GROUP_SIM = 0.80


# ---------- utilidades de color (son medidas, no decisiones) ----------

def to_lab(rgb):
    import cv2
    arr = np.asarray(rgb, dtype=np.float32).reshape(-1, 1, 3) / 255.0
    return cv2.cvtColor(arr, cv2.COLOR_RGB2Lab).reshape(-1, 3)


def delta_e(a, b) -> float:
    la, lb = to_lab(np.array([a, b]))
    return float(np.linalg.norm(la - lb))


def luminance(c) -> float:
    c = np.asarray(c, dtype=np.float64) / 255.0
    c = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    return float(0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2])


def contrast_ratio(a, b) -> float:
    la, lb = sorted((luminance(a), luminance(b)), reverse=True)
    return (la + 0.05) / (lb + 0.05)


def hex_of(c) -> str:
    r, g, b = (int(round(float(v))) for v in c[:3])
    return f"#{r:02X}{g:02X}{b:02X}"


def fill_color(rgb, box):
    """Color más frecuente del interior del elemento."""
    x, y, w, h = box
    m = 3 if min(w, h) > 10 else 0
    inner = rgb[max(0, y + m):y + h - m, max(0, x + m):x + w - m].reshape(-1, 3)
    if not len(inner):
        return np.array([255, 255, 255])
    packed = (inner[:, 0].astype(np.int32) << 16) | (inner[:, 1].astype(np.int32) << 8) | inner[:, 2]
    vals, counts = np.unique(packed, return_counts=True)
    v = int(vals[np.argmax(counts)])
    return np.array([(v >> 16) & 255, (v >> 8) & 255, v & 255])


def text_colors(rgb, box, pad=3):
    """(color del texto, color de fondo) de una caja de texto."""
    H, W = rgb.shape[:2]
    x, y, w, h = box
    x0, y0, x1, y1 = max(0, x - pad), max(0, y - pad), min(W, x + w + pad), min(H, y + h + pad)
    crop = rgb[y0:y1, x0:x1].reshape(-1, 3).astype(np.float32)
    if len(crop) < 10:
        return np.zeros(3), np.zeros(3)
    ring = np.concatenate([rgb[y0, x0:x1], rgb[y1 - 1, x0:x1], rgb[y0:y1, x0], rgb[y0:y1, x1 - 1]]).astype(np.float32)
    bg = np.median(ring, axis=0)
    dist = np.linalg.norm(crop - bg, axis=1)
    fg = np.median(crop[dist >= np.percentile(dist, 90)], axis=0)
    return fg, bg


# ---------- grupos con DINOv2 ----------

def group_blocks(boxes, emb):
    """Agrupa por parecido de DINOv2 (+ tamaño compatible)."""
    groups, cents = [], []
    for i, (x, y, w, h) in enumerate(boxes):
        best, bs = None, GROUP_SIM
        for g, (mem, c) in enumerate(zip(groups, cents)):
            _, _, w0, h0 = boxes[mem[0]]
            if max(h, h0) > 1.5 * min(h, h0) or max(w, w0) > 3 * min(w, w0):
                continue
            s = float(emb[i] @ c)
            if s >= bs:
                best, bs = g, s
        if best is None:
            groups.append([i])
            cents.append(emb[i].copy())
        else:
            groups[best].append(i)
            c = emb[groups[best]].mean(0)
            cents[best] = c / (np.linalg.norm(c) or 1)
    return groups


def _ov(a0, a1, b0, b1):
    return max(0, min(a1, b1) - max(a0, b0))


N_BLOCK_FEATURES = 26


def block_features(i, members, boxes, fills, emb):
    """Cómo se compara el elemento i con sus gemelos. Devuelve (vector, medidas útiles para la corrección)."""
    x, y, w, h = boxes[i]
    sib = [j for j in members if j != i]
    B = np.array([boxes[j] for j in sib], dtype=np.float32)
    row = [j for j in sib if _ov(y, y + h, boxes[j][1], boxes[j][1] + boxes[j][3]) > 0.5 * min(h, boxes[j][3])]
    col = [j for j in sib if _ov(x, x + w, boxes[j][0], boxes[j][0] + boxes[j][2]) > 0.5 * min(w, boxes[j][2])]

    def d(js, k, val):
        if not js:
            return 0.0, 0.0
        vals = np.array([boxes[j][k] + (boxes[j][2 + k % 2] if val == "end" else 0) for j in js], dtype=np.float32)
        me = float(np.median(vals))
        mine = (x, y)[k % 2] + ((w, h)[k % 2] if val == "end" else 0)
        same = float(np.mean(np.abs(vals - me) <= 1))
        return mine - me, same

    d_top, s_top = d(row, 1, "start")
    d_bot, _ = d(row, 1, "end")
    d_left, s_left = d(col, 0, "start")
    d_right, _ = d(col, 0, "end")
    mh, mw = float(np.median(B[:, 3])), float(np.median(B[:, 2]))
    s_h, s_w = float(np.mean(np.abs(B[:, 3] - mh) <= 1)), float(np.mean(np.abs(B[:, 2] - mw) <= 1))
    # separación con el vecino de fila (los errores de margen cambian el hueco)
    gaps = sorted(boxes[j][0] for j in row)
    left_n = [boxes[j] for j in row if boxes[j][0] < x]
    gap_l = x - max(b[0] + b[2] for b in left_n) if left_n else 0.0
    rowg = sorted([boxes[j] for j in row] + [boxes[i]], key=lambda b: b[0])
    gl = [b2[0] - (b1[0] + b1[2]) for b1, b2 in zip(rowg, rowg[1:])]
    d_gap = gap_l - float(np.median(gl)) if (left_n and gl) else 0.0
    sib_fill = np.median(np.array([fills[j] for j in sib], dtype=np.float32), axis=0)
    de = delta_e(fills[i], sib_fill)
    spread = float(np.median([delta_e(fills[j], sib_fill) for j in sib]))
    c = emb[sib].mean(0)
    c /= np.linalg.norm(c) or 1
    cos = float(emb[i] @ c)
    v = [d_top, d_bot, d_left, d_right, abs(d_top), abs(d_bot), abs(d_left), abs(d_right),
         float(bool(row)), float(bool(col)), s_top, s_left,
         h - mh, w - mw, abs(h - mh), abs(w - mw), (h - mh) / max(mh, 1), (w - mw) / max(mw, 1), s_h, s_w,
         d_gap, abs(d_gap), de, spread, cos, len(members)]
    info = {"d_top": d_top, "d_left": d_left, "has_row": bool(row), "has_col": bool(col),
            "mh": mh, "mw": mw, "sib_fill": sib_fill, "row": row, "col": col, "sib": sib}
    return np.array(v, dtype=np.float32), info


N_TEXT_FEATURES = 6


def text_features(rgb, box):
    fg, bg = text_colors(rgb, box)
    r = contrast_ratio(fg, bg)
    return np.array([r, np.log(r), box[3], 1.0 / max(r, 1e-3), delta_e(fg, bg), luminance(bg)], dtype=np.float32), (fg, bg, r)


# ---------- redes ----------

class MLP(nn.Module):
    def __init__(self, n_in, n_out, hidden=64):
        super().__init__()
        self.norm = nn.BatchNorm1d(n_in)
        self.net = nn.Sequential(nn.Linear(n_in, hidden), nn.ReLU(), nn.Dropout(0.1),
                                 nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, n_out))

    def forward(self, x):
        return self.net(self.norm(x))


class Judge:
    """Las 3 redes entrenadas + los umbrales elegidos en validación."""

    def __init__(self, path):
        ck = torch.load(path, weights_only=False)
        self.block = MLP(N_BLOCK_FEATURES, len(BLOCK_CLASSES)).eval()
        self.block.load_state_dict(ck["block"])
        self.text = MLP(N_TEXT_FEATURES, 2, hidden=16).eval()
        self.text.load_state_dict(ck["text"])
        self.types = nn.Linear(ck["emb_dim"], len(TYPES)).eval()
        self.types.load_state_dict(ck["types"])
        self.thr = ck["thresholds"]
        self.meta = ck.get("meta", {})

    @torch.inference_mode()
    def predict_blocks(self, X):
        return torch.softmax(self.block(torch.from_numpy(np.asarray(X, dtype=np.float32))), -1).numpy()

    @torch.inference_mode()
    def predict_text(self, X):
        return torch.softmax(self.text(torch.from_numpy(np.asarray(X, dtype=np.float32))), -1).numpy()[:, 1]

    @torch.inference_mode()
    def predict_types(self, emb):
        p = torch.softmax(self.types(torch.from_numpy(np.asarray(emb, dtype=np.float32))), -1).numpy()
        return [(TYPES[int(k)], float(p[n, k])) for n, k in enumerate(p.argmax(1))]
