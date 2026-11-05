"""Detección de elementos de UI y revisión de alineación / espaciado con precisión de píxel."""
from dataclasses import dataclass
from itertools import combinations
from math import ceil

import cv2
import numpy as np

from app.services.findings import Finding


@dataclass
class Element:
    x: int
    y: int
    w: int
    h: int
    kind: str  # "bloque" (botón, tarjeta, input, imagen) | "texto"

    @property
    def left(self): return self.x
    @property
    def top(self): return self.y
    @property
    def right(self): return self.x + self.w
    @property
    def bottom(self): return self.y + self.h
    @property
    def cx(self): return self.x + self.w / 2
    @property
    def cy(self): return self.y + self.h / 2
    @property
    def area(self): return self.w * self.h
    @property
    def box(self): return [int(self.x), int(self.y), int(self.w), int(self.h)]

    def label(self):
        return "El bloque" if self.kind == "bloque" else "El texto"


def _iou(a: Element, b: Element) -> float:
    x1, y1 = max(a.x, b.x), max(a.y, b.y)
    x2, y2 = min(a.right, b.right), min(a.bottom, b.bottom)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    return inter / float(a.area + b.area - inter or 1)


def _dedupe(elems: list[Element], thr=0.8) -> list[Element]:
    out: list[Element] = []
    for e in sorted(elems, key=lambda e: -e.area):
        if all(_iou(e, o) < thr for o in out):
            out.append(e)
    return out


def _merge_text_lines(texts: list[Element]) -> list[Element]:
    """Une palabras de la misma línea (los titulares grandes tienen espacios más anchos)."""
    texts = sorted(texts, key=lambda t: (t.y, t.x))
    merged = True
    while merged:
        merged = False
        out: list[Element] = []
        for t in sorted(texts, key=lambda t: t.x):
            for o in out:
                same_line = _overlap(t.y, t.bottom, o.y, o.bottom) > 0.6 * min(t.h, o.h)
                gap = max(t.x, o.x) - min(t.right, o.right)
                if same_line and gap < 0.8 * max(t.h, o.h):
                    x0, y0 = min(t.x, o.x), min(t.y, o.y)
                    o.w, o.h = max(t.right, o.right) - x0, max(t.bottom, o.bottom) - y0
                    o.x, o.y = x0, y0
                    merged = True
                    break
            else:
                out.append(Element(t.x, t.y, t.w, t.h, t.kind))
        texts = out
    return texts


def detect_elements(rgb: np.ndarray, max_elements=160) -> list[Element]:
    H, W = rgb.shape[:2]
    # gradiente por canal: detecta también fondos de tarjeta muy sutiles (#F1F5F9 sobre blanco)
    grad = cv2.morphologyEx(rgb, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)).max(axis=2)
    edges = (grad > 7).astype(np.uint8) * 255

    # 1) Bloques: contornos casi rectangulares (botones, inputs, tarjetas, imágenes).
    blocks: list[Element] = []
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    cnts, _ = cv2.findContours(closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if w < 16 or h < 12 or w * h > 0.9 * W * H:
            continue
        if cv2.contourArea(c) / float(w * h) > 0.88:
            # el gradiente marca 1px por fuera del borde real → lo compensamos
            blocks.append(Element(x + 1, y + 1, w - 2, h - 2, "bloque"))
    blocks = _dedupe(blocks)

    # 2) Texto: se unen letras en líneas con una dilatación horizontal.
    kx = max(5, W // 160)
    lines = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_RECT, (kx, 1)))
    texts: list[Element] = []
    cnts, _ = cv2.findContours(lines, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if w < 12 or h < 6 or h > 120 or w * h > 0.5 * W * H:
            continue
        if cv2.contourArea(c) / float(w * h) > 0.88:  # rectángulo → ya es bloque
            continue
        # compensamos la dilatación horizontal para no sesgar el borde izquierdo/derecho
        pad = kx // 2 + 1
        texts.append(Element(x + pad, y + 1, max(1, w - 2 * pad), h - 2, "texto"))
    texts = _merge_text_lines(_dedupe(texts))
    texts = [t for t in texts if t.w >= 10 and t.h >= 8 and all(_iou(t, b) < 0.6 for b in blocks)]

    elems = blocks + texts
    elems.sort(key=lambda e: -e.area)
    return elems[:max_elements]


def _overlap(a0, a1, b0, b1) -> float:
    return max(0, min(a1, b1) - max(a0, b0))


HORIZONTAL = {"left": "borde izquierdo", "right": "borde derecho", "cx": "centro horizontal"}
VERTICAL = {"top": "borde superior", "bottom": "borde inferior", "cy": "centro vertical"}


def check_alignment(elems: list[Element], tol=4) -> list[Finding]:
    """Busca pares de elementos que *casi* están alineados (1..tol px de diferencia)."""
    candidates = {}
    # nº de *bloques* alineados exactamente en cada coordenada (el texto no cuenta:
    # el espacio interno de cada letra mueve su borde 1-3px)
    exact: dict = {}
    for attr in list(HORIZONTAL) + list(VERTICAL):
        for e in elems:
            if e.kind == "bloque":
                k = (attr, round(getattr(e, attr)))
                exact[k] = exact.get(k, 0) + 1

    for a, b in combinations(elems, 2):
        for attr in list(HORIZONTAL) + list(VERTICAL):
            horizontal = attr in HORIZONTAL
            if horizontal:
                # alineación de columna: uno encima del otro
                if _overlap(a.y, a.bottom, b.y, b.bottom) > 0:
                    continue
            elif _overlap(a.x, a.right, b.x, b.right) > 0:
                continue  # alineación de fila: uno al lado del otro
            # el texto sólo tiene sentido revisarlo en su borde de anclaje
            if "texto" in (a.kind, b.kind) and attr not in ("left", "top", "cy"):
                continue
            va, vb = getattr(a, attr), getattr(b, attr)
            d = vb - va
            # margen de tolerancia del texto: crece con el tamaño de la fuente
            min_d = max([1] + [max(2, ceil(0.1 * e.h)) for e in (a, b) if e.kind == "texto"])
            if not (min_d <= abs(d) <= tol):
                continue
            # ¿quién está mal? el que no coincide con otros bloques; si no, el más pequeño
            ca, cb = exact.get((attr, round(va)), 0), exact.get((attr, round(vb)), 0)
            off, ref = (b, a) if (ca > cb or (ca == cb and b.area <= a.area)) else (a, b)
            delta = getattr(off, attr) - getattr(ref, attr)
            key = (id(off), horizontal)  # un hallazgo por elemento y eje
            conf = 2 if a.kind == b.kind == "bloque" else 1
            conf += 1 if max(ca, cb) >= 2 else 0
            dist = (b.y - a.y) if horizontal else (b.x - a.x)
            prio = (conf, attr in ("left", "top"), -abs(dist))  # mejor: el vecino más cercano
            if key not in candidates or candidates[key][0] < prio:
                candidates[key] = (prio, off, ref, attr, delta)

    out = []
    offenders = set()
    for (conf, *_), off, ref, attr, delta in sorted(candidates.values(), key=lambda c: c[0], reverse=True):
        if id(ref) in offenders or len(out) >= 12:
            continue  # no usar como referencia algo que ya está mal colocado
        offenders.add(id(off))
        horizontal = attr in HORIZONTAL
        px = abs(delta)
        px_txt = f"{px:g}px"
        if horizontal:
            where = "a la derecha" if delta > 0 else "a la izquierda"
            fix = f"Muévelo {px_txt} a la {'izquierda' if delta > 0 else 'derecha'}"
            name = HORIZONTAL[attr]
        else:
            where = "más abajo" if delta > 0 else "más arriba"
            fix = f"Muévelo {px_txt} hacia {'arriba' if delta > 0 else 'abajo'}"
            name = VERTICAL[attr]
        rel = "el elemento de arriba/abajo" if horizontal else "el elemento de al lado"
        out.append(Finding(
            category="alineacion",
            severity="alta" if conf >= 3 else "media" if conf == 2 else "baja",
            title=f"Desalineado {px_txt} ({name})",
            detail=f"{off.label()} en x={off.x}, y={off.y} está {px_txt} {where} que {rel} "
                   f"(x={ref.x}, y={ref.y}). Casi coinciden, así que probablemente es un error.",
            suggestion=f"{fix} para que su {name} coincida.",
            box=off.box,
            ref_box=ref.box,
        ))
    return out


def check_spacing(elems: list[Element]) -> list[Finding]:
    """Detecta separaciones que casi son iguales (p.ej. 24px vs 23px)."""
    gaps = []  # (gap, eje, a, b)
    # sólo bloques: el espacio entre líneas de texto depende de la fuente y mete ruido
    blocks = [e for e in elems if e.kind == "bloque"]
    for a, b in combinations(blocks, 2):
        if a.bottom <= b.y or b.bottom <= a.y:  # apilados verticalmente
            top, bot = (a, b) if a.bottom <= b.y else (b, a)
            if _overlap(a.x, a.right, b.x, b.right) > 0.5 * min(a.w, b.w):
                gaps.append((bot.y - top.bottom, "vertical", top, bot))
        elif a.right <= b.x or b.right <= a.x:  # en la misma fila
            l, r = (a, b) if a.right <= b.x else (b, a)
            if _overlap(a.y, a.bottom, b.y, b.bottom) > 0.5 * min(a.h, b.h):
                gaps.append((r.x - l.right, "horizontal", l, r))

    # sólo vecinos inmediatos: descartamos gaps que tienen otro elemento en medio
    def nearest(gs):
        best = {}
        for g in gs:
            k = id(g[2])
            if g[0] >= 0 and (k not in best or g[0] < best[k][0]):
                best[k] = g
        return list(best.values())

    out = []
    for axis in ("vertical", "horizontal"):
        gs = [g for g in nearest([g for g in gaps if g[1] == axis]) if 4 <= g[0] <= 160]
        counts: dict[int, int] = {}
        for g in gs:
            counts[g[0]] = counts.get(g[0], 0) + 1
        for g, _, a, b in gs:
            for other, n in counts.items():
                on_grid = other % 4 == 0 and g % 4 != 0  # rejilla de 4/8px
                if 1 <= abs(g - other) <= 2 and (n > counts[g] or (n == counts[g] and on_grid)):
                    x0, y0 = min(a.x, b.x), min(a.y, b.y)
                    x1, y1 = max(a.right, b.right), max(a.bottom, b.bottom)
                    out.append(Finding(
                        category="espaciado",
                        severity="media",
                        title=f"Espaciado inconsistente: {g}px vs {other}px",
                        detail=f"La separación {axis} entre estos dos elementos es {g}px, pero en "
                               f"{'otra parte' if n == 1 else f'{n} otros lugares'} de la página usas {other}px.",
                        suggestion=f"Ajusta la separación a {other}px "
                                   f"({'+' if other > g else '-'}{abs(other - g)}px) para mantener el ritmo.",
                        box=[x0, y0, x1 - x0, y1 - y0],
                    ))
                    break
    return out[:8]


def check_centering(elems: list[Element], width: int) -> list[Finding]:
    out = []
    mid = width / 2
    for e in elems:
        if e.w > 0.9 * width:
            continue
        d = e.cx - mid
        min_d = 2 if e.kind == "texto" else 1
        if min_d <= abs(d) <= 4:
            side = "derecha" if d > 0 else "izquierda"
            out.append(Finding(
                category="alineacion",
                severity="media" if e.kind == "bloque" else "baja",
                title=f"Casi centrado ({abs(d):g}px a la {side})",
                detail=f"{e.label()} en x={e.x}, y={e.y} tiene su centro a {abs(d):g}px del centro de la página.",
                suggestion=f"Si debe ir centrado, muévelo {abs(d):g}px a la {'izquierda' if d > 0 else 'derecha'} "
                           f"(revisa margin/padding asimétricos o un scrollbar).",
                box=e.box,
            ))
    return out[:5]
