"""Comparador con DINOv2 como motor: empareja cada elemento del diseño con su pareja en la implementación.

Paso a paso:
1. Se detectan los elementos (bloques y textos) en ambas imágenes.
2. DINOv2 convierte cada elemento en un vector que describe "qué es" (botón azul, tarjeta, título...).
3. Se emparejan diseño ↔ implementación por parecido según DINOv2 (+ cercanía como desempate).
4. De cada pareja se miden las diferencias: posición, tamaño, color y contenido.
   - Elemento del diseño sin pareja     → falta en la implementación.
   - Elemento de la implementación sin pareja → sobra (no está en el diseño).

DINOv2 decide *qué es cada cosa y cuál es su pareja*; el código sólo pone los números (px, hex).
"""
from math import hypot

import cv2
import numpy as np

from app.services.colors import delta_e, hex_of
from app.services.components import _fill, _noun
from app.services.findings import Finding
from app.services.layout import Element, _iou, detect_elements

MAX_ELEMENTS = 90
MIN_SIMILARITY = 0.55  # por debajo, DINOv2 no los considera el mismo elemento
CONTENT_CHANGED = 0.80  # emparejados pero con contenido distinto (texto, ícono, imagen)
MAX_DISTANCE = 0.25  # fracción de la diagonal: más lejos no se considera la misma pieza


def _elements(rgb: np.ndarray) -> list[Element]:
    H, W = rgb.shape[:2]
    els = [e for e in detect_elements(rgb) if e.w >= 12 and e.h >= 8 and e.area < 0.5 * W * H]
    return els[:MAX_ELEMENTS]


def _name(e: Element, rgb=None) -> tuple[str, str]:
    """(nombre, artículo) legible del elemento."""
    if e.kind == "texto":
        return ("título" if e.h >= 28 else "texto"), "el"
    n = _noun([e], rgb)[0]
    return n, "la" if n == "tarjeta" else "el"


def _match(D, I, ED, EI, W, H):
    """Emparejamiento voraz por (similitud DINOv2 − penalización por distancia)."""
    diag = hypot(W, H)
    sim = ED @ EI.T
    pairs = []
    for a, d in enumerate(D):
        for b, i in enumerate(I):
            s = float(sim[a, b])
            if s < MIN_SIMILARITY or d.kind != i.kind:
                continue
            if max(d.h, i.h) > 2 * min(d.h, i.h) or max(d.w, i.w) > 2 * min(d.w, i.w):
                continue
            dist = hypot(i.cx - d.cx, i.cy - d.cy) / diag
            if dist > MAX_DISTANCE:
                continue
            pairs.append((s - 1.5 * dist, s, a, b))
    pairs.sort(reverse=True)
    used_a, used_b, out = set(), set(), []
    for _, s, a, b in pairs:
        if a in used_a or b in used_b:
            continue
        used_a.add(a)
        used_b.add(b)
        out.append((a, b, s))
    return out


def _global_shift(D, I, matches):
    """Si casi todo se movió lo mismo, es la página entera (margen del contenedor)."""
    good = [(I[b].x - D[a].x, I[b].y - D[a].y) for a, b, s in matches if s >= 0.8]
    if len(good) < 4:
        return 0, 0
    dx, dy = (int(np.median([g[k] for g in good])) for k in (0, 1))
    agree = sum(abs(gx - dx) <= 1 and abs(gy - dy) <= 1 for gx, gy in good)
    return (dx, dy) if agree >= 0.6 * len(good) and (dx or dy) else (0, 0)


def _dir(dx, dy):
    where, fix = [], []
    if dx:
        where.append(f"{abs(dx)}px más a la {'derecha' if dx > 0 else 'izquierda'}")
        fix.append(f"{abs(dx)}px a la {'izquierda' if dx > 0 else 'derecha'}")
    if dy:
        where.append(f"{abs(dy)}px más {'abajo' if dy > 0 else 'arriba'}")
        fix.append(f"{abs(dy)}px hacia {'arriba' if dy > 0 else 'abajo'}")
    return " y ".join(where), " y ".join(fix)


def _region_diff(a: np.ndarray, b: np.ndarray, box, shift=(0, 0)) -> float:
    """Diferencia media (0-255) entre la zona `box` del diseño y la misma zona (desplazada) de la implementación."""
    H, W = a.shape[:2]
    x, y, w, h = box
    sx, sy = shift
    x0, y0, x1, y1 = max(0, x), max(0, y), min(W, x + w), min(H, y + h)
    bx0, by0 = max(0, x0 + sx), max(0, y0 + sy)
    bx1, by1 = min(W, x1 + sx), min(H, y1 + sy)
    ww, hh = min(x1 - x0, bx1 - bx0), min(y1 - y0, by1 - by0)
    if ww <= 0 or hh <= 0:
        return 255.0
    ca = a[y0:y0 + hh, x0:x0 + ww].astype(np.int16)
    cb = b[by0:by0 + hh, bx0:bx0 + ww].astype(np.int16)
    return float(np.abs(ca - cb).max(axis=2).mean())


def _ink(crop: np.ndarray) -> np.ndarray:
    """Máscara de 'tinta' (letras) de un recorte: lo que se aleja del color de fondo."""
    border = np.concatenate([crop[0], crop[-1], crop[:, 0], crop[:, -1]]).astype(np.float32)
    bg = np.median(border, axis=0)
    return np.linalg.norm(crop.astype(np.float32) - bg, axis=2) > 40


def _text_changed(a: np.ndarray, b: np.ndarray, da: Element, db: Element) -> bool:
    """¿Cambió el texto? Compara la forma de las letras (no el color): un fondo distinto no cuenta."""
    m = 3
    def crop(img, e):
        H, W = img.shape[:2]
        return img[max(0, e.y - m):min(H, e.bottom + m), max(0, e.x - m):min(W, e.right + m)]
    ca, cb = crop(a, da), crop(b, db)
    h, w = min(ca.shape[0], cb.shape[0]), min(ca.shape[1], cb.shape[1])
    if abs(ca.shape[1] - cb.shape[1]) > 0.15 * max(ca.shape[1], cb.shape[1]):
        return True  # el renglón cambió mucho de largo: otro texto
    ia, ib = _ink(ca)[:h, :w], _ink(cb)[:h, :w]
    union = (ia | ib).sum()
    return bool(union) and (ia ^ ib).sum() / union > 0.35


def _contains(outer: Element, inner: Element, m=2) -> bool:
    return outer is not inner and outer.x - m <= inner.x and outer.y - m <= inner.y and \
        inner.right <= outer.right + m and inner.bottom <= outer.bottom + m


def _drop_children(fs: list[Finding]) -> list[Finding]:
    """Si falta un botón, no reportamos también "falta su texto": el hallazgo es el botón."""
    def inside(a, b):
        return a is not b and b[0] - 2 <= a[0] and b[1] - 2 <= a[1] and             a[0] + a[2] <= b[0] + b[2] + 2 and a[1] + a[3] <= b[1] + b[3] + 2
    return [f for f in fs if not any(inside(f.box, o.box) for o in fs)]


def compare_dino(design: np.ndarray, impl: np.ndarray, dino) -> tuple[list[Finding], dict]:
    H, W = design.shape[:2]
    D, I = _elements(design), _elements(impl)
    ED = dino.embed_boxes(design, [e.box for e in D])
    EI = dino.embed_boxes(impl, [e.box for e in I])
    matches = _match(D, I, ED, EI, W, H)
    gdx, gdy = _global_shift(D, I, matches)

    findings: list[Finding] = []
    if gdx or gdy:
        where, fix = _dir(gdx, gdy)
        findings.append(Finding(
            category="diferencia", severity="alta", source="dinov2",
            title=f"Toda la página está desplazada {where}",
            detail=f"DINOv2 emparejó los elementos de ambas imágenes y casi todos se movieron igual ({where}). "
                   f"No es un elemento: es el contenedor completo.",
            suggestion=f"Revisa el margin/padding del contenedor raíz (body, main o el layout): muévelo {fix}.",
        ))

    # --- 1) Parejas: ¿qué cambió en cada elemento? ---
    moved = {}  # índice diseño -> (dx, dy) ya descontado el desplazamiento global
    per_pair = []
    for a, b, s in matches:
        d, i = D[a], I[b]
        tol = 1 if d.kind == "bloque" else 2  # el borde del texto baila 1px por el antialias
        dx, dy = i.x - d.x - gdx, i.y - d.y - gdy
        dx = dx if abs(dx) >= tol else 0
        dy = dy if abs(dy) >= tol else 0
        dw = i.w - d.w if abs(i.w - d.w) >= tol else 0
        dh = i.h - d.h if abs(i.h - d.h) >= tol else 0
        if dx or dy:
            moved[a] = (dx, dy)
        per_pair.append((a, b, s, dx, dy, dw, dh))

    for a, b, s, dx, dy, dw, dh in per_pair:
        d, i = D[a], I[b]
        # si su contenedor se movió igual, el hallazgo es del contenedor (no repetimos por cada hijo)
        if (dx or dy) and any(_contains(D[p], d) and abs(mx - dx) <= 1 and abs(my - dy) <= 1
                              for p, (mx, my) in moved.items() if p != a):
            dx = dy = 0
        name, art = _name(d, design)
        Name = name.capitalize()
        changes, fixes, titles, sev = [], [], [], "baja"
        if dx or dy:
            where, fix = _dir(dx, dy)
            changes.append(f"está {where}")
            fixes.append(f"muévelo {fix} (revisa margin, padding, gap o line-height)")
            titles.append(f"{Name} desplazad{'a' if art == 'la' else 'o'}: {where}")
            sev = "alta" if max(abs(dx), abs(dy)) >= 3 else "media"
        text_changed = d.kind == "texto" and _text_changed(design, impl, d, i)
        if text_changed:
            dw = dh = 0  # el tamaño cambia porque cambió el texto; ese es el hallazgo
        if dw or dh:
            changes.append(f"mide {i.w}×{i.h}px en vez de {d.w}×{d.h}px")
            fixes.append(f"ajusta su tamaño a {d.w}×{d.h}px (width/height, padding o font-size)")
            titles.append(f"{Name} con otro tamaño: {i.w}×{i.h} vs {d.w}×{d.h}px")
            sev = "alta" if max(abs(dw), abs(dh)) >= 4 else ("media" if sev == "baja" else sev)
        color_changed = False
        if d.kind == "bloque":
            c1, c2 = _fill(design, d), _fill(impl, i)
            if delta_e(c1, c2) > 3:
                color_changed = True
                changes.append(f"su fondo es {hex_of(c2)} en vez de {hex_of(c1)}")
                fixes.append(f"cambia el color de fondo a {hex_of(c1)}")
                titles.append(f"{Name} con otro color: {hex_of(c1)} → {hex_of(c2)}")
                sev = "media" if sev == "baja" else sev
        # contenido distinto: DINOv2 los empareja, pero ya no "se ven igual" (y los píxeles lo confirman)
        if text_changed:
            changes.append("el texto es distinto al del diseño")
            fixes.append("copia el texto exacto del diseño (revisa también tipografía y tamaño de fuente)")
            titles.append(f"{Name} distinto al diseño")
            sev = "media" if sev == "baja" else sev
        elif s < CONTENT_CHANGED and _region_diff(design, impl, d.box, (i.x - d.x, i.y - d.y)) > 8:
            changes.append(f"su contenido cambió (DINOv2: {s:.0%} de parecido)")
            fixes.append("revisa texto, ícono, imagen o tipografía contra el diseño")
            titles.append(f"{Name} con contenido distinto")
            sev = "media" if sev == "baja" else sev
        if not changes:
            continue
        category = "alineacion" if (dx or dy) else "espaciado" if (dw or dh) else "color" if color_changed else "diferencia"
        fix_text = fixes[0][0].upper() + fixes[0][1:]
        if len(fixes) > 1:
            fix_text += ". Además, " + "; ".join(fixes[1:])
        findings.append(Finding(
            category=category, severity=sev, source="dinov2", confidence=s,
            title=titles[0],
            detail=f"DINOv2 reconoce {art} {name} del diseño en la implementación ({s:.0%} de parecido). "
                   f"En la implementación {', '.join(changes)}.",
            suggestion=fix_text + ".",
            box=i.box, ref_box=d.box,
        ))

    # --- 2) Sin pareja: falta o sobra ---
    ma = {a for a, _, _ in matches}
    mb = {b for _, b, _ in matches}
    miss, extra = [], []
    for a, d in enumerate(D):
        if a in ma or d.area < 300:
            continue
        if any(_iou(d, I[b]) > 0.5 for b in mb):  # la detección lo partió distinto, pero ahí está
            continue
        if _region_diff(design, impl, d.box, (gdx, gdy)) < 6:  # los píxeles confirman que sigue ahí
            continue
        name, art = _name(d, design)
        miss.append(Finding(
            category="diferencia", severity="alta", source="dinov2",
            title=f"Falta {art} {name} del diseño",
            detail=f"DINOv2 no encontró en la implementación ningún elemento parecido {'al' if art == 'el' else 'a la'} {name} que en el diseño "
                   f"está en x={d.x}, y={d.y} ({d.w}×{d.h}px).",
            suggestion="Agrégalo, o revisa si quedó oculto (display:none, overflow, z-index) o si cambió tanto que ya no se reconoce.",
            box=[d.x + gdx, d.y + gdy, d.w, d.h], ref_box=d.box,
        ))
    for b, i in enumerate(I):
        if b in mb or i.area < 300:
            continue
        if any(_iou(i, D[a]) > 0.5 for a in ma):
            continue
        back = [i.x - gdx, i.y - gdy, i.w, i.h]
        if _region_diff(design, impl, back, (gdx, gdy)) < 6:
            continue
        name, art = _name(i, impl)
        extra.append(Finding(
            category="diferencia", severity="media", source="dinov2",
            title=f"{name.capitalize()} que no está en el diseño",
            detail=f"En la implementación hay {'un' if art == 'el' else 'una'} {name} en x={i.x}, y={i.y} ({i.w}×{i.h}px) "
                   f"y DINOv2 no le encontró pareja en el diseño.",
            suggestion="Si no es intencional, quítalo; si lo es, actualiza el diseño para que quede documentado.",
            box=i.box,
        ))
    findings += _drop_children(miss)[:6] + _drop_children(extra)[:6]

    return findings[:20], {
        "design_elements": len(D),
        "impl_elements": len(I),
        "matched": len(matches),
        "global_shift": [gdx, gdy],
    }
