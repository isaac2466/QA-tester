"""Consistencia de componentes con DINOv2.

1. Cada bloque detectado (botón, tarjeta, input, icono...) se pasa por DINOv2 y se obtiene
   un vector que describe "qué tipo de cosa es".
2. Los vectores parecidos se agrupan: así sabemos que 5 rectángulos son *el mismo* botón
   aunque tengan textos distintos.
3. Dentro de cada grupo se mide alto, ancho y color de relleno, y se señala el que no
   coincide con la mayoría (p.ej. 4 botones de 44px y uno de 41px).
"""
from math import ceil

import numpy as np

from app.services.colors import delta_e, hex_of
from app.services.findings import Finding
from app.services.layout import Element

SIMILARITY = 0.80  # coseno mínimo para considerar dos elementos "el mismo componente"
MAX_ELEMENTS = 60
MIN_GROUP = 3  # con menos de 3 no hay mayoría clara


def _candidates(elems: list[Element], W: int, H: int) -> list[Element]:
    out = [e for e in elems if e.kind == "bloque" and e.w >= 20 and e.h >= 16 and e.area < 0.2 * W * H]
    return out[:MAX_ELEMENTS]


def _compatible(a: Element, b: Element) -> bool:
    """Un botón y una tarjeta pueden parecerse a DINOv2; el tamaño los separa."""
    return max(a.h, b.h) <= 1.5 * min(a.h, b.h) and max(a.w, b.w) <= 3 * min(a.w, b.w)


def _cluster(elems: list[Element], emb: np.ndarray) -> list[list[int]]:
    """Agrupación voraz por centroide (evita el 'encadenamiento' de un union-find)."""
    groups: list[list[int]] = []
    centroids: list[np.ndarray] = []
    for i in range(len(elems)):
        best, best_sim = None, SIMILARITY
        for g, (members, c) in enumerate(zip(groups, centroids)):
            sim = float(emb[i] @ c)
            if sim >= best_sim and _compatible(elems[i], elems[members[0]]):
                best, best_sim = g, sim
        if best is None:
            groups.append([i])
            centroids.append(emb[i].copy())
        else:
            groups[best].append(i)
            c = emb[groups[best]].mean(0)
            centroids[best] = c / (np.linalg.norm(c) or 1)
    return groups


def _fill(rgb: np.ndarray, e: Element) -> np.ndarray:
    """Color de relleno: el color exacto más frecuente del interior (el texto es minoría)."""
    m = 3 if min(e.w, e.h) > 10 else 0
    inner = rgb[e.y + m:e.bottom - m, e.x + m:e.right - m].reshape(-1, 3)
    if not len(inner):
        inner = rgb[e.y:e.bottom, e.x:e.right].reshape(-1, 3)
    packed = (inner[:, 0].astype(np.int32) << 16) | (inner[:, 1].astype(np.int32) << 8) | inner[:, 2]
    vals, counts = np.unique(packed, return_counts=True)
    v = int(vals[np.argmax(counts)])
    return np.array([(v >> 16) & 255, (v >> 8) & 255, v & 255])


def _noun(members: list[Element]) -> tuple[str, str]:
    h = float(np.median([e.h for e in members]))
    w = float(np.median([e.w for e in members]))
    if h <= 48 and 0.7 <= w / h <= 1.4:
        return "icono", "iconos"
    if 20 <= h <= 72 and w / h >= 1.5:
        return "botón", "botones"
    if h >= 80 and w >= 120:
        return "tarjeta", "tarjetas"
    return "elemento", "elementos"


def _majority(values: list[int]) -> tuple[int, int] | None:
    """Valor que comparte la mayoría (≥60%, mínimo 2). Devuelve (valor, cuántos)."""
    vals, counts = np.unique(values, return_counts=True)
    i = int(np.argmax(counts))
    if counts[i] >= max(2, ceil(0.6 * len(values))):
        return int(vals[i]), int(counts[i])
    return None


def _nearest(e: Element, refs: list[Element]) -> Element:
    return min(refs, key=lambda r: abs(r.cx - e.cx) + abs(r.cy - e.cy))


def _size_findings(members: list[Element], noun, attr: str) -> list[Finding]:
    name = "alto" if attr == "h" else "ancho"
    maj = _majority([getattr(e, attr) for e in members])
    if not maj:
        return []
    value, n = maj
    out = []
    for e in members:
        d = getattr(e, attr) - value
        if not (1 <= abs(d) <= max(12, 0.25 * value)):
            continue  # igual a la mayoría, o tan distinto que seguramente es otra variante
        ref = _nearest(e, [m for m in members if getattr(m, attr) == value])
        prop = "padding vertical, line-height o border" if attr == "h" else "width, padding horizontal o el texto"
        out.append(Finding(
            category="consistencia",
            severity="media" if abs(d) >= 2 else "baja",
            title=f"{noun[0].capitalize()} con otro {name}: {getattr(e, attr)}px vs {value}px",
            detail=f"DINOv2 reconoce {len(members)} {noun[1]} que son el mismo componente. {n} miden "
                   f"{value}px de {name}, pero este (x={e.x}, y={e.y}) mide {getattr(e, attr)}px.",
            suggestion=f"Ajusta su {name} a {value}px ({'+' if d < 0 else '-'}{abs(d)}px): revisa {prop}. "
                       f"Si es el mismo componente, debería usar la misma clase o estilo.",
            box=e.box,
            ref_box=ref.box,
            fix={"type": "resize", "box": e.box, "ref": ref.box,
                 "w": value if attr == "w" else e.w, "h": value if attr == "h" else e.h},
        ))
    return out


def _color_findings(rgb: np.ndarray, members: list[Element], noun) -> list[Finding]:
    fills = [_fill(rgb, e) for e in members]
    # grupo de color mayoritario: los que se ven idénticos (ΔE < 1) al más común
    best, best_idx = None, []
    for f in fills:
        idx = [j for j, g in enumerate(fills) if delta_e(f, g) < 1]
        if len(idx) > len(best_idx):
            best, best_idx = f, idx
    if len(best_idx) < max(2, ceil(0.6 * len(members))):
        return []
    out = []
    for j, (e, f) in enumerate(zip(members, fills)):
        if j in best_idx:
            continue
        de = delta_e(best, f)
        if not (1.5 < de < 20):
            continue  # muy distinto = variante intencional (primario vs secundario)
        ref = _nearest(e, [members[k] for k in best_idx])
        out.append(Finding(
            category="consistencia",
            severity="media",
            title=f"{noun[0].capitalize()} con otro color: {hex_of(f)} vs {hex_of(best)}",
            detail=f"DINOv2 reconoce {len(members)} {noun[1]} iguales. {len(best_idx)} usan {hex_of(best)} de fondo, "
                   f"pero este (x={e.x}, y={e.y}) usa {hex_of(f)} (ΔE {de:.1f}): casi igual, pero no es el mismo color.",
            suggestion=f"Cambia su fondo a {hex_of(best)} o usa la misma variable de color que los demás.",
            box=e.box,
            ref_box=ref.box,
            fix={"type": "recolor", "box": e.box, "from": hex_of(f), "to": hex_of(best)},
        ))
    return out


def check_components(rgb: np.ndarray, elems: list[Element], dino) -> tuple[list[Finding], list[dict]]:
    H, W = rgb.shape[:2]
    cands = _candidates(elems, W, H)
    if len(cands) < MIN_GROUP:
        return [], []
    emb = dino.embed_boxes(rgb, [e.box for e in cands])
    groups = []
    findings: list[Finding] = []
    for idx in _cluster(cands, emb):
        if len(idx) < 2:
            continue
        # confianza = qué tanto se parece cada elemento al resto de su grupo según DINOv2
        c = emb[idx].mean(0)
        c /= np.linalg.norm(c) or 1
        conf = {tuple(cands[i].box): float(emb[i] @ c) for i in idx}
        members = sorted((cands[i] for i in idx), key=lambda e: (e.y, e.x))
        noun = _noun(members)
        groups.append({"label": noun[1] if len(members) > 1 else noun[0], "count": len(members),
                       "boxes": [e.box for e in members]})
        if len(members) < MIN_GROUP:
            continue
        found = _size_findings(members, noun, "h")
        flagged = {tuple(f.box) for f in found}
        found += [f for f in _size_findings(members, noun, "w") if tuple(f.box) not in flagged]
        flagged |= {tuple(f.box) for f in found}
        found += [f for f in _color_findings(rgb, members, noun) if tuple(f.box) not in flagged]
        for f in found:
            f.source, f.confidence = "dinov2", conf.get(tuple(f.box))
        findings += found
    groups.sort(key=lambda g: -g["count"])
    return findings[:10], groups
