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
    ra, rb = a.w / a.h, b.w / b.h  # proporción: un botón del menú no es el mismo componente que "Filtros"
    return max(a.h, b.h) <= 1.5 * min(a.h, b.h) and max(a.w, b.w) <= 3 * min(a.w, b.w) and max(ra, rb) <= 1.6 * min(ra, rb)


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


def _noun(members: list[Element], rgb: np.ndarray | None = None) -> tuple[str, str]:
    """Nombre del componente. Sólo dice "botón" si además tiene fondo de color de acento
    (morado, azul, verde...): una fila de tabla o un buscador blanco tienen forma de botón pero no lo son."""
    from app.services.colors import _chroma_hue
    h = float(np.median([e.h for e in members]))
    w = float(np.median([e.w for e in members]))
    accent = rgb is not None and float(np.median([_chroma_hue(_fill(rgb, e))[0] for e in members])) > 30
    if h <= 32 and 0.7 <= w / h <= 1.4:
        return "icono", "iconos"
    if h >= 60 and w >= 120:  # tarjetas de estadísticas (KPI), paneles
        return "tarjeta", "tarjetas"
    if 20 <= h < 60 and 1.5 <= w / h <= 8 and accent:
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


def _row_offsets(members, noun) -> list[Finding]:
    """Hermanos en la misma fila que empiezan a distinta altura (p. ej. 3 tarjetas desfasadas 10-40px)."""
    out = []
    row = [m for m in members if abs(m.cy - members[0].cy) < 0.6 * max(m.h, members[0].h)]
    if len(row) < 3:
        return out
    top = float(np.median([m.y for m in row]))
    for m in row:
        d = m.y - top
        if 5 <= abs(d) <= 0.6 * m.h:
            out.append(Finding(
                category="alineacion", severity="media", source="dinov2",
                title=f"{noun[0].capitalize()} a distinta altura ({abs(d):.0f}px)",
                detail=f"DINOv2 agrupa {len(row)} {noun[1]} en la misma fila, pero este empieza {abs(d):.0f}px "
                       f"{'más abajo' if d > 0 else 'más arriba'} que los demás.",
                suggestion=f"Alinea su borde superior con el de sus hermanos (muévelo {abs(d):.0f}px hacia "
                           f"{'arriba' if d > 0 else 'abajo'}) o usa align-items: start en el contenedor.",
                box=m.box, fix={"type": "move", "box": m.box, "dx": 0, "dy": -round(d)},
            ))
    return out


def check_button_styles(rgb: np.ndarray, elems: list[Element]) -> list[Finding]:
    """Inventario de botones: cuántos estilos distintos (color + alto) hay en la página."""
    from app.services.colors import _chroma_hue
    btns = [e for e in elems if e.kind == "bloque" and 20 <= e.h <= 64 and 1.5 <= e.w / e.h <= 10
            and _chroma_hue(_fill(rgb, e))[0] > 30]
    styles: list[list] = []
    for e in btns:
        f = _fill(rgb, e)
        for s in styles:
            if delta_e(s[0], f) < 10 and abs(s[1] - e.h) <= 4:
                s[2].append(e)
                break
        else:
            styles.append([f, e.h, [e]])
    if len(styles) <= 3:
        return []
    desc = ", ".join(f"{hex_of(f)} ({h}px de alto)" for f, h, _ in styles[:6])
    return [Finding(
        category="sistema", severity="media",
        title=f"{len(styles)} estilos de botón distintos",
        detail=f"Hay {len(btns)} botones con {len(styles)} combinaciones de color y tamaño: {desc}. "
               f"Así ninguno se ve como la acción principal.",
        suggestion="Define 2-3 estilos: uno primario (la acción principal, un solo color fuerte), uno secundario "
                   "(contorno o color neutro) y, si hace falta, uno de texto. Usa el mismo alto para todos.",
    )]


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
        noun = _noun(members, rgb)
        groups.append({"label": noun[1] if len(members) > 1 else noun[0], "count": len(members),
                       "boxes": [e.box for e in members]})
        if len(members) < MIN_GROUP:
            continue
        found = _row_offsets(members, noun)
        found += _size_findings(members, noun, "h")
        flagged = {tuple(f.box) for f in found}
        found += [f for f in _size_findings(members, noun, "w") if tuple(f.box) not in flagged]
        flagged |= {tuple(f.box) for f in found}
        found += [f for f in _color_findings(rgb, members, noun) if tuple(f.box) not in flagged]
        for f in found:
            f.source, f.confidence = "dinov2", conf.get(tuple(f.box))
        findings += found
    groups.sort(key=lambda g: -g["count"])
    return findings[:10], groups
