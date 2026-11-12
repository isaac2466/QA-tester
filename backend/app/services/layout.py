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


def _color_regions(rgb: np.ndarray) -> list[Element]:
    H, W = rgb.shape[:2]
    q = (rgb // 8).astype(np.int32)  # agrupa tonos casi iguales (ruido de compresión)
    packed = (q[..., 0] << 10) | (q[..., 1] << 5) | q[..., 2]
    vals, counts = np.unique(packed, return_counts=True)
    page_bg = vals[np.argmax(counts)]
    out = []
    for v in vals[counts >= 300]:
        if v == page_bg:
            continue
        mask = (packed == v).astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))  # tapa texto e íconos de adentro
        n, _, st, _ = cv2.connectedComponentsWithStats(mask, 8)
        for i in range(1, n):
            x, y, w, h, area = st[i]
            if w < 16 or h < 12 or w * h > 0.5 * W * H:
                continue
            if area / float(w * h) > 0.6:  # región "llena": un bloque, no una línea o un degradado
                out.append(Element(int(x), int(y), int(w), int(h), "bloque"))
    return out


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
        fill = cv2.contourArea(c) / float(w * h)
        round_shape = 0.8 <= w / h <= 1.25 and fill > 0.72  # avatares e íconos circulares (π/4 ≈ 0.785)
        if fill > 0.88 or round_shape:
            # el gradiente marca 1px por fuera del borde real → lo compensamos
            blocks.append(Element(x + 1, y + 1, w - 2, h - 2, "bloque"))
    # 1b) Bloques por COLOR: zonas grandes de un color uniforme distinto al fondo (tarjetas de color,
    #     botones, paneles en modo oscuro). El método de bordes falla cuando el contorno sale "roto".
    blocks += _color_regions(rgb)
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
    # una "línea" que contiene varias líneas más chicas es un marco (p. ej. todo el encabezado unido por un
    # botón que toca su borde): se quita y quedan las palabras sueltas (Inicio · Precios · Contacto…)
    def _contains(a, b):
        return a is not b and a.x <= b.x and a.y <= b.y and b.right <= a.right and b.bottom <= a.bottom
    texts = [t for t in texts if t.w < 0.5 * W or sum(_contains(t, o) for o in texts) < 2]
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
            # (de la rama de Chris) sólo elementos del mismo tipo: un icono vs un texto no es un error
            if a.kind != b.kind:
                continue
            # texto: sólo el borde izquierdo de líneas apiladas (el borde superior depende de mayúsculas/ascendentes)
            if a.kind == "texto" and attr != "left":
                continue
            # sólo vecinos cercanos: dos cosas lejanas a 2px es casualidad, no un error
            gap = (max(a.y, b.y) - min(a.bottom, b.bottom)) if horizontal else (max(a.x, b.x) - min(a.right, b.right))
            if gap > 120:
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
            fix={"type": "move", "box": off.box,
                 "dx": -round(delta) if horizontal else 0, "dy": 0 if horizontal else -round(delta)},
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
                        # se mueve el segundo elemento (b) para que la separación sea `other`
                        fix={"type": "move", "box": b.box,
                             "dx": (other - g) if axis == "horizontal" else 0,
                             "dy": (other - g) if axis == "vertical" else 0},
                    ))
                    break
    return out[:8]


def check_centering(elems: list[Element], width: int) -> list[Finding]:
    out = []
    mid = width / 2
    for e in elems:
        if e.kind != "bloque" or e.w > 0.9 * width:
            continue
        d = e.cx - mid
        # 1px suele ser redondeo (escala de Windows al 125%, anchos impares)
        if 2 <= abs(d) <= 4:
            side = "derecha" if d > 0 else "izquierda"
            out.append(Finding(
                category="alineacion",
                severity="media" if e.kind == "bloque" else "baja",
                title=f"Casi centrado ({abs(d):g}px a la {side})",
                detail=f"{e.label()} en x={e.x}, y={e.y} tiene su centro a {abs(d):g}px del centro de la página.",
                suggestion=f"Si debe ir centrado, muévelo {abs(d):g}px a la {'izquierda' if d > 0 else 'derecha'} "
                           f"(revisa margin/padding asimétricos o un scrollbar).",
                box=e.box,
                fix={"type": "move", "box": e.box, "dx": -round(d), "dy": 0},
            ))
    return out[:5]


def check_text_rows(elems: list[Element], max_findings=4) -> list[Finding]:
    """Filas de textos parecidos (menús, enlaces): si casi todos coinciden y uno se sale, es error.

    Ej.: "Funciones · Planes · Contacto" con "Planes" 20px más abajo.
    """
    page_w = max((e.right for e in elems), default=1)
    # (una línea de 1500px de ancho es un borde entre secciones, no un texto)
    texts = [e for e in elems if e.kind == "texto" and 10 <= e.h <= 60 and e.w < 0.5 * page_w]
    # un texto con otro justo arriba o abajo es parte de una lista o párrafo, no de un menú
    def in_column(e):
        return any(o is not e and _overlap(e.x, e.right, o.x, o.right) > 0.3 * min(e.w, o.w)
                   and 0 < abs(o.cy - e.cy) <= 2.2 * max(e.h, o.h) for o in texts)
    # el texto de un botón se mueve con su botón (eso lo revisa la regla de botones)
    blocks = [b for b in elems if b.kind == "bloque" and b.h <= 100]
    def in_button(e):
        return any(b.x - 2 <= e.x and b.y - 2 <= e.y and e.right <= b.right + 2 and e.bottom <= b.bottom + 2 for b in blocks)
    texts = sorted([e for e in texts if not in_column(e) and not in_button(e)], key=lambda e: e.x)
    out, used = [], set()
    for a in texts:
        if id(a) in used:
            continue
        row = [b for b in texts if abs(b.h - a.h) <= max(3, 0.2 * a.h) and abs(b.cy - a.cy) <= 2 * a.h
               and (b is a or _overlap(a.x, a.right, b.x, b.right) == 0)]
        row.sort(key=lambda e: e.x)
        # vecinos razonablemente cerca (un menú, no dos textos en extremos opuestos)
        chain = [row[0]] if row else []
        for b in row[1:]:
            if b.x - chain[-1].right < 260:
                chain.append(b)
        if len(chain) < 3:
            continue
        # referencia: el grupo más grande que sí está alineado; si todos están sueltos, la mediana
        thr = lambda e: max(4, 0.25 * e.h)
        best = max(([o for o in chain if abs(o.cy - e.cy) <= 2] for e in chain), key=len)
        med = float(np.mean([e.cy for e in best])) if len(best) >= 2 else float(np.median([e.cy for e in chain]))
        aligned = [e for e in chain if abs(e.cy - med) < thr(e)]
        if not aligned:
            continue
        for e in chain:
            used.add(id(e))
            d = e.cy - med
            if thr(e) <= abs(d) <= 2 * e.h:
                ref = min(aligned, key=lambda r: abs(r.cx - e.cx))
                out.append(Finding(
                    category="alineacion", severity="alta",
                    title=f"Texto fuera de línea ({abs(d):.0f}px)",
                    detail=f"En una fila de {len(chain)} textos ({len(aligned)} alineados entre sí), este está "
                           f"{abs(d):.0f}px más {'abajo' if d > 0 else 'arriba'} que los demás. Típico de un menú con un enlace movido.",
                    suggestion=f"Súbelo {abs(d):.0f}px" if d > 0 else f"Bájalo {abs(d):.0f}px",
                    box=e.box, ref_box=ref.box,
                    fix={"type": "move", "box": e.box, "dx": 0, "dy": -round(d)},
                ))
    return out[:max_findings]


def check_stacked(elems: list[Element]) -> list[Finding]:
    """Un botón justo debajo de otro debe alinearse a su izquierda, centro o derecha (p. ej. "Ver demo")."""
    blocks = [e for e in elems if e.kind == "bloque" and 20 <= e.h <= 80 and e.w >= 40]
    out = []
    for a in blocks:
        for b in blocks:
            if b is a or not (0 <= b.y - a.bottom <= 40) or _overlap(a.x, a.right, b.x, b.right) < 0.3 * min(a.w, b.w):
                continue
            opts = {"izquierda": b.x - a.x, "centro": b.cx - a.cx, "derecha": b.right - a.right}
            name, d = min(opts.items(), key=lambda kv: abs(kv[1]))
            if 5 <= abs(d) <= 0.5 * a.w:
                out.append(Finding(
                    category="alineacion", severity="media",
                    title=f"Botón suelto: no se alinea con el de arriba ({abs(d):.0f}px)",
                    detail=f"Está justo debajo de otro botón pero no coincide ni a la izquierda, ni al centro, ni a la derecha. "
                           f"Lo más cercano es alinearlo a la {name} ({abs(d):.0f}px).",
                    suggestion=f"Muévelo {abs(d):.0f}px a la {'izquierda' if d > 0 else 'derecha'} para alinearlo a la {name}, "
                               f"o agrúpalos en un mismo contenedor (flex/column con align-items).",
                    box=b.box, ref_box=a.box, fix={"type": "move", "box": b.box, "dx": -round(d), "dy": 0},
                ))
    return out[:4]


def _inter(a: Element, b: Element) -> float:
    return max(0, min(a.right, b.right) - max(a.x, b.x)) * max(0, min(a.bottom, b.bottom) - max(a.y, b.y))


def _inside(a: Element, b: Element, m=3) -> bool:
    return b.x - m <= a.x and b.y - m <= a.y and a.right <= b.right + m and a.bottom <= b.bottom + m


def check_overlaps(rgb: np.ndarray, elems: list[Element], max_findings=5) -> list[Finding]:
    """Elementos encimados (uno tapa parte de otro) y textos cortados en el borde de la pantalla."""
    H, W = rgb.shape[:2]
    big = 0.25 * W * H  # secciones y fondos contienen todo: no cuentan como "encima"
    blocks = [b for b in elems if b.kind == "bloque" and b.area < big]
    texts = [t for t in elems if t.kind == "texto" and 8 <= t.h <= 80 and t.w < 0.6 * W]
    out = []
    all_blocks = [b for b in elems if b.kind == "bloque"]

    def same_mockup(a, b):
        """Los dos dentro de una misma tarjeta alta (ilustración, maqueta de dashboard): ahí encimar es parte del dibujo."""
        return any(p is not a and p is not b and p.h >= 100 and _inside(a, p, 6) and _inside(b, p, 6) for p in all_blocks)
    # texto tapado en parte por un bloque que no es su contenedor
    for t in texts:
        for b in blocks:
            i = _inter(t, b)
            if i < 0.15 * t.area or i > 0.85 * t.area or _inside(t, b) or _inside(b, t) or same_mockup(t, b):
                continue
            out.append(Finding(
                category="alineacion", severity="alta",
                title="Elemento encimado: tapa parte de un texto",
                detail=f"Un elemento ({b.w}×{b.h}px en x={b.x}, y={b.y}) cubre el {i / t.area:.0%} de un texto que no está dentro de él.",
                suggestion="Separa los elementos (margin, gap o z-index) o ajusta el ancho del contenedor para que el texto no quede debajo.",
                box=t.box, ref_box=b.box,
            ))
            break
    # bloque encimado sobre otro bloque (burbujas, banners, ilustraciones)
    for a in blocks:
        for b in blocks:
            if a is b or a.area > b.area:
                continue
            i = _inter(a, b)
            if 0.08 * a.area <= i <= 0.7 * a.area and not _inside(a, b) and not _inside(b, a) and not same_mockup(a, b):
                out.append(Finding(
                    category="alineacion", severity="alta",
                    title="Elemento encimado sobre otro",
                    detail=f"Este elemento tapa el {i / a.area:.0%} de otro elemento ({b.w}×{b.h}px en x={b.x}, y={b.y}).",
                    suggestion="Si es un elemento flotante (chat, cookies, avisos), deja espacio abajo o muévelo para que no tape botones ni texto.",
                    box=a.box, ref_box=b.box,
                ))
                break
    # texto cortado por el borde de la pantalla
    for t in texts:
        if t.x <= 1 or t.right >= W - 1:
            out.append(Finding(
                category="alineacion", severity="alta",
                title="Texto cortado en el borde de la pantalla",
                detail="Este texto toca el borde de la imagen: probablemente se sale de la pantalla y se corta.",
                suggestion="Permite que el texto haga salto de línea (sin white-space: nowrap), reduce su tamaño en móvil o agrega padding lateral.",
                box=t.box,
            ))
    return out[:max_findings]


def check_button_row(elems: list[Element], tilt_findings: list) -> list[Finding]:
    """Botones lado a lado (Iniciar sesión · Registrarse) a distinta altura.

    Si uno además está girado, su corrección de giro también lo sube/baja a la altura del otro.
    """
    tilted = [f for f in tilt_findings if f.fix and f.fix.get("type") == "rotate"]

    def tilt_of(e):
        for f in tilted:
            x, y, w, h = f.box
            if x - 2 <= e.x and y - 2 <= e.y and e.right <= x + w + 2 and e.bottom <= y + h + 2:
                return f
        return None

    page_w = max((e.right for e in elems), default=1)
    btns = []
    for e in elems:
        if e.kind != "bloque" or not (24 <= e.h <= 400 and 50 <= e.w <= 0.45 * page_w):
            continue
        if e.w < (1.8 if e.h <= 100 else 1.2) * e.h:  # botones y tarjetas son anchos; círculos/íconos/fotos no
            continue
        f = tilt_of(e)
        box = list(f.box) if f else list(e.box)
        if any(b["box"] == box for b in btns):
            continue
        btns.append({"box": box, "tilt": f, "cy": box[1] + box[3] / 2, "h": box[3]})
    btns.sort(key=lambda b: b["box"][0])

    # filas: elementos de tamaño parecido, uno junto al otro (botones de un menú, tarjetas de una sección)
    rows, used = [], set()
    for a in btns:
        if id(a) in used:
            continue
        row = [a]
        for b in btns:
            last = row[-1]
            lx, ly, lw, lh = last["box"]
            bx, by, bw, bh = b["box"]
            gap = bx - (lx + lw)
            if id(b) in used or b is last or not (-6 <= gap <= max(80, 0.6 * lw)):
                continue
            if not (0.65 <= bh / lh <= 1.5 and 0.6 <= bw / lw <= 1.6) or _overlap(ly, ly + lh, by, by + bh) <= 0:
                continue
            row.append(b)
        if len(row) >= 2:
            used.update(id(r) for r in row)
            rows.append(row)

    out = []
    for row in rows:
        if len(row) >= 3:  # la mayoría manda
            ref_cy = float(np.median([r["cy"] for r in row]))
            offs = [(r, ref_cy - r["cy"]) for r in row]
        else:  # dos: se mueve el girado (o el de la derecha)
            a, b = row
            off = b if (b["tilt"] or not a["tilt"]) else a
            other = a if off is b else b
            offs = [(off, other["cy"] - off["cy"])]
        for off, dy in offs:
            h = off["h"]
            if not (max(4, 0.15 * h) <= abs(dy) <= h):
                continue
            ref = min((r for r in row if r is not off), key=lambda r: abs(r["box"][0] - off["box"][0]))
            if off["tilt"]:
                off["tilt"].fix["dy"] = round(dy)  # el giro y el ajuste de altura van juntos
                fix = None
            else:
                fix = {"type": "move", "box": off["box"], "dx": 0, "dy": round(dy)}
            kind = "botón" if h <= 100 else "elemento"
            out.append(Finding(
                category="alineacion", severity="media",
                title=f"Botón desalineado con su vecino ({abs(dy):.0f}px)" if h <= 100
                      else f"Tarjeta fuera de la fila ({abs(dy):.0f}px)",
                detail=f"Este {kind} está {abs(dy):.0f}px más {'abajo' if dy < 0 else 'arriba'} que los demás de su fila; "
                       "en una misma fila deberían compartir la misma altura.",
                suggestion=f"{'Súbelo' if dy < 0 else 'Bájalo'} {abs(dy):.0f}px para alinearlo con los demás.",
                box=off["box"], ref_box=ref["box"], fix=fix,
            ))
    return out
