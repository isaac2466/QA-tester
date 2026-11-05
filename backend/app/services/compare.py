"""Compara el diseño (mockup) con la implementación real, píxel a píxel y con DINOv2."""
import cv2
import numpy as np

from app.services.colors import delta_e, hex_of
from app.services.findings import Finding
from app.services.layout import detect_elements


def _shift(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float]:
    """Desplazamiento (dx, dy) de `b` respecto a `a` y la confianza (0..1)."""
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    win = cv2.createHanningWindow(a.shape[::-1], cv2.CV_32F) if min(a.shape) > 8 else None
    (dx, dy), resp = cv2.phaseCorrelate(a, b, win)
    return float(dx), float(dy), float(resp)


def _dir_text(dx: float, dy: float) -> tuple[str, str]:
    parts, fix = [], []
    if abs(dx) >= 0.5:
        n = f"{round(abs(dx))}px"
        parts.append(f"{n} más a la {'derecha' if dx > 0 else 'izquierda'}")
        fix.append(f"{n} a la {'izquierda' if dx > 0 else 'derecha'}")
    if abs(dy) >= 0.5:
        n = f"{round(abs(dy))}px"
        parts.append(f"{n} más {'abajo' if dy > 0 else 'arriba'}")
        fix.append(f"{n} hacia {'arriba' if dy > 0 else 'abajo'}")
    return " y ".join(parts), " y ".join(fix)


THR = 12  # diferencia mínima por canal (0-255) para considerar un píxel distinto


def _snap_to_elements(boxes, elems, page_area):
    """Agranda cada zona distinta al elemento de UI más pequeño que la contiene
    (así un botón movido es *un* hallazgo, no tres trozos de borde y texto)."""
    out = []
    for x, y, w, h in boxes:
        cx, cy = x + w / 2, y + h / 2
        best = None
        for e in elems:
            if e.kind == "bloque" and e.area < 0.25 * page_area and                     e.x - 4 <= cx <= e.right + 4 and e.y - 4 <= cy <= e.bottom + 4:
                if best is None or e.area < best.area:
                    best = e
        if best is None:
            out.append([x, y, w, h])
        else:
            x0, y0 = min(x, best.x), min(y, best.y)
            out.append([x0, y0, max(x + w, best.right) - x0, max(y + h, best.bottom) - y0])
    return out


def _merge_boxes(boxes: list[list[int]], gap=6) -> list[list[int]]:
    boxes = [b[:] for b in boxes]
    changed = True
    while changed:
        changed = False
        out = []
        while boxes:
            x, y, w, h = boxes.pop()
            for o in out:
                ox, oy, ow, oh = o
                if x <= ox + ow + gap and ox <= x + w + gap and y <= oy + oh + gap and oy <= y + h + gap:
                    nx, ny = min(x, ox), min(y, oy)
                    o[:] = [nx, ny, max(x + w, ox + ow) - nx, max(y + h, oy + oh) - ny]
                    changed = True
                    break
            else:
                out.append([x, y, w, h])
        boxes = out
    return boxes


def compare(design: np.ndarray, impl: np.ndarray, dino=None):
    findings: list[Finding] = []
    H, W = design.shape[:2]
    resized = impl.shape[:2] != design.shape[:2]
    if resized:
        findings.append(Finding(
            category="diferencia", severity="media",
            title=f"Tamaños distintos: diseño {W}×{H}, implementación {impl.shape[1]}×{impl.shape[0]}",
            detail="Se reescaló la implementación al tamaño del diseño para compararlas; las medidas en px "
                   "son aproximadas. Para precisión de 1px exporta ambas al mismo tamaño (y mismo zoom).",
            suggestion="Haz la captura con el mismo ancho de viewport y escala 1× (o 2× en ambas).",
        ))
        impl = cv2.resize(impl, (W, H), interpolation=cv2.INTER_AREA)

    ga = cv2.cvtColor(design, cv2.COLOR_RGB2GRAY)
    gb = cv2.cvtColor(impl, cv2.COLOR_RGB2GRAY)

    # 1) Desplazamiento global
    gdx, gdy, gresp = _shift(ga, gb)
    if gresp > 0.25 and (abs(gdx) >= 0.5 or abs(gdy) >= 0.5) and max(abs(gdx), abs(gdy)) < 0.2 * W:
        where, fix = _dir_text(gdx, gdy)
        findings.append(Finding(
            category="diferencia", severity="alta",
            title="Toda la página está desplazada",
            detail=f"La implementación completa está {where} que el diseño.",
            suggestion=f"Revisa márgenes/padding del contenedor raíz: hay que moverla {fix}.",
        ))

    # 2) Regiones distintas, una por una
    diff = np.max(cv2.absdiff(design, impl), axis=2)
    mask = (diff > THR).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    mask = cv2.dilate(mask, np.ones((5, 5), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    boxes = [list(map(int, stats[i, :4])) for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= 12]
    boxes = _snap_to_elements(boxes, detect_elements(design) + detect_elements(impl), W * H)
    boxes = sorted(_merge_boxes(boxes), key=lambda b: -b[2] * b[3])[:15]
    changed_px = float((diff > THR).mean())

    for x, y, w, h in boxes:
        m = 10
        x0, y0, x1, y1 = max(0, x - m), max(0, y - m), min(W, x + w + m), min(H, y + h + m)
        ca, cb = ga[y0:y1, x0:x1], gb[y0:y1, x0:x1]
        dx, dy, resp = _shift(ca, cb)
        box = [x, y, w, h]
        if resp > 0.2 and 0.5 <= max(abs(dx), abs(dy)) <= 30:
            where, fix = _dir_text(dx, dy)
            findings.append(Finding(
                category="alineacion", severity="alta",
                title=f"Elemento desplazado: {where}",
                detail=f"En la implementación, el elemento en x={x}, y={y} está {where} que en el diseño.",
                suggestion=f"Muévelo {fix} (revisa margin, padding, gap o line-height).",
                box=box,
            ))
            continue
        sub = (diff[y:y + h, x:x + w] > THR)
        ca_rgb = design[y:y + h, x:x + w][sub]
        cb_rgb = impl[y:y + h, x:x + w][sub]
        if len(ca_rgb):
            c1, c2 = np.median(ca_rgb, axis=0), np.median(cb_rgb, axis=0)
            if delta_e(c1, c2) > 3 and np.corrcoef(ca.ravel(), cb.ravel())[0, 1] > 0.6:
                findings.append(Finding(
                    category="color", severity="media",
                    title=f"Color distinto al diseño: {hex_of(c1)} → {hex_of(c2)}",
                    detail=f"En x={x}, y={y} el diseño usa {hex_of(c1)} pero la implementación muestra {hex_of(c2)}.",
                    suggestion=f"Cambia el color a {hex_of(c1)}.",
                    box=box,
                ))
                continue
        findings.append(Finding(
            category="diferencia", severity="media" if w * h > 400 else "baja",
            title="Contenido distinto al diseño",
            detail=f"La zona x={x}, y={y} ({w}×{h}px) no coincide con el diseño (texto, icono, tamaño o forma).",
            suggestion="Compara esta zona con el mockup: puede faltar un elemento, cambiar la tipografía o el tamaño.",
            box=box,
        ))

    # 3) Similitud semántica con DINOv2 (resistente a ruido de antialias / compresión)
    semantic_map, semantic_score = None, None
    if dino is not None:
        from PIL import Image
        sim, semantic_score = dino.patch_similarity(Image.fromarray(design), Image.fromarray(impl))
        semantic_map = cv2.resize(np.clip((1 - sim) / 0.5, 0, 1).astype(np.float32), (W, H),
                                  interpolation=cv2.INTER_NEAREST)

    return findings, impl, {
        "changed_pixels": round(changed_px, 4),
        "semantic_similarity": None if semantic_score is None else round(semantic_score, 4),
        "resized": resized,
    }, semantic_map
