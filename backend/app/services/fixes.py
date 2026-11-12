"""Vista corregida: aplica sobre la captura las correcciones concretas de cada hallazgo.

- SAM (Meta) da la silueta exacta de cada elemento → se puede mover / redimensionar sin arrastrar
  el fondo ni cortar las esquinas redondeadas.
- El hueco que deja un elemento movido se rellena con el fondo (inpainting de OpenCV).
- Los colores se corrigen conservando el antialias de bordes y letras.

No es una imagen inventada: son los mismos píxeles de tu captura con los cambios aplicados.
"""
import cv2
import numpy as np


def _rgb(hex_: str) -> np.ndarray:
    h = hex_.lstrip("#")
    return np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)], dtype=np.float32)


def _clip_box(box, W, H, pad=0):
    x, y, w, h = box
    return max(0, x - pad), max(0, y - pad), min(W, x + w + pad), min(H, y + h + pad)


def _rect_mask(shape, box, pad=0):
    m = np.zeros(shape[:2], dtype=bool)
    x0, y0, x1, y1 = _clip_box(box, shape[1], shape[0], pad)
    m[y0:y1, x0:x1] = True
    return m


def _element_mask(shape, box, sam_mask):
    """Máscara de SAM limitada a la caja (+2px); si SAM falló, el rectángulo de la caja."""
    rect = _rect_mask(shape, box, pad=2)
    if sam_mask is not None:
        m = sam_mask & rect
        x0, y0, x1, y1 = _clip_box(box, shape[1], shape[0])
        if m[y0:y1, x0:x1].mean() >= 0.5:  # SAM encontró el elemento
            return cv2.dilate(m.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool) & rect
    return _rect_mask(shape, box)


def _recolor(img, mask, src, dst, tol=30.0):
    """Cambia `src` por `dst` dentro de la máscara, desplazando también los tonos cercanos (antialias)."""
    px = img[mask].astype(np.float32)
    near = np.linalg.norm(px - src, axis=1) < tol
    px[near] += dst - src
    img[mask] = np.clip(px, 0, 255).astype(np.uint8)


def _text_color(img, box, bg, fg, dst):
    """Reemplaza el color de las letras manteniendo su forma: mezcla fondo→nuevo color según la 'tinta'."""
    H, W = img.shape[:2]
    x0, y0, x1, y1 = _clip_box(box, W, H, pad=2)
    crop = img[y0:y1, x0:x1].astype(np.float32)
    span = float(np.linalg.norm(fg - bg)) or 1.0
    alpha = np.clip(np.linalg.norm(crop - bg, axis=2) / span, 0, 1)[..., None]
    img[y0:y1, x0:x1] = np.clip(bg * (1 - alpha) + dst * alpha, 0, 255).astype(np.uint8)


def _uniform_lines(patch: np.ndarray, axis: int, margin: int = 10) -> list[int]:
    """Filas (axis=0) o columnas (axis=1) que son puro fondo del elemento: ni texto ni bordes."""
    p = patch.astype(np.float32)
    if axis == 1:
        p = p.transpose(1, 0, 2)
    n, m = p.shape[:2]
    inner = p[:, margin:m - margin] if m > 2 * margin + 4 else p
    std = inner.std(axis=1).max(axis=1)
    lo, hi = max(2, n // 8), n - max(2, n // 8)  # lejos de las esquinas redondeadas
    return [i for i in range(lo, hi) if std[i] < 3]


def _content_start(patch: np.ndarray, axis: int) -> int:
    """Primera fila/columna con contenido (texto o ícono) después del borde."""
    uniform = set(_uniform_lines(patch, axis))
    n = patch.shape[axis]
    lo = max(2, n // 8)
    return next((i for i in range(lo, n - lo) if i not in uniform), n // 2)


def _stretch(patch: np.ndarray, mask: np.ndarray, target: int, axis: int, before: int | None = None):
    """Cambia el tamaño en un eje duplicando/quitando filas de puro fondo (9-slice): el texto no se deforma."""
    n = patch.shape[axis]
    k = target - n
    if k == 0:
        return patch, mask
    lines = _uniform_lines(patch, axis)
    if not lines:  # sin franjas lisas: escalado normal
        size = (patch.shape[1], target) if axis == 0 else (target, patch.shape[0])
        return (cv2.resize(patch, size, interpolation=cv2.INTER_LINEAR),
                cv2.resize(mask.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST).astype(bool))
    mid = n / 2
    upper = [i for i in lines if i < mid] or lines
    lower = [i for i in lines if i >= mid] or lines
    # cuánto va antes del contenido: lo que diga la referencia, o mitad y mitad (centrado)
    b = (abs(k) + 1) // 2 if before is None else int(np.clip(before if k > 0 else -before, 0, abs(k)))
    cuts = [(max(upper), b), (min(lower), abs(k) - b)]
    for at, cnt in sorted(cuts, reverse=True):  # de atrás hacia adelante: los índices no se mueven
        if cnt == 0:
            continue
        if k > 0:
            rep = np.repeat(np.take(patch, [at], axis=axis), cnt, axis=axis)
            mrep = np.repeat(np.take(mask, [at], axis=axis), cnt, axis=axis)
            patch = np.concatenate([np.take(patch, range(at), axis=axis), rep,
                                    np.take(patch, range(at, patch.shape[axis]), axis=axis)], axis=axis)
            mask = np.concatenate([np.take(mask, range(at), axis=axis), mrep,
                                   np.take(mask, range(at, mask.shape[axis]), axis=axis)], axis=axis)
        else:
            keep = [i for i in range(patch.shape[axis]) if not (at <= i < at + cnt)]
            patch, mask = np.take(patch, keep, axis=axis), np.take(mask, keep, axis=axis)
    return patch, mask


def apply_fixes(rgb: np.ndarray, findings: list[dict], sam=None) -> tuple[np.ndarray, list[int]]:
    """Devuelve (imagen corregida, ids de los hallazgos aplicados)."""
    img = rgb.copy()
    H, W = img.shape[:2]
    applied: list[int] = []
    touched: set = set()  # una corrección geométrica por elemento

    with_fix = [f for f in findings if f.get("fix")]
    global_done = set()
    # 1) colores (no mueven nada)
    for f in with_fix:
        fx = f["fix"]
        if fx["type"] == "recolor_global":
            src, dst = _rgb(fx["from"]), _rgb(fx["to"])
            m = np.all(img == src.astype(np.uint8), axis=-1)
            if m.any():
                img[m] = dst.astype(np.uint8)
                applied.append(f["id"])
                global_done.add((fx["from"], fx["to"]))
        elif fx["type"] == "recolor_many":
            boxes = [it["box"] for it in fx["items"]]
            masks = sam.masks(img, boxes) if sam else [None] * len(boxes)
            for it, sm in zip(fx["items"], masks):
                if it.get("ink"):  # letras oscuras → blancas (antes de cambiar el fondo)
                    _text_color(img, it["box"], _rgb(it["from"]), _rgb(it["ink"]), np.array([255.0, 255.0, 255.0]))
                _recolor(img, _element_mask(img.shape, it["box"], sm), _rgb(it["from"]), _rgb(it["to"]), tol=45)
            applied.append(f["id"])
        elif fx["type"] == "text_color":
            _text_color(img, fx["box"], _rgb(fx["bg"]), _rgb(fx["fg"]), _rgb(fx["to"]))
            applied.append(f["id"])

    # 2) geometría: mover / redimensionar con la silueta de SAM
    geo = []
    for f in with_fix:
        fx = f["fix"]
        key = tuple(fx.get("box") or ())
        if fx["type"] == "recolor" and (fx["from"], fx["to"]) in global_done:
            applied.append(f["id"])  # ya lo resolvió la unificación de color
            continue
        if fx["type"] in ("move", "resize", "recolor", "rotate") and key not in touched:
            touched.add(key)
            geo.append(f)
    # no tocar elementos que están dentro de otro que también se corrige (se moverían dos veces)
    def inside(a, b):
        return a != b and b[0] <= a[0] and b[1] <= a[1] and a[0] + a[2] <= b[0] + b[2] and a[1] + a[3] <= b[1] + b[3]
    geo = [f for f in geo if not any(inside(f["fix"]["box"], o["fix"]["box"]) for o in geo
                                     if o["fix"]["type"] != "recolor")]

    masks = sam.masks(img, [f["fix"]["box"] for f in geo]) if (sam and geo) else [None] * len(geo)
    pieces = []
    erase = np.zeros((H, W), dtype=np.uint8)
    for f, sm in zip(geo, masks):
        fx = f["fix"]
        m = _element_mask(img.shape, fx["box"], sm)
        if fx["type"] == "recolor":
            _recolor(img, m, _rgb(fx["from"]), _rgb(fx["to"]))
            applied.append(f["id"])
            continue
        x0, y0, x1, y1 = _clip_box(fx["box"], W, H, pad=2)
        patch, pmask = img[y0:y1, x0:x1].copy(), m[y0:y1, x0:x1].copy()
        if fx["type"] == "rotate":  # enderezar: girar la pieza (y su silueta) sobre su centro
            ph, pw = patch.shape[:2]
            rot = cv2.getRotationMatrix2D((pw / 2, ph / 2), float(fx["angle"]), 1.0)
            patch = cv2.warpAffine(patch, rot, (pw, ph), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            pmask = cv2.warpAffine(pmask.astype(np.uint8), rot, (pw, ph), flags=cv2.INTER_NEAREST).astype(bool)
            nx, ny = x0, y0
        elif fx["type"] == "move":
            nx, ny = x0 + int(fx["dx"]), y0 + int(fx["dy"])
        else:  # resize: se escala la pieza al tamaño de sus gemelos (anclada arriba a la izquierda)
            bx, by, bw, bh = fx["box"]
            before = [None, None]
            if fx.get("ref"):  # que el contenido quede donde está en su gemelo bien hecho
                rx0, ry0, rx1, ry1 = _clip_box(fx["ref"], W, H, pad=2)
                ref = img[ry0:ry1, rx0:rx1]
                before = [_content_start(ref, a) - _content_start(patch, a) for a in (0, 1)]
            patch, pmask = _stretch(patch, pmask, patch.shape[0] + int(fx["h"]) - bh, axis=0, before=before[0])
            patch, pmask = _stretch(patch, pmask, patch.shape[1] + int(fx["w"]) - bw, axis=1, before=before[1])
            nx, ny = x0, y0
        erase |= cv2.dilate(m.astype(np.uint8), np.ones((5, 5), np.uint8))
        pieces.append((patch, pmask, nx, ny))
        applied.append(f["id"])

    if pieces:
        # el hueco se rellena con el fondo que lo rodea (en UI casi siempre es plano)
        img = cv2.inpaint(img, erase, 3, cv2.INPAINT_TELEA)
        for patch, pmask, nx, ny in pieces:
            ph, pw = patch.shape[:2]
            tx0, ty0 = max(0, nx), max(0, ny)
            tx1, ty1 = min(W, nx + pw), min(H, ny + ph)
            if tx1 <= tx0 or ty1 <= ty0:
                continue
            sub = patch[ty0 - ny:ty1 - ny, tx0 - nx:tx1 - nx]
            sm = pmask[ty0 - ny:ty1 - ny, tx0 - nx:tx1 - nx]
            region = img[ty0:ty1, tx0:tx1]
            region[sm] = sub[sm]

    return img, sorted(set(applied))
