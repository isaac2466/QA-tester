"""Paleta, tonos casi duplicados, choques de color y contraste WCAG."""
import cv2
import numpy as np

from app.services.findings import Finding
from app.services.layout import Element


def hex_of(rgb) -> str:
    r, g, b = (int(round(v)) for v in rgb[:3])
    return f"#{r:02X}{g:02X}{b:02X}"


def to_lab(rgb: np.ndarray) -> np.ndarray:
    arr = np.asarray(rgb, dtype=np.float32).reshape(-1, 1, 3) / 255.0
    return cv2.cvtColor(arr, cv2.COLOR_RGB2Lab).reshape(-1, 3)


def delta_e(c1, c2) -> float:
    l1, l2 = to_lab(np.array([c1, c2]))
    return float(np.linalg.norm(l1 - l2))


def luminance(rgb) -> float:
    c = np.asarray(rgb, dtype=np.float64) / 255.0
    c = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    return float(0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2])


def contrast_ratio(a, b) -> float:
    la, lb = sorted((luminance(a), luminance(b)), reverse=True)
    return (la + 0.05) / (lb + 0.05)


def dominant_flat_colors(rgb: np.ndarray, min_share=0.0015):
    """Colores exactos que cubren áreas planas (típico de UI). Devuelve [(rgb, share)]."""
    px = rgb.reshape(-1, 3)
    packed = (px[:, 0].astype(np.int32) << 16) | (px[:, 1].astype(np.int32) << 8) | px[:, 2]
    vals, counts = np.unique(packed, return_counts=True)
    total = len(packed)
    keep = counts >= max(min_share * total, 40)
    vals, counts = vals[keep], counts[keep]
    order = np.argsort(-counts)
    out = []
    for v, c in zip(vals[order], counts[order]):
        out.append((np.array([(v >> 16) & 255, (v >> 8) & 255, v & 255]), c / total))
    return out


def palette(rgb: np.ndarray, k=8):
    flat = dominant_flat_colors(rgb)
    coverage = sum(s for _, s in flat)
    if coverage >= 0.4:
        # UI "plana": agrupamos colores idénticos a la vista (ΔE < 2.3)
        groups = []
        for c, s in flat:
            for g in groups:
                if delta_e(g[0], c) < 2.3:
                    g[1] += s
                    break
            else:
                groups.append([c, s])
        return [(c, s) for c, s in groups[:10]], flat, True
    # imagen fotográfica / con degradados → k-means
    small = cv2.resize(rgb, (160, max(1, int(160 * rgb.shape[0] / rgb.shape[1]))))
    data = small.reshape(-1, 3).astype(np.float32)
    _, labels, centers = cv2.kmeans(data, k, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0),
                                    3, cv2.KMEANS_PP_CENTERS)
    shares = np.bincount(labels.ravel(), minlength=k) / len(labels)
    order = np.argsort(-shares)
    return [(centers[i], shares[i]) for i in order], flat, False


def _chroma_hue(rgb):
    L, a, b = to_lab(np.array([rgb]))[0]
    return float(np.hypot(a, b)), float(np.degrees(np.arctan2(b, a)) % 360), float(L)


def _bbox_of_color(rgb: np.ndarray, color) -> list[int] | None:
    mask = np.all(rgb == np.asarray(color, dtype=rgb.dtype), axis=-1).astype(np.uint8)
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if n <= 1:
        return None
    i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x, y, w, h = stats[i, :4]
    return [int(x), int(y), int(w), int(h)]


def _is_solid_surface(rgb: np.ndarray, color) -> bool:
    """¿El color forma una superficie sólida (tarjeta, botón, fondo) y no una franja de degradado?"""
    mask = np.all(rgb == np.asarray(color, dtype=rgb.dtype), axis=-1).astype(np.uint8)
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, 4)
    if n <= 1:
        return False
    i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x, y, w, h, area = stats[i]
    # una superficie real es grande y "rellena" su caja; un degradado deja franjas finas
    return area >= max(600, 0.001 * mask.size) and area / float(w * h) > 0.6 and min(w, h) >= 12


def check_colors(rgb: np.ndarray):
    pal, flat, is_flat = palette(rgb)
    findings: list[Finding] = []

    if is_flat:
        # Tonos casi idénticos: suele indicar un color "hardcodeado" en vez del token.
        seen = set()
        page_bg = hex_of(flat[0][0])  # una "superficie" sutil sobre el fondo es intencional
        for i, (c1, s1) in enumerate(flat[:24]):
            for c2, s2 in flat[i + 1:24]:
                de = delta_e(c1, c2)
                if page_bg in (hex_of(c1), hex_of(c2)):
                    continue
                if 0.3 < de < 3.5 and min(s1, s2) > 0.002 and                         _is_solid_surface(rgb, c1) and _is_solid_surface(rgb, c2):
                    minor = c2 if s2 < s1 else c1
                    major = c1 if s2 < s1 else c2
                    k = hex_of(minor)
                    if k in seen:
                        continue
                    seen.add(k)
                    findings.append(Finding(
                        category="color", severity="media",
                        title=f"Dos tonos casi iguales: {hex_of(major)} y {hex_of(minor)}",
                        detail=f"La diferencia es mínima (ΔE {de:.1f}): a simple vista parecen el mismo color, "
                               f"pero no lo son. Suele pasar al copiar un color a mano en vez de usar la variable/token.",
                        suggestion=f"Unifica {hex_of(minor)} → {hex_of(major)} (o usa la misma variable CSS).",
                        box=_bbox_of_color(rgb, minor),
                    ))
        findings = findings[:6]

    # Demasiados colores de acento
    accents = [(c, s) for c, s in pal if s > 0.004 and _chroma_hue(c)[0] > 25]
    hue_groups: list[float] = []
    for c, _ in accents:
        h = _chroma_hue(c)[1]
        if all(min(abs(h - g), 360 - abs(h - g)) > 30 for g in hue_groups):
            hue_groups.append(h)
    if len(hue_groups) > 3:
        findings.append(Finding(
            category="color", severity="media",
            title=f"Demasiados colores de acento ({len(hue_groups)} tonos distintos)",
            detail="Hay varios colores saturados compitiendo entre sí; la vista no sabe a dónde ir.",
            suggestion="Limita la paleta a 1 color primario + 1 secundario y neutros (grises). "
                       "Reserva el primario para la acción principal (CTA).",
        ))

    # Colores complementarios saturados con luminosidad parecida → "vibran"
    sat = [(c, s) for c, s in pal if s > 0.01 and _chroma_hue(c)[0] > 40]
    for i, (c1, _) in enumerate(sat):
        for c2, _ in sat[i + 1:]:
            ch1, h1, l1 = _chroma_hue(c1)
            ch2, h2, l2 = _chroma_hue(c2)
            dh = min(abs(h1 - h2), 360 - abs(h1 - h2))
            if dh > 140 and abs(l1 - l2) < 15:
                findings.append(Finding(
                    category="color", severity="media",
                    title=f"Estos colores chocan: {hex_of(c1)} y {hex_of(c2)}",
                    detail="Son colores opuestos, muy saturados y con la misma luminosidad: juntos 'vibran' "
                           "y cansan la vista.",
                    suggestion="Baja la saturación de uno de los dos, o separa su luminosidad (uno claro, otro oscuro).",
                ))
                break

    return findings, [{"hex": hex_of(c), "share": round(float(s), 4)} for c, s in pal]


def check_contrast(rgb: np.ndarray, elems: list[Element]) -> list[Finding]:
    H, W = rgb.shape[:2]
    out = []
    for e in elems:
        if e.kind != "texto" or not (8 <= e.h <= 90):
            continue
        x0, y0 = max(0, e.x - 3), max(0, e.y - 3)
        x1, y1 = min(W, e.right + 3), min(H, e.bottom + 3)
        crop = rgb[y0:y1, x0:x1].reshape(-1, 3).astype(np.float32)
        if len(crop) < 30:
            continue
        ring = np.concatenate([rgb[y0, x0:x1], rgb[y1 - 1, x0:x1], rgb[y0:y1, x0], rgb[y0:y1, x1 - 1]])
        # el texto real va sobre un fondo liso; si el borde varía mucho es una imagen/degradado
        if ring.astype(np.float32).std(axis=0).max() > 10:
            continue
        bg = np.median(ring.astype(np.float32), axis=0)
        dist = np.linalg.norm(crop - bg, axis=1)
        if dist.max() < 10:
            continue
        fg = np.median(crop[dist >= np.percentile(dist, 90)], axis=0)
        ratio = contrast_ratio(fg, bg)
        large = e.h >= 24  # aprox. ≥18pt
        need = 3.0 if large else 4.5
        if ratio < need:
            out.append((ratio, Finding(
                category="contraste",
                severity="alta" if ratio < 3 else "media",
                title=f"Contraste bajo {ratio:.1f}:1 (mínimo WCAG AA {need:g}:1)",
                detail=f"Texto {hex_of(fg)} sobre fondo {hex_of(bg)} en x={e.x}, y={e.y}. "
                       f"Será difícil de leer, sobre todo en móvil o con sol.",
                suggestion="Oscurece el texto o aclara el fondo hasta llegar a "
                           f"{need:g}:1 (puedes verificarlo con cualquier contrast checker).",
                box=e.box,
            )))
    out.sort(key=lambda t: t[0])
    return [f for _, f in out[:8]]
