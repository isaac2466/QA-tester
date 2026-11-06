"""Interpreta el mapa de saliencia de DINOv2: qué llama la atención y qué no."""
import cv2
import numpy as np

from app.services.colors import _chroma_hue
from app.services.findings import Finding
from app.services.layout import Element


def _fold_height(w: int, h: int) -> tuple[int, str]:
    """Altura del primer pantallazo (lo que se ve sin hacer scroll)."""
    if h > 1.25 * w:  # captura de página completa
        fold = int(w * 16 / 9) if w < 700 else int(w * 9 / 16)
        return min(h, fold), "lo que se ve sin hacer scroll"
    return h // 3, "el tercio superior"


def hotspots(sal: np.ndarray, thr=0.6, n=3):
    mask = (sal >= thr).astype(np.uint8)
    cnt, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    total = float(sal.sum()) or 1.0
    spots = []
    for i in range(1, cnt):
        x, y, w, h, area = stats[i]
        if area < 0.002 * sal.size:
            continue
        spots.append({"box": [int(x), int(y), int(w), int(h)],
                      "share": float(sal[labels == i].sum() / total)})
    spots.sort(key=lambda s: -s["share"])
    return spots[:n]


def cta_candidates(rgb: np.ndarray, elems: list[Element]) -> list[Element]:
    W = rgb.shape[1]
    out = []
    for e in elems:
        if e.kind != "bloque" or not (24 <= e.h <= 90) or not (1.8 <= e.w / e.h <= 10) or e.w > 0.5 * W:
            continue
        inner = rgb[e.y + e.h // 4: e.bottom - e.h // 4, e.x + 4: e.right - 4].reshape(-1, 3)
        if len(inner) and _chroma_hue(np.median(inner, axis=0))[0] > 30:
            out.append(e)
    return out


def check_attention(rgb: np.ndarray, sal: np.ndarray, elems: list[Element]):
    H, W = sal.shape
    findings: list[Finding] = []
    total = float(sal.sum()) or 1.0

    # 1) ¿Hay un punto focal claro?
    vals = np.sort(sal.ravel())[::-1]
    area_50 = float(np.searchsorted(np.cumsum(vals), 0.5 * total) / vals.size)
    spots = hotspots(sal)
    if area_50 > 0.30:
        findings.append(Finding(
            category="atencion", severity="alta",
            title="No hay un punto focal claro",
            detail=f"La atención está repartida por toda la página (hace falta el {area_50:.0%} del área "
                   f"para juntar la mitad de la atención). Nada destaca, así que la landing no 'engancha'.",
            suggestion="Crea jerarquía: un titular más grande y con más peso, una imagen protagonista "
                       "y un CTA con el color más llamativo de la página. Reduce lo que compite con ellos.",
        ))
    elif area_50 < 0.06 and spots:
        findings.append(Finding(
            category="atencion", severity="baja",
            title="Sólo una zona llama la atención",
            detail=f"Casi toda la atención se concentra en un área pequeña ({spots[0]['share']:.0%} de la atención "
                   f"en una sola zona). El resto de la página pasa desapercibido.",
            suggestion="Si esa zona es tu mensaje principal, perfecto. Si no, revisa que el titular y el "
                       "botón principal no queden eclipsados por una imagen o un elemento decorativo.",
            box=spots[0]["box"],
        ))

    # 2) ¿El primer pantallazo atrae?
    fold, fold_name = _fold_height(W, H)
    top_share = float(sal[:fold].sum() / total)
    expected = fold / H
    if expected < 0.9 and top_share < 0.8 * expected:
        findings.append(Finding(
            category="atencion", severity="alta",
            title="Lo más llamativo está abajo, no al inicio",
            detail=f"Sólo el {top_share:.0%} de la atención cae en {fold_name}, que ocupa el {expected:.0%} de "
                   f"la página. El usuario puede irse antes de ver lo interesante.",
            suggestion="Sube tu propuesta de valor y el CTA al hero, o dale más contraste/tamaño al hero.",
            box=[0, 0, W, fold],
        ))

    # 3) ¿El botón principal (CTA) destaca?
    mean_sal = float(sal.mean())
    for e in cta_candidates(rgb, elems)[:3]:
        s = float(sal[e.y:e.bottom, e.x:e.right].mean())
        if s < mean_sal * 1.1:
            findings.append(Finding(
                category="atencion", severity="media",
                title="Este botón no destaca",
                detail=f"El botón en x={e.x}, y={e.y} recibe menos atención que el promedio de la página "
                       f"({s / (mean_sal or 1):.1f}× el promedio). Si es tu llamada a la acción, se pierde.",
                suggestion="Hazlo más grande, dale más espacio alrededor o usa un color con más contraste "
                           "respecto a lo que lo rodea.",
                box=e.box,
            ))

    for f in findings:
        f.source = "dinov2"  # todo sale del mapa de atención del modelo
    return findings, {
        "focus_area_50": round(area_50, 3),
        "top_share": round(top_share, 3),
        "hotspots": spots,
    }
