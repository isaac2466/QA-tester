import base64

import cv2
import numpy as np

COLORS = {"alta": (230, 57, 70), "media": (245, 158, 11), "baja": (59, 130, 246)}  # RGB


def to_data_url(rgb: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    return "data:image/png;base64," + base64.b64encode(buf).decode()


def heatmap_overlay(rgb: np.ndarray, m: np.ndarray, alpha=0.5) -> np.ndarray:
    heat = cv2.applyColorMap((np.clip(m, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(rgb, 1 - alpha, heat, alpha, 0)


def annotate(rgb: np.ndarray, findings: list[dict]) -> np.ndarray:
    img = rgb.copy()
    t = max(2, round(min(img.shape[:2]) / 400))
    for f in findings:
        if not f.get("box"):
            continue
        color = COLORS.get(f["severity"], COLORS["baja"])
        x, y, w, h = f["box"]
        if f.get("ref_box"):
            rx, ry, rw, rh = f["ref_box"]
            cv2.rectangle(img, (rx - 2, ry - 2), (rx + rw + 1, ry + rh + 1), (120, 120, 120), 1, cv2.LINE_AA)
        cv2.rectangle(img, (x - 2, y - 2), (x + w + 1, y + h + 1), color, t, cv2.LINE_AA)
        label = str(f["id"])
        scale = 0.5 * t
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, t)
        ly = max(th + 6, y - 4)
        cv2.rectangle(img, (x - 2, ly - th - 6), (x + tw + 6, ly + 2), color, -1)
        cv2.putText(img, label, (x + 2, ly - 2), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), t, cv2.LINE_AA)
    return img


GROUP_COLORS = [(124, 58, 237), (5, 150, 105), (219, 39, 119), (234, 88, 12), (8, 145, 178),
                (101, 163, 13), (79, 70, 229), (190, 24, 93)]  # RGB


def components_overlay(rgb: np.ndarray, groups: list[dict]) -> np.ndarray:
    """Pinta cada grupo de componentes (DINOv2) con su color y su letra: A1, A2, B1..."""
    img = (rgb.astype(np.float32) * 0.55 + 255 * 0.45).astype(np.uint8)  # fondo atenuado
    t = max(2, round(min(img.shape[:2]) / 400))
    scale = 0.45 * t
    groups = [g for g in groups if g["count"] >= 3]
    for gi, g in enumerate(groups[:len(GROUP_COLORS)]):
        color = GROUP_COLORS[gi]
        letter = chr(ord("A") + gi)
        for k, (x, y, w, h) in enumerate(g["boxes"]):
            img[y:y + h, x:x + w] = rgb[y:y + h, x:x + w]  # el componente se ve sin atenuar
            cv2.rectangle(img, (x - 2, y - 2), (x + w + 1, y + h + 1), color, t, cv2.LINE_AA)
            label = f"{letter}{k + 1}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, t)
            ly = max(th + 6, y - 4)
            cv2.rectangle(img, (x - 2, ly - th - 6), (x + tw + 6, ly + 2), color, -1)
            cv2.putText(img, label, (x + 2, ly - 2), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), t, cv2.LINE_AA)
    return img
