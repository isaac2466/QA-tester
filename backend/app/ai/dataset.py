"""Generador de pantallas sintéticas para ENTRENAR la IA.

Dibuja interfaces al azar (menú lateral, botones, tarjetas, tablas con avatares y etiquetas,
paginación, buscador...) y a algunas les mete errores a propósito. Como el generador sabe
dónde dibujó cada cosa y qué error metió, cada pantalla sale con sus etiquetas:

    pantalla_0001.png  +  pantalla_0001.json  {"elements": [{"box", "type", "group", "error"}...]}

Uso:
    python -m app.ai.dataset --n 300 --out ai_data/train
"""
import argparse
import colorsys
import json
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

TYPES = ["boton", "tarjeta", "etiqueta", "avatar", "campo", "icono", "pagina", "texto"]
ERRORS = ["ninguno", "desalineado", "tamano", "color", "contraste"]

FONTS = ["arial.ttf", "segoeui.ttf", "calibri.ttf", "verdana.ttf", "tahoma.ttf"]
BOLD = {"arial.ttf": "arialbd.ttf", "segoeui.ttf": "segoeuib.ttf", "calibri.ttf": "calibrib.ttf",
        "verdana.ttf": "verdanab.ttf", "tahoma.ttf": "tahomabd.ttf"}
WORDS = ("Proyecto Diseño Equipo Reporte Cliente Tarea Inicio Ventas Pedido Usuario Producto Panel Ajustes "
         "Nuevo Guardar Editar Ver Enviar Crear Buscar Filtros Exportar Agregar Plan Activo Pendiente "
         "Completado Revisión Entrega Estado Total Mensual Equipo Marketing Soporte Ana Luis Carla Diego").split()


def _font(name, size, bold=False):
    try:
        return ImageFont.truetype(BOLD.get(name, name) if bold else name, size)
    except OSError:
        return ImageFont.truetype("arial.ttf", size)


def _rand_color(sat=(0.5, 0.9), val=(0.45, 0.85)):
    h = random.random()
    r, g, b = colorsys.hsv_to_rgb(h, random.uniform(*sat), random.uniform(*val))
    return int(r * 255), int(g * 255), int(b * 255)


def _soft(c, k=0.85):
    return tuple(int(v + (255 - v) * k) for v in c)


def _lum(c):
    def ch(v):
        v /= 255
        return v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4
    r, g, b = (ch(v) for v in c)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _contrast(a, b):
    la, lb = sorted((_lum(a), _lum(b)), reverse=True)
    return (la + 0.05) / (lb + 0.05)


def _words(n):
    return " ".join(random.choice(WORDS) for _ in range(n))


class Page:
    """Una pantalla en construcción: lista de componentes con sus cajas y estilos."""

    def __init__(self, W, H):
        self.W, self.H = W, H
        self.bg = random.choice([(255, 255, 255), (248, 249, 251), (246, 247, 251), (250, 250, 250), (243, 244, 246)])
        self.font = random.choice(FONTS)
        self.accent = _rand_color()
        self.ink = random.choice([(17, 24, 39), (30, 30, 30), (15, 23, 42), (40, 44, 52)])
        self.muted = random.choice([(100, 106, 120), (107, 114, 128), (90, 96, 110)])
        self.items = []  # dicts: type, box, group, style, text, error, children
        self.group_n = 0

    def group(self):
        self.group_n += 1
        return self.group_n

    def add(self, type_, box, group=None, **style):
        it = {"type": type_, "box": list(box), "group": group, "error": "ninguno", **style}
        self.items.append(it)
        return it


# ---------- componentes ----------

def sidebar(p: Page, w):
    p.add("panel", (0, 0, w, p.H), fill=random.choice([(255, 255, 255), (250, 250, 252), (17, 24, 39)]))
    dark = p.items[-1]["fill"][0] < 100
    p.add("texto", (28, 34, 0, 0), text=random.choice(["Forma", "Nova", "Orbit", "Flux", "Kairo"]),
          size=random.randint(22, 30), bold=True, color=(255, 255, 255) if dark else p.ink)
    g = p.group()
    y = 120
    sel = random.randrange(5)
    for i in range(random.randint(4, 6)):
        h = random.choice([40, 44, 48])
        if i == sel:
            p.add("boton", (14, y, w - 28, h), group=None, fill=_soft(p.accent, 0.8), radius=8, variant=True)
        p.add("icono", (28, y + h // 2 - 9, 18, 18), group=g, fill=p.accent if i == sel else p.muted, radius=4)
        p.add("texto", (60, y + h // 2, 0, 0), text=random.choice(WORDS), size=random.randint(15, 18),
              color=p.accent if i == sel else ((220, 220, 230) if dark else p.ink), anchor="lm")
        y += h + random.choice([12, 16, 20])


def header(p: Page, x0, x1, y):
    p.add("texto", (x0, y, 0, 0), text=random.choice(WORDS), size=random.randint(30, 40), bold=True, color=p.ink)
    p.add("texto", (x0, y + 50, 0, 0), text=_words(random.randint(3, 6)), size=random.randint(16, 19), color=p.muted)
    bw, bh = random.randint(150, 210), random.choice([40, 44, 48])
    b = p.add("boton", (x1 - bw, y + 4, bw, bh), fill=p.accent, radius=random.choice([6, 8, 10]))
    b["label"] = ("+ " if random.random() < .5 else "") + random.choice(["Nuevo", "Crear", "Agregar"]) + " " + random.choice(WORDS).lower()
    b["label_color"] = (255, 255, 255)
    return y + 100


def stat_cards(p: Page, x0, x1, y):
    n = random.randint(3, 4)
    gap = random.choice([16, 20, 24])
    w = (x1 - x0 - gap * (n - 1)) // n
    h = random.randint(110, 150)
    g = p.group()
    fill = random.choice([(255, 255, 255), (250, 250, 252)])
    border = random.choice([(229, 231, 235), (226, 232, 240), None])
    for i in range(n):
        c = p.add("tarjeta", (x0 + i * (w + gap), y, w, h), group=g, fill=fill, border=border, radius=random.choice([8, 12]))
        c["children"] = [("texto", 24, 28, _words(2), 16, p.muted, False),
                         ("texto", 24, 64, str(random.randint(3, 99)) + random.choice(["", "%", ""]), 36, p.ink, True)]
    return y + h + random.choice([28, 36, 40])


def button_row(p: Page, x0, x1, y):
    n = random.randint(3, 6)
    w, h = random.randint(110, 170), random.choice([36, 40, 44, 48])
    gap = random.choice([12, 16, 20, 24])
    g = p.group()
    filled = random.random() < 0.6
    for i in range(n):
        b = p.add("boton", (x0 + i * (w + gap), y, w, h), group=g, radius=random.choice([4, 6, 8, 22]),
                  fill=p.accent if filled else p.bg, border=None if filled else p.accent)
        b["label"], b["label_color"] = random.choice(WORDS), (255, 255, 255) if filled else p.accent
        if x0 + (i + 1) * (w + gap) > x1:
            p.items.pop()
            break
    return y + h + random.choice([28, 36])


def search_row(p: Page, x0, x1, y):
    h = random.choice([40, 44, 48])
    w = int((x1 - x0) * random.uniform(0.4, 0.65))
    p.add("campo", (x0, y, w, h), fill=(255, 255, 255), border=(209, 213, 219), radius=8,
          placeholder=random.choice(["Buscar...", "Buscar proyecto...", "Escribe aquí..."]))
    bw = random.randint(100, 130)
    b = p.add("boton", (x1 - bw, y, bw, h), fill=(255, 255, 255), border=(209, 213, 219), radius=8)
    b["label"], b["label_color"] = random.choice(["Filtros", "Ordenar", "Exportar"]), p.ink
    return y + h + random.choice([24, 32])


def table(p: Page, x0, x1, y):
    rows = random.randint(3, 6)
    rh = random.choice([56, 64, 72, 80])
    p.add("tarjeta", (x0, y, x1 - x0, rh * (rows + 1)), group=None, fill=(255, 255, 255), border=(229, 231, 235), radius=10)
    cols = [x0 + 28, x0 + int((x1 - x0) * .32), x0 + int((x1 - x0) * .58), x0 + int((x1 - x0) * .8)]
    for c, name in zip(cols, ["Nombre", "Responsable", "Estado", "Fecha"]):
        p.add("texto", (c, y + rh // 2, 0, 0), text=name, size=16, color=p.muted, anchor="lm")
    ga, gb = p.group(), p.group()
    av = random.choice([36, 40, 44])
    statuses = [(_rand_color(), random.choice(["Activo", "En curso"])), (_rand_color(), "Pendiente"), (_rand_color(), "Revisión")]
    bw, bh = random.randint(100, 130), random.choice([28, 30, 32])
    for r in range(rows):
        cy = y + rh * (r + 1) + rh // 2
        p.add("linea", (x0, y + rh * (r + 1), x1 - x0, 1), fill=(235, 236, 240))
        p.add("texto", (cols[0], cy, 0, 0), text=_words(2), size=17, bold=True, color=p.ink, anchor="lm")
        a = p.add("avatar", (cols[1], cy - av // 2, av, av), group=ga, fill=_soft(p.accent, 0.82), radius=av // 2)
        a["label"], a["label_color"] = random.choice("ABCDLMRT") + random.choice("ABCDLMRT"), p.accent
        p.add("texto", (cols[1] + av + 14, cy, 0, 0), text=_words(2), size=17, color=p.ink, anchor="lm")
        col, txt = random.choice(statuses)
        e = p.add("etiqueta", (cols[2], cy - bh // 2, bw, bh), group=gb, fill=_soft(col, 0.75), radius=bh // 2, variant=True)
        e["label"], e["label_color"] = txt, tuple(int(v * 0.6) for v in col)
        p.add("texto", (cols[3], cy, 0, 0), text=f"{random.randint(1, 28):02d} {random.choice(['ene', 'nov', 'dic'])} 2026",
              size=17, color=p.ink, anchor="lm")
    return y + rh * (rows + 1) + 24


def pagination(p: Page, x0, x1, y):
    n = random.randint(3, 5)
    s = random.choice([36, 40, 44])
    g = p.group()
    act = random.randrange(n)
    for i in range(n):
        x = x1 - (n - i) * (s + 12)
        b = p.add("pagina", (x, y, s, s), group=g, radius=8, fill=p.accent if i == act else (255, 255, 255),
                  border=None if i == act else (209, 213, 219), variant=i == act)
        b["label"], b["label_color"] = str(i + 1), (255, 255, 255) if i == act else p.ink
    p.add("texto", (x0, y + s // 2, 0, 0), text=f"Mostrando {random.randint(3, 9)} de {random.randint(10, 60)}",
          size=16, color=p.muted, anchor="lm")
    return y + s + 30


def card_grid(p: Page, x0, x1, y):
    n = random.randint(2, 4)
    gap = random.choice([16, 20, 24])
    w = (x1 - x0 - gap * (n - 1)) // n
    h = random.randint(140, 220)
    g = p.group()
    for i in range(n):
        c = p.add("tarjeta", (x0 + i * (w + gap), y, w, h), group=g, fill=random.choice([(255, 255, 255), (241, 245, 249)]),
                  border=(229, 231, 235), radius=12)
        c["children"] = [("texto", 20, 24, _words(2), 19, p.ink, True), ("texto", 20, 58, _words(3), 15, p.muted, False)]
    return y + h + 30


# ---------- errores ----------

def _shift_color(c, mag):
    import math
    v = [random.uniform(-1, 1) for _ in range(3)]
    n = math.sqrt(sum(x * x for x in v)) or 1
    return tuple(max(0, min(255, int(ch + mag * x / n))) for ch, x in zip(c, v))


def plant_errors(p: Page, max_errors=3):
    groups = {}
    for it in p.items:
        if it["group"] and not it.get("variant"):
            groups.setdefault(it["group"], []).append(it)
    groups = [g for g in groups.values() if len(g) >= 3]
    random.shuffle(groups)
    planted = 0
    for g in groups:
        if planted >= max_errors or random.random() < 0.35:
            continue
        it = random.choice(g)
        kind = random.choice(["desalineado", "tamano", "color"])
        x, y, w, h = it["box"]
        if kind == "desalineado":
            d = random.choice([-1, 1]) * random.randint(2, 6)
            it["box"] = [x + d, y, w, h] if random.random() < .5 else [x, y + d, w, h]
            it["error_param"] = {"dx": it["box"][0] - x, "dy": it["box"][1] - y}
        elif kind == "tamano":
            d = random.choice([-1, 1]) * random.randint(3, 8)
            it["box"] = [x, y, w, h + d] if random.random() < .6 or it["type"] == "avatar" else [x, y, w + d, h]
            if it["type"] in ("avatar", "pagina", "icono"):  # cuadrados: cambian ambos lados
                it["box"] = [x, y, w + d, h + d]
            it["error_param"] = {"dw": it["box"][2] - w, "dh": it["box"][3] - h}
        else:
            base = it.get("fill") if it.get("fill") not in (None, p.bg) else it.get("border")
            if base is None:
                continue
            new = _shift_color(base, random.uniform(10, 30))
            it["fill" if it.get("fill") not in (None, p.bg) else "border"] = new
            it["error_param"] = {"from": base, "to": new}
        it["error"] = kind
        planted += 1
    # texto poco legible
    texts = [it for it in p.items if it["type"] == "texto" and it.get("size", 16) <= 22]
    if texts and random.random() < 0.45:
        t = random.choice(texts)
        for _ in range(30):
            c = tuple(min(255, int(v + random.uniform(0.55, 0.85) * (255 - v))) for v in t["color"])
            if _contrast(c, p.bg) < 2.6:
                t["color"], t["error"], t["error_param"] = c, "contraste", {"ratio": round(_contrast(c, p.bg), 2)}
                break


# ---------- dibujo ----------

def render(p: Page):
    img = Image.new("RGB", (p.W, p.H), p.bg)
    d = ImageDraw.Draw(img)
    out = []

    def text(x, y, s, size, color, bold=False, anchor="la"):
        f = _font(p.font, size, bold)
        d.text((x, y), s, fill=color, font=f, anchor=anchor)
        return d.textbbox((x, y), s, font=f, anchor=anchor)

    for it in p.items:
        x, y, w, h = it["box"]
        t = it["type"]
        if t == "texto":
            l, tp, r, b = text(x, y, it["text"], it["size"], it["color"], it.get("bold", False), it.get("anchor", "la"))
            out.append({"box": [l, tp, r - l, b - tp], "type": "texto", "group": None, "error": it["error"],
                        "error_param": it.get("error_param")})
            continue
        if t in ("panel", "linea"):
            d.rectangle([x, y, x + w - 1, y + h - 1], fill=it["fill"])
            continue
        d.rounded_rectangle([x, y, x + w - 1, y + h - 1], it.get("radius", 0), fill=it.get("fill"),
                            outline=it.get("border"), width=1 if t in ("tarjeta", "campo", "pagina") else 2)
        out.append({"box": [x, y, w, h], "type": t, "group": it["group"], "error": it["error"],
                    "error_param": it.get("error_param"), "variant": bool(it.get("variant"))})
        if it.get("label"):
            size = 14 if t in ("etiqueta", "avatar") else 16
            l, tp, r, b = text(x + w / 2, y + h / 2, it["label"], size, it["label_color"], t != "boton", "mm")
            out.append({"box": [l, tp, r - l, b - tp], "type": "texto", "group": None, "error": "ninguno"})
        if it.get("placeholder"):
            text(x + 44, y + h / 2, it["placeholder"], 16, (150, 155, 165), anchor="lm")
            d.ellipse([x + 16, y + h / 2 - 8, x + 30, y + h / 2 + 6], outline=(120, 125, 135), width=2)
        for (_, ox, oy, s, size, color, bold) in it.get("children", []):
            l, tp, r, b = text(x + ox, y + oy, s, size, color, bold)
            out.append({"box": [l, tp, r - l, b - tp], "type": "texto", "group": None, "error": "ninguno"})
    return img, out


def make_page(seed=None):
    if seed is not None:
        random.seed(seed)
    W = random.choice([1280, 1366, 1440, 1536])
    H = random.randint(820, 1100)
    p = Page(W, H)
    x0 = 32
    if random.random() < 0.7:
        sw = random.randint(220, 290)
        sidebar(p, sw)
        x0 = sw + random.randint(32, 48)
    x1 = W - random.randint(24, 40)
    y = header(p, x0, x1, random.randint(36, 56))
    blocks = [stat_cards, button_row, search_row, table, card_grid, pagination]
    random.shuffle(blocks)
    for blk in blocks:
        if y > H - 120:
            break
        if random.random() < 0.8:
            y = blk(p, x0, x1, y)
    if random.random() < 0.8:
        plant_errors(p)
    img, els = render(p)
    els = [e for e in els if e["box"][2] >= 6 and e["box"][3] >= 6 and e["box"][1] + e["box"][3] <= H]
    return img, {"size": [W, H], "elements": els}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--out", default="ai_data/train")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for i in range(a.n):
        img, meta = make_page(a.seed * 100000 + i)
        img.save(out / f"pantalla_{i:04d}.png")
        (out / f"pantalla_{i:04d}.json").write_text(json.dumps(meta), encoding="utf-8")
    print(f"{a.n} pantallas en {out}")
