"""Genera capturas de prueba con errores plantados a propósito.

    python tests/make_samples.py  -> tests/samples/*.png
"""
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).parent / "samples"


def font(size):
    for name in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def page(btn_shift=0, cta_color=(37, 99, 235), card_gap_bug=True, title_y=60):
    W, H = 1200, 800
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, W, 56], fill=(15, 23, 42))
    d.text((32, 16), "MiMarca", fill="white", font=font(22))
    d.text((80, title_y + 40), "Lanza tu producto mas rapido", fill=(15, 23, 42), font=font(44))
    d.text((80, title_y + 110), "La herramienta que tu equipo necesita.", fill=(71, 85, 105), font=font(22))
    # texto con bajo contraste (gris claro sobre blanco)
    d.text((80, title_y + 150), "Sin tarjeta de credito. Cancela cuando quieras.", fill=(203, 213, 225), font=font(18))
    # dos botones apilados: el segundo está btn_shift px corrido
    d.rounded_rectangle([80, 300, 300, 352], 8, fill=cta_color)
    d.text((120, 314), "Empezar gratis", fill="white", font=font(20))
    d.rounded_rectangle([80 + btn_shift, 372, 300 + btn_shift, 424], 8, outline=(37, 99, 235), width=2)
    d.text((130 + btn_shift, 386), "Ver demo", fill=(37, 99, 235), font=font(20))
    # tres tarjetas en fila: gaps 24 y 23 (bug) / 24
    xs = [80, 80 + 320 + 24, 80 + 2 * 320 + 24 + (23 if card_gap_bug else 24)]
    xs.append(80 + 3 * 320)  # dummy
    for i, x in enumerate(xs[:3]):
        fill = (241, 245, 249) if i != 1 else (238, 242, 247)  # tono casi idéntico en la 2ª
        d.rounded_rectangle([x, 480, x + 319, 700], 12, fill=fill)
        d.text((x + 24, 500), f"Ventaja {i + 1}", fill=(15, 23, 42), font=font(22))
    return img


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    page(btn_shift=1).save(OUT / "landing_con_errores.png")
    page(btn_shift=0, card_gap_bug=False).save(OUT / "diseno.png")
    page(btn_shift=3, cta_color=(29, 78, 216), card_gap_bug=False).save(OUT / "implementacion.png")
    print("ok ->", OUT)
