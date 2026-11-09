"""Analiza una captura (reglas + DINOv2) y pide a Qwen que explique los hallazgos, sin levantar el servidor.

Desde la carpeta backend/ (con Ollama abierto):
    python tests/run_ia.py tests/samples/real_softtek.png [cuantos]
"""
import sys
import time

import numpy as np
from PIL import Image

sys.path.insert(0, ".")
from app.services import dino, vlm  # noqa: E402
from app.services.components import check_components  # noqa: E402
from app.services.colors import check_colors, check_contrast  # noqa: E402
from app.services.layout import check_alignment, check_centering, check_spacing, detect_elements  # noqa: E402

img = Image.open(sys.argv[1]).convert("RGB")
n = int(sys.argv[2]) if len(sys.argv) > 2 else 4
rgb = np.asarray(img)
el = detect_elements(rgb)
fs = check_alignment(el) + check_centering(el, rgb.shape[1]) + check_spacing(el)
fs += check_colors(rgb)[0] + check_contrast(rgb, el)
dino.load()
if dino.get():
    fs += check_components(rgb, el, dino.get())[0]
dicts = [f.to_dict(i + 1) for i, f in enumerate(fs)][:n]

print("Qwen:", vlm.status())
t0 = time.perf_counter()
out = vlm.explain(rgb, dicts)
print(f"Qwen tardó {time.perf_counter() - t0:.1f}s para {len(dicts)} hallazgos\n")
for f in dicts:
    e = out.get(f["id"], {})
    print(f"#{f['id']} [{f['category']}] {f['title']}")
    print(f"   regla:   {f['detail']}")
    print(f"   Qwen ve: {e.get('contenido')} '{e.get('texto', '')}'")
    print(f"   Qwen:    {e.get('veredicto', '(sin respuesta)')} - {e.get('razon', '')}")
    for k in ("que_pasa", "por_que_importa", "como_arreglarlo"):
        print(f"   {k}: {e.get(k, '')}")
    print()
