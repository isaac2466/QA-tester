"""Prueba los analizadores sin levantar el servidor: python tests/run_local.py <imagen>"""
import sys
import numpy as np
from PIL import Image
sys.path.insert(0, ".")
from app.services.layout import detect_elements, check_alignment, check_centering, check_spacing
from app.services.colors import check_colors, check_contrast

rgb = np.asarray(Image.open(sys.argv[1]).convert("RGB"))
el = detect_elements(rgb)
for e in el:
    print(" ", e)
fs = check_alignment(el) + check_centering(el, rgb.shape[1]) + check_spacing(el)
cf, pal = check_colors(rgb)
fs += cf + check_contrast(rgb, el)
print("paleta:", pal)
for f in fs:
    print(f"- [{f.severity}] {f.title}\n    {f.detail}\n    -> {f.suggestion}")
