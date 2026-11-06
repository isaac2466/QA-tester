"""Analiza una captura con reglas + IA local (Ollama) sin levantar el servidor.

    python tests/run_ia.py tests/samples/real_softtek.png
"""
import sys
import time

import numpy as np
from PIL import Image

sys.path.insert(0, ".")
from app.services import vlm  # noqa: E402
from app.services.colors import check_colors, check_contrast  # noqa: E402
from app.services.layout import check_alignment, check_centering, check_spacing, detect_elements  # noqa: E402
from app.services.render import annotate  # noqa: E402

img = Image.open(sys.argv[1]).convert("RGB")
rgb = np.asarray(img)
el = detect_elements(rgb)
fs = check_alignment(el) + check_centering(el, rgb.shape[1]) + check_spacing(el)
fs += check_colors(rgb)[0] + check_contrast(rgb, el)
cands = [f.to_dict(i + 1) for i, f in enumerate(fs)]

print("Estado IA:", vlm.status())
t0 = time.time()
confirmed, rejected, reasons, new, summary = vlm.review(annotate(rgb, cands), cands, img.size)
print(f"\nTiempo IA: {time.time() - t0:.0f}s\n")
print("== Mediciones verificadas por la IA ==")
for c in cands:
    tag = "REAL " if c["id"] in confirmed else "FALSO" if c["id"] in rejected else "¿?   "
    print(f"[{tag}] #{c['id']} {c['title']}\n         razón: {reasons.get(c['id'], '-')}")
print("\n== Hallazgos propios de la IA ==")
for f in new:
    print(f"- [{f.severity}] {f.title}\n    {f.detail}\n    -> {f.suggestion}  caja={f.box}")
print("\nResumen:", summary)
