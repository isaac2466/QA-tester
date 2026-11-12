"""Prueba la revisión de página completa con Qwen, sin abrir la app.

    cd backend
    python tests/revisar_qwen.py ruta\a\captura.png

Tarda varios minutos en CPU. Necesita Ollama abierto y el modelo: ollama pull qwen3-vl:4b-instruct
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from app.services import vlm  # noqa: E402

if len(sys.argv) < 2:
    sys.exit(__doc__)
print("Qwen:", vlm.status())
rgb = np.asarray(Image.open(sys.argv[1]).convert("RGB"))
t = time.time()
found, summary = vlm.review_page(rgb)
print(f"\nTardó {time.time() - t:.0f} s · {len(found)} problemas\n")
for f in found:
    print(f"- [{f['category']} · {f['severity']}] {f['title']}\n    {f['detail']}\n    → {f['suggestion']}")
print("\nResumen:", summary)
