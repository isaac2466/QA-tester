"""Prueba el análisis completo SIN abrir la página web (usa el mismo código que el servidor).

    cd backend
    python tests/probar.py tests/samples/landing_con_errores.png
    python tests/probar.py mi_captura.png otra.png

Muestra, por cada imagen: lo que se ve en la app, lo que la IA descartó y por qué, y los grupos de DINOv2.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402

if len(sys.argv) < 2:
    sys.exit(__doc__)

with TestClient(app) as c:
    for src in sys.argv[1:]:
        t = time.time()
        with open(src, "rb") as f:
            r = c.post("/api/analyze", files={"file": (Path(src).name, f, "image/png")})
        if r.status_code != 200:
            print(f"\n✗ {src}: error {r.status_code} {r.text[:200]}")
            continue
        r = r.json()
        shown = [x for x in r["findings"] if x["verdict"] != "descartado"]
        gone = [x for x in r["findings"] if x["verdict"] == "descartado"]
        print(f"\n=== {Path(src).name}  ({time.time() - t:.0f} s)  puntaje {r['score']}/100")
        det = r.get("detector", {})
        print("Detector entrenado:", f"SÍ, aportó {det.get('elementos_del_modelo', 0)} elementos" if det.get("entrenado") else "no (falta app/ai/weights/detector.pt)")
        print("Grupos DINOv2:", ", ".join(f"{g['count']} {g['label']}" for g in r["components"]) or "ninguno")
        print(f"\nSE MUESTRAN ({len(shown)}):")
        for x in shown:
            print(f"  #{x['id']} [{x['severity']}] {x['title']}  ({x['source']})")
        print(f"\nDESCARTADOS ({len(gone)}):")
        for x in gone:
            print(f"  #{x['id']} {x['title']}\n      → {x['verdict_reason']}")
        print(f"\nCorrecciones en la vista corregida: {len(r['fixes_applied'])} · Marcas aprendidas: {r['learning']['marcas']}")
