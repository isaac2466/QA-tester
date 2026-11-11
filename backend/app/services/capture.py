"""Captura una URL en varios tamaños (móvil, tablet, escritorio) con Playwright.

Playwright corre en un proceso aparte: así no choca con el event loop de uvicorn en Windows
y, si el navegador se cuelga, se puede matar sin tumbar el backend.
"""
import json
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlparse

from app.services.capture_worker import SIZES

BACKEND = Path(__file__).resolve().parents[2]
TIMEOUT = 150  # segundos para todas las capturas juntas


def normalize_url(raw: str) -> str:
    url = raw.strip()
    if not url:
        raise ValueError("Escribe la dirección de la página.")
    if "://" not in url:
        url = "https://" + url  # "misitio.com" → "https://misitio.com"
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise ValueError("La dirección debe empezar con http:// o https://")
    return url


def capture(url: str, sizes: list[str]) -> list[dict]:
    sizes = [s for s in dict.fromkeys(sizes) if s in SIZES]
    if not sizes:
        raise ValueError("Elige al menos un tamaño.")
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "app.services.capture_worker", json.dumps({"url": url, "sizes": sizes})],
            cwd=BACKEND, capture_output=True, timeout=TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError("Las capturas tardaron demasiado. Prueba con menos tamaños.")
    try:
        out = json.loads(proc.stdout.decode("utf-8"))
    except json.JSONDecodeError:
        tail = proc.stderr.decode("utf-8", "replace").strip().splitlines()[-1:] or ["sin detalles"]
        raise RuntimeError(f"No se pudo tomar la captura ({tail[0]}).")
    if "error" in out:
        raise RuntimeError(out["error"])
    return out["shots"]
