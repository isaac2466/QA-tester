"""Toma capturas de una URL en varios tamaños. Corre como proceso aparte (ver capture.py).

    python -m app.services.capture_worker '{"url": "https://...", "sizes": ["movil", "escritorio"]}'

Imprime en stdout un JSON: {"shots": [{"size", "label", "width", "height", "png_b64"}]} o {"error": "..."}.
Usa el Edge o Chrome que ya tiene la computadora; si no hay, el Chromium de Playwright.
"""
import base64
import json
import sys

from playwright.sync_api import Error as PWError, TimeoutError as PWTimeout, sync_playwright

# ancho, alto de la ventana y si se comporta como teléfono/tablet (user agent y táctil)
SIZES = {
    "movil": ("Móvil", 390, 844, True),
    "tablet": ("Tablet", 820, 1180, True),
    "escritorio": ("Escritorio", 1440, 900, False),
}
MOBILE_UA = ("Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15 "
             "(KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1")
TABLET_UA = ("Mozilla/5.0 (iPad; CPU OS 17_0 like Mac OS X) AppleWebKit/605.1.15 "
             "(KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1")
MAX_HEIGHT = 4000  # el análisis acepta hasta 4096 px por lado: páginas muy largas se recortan


def _browser(p):
    errors = []
    for channel in ("msedge", "chrome", None):
        try:
            return p.chromium.launch(channel=channel) if channel else p.chromium.launch()
        except PWError as e:
            errors.append(f"{channel or 'chromium'}: {str(e).splitlines()[0]}")
    raise RuntimeError("No se encontró un navegador para tomar la captura. Instala Edge o Chrome, "
                       "o ejecuta: python -m playwright install chromium")


def main(args: dict) -> dict:
    url, sizes = args["url"], args["sizes"]
    shots = []
    with sync_playwright() as p:
        browser = _browser(p)
        try:
            for key in sizes:
                label, w, h, mobile = SIZES[key]
                ctx = browser.new_context(
                    viewport={"width": w, "height": h}, device_scale_factor=1, is_mobile=mobile, has_touch=mobile,
                    user_agent=(MOBILE_UA if key == "movil" else TABLET_UA) if mobile else None, locale="es-MX",
                )
                page = ctx.new_page()
                try:
                    try:
                        page.goto(url, wait_until="networkidle", timeout=30000)
                    except PWTimeout:
                        # sitios con conexiones abiertas (chat, analytics) nunca quedan "quietos": basta con que cargue
                        page.goto(url, wait_until="load", timeout=30000)
                    # baja hasta el final para que carguen las imágenes "lazy" y vuelve arriba
                    page.evaluate("""async () => {
                        for (let y = 0; y < Math.min(document.body.scrollHeight, 4000); y += 600) {
                            window.scrollTo(0, y); await new Promise(r => setTimeout(r, 120));
                        }
                        window.scrollTo(0, 0);
                    }""")
                    page.wait_for_timeout(800)
                    full = page.evaluate("Math.max(document.body.scrollHeight, document.documentElement.scrollHeight)")
                    height = max(h, min(int(full), MAX_HEIGHT))
                    png = page.screenshot(full_page=True, clip={"x": 0, "y": 0, "width": w, "height": height})
                finally:
                    ctx.close()
                shots.append({"size": key, "label": label, "width": w, "height": height, "truncated": full > MAX_HEIGHT,
                              "png_b64": base64.b64encode(png).decode()})
        finally:
            browser.close()
    return {"shots": shots}


if __name__ == "__main__":
    try:
        out = main(json.loads(sys.argv[1]))
    except PWTimeout:
        out = {"error": "La página tardó demasiado en cargar (más de 30 s)."}
    except PWError as e:
        msg = str(e).splitlines()[0]
        if "ERR_NAME_NOT_RESOLVED" in msg:
            msg = "No se encontró ese sitio. Revisa que la URL esté bien escrita."
        elif "ERR_CONNECTION_REFUSED" in msg:
            msg = "El sitio rechazó la conexión. ¿Está encendido el servidor?"
        out = {"error": msg}
    except RuntimeError as e:
        out = {"error": str(e)}
    sys.stdout.write(json.dumps(out))
