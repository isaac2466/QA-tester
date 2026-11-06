"""Revisión con un modelo de visión + lenguaje local (Ollama).

Las reglas (OpenCV) miden con precisión de píxel pero no entienden qué están viendo:
confunden una imagen 3D con texto o un degradado con colores mal elegidos. El modelo
de visión sí entiende la página, así que:

  1. Verifica cada candidato de las reglas (¿es un error real o un falso positivo?).
  2. Añade su propia revisión de diseño: jerarquía, CTA, colores, legibilidad, consistencia.
"""
import base64
import json
import urllib.error
import urllib.request

import cv2
import numpy as np

from app.core.config import OLLAMA_MODEL, OLLAMA_TIMEOUT, OLLAMA_URL, VLM_MAX_SIDE
from app.services.findings import Finding

CATEGORIES = ["alineacion", "espaciado", "color", "contraste", "atencion", "tipografia", "jerarquia", "consistencia"]

# Lo que la IA SÍ puede ver bien: qué tipo de cosa hay dentro de un recuadro.
# Lo que NO puede ver: 1px de diferencia o dos grises casi iguales (eso lo miden las reglas,
# que son exactas). Por eso la IA clasifica y una regla fija decide.
ELEMENT_TYPES = ["texto", "boton", "tarjeta_o_panel", "icono", "indicador_o_decoracion",
                 "imagen_o_ilustracion", "degradado_o_fondo", "marca_de_agua_o_sistema"]
NOT_PAGE_UI = {"imagen_o_ilustracion", "degradado_o_fondo", "marca_de_agua_o_sistema"}
HAS_TEXT = {"texto", "boton"}

SCHEMA = {
    "type": "object",
    "properties": {
        "recuadros": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "contenido": {"type": "string", "enum": ELEMENT_TYPES},
                    "razon": {"type": "string"},
                },
                "required": ["id", "contenido", "razon"],
            },
        },
        "hallazgos": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "categoria": {"type": "string", "enum": CATEGORIES},
                    "severidad": {"type": "string", "enum": ["alta", "media", "baja"]},
                    "titulo": {"type": "string"},
                    "detalle": {"type": "string"},
                    "sugerencia": {"type": "string"},
                    "caja": {"type": "array", "items": {"type": "integer"}},
                },
                "required": ["categoria", "severidad", "titulo", "detalle", "sugerencia"],
            },
        },
        "resumen": {"type": "string"},
    },
    "required": ["recuadros", "hallazgos", "resumen"],
}

PROMPT = """Eres un tester de QA visual y diseñador UI/UX senior. Revisas la captura de pantalla de una página web.
Responde SIEMPRE en español y SOLO con el JSON pedido.

TAREA 1 – "recuadros": en la imagen hay recuadros de colores con un número. Para CADA número de esta lista
di qué hay DENTRO de ese recuadro ("contenido"):
- texto: letras de la página (titulares, párrafos, enlaces del menú)
- boton: un botón de la página (con o sin texto)
- tarjeta_o_panel: una tarjeta, caja, input o sección de la interfaz
- icono: un icono pequeño
- indicador_o_decoracion: puntos de carrusel, separadores, líneas decorativas
- imagen_o_ilustracion: parte de una foto, render 3D o ilustración
- degradado_o_fondo: sólo fondo o un degradado, sin contenido
- marca_de_agua_o_sistema: cosas que NO son de la página, como "Activar Windows" o la barra del navegador
"razon": máximo 10 palabras describiendo lo que ves en el recuadro.

Recuadros:
{candidatos}

TAREA 2 – "hallazgos": escribe entre 1 y 3 mejoras de diseño concretas que veas en la página.
IMPORTANTE: los recuadros de colores con números los dibujamos NOSOTROS encima de la captura;
NO son parte del diseño. Ignóralos por completo en esta tarea (no opines de su color ni de su posición).
Temas a revisar:
jerarquía visual (¿qué se ve primero? ¿el mensaje principal destaca?), botón principal (CTA) poco visible,
legibilidad, colores que no combinan o no siguen la marca, inconsistencias entre elementos parecidos,
tipografía, espacios amontonados o vacíos.
- Sé concreto: di QUÉ elemento (por su texto o posición) y QUÉ cambiar exactamente.
- Sólo lo que veas claramente y estés seguro. Mejor 1 observación correcta que 3 dudosas. No reportes nada del sistema operativo o del navegador.
- "caja" (opcional): [x1, y1, x2, y2] del elemento en coordenadas de 0 a 1000 relativas a la imagen.

"resumen": 1-2 frases con tu opinión general del diseño (qué funciona y qué es lo más urgente)."""


def _encode(rgb: np.ndarray) -> str:
    h, w = rgb.shape[:2]
    scale = min(1.0, VLM_MAX_SIDE / max(h, w))
    if scale < 1:
        rgb = cv2.resize(rgb, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    return base64.b64encode(buf).decode()


def _post(path: str, payload: dict | None = None, timeout: float = 5):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(OLLAMA_URL + path, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def status() -> dict:
    try:
        tags = _post("/api/tags", timeout=3)
    except (urllib.error.URLError, OSError):
        return {"available": False, "model": OLLAMA_MODEL,
                "error": "Ollama no está corriendo (ábrelo o ejecuta 'ollama serve')."}
    names = {m.get("name") for m in tags.get("models", [])} | {m.get("model") for m in tags.get("models", [])}
    if OLLAMA_MODEL not in names and f"{OLLAMA_MODEL}:latest" not in names:
        return {"available": False, "model": OLLAMA_MODEL,
                "error": f"Falta el modelo. Ejecuta: ollama pull {OLLAMA_MODEL}"}
    return {"available": True, "model": OLLAMA_MODEL}


def _describe(f: dict) -> str:
    x, y, w, h = f["box"]
    return f"#{f['id']} (recuadro en x={x}, y={y}, {w}×{h}px)"


def review(annotated_rgb: np.ndarray, candidates: list[dict], size: tuple[int, int]):
    """Devuelve (ids_confirmados, razones_por_id, hallazgos_nuevos, resumen)."""
    # sólo los que tienen recuadro (la IA clasifica lo que hay dentro); el resto queda "sin verificar"
    candidates = [c for c in candidates if c.get("box")][:12]
    cand_txt = "\n".join(_describe(f) for f in candidates) or "(no hay candidatos)"
    payload = {
        "model": OLLAMA_MODEL,
        "stream": False,
        "think": False,
        "format": SCHEMA,
        "keep_alive": "30m",  # deja el modelo en memoria: el 2º análisis no espera a que cargue
        "options": {"temperature": 0, "num_ctx": 8192, "num_predict": 1200},
        "messages": [{
            "role": "user",
            "content": PROMPT.replace("{candidatos}", cand_txt),
            "images": [_encode(annotated_rgb)],
        }],
    }
    resp = _post("/api/chat", payload, timeout=OLLAMA_TIMEOUT)
    content = resp.get("message", {}).get("content", "")
    if not content.strip():
        raise RuntimeError(
            "el modelo no devolvió respuesta (si usas una versión 'thinking', cambia a una '-instruct')")
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        raise RuntimeError("el modelo devolvió una respuesta incompleta (se cortó). Intenta de nuevo.")

    by_id = {f["id"]: f for f in candidates}
    confirmed, rejected, reasons = set(), set(), {}
    for r in data.get("recuadros", []):
        i, kind = r.get("id"), r.get("contenido")
        if i not in by_id or kind not in ELEMENT_TYPES:
            continue  # si el modelo omitió o inventó un id, ese hallazgo queda "sin verificar"
        f = by_id[i]
        if kind in NOT_PAGE_UI:
            ok, why = False, {
                "imagen_o_ilustracion": "Cae sobre una imagen o ilustración, no sobre la interfaz",
                "degradado_o_fondo": "Es un degradado o fondo, no un color elegido",
                "marca_de_agua_o_sistema": "No es parte de la página (sistema o navegador)",
            }[kind]
        elif f["category"] == "contraste" and kind not in HAS_TEXT:
            ok, why = False, "La regla de contraste es para texto y aquí no hay texto"
        else:
            ok, why = True, "Medición exacta sobre un elemento real de la interfaz"
        (confirmed if ok else rejected).add(i)
        reasons[i] = f"{why}. La IA ve: {r.get('razon', '').strip()}"

    W, H = size
    new = []
    for h in data.get("hallazgos", [])[:3]:
        box = None
        c = h.get("caja")
        if isinstance(c, list) and len(c) == 4 and all(isinstance(v, (int, float)) for v in c):
            x1, y1, x2, y2 = (max(0, min(1000, v)) for v in c)
            if x2 > x1 and y2 > y1:
                box = [round(x1 * W / 1000), round(y1 * H / 1000), round((x2 - x1) * W / 1000), round((y2 - y1) * H / 1000)]
        new.append(Finding(
            category=h.get("categoria", "jerarquia"),
            severity=h.get("severidad", "media"),
            title=h.get("titulo", "").strip() or "Observación de diseño",
            detail=h.get("detalle", "").strip(),
            suggestion=h.get("sugerencia", "").strip(),
            box=box,
            source="ia",
        ))
    return confirmed, rejected, reasons, new, data.get("resumen", "").strip()
