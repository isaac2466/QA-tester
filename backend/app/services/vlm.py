"""Qwen3-VL (local, vía Ollama) explica los hallazgos y opina si son falsos positivos.

DINOv2 y las reglas miden con precisión de píxel, pero sólo dejan un texto corto y no
entienden qué están viendo (pueden confundir una ilustración con un botón). Qwen sí
entiende la imagen, así que para cada hallazgo:

  1. Lo explica con más detalle: qué pasa, por qué importa y cómo arreglarlo.
  2. Da un veredicto: "real", "falso_positivo" o "dudoso".

Para que vea los detalles de pocos píxeles le mandamos un acercamiento de cada hallazgo.
En CPU cada imagen le cuesta ~75 s a Qwen (Ollama la reescala a ~1000 tokens, sin importar
su tamaño), así que juntamos los acercamientos de un lote en UNA sola imagen tipo mosaico.
"""
import base64
import json
import threading
import urllib.error
import urllib.request

import cv2
import numpy as np

from app.core.config import OLLAMA_MODEL, OLLAMA_TIMEOUT, OLLAMA_URL, VLM_MAX_SIDE
from app.services.render import annotate

VERDICTS = ["real", "falso_positivo", "dudoso"]

# Un modelo de 4B no es bueno decidiendo "¿es error?" en abstracto, pero sí diciendo QUÉ ve.
# Por eso Qwen clasifica el contenido del recuadro y una regla fija decide los casos claros.
ELEMENT_TYPES = ["texto", "boton", "tarjeta_o_panel", "campo_de_formulario", "icono",
                 "indicador_o_decoracion", "imagen_o_ilustracion", "degradado_o_fondo",
                 "marca_de_agua_o_sistema"]
NOT_PAGE_UI = {
    "imagen_o_ilustracion": "Cae sobre una imagen o ilustración, no sobre la interfaz",
    "degradado_o_fondo": "Es un fondo o degradado, no un elemento de la interfaz",
    "marca_de_agua_o_sistema": "No es parte de la página (marca de agua, sistema o navegador)",
}
HAS_TEXT = {"texto", "boton", "campo_de_formulario"}
FINDING_COLORS = {"alta": (220, 38, 38), "media": (217, 119, 6), "baja": (37, 99, 235)}  # RGB
TILE = 512  # lado de cada acercamiento dentro del mosaico (2 columnas → 1024 px de ancho)

SCHEMA = {
    "type": "object",
    "properties": {
        "hallazgos": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "contenido": {"type": "string", "enum": ELEMENT_TYPES},
                    # maxLength: si Qwen entra en bucle repitiendo texto, el campo se cierra solo
                    "texto_visible": {"type": "string", "maxLength": 80},
                    "veredicto": {"type": "string", "enum": VERDICTS},
                    "razon": {"type": "string", "maxLength": 160},
                    "que_pasa": {"type": "string", "maxLength": 400},
                    "por_que_importa": {"type": "string", "maxLength": 320},
                    "como_arreglarlo": {"type": "string", "maxLength": 320},
                },
                "required": ["id", "contenido", "texto_visible", "veredicto", "razon",
                             "que_pasa", "por_que_importa", "como_arreglarlo"],
            },
        },
    },
    "required": ["hallazgos"],
}

PROMPT = """Eres un tester de QA visual y diseñador UI/UX senior. Una herramienta automática revisó
la captura de una página web y reportó los hallazgos de abajo. Tu trabajo es revisarlos uno por uno.
Responde SIEMPRE en español y SOLO con el JSON pedido.

IMAGEN: {imagen}

HALLAZGOS:
{hallazgos}

Para CADA hallazgo, mira su cuadro y devuelve EN ESTE ORDEN:
- "id": el número del hallazgo.
- "contenido": qué hay DENTRO del recuadro de color:
    texto (letras de la página), boton, tarjeta_o_panel, campo_de_formulario, icono,
    indicador_o_decoracion (puntos o barra de carrusel, separadores, líneas decorativas),
    imagen_o_ilustracion (foto, render 3D, ilustración), degradado_o_fondo (sólo fondo),
    marca_de_agua_o_sistema (cosas que NO son de la página, como "Activar Windows" o la barra del navegador).
- "texto_visible": el texto que alcanzas a leer dentro del recuadro, tal cual ("" si no hay texto).
- "veredicto":
    "real" si en la imagen se ve que el problema existe y afecta a la interfaz;
    "falso_positivo" si la herramienta se equivocó: el recuadro cae sobre una foto, ilustración,
      degradado, fondo, marca de agua o algo del sistema/navegador; o compara elementos que NO son
      del mismo tipo (p. ej. un botón con una tarjeta); o la diferencia es intencional del diseño
      (p. ej. el botón principal es más grande a propósito);
    "dudoso" si no se alcanza a ver con claridad.
- "razon": máximo 12 palabras: qué ves en el recuadro y por qué das ese veredicto.
Luego explícalo para alguien que NO es diseñador, en tres partes de 1 o 2 frases cada una:
- "que_pasa": describe el problema con tus palabras: QUÉ elemento (por su texto, tipo y posición:
  "el botón «Ver demo» de la parte de arriba") y QUÉ tiene mal, usando las medidas del hallazgo
  (px, colores, contraste). Si compara dos elementos, di cuáles son.
- "por_que_importa": la consecuencia concreta para quien visita la página. Ejemplos de consecuencias:
  se ve descuidado o poco profesional, cuesta leerlo, no se nota que es clicable, rompe el ritmo
  visual, distrae la mirada, da desconfianza, falla accesibilidad (WCAG) para personas con baja visión.
- "como_arreglarlo": el cambio exacto (cuántos px mover, qué tamaño o color usar) y dónde hacerlo
  (CSS, Figma o el componente).
Cada hallazgo es distinto: no copies la misma frase en varios. Si es falso positivo, en "que_pasa"
di qué es en realidad lo que hay en el recuadro y en "por_que_importa" por qué se puede ignorar.

Las medidas (px, colores, contraste) son exactas: no las cuestiones, sólo decide si el elemento medido
es el correcto y si el problema importa."""

IMG_MOSAIC = """un mosaico con un acercamiento de cada hallazgo. Arriba a la izquierda de cada cuadro hay
una etiqueta negra con el número del hallazgo (#N). Dentro del cuadro, el recuadro de color marca el
elemento reportado y el recuadro gris delgado (si hay) el elemento con el que se comparó. Esos recuadros los
dibujamos NOSOTROS; NO son parte del diseño."""

IMG_PAGE = """la página completa. Los recuadros de colores con número los dibujamos NOSOTROS para
señalar hallazgos; NO son parte del diseño."""


def _encode(rgb: np.ndarray, max_side: int) -> str:
    h, w = rgb.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale < 1:
        rgb = cv2.resize(rgb, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    return base64.b64encode(buf).decode()


def _crop(rgb: np.ndarray, f: dict) -> np.ndarray:
    """Acercamiento del hallazgo (y de su referencia, si la tiene) con contexto alrededor.

    Dibuja sólo el recuadro de ESTE hallazgo, por fuera del elemento y sin número, para no
    tapar lo que Qwen tiene que leer.
    """
    H, W = rgb.shape[:2]
    boxes = [b for b in (f.get("box"), f.get("ref_box")) if b]
    x1 = min(b[0] for b in boxes)
    y1 = min(b[1] for b in boxes)
    x2 = max(b[0] + b[2] for b in boxes)
    y2 = max(b[1] + b[3] for b in boxes)
    m = max(32, round(0.4 * max(x2 - x1, y2 - y1)))
    x1, y1, x2, y2 = max(0, x1 - m), max(0, y1 - m), min(W, x2 + m), min(H, y2 + m)
    crop = np.ascontiguousarray(rgb[y1:y2, x1:x2])
    t = max(2, round(max(x2 - x1, y2 - y1) / 150))  # grosor que se siga viendo tras escalar
    if f.get("ref_box"):
        rx, ry, rw, rh = f["ref_box"]
        cv2.rectangle(crop, (rx - x1 - 3, ry - y1 - 3), (rx - x1 + rw + 2, ry - y1 + rh + 2),
                      (150, 150, 150), max(1, t // 2))
    bx, by, bw, bh = f["box"]
    color = FINDING_COLORS.get(f.get("severity"), FINDING_COLORS["baja"])
    cv2.rectangle(crop, (bx - x1 - t - 2, by - y1 - t - 2), (bx - x1 + bw + t + 1, by - y1 + bh + t + 1), color, t)
    return crop


def _mosaic(rgb: np.ndarray, findings: list[dict]) -> np.ndarray:
    """Junta los acercamientos en una cuadrícula de 2 columnas con el número de cada hallazgo."""
    cols = 1 if len(findings) == 1 else 2
    rows = -(-len(findings) // cols)
    sheet = np.full((rows * TILE, cols * TILE, 3), 255, np.uint8)
    for k, f in enumerate(findings):
        crop = _crop(rgb, f)
        h, w = crop.shape[:2]
        s = (TILE - 16) / max(h, w)
        crop = cv2.resize(crop, (max(1, round(w * s)), max(1, round(h * s))),
                          interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
        h, w = crop.shape[:2]
        r, c = divmod(k, cols)
        y0, x0 = r * TILE + (TILE - h) // 2, c * TILE + (TILE - w) // 2
        sheet[y0:y0 + h, x0:x0 + w] = crop
        cv2.rectangle(sheet, (c * TILE, r * TILE), ((c + 1) * TILE - 1, (r + 1) * TILE - 1), (200, 200, 200), 2)
        label = f"#{f['id']}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.1, 3)
        cv2.rectangle(sheet, (c * TILE, r * TILE), (c * TILE + tw + 16, r * TILE + th + 16), (0, 0, 0), -1)
        cv2.putText(sheet, label, (c * TILE + 8, r * TILE + th + 8), cv2.FONT_HERSHEY_SIMPLEX, 1.1,
                    (255, 255, 255), 3, cv2.LINE_AA)
    return sheet


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
    return f"#{f['id']} [{f.get('category', '')}] {f.get('title', '')}. {f.get('detail', '')}"


def _salvage(content: str) -> list[dict]:
    """Si la respuesta se cortó, recupera los objetos de "hallazgos" que sí quedaron completos."""
    start = content.find("[")
    if start < 0:
        return []
    dec, items, i = json.JSONDecoder(), [], start + 1
    while True:
        while i < len(content) and content[i] in " \t\r\n,":
            i += 1
        if i >= len(content) or content[i] != "{":
            return items
        try:
            obj, i = dec.raw_decode(content, i)
        except json.JSONDecodeError:
            return items  # el último quedó a medias: lo descartamos
        if isinstance(obj, dict):
            items.append(obj)


class Cancelled(Exception):
    """El navegador ya no espera la respuesta (se cerró o se pidió un análisis nuevo)."""


def _chat(payload: dict, cancel: threading.Event | None) -> tuple[str, str | None]:
    """Llama a Ollama en modo streaming para poder cortar la generación si se cancela.

    Cerrar la conexión hace que Ollama deje de generar, así Qwen no se queda ocupado
    con un análisis viejo mientras el nuevo espera en la fila.
    """
    req = urllib.request.Request(OLLAMA_URL + "/api/chat", data=json.dumps({**payload, "stream": True}).encode(),
                                 headers={"Content-Type": "application/json"})
    parts, done_reason = [], None
    with urllib.request.urlopen(req, timeout=OLLAMA_TIMEOUT) as r:
        for line in r:
            if cancel is not None and cancel.is_set():
                raise Cancelled()
            if not line.strip():
                continue
            chunk = json.loads(line)
            if chunk.get("error"):
                raise RuntimeError(f"Ollama: {chunk['error']}")
            parts.append(chunk.get("message", {}).get("content", ""))
            if chunk.get("done"):
                done_reason = chunk.get("done_reason")
                break
    return "".join(parts), done_reason


def explain(rgb: np.ndarray, findings: list[dict], cancel: threading.Event | None = None) -> dict[int, dict]:
    """Devuelve {id: {"veredicto", "razon", "que_pasa", "por_que_importa", "como_arreglarlo", ...}}."""
    if not findings:
        return {}
    with_box = [f for f in findings if f.get("box")]
    # si ninguno tiene recuadro (p. ej. "la paleta tiene demasiados colores"), Qwen ve la página entera
    image, how = (_mosaic(rgb, with_box), IMG_MOSAIC) if with_box else (annotate(rgb, findings), IMG_PAGE)
    prompt = PROMPT.replace("{imagen}", how).replace("{hallazgos}", "\n".join(_describe(f) for f in findings))
    payload = {
        "model": OLLAMA_MODEL,
        "think": False,
        "format": SCHEMA,
        "keep_alive": "30m",  # deja el modelo en memoria: el siguiente lote no espera a que cargue
        "options": {"temperature": 0, "repeat_penalty": 1.15, "num_ctx": 8192,
                    "num_predict": 380 * len(findings) + 150},
        "messages": [{"role": "user", "content": prompt, "images": [_encode(image, VLM_MAX_SIDE)]}],
    }
    content, done_reason = _chat(payload, cancel)
    if not content.strip():
        raise RuntimeError("Qwen no devolvió respuesta (si usas una versión 'thinking', cambia a una '-instruct').")
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        data = {"hallazgos": _salvage(content)}  # nos quedamos con los hallazgos que sí completó
        if not data["hallazgos"]:
            why = "se le acabó el límite de texto" if done_reason == "length" else "la respuesta llegó incompleta"
            raise RuntimeError(f"Qwen no terminó su respuesta ({why}).")

    by_id = {f["id"]: f for f in findings}
    out = {}
    for h in data.get("hallazgos", []):
        f = by_id.get(h.get("id"))
        if not f or h.get("veredicto") not in VERDICTS:
            continue  # si el modelo inventa un id, lo ignoramos
        kind = h.get("contenido")
        verdict, reason = h["veredicto"], (h.get("razon") or "").strip()
        what, why, how = ((h.get(k) or "").strip() for k in ("que_pasa", "por_que_importa", "como_arreglarlo"))
        # reglas fijas sobre lo que Qwen dice que VE (más fiable que su veredicto directo)
        forced = None
        if f.get("box") and kind in NOT_PAGE_UI:
            forced = NOT_PAGE_UI[kind]
        elif f.get("category") == "contraste" and kind in ELEMENT_TYPES and kind not in HAS_TEXT:
            forced = "La regla de contraste es para texto y aquí no hay texto"
        if forced:
            # la explicación de Qwen pudo asumir que era un error real: la reemplazamos para no contradecirnos
            # Si Qwen ya decía "falso positivo", su texto lo explica. Si decía "real", su "por qué importa"
            # y "cómo arreglarlo" asumen un error, así que sólo dejamos su descripción de lo que ve.
            if verdict != "falso_positivo":
                why, how = "", ""
            verdict, reason = "falso_positivo", forced
        out[f["id"]] = {
            "veredicto": verdict,
            "razon": reason,
            "que_pasa": what,
            "por_que_importa": why,
            "como_arreglarlo": how,
            "explicacion": " ".join(x for x in (what, why, how) if x),  # texto corrido (reporte, compatibilidad)
            "contenido": kind,
            "texto": (h.get("texto_visible") or "").strip(),
        }
    return out
