import asyncio
import io
import json
import threading
import traceback
import time
import re
from datetime import datetime, timezone

import numpy as np
from fastapi import APIRouter, Body, File, Form, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel
from PIL import Image, UnidentifiedImageError

from app.ai import feedback, verifier as verifier_mod
from app.api.projects import own_project
from app.auth import CurrentUser, create_session, hash_password, verify_password
from app.ai.detector import status as detector_status, trained_elements
from app.core.config import MAX_IMAGE_SIDE
from app.db import execute, fetch_all, fetch_one, is_database_error
from app.services import dino, sam, vlm
from app.services.attention import check_attention
from app.services.colors import check_colors, check_contrast
from app.services.compare import compare
from app.services.components import check_button_styles, check_components
from app.services.findings import score
from app.services.fixes import apply_fixes
from app.services.layout import check_alignment, check_centering, check_spacing, detect_elements
from app.services.render import annotate, components_overlay, heatmap_overlay, to_data_url

router = APIRouter(prefix="/api")

ORDER = {"alta": 0, "media": 1, "baja": 2}


async def _read_image(file: UploadFile) -> Image.Image:
    data = await file.read()
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except (UnidentifiedImageError, OSError):
        raise HTTPException(400, f"'{file.filename}' no es una imagen válida.")
    img = img.convert("RGB")
    if max(img.size) > MAX_IMAGE_SIDE:
        raise HTTPException(413, f"La imagen mide {img.size[0]}×{img.size[1]}; el máximo es {MAX_IMAGE_SIDE}px por lado.")
    return img


_verifier = None

REASONS = {
    "imagen": "Cae sobre una foto o ilustración, no sobre la interfaz",
    "degradado": "Es un degradado o fondo, no un elemento de la interfaz",
    "sistema": "No es parte de la página (sistema o navegador)",
}


class Credentials(BaseModel):
    email: str
    password: str


class HistoryEntry(BaseModel):
    mode: str
    names: list[str]
    result: dict
    project_id: int | None = None
    viewport: str | None = None  # "movil" | "tablet" | "escritorio" (si vino de una URL)
    source_url: str | None = None


def _verify(findings, rgb):
    """DINOv2 revisa cada hallazgo: ¿es interfaz real? ¿se parece a algo que marcaste como 'No es error'?"""
    global _verifier
    model = dino.get()
    if model is None:
        return
    if _verifier is None:
        _verifier = verifier_mod.load(model)
    boxed = [f for f in findings if f.box][:40]
    if not boxed:
        return
    emb = model.embed_boxes(rgb, [f.box for f in boxed])
    kinds = _verifier.classify_emb(emb) if _verifier else [(None, 0.0)] * len(boxed)
    for f, e, (kind, p) in zip(boxed, emb, kinds):
        f._emb = e
        if kind in REASONS:
            f.verdict, f.verdict_reason = "descartado", f"{REASONS[kind]} (DINOv2, {p:.0%})."
        elif kind and f.category == "contraste" and kind != "texto":
            f.verdict, f.verdict_reason = "descartado", f"La revisión de contraste es para texto y DINOv2 ve un {kind} ({p:.0%})."
        elif kind:
            f.verdict, f.verdict_reason = "confirmado", f"DINOv2 confirma que es un {kind} real de la interfaz ({p:.0%})."
        p_err = feedback.net_predict(e, f.category)
        if p_err is not None and p_err < 0.1 and f.verdict != "descartado":
            f.verdict, f.verdict_reason = "descartado", f"La red neuronal entrenada con tus marcas cree que no es error ({1 - p_err:.0%})."
        m = feedback.match(e, f.category)
        if m and m[1] == "no_es_error":
            f.verdict, f.verdict_reason = "descartado", f"Se parece {m[0]:.0%} a un hallazgo que marcaste como «No es error»."


def _strict(findings, group_of):
    """Modo estricto: sólo queda lo que tiene respaldo de DINOv2.

    - Hallazgos de DINOv2 (consistencia, atención) → se quedan.
    - Contraste → se queda (es la fórmula WCAG exacta y DINOv2 ya confirmó que es texto).
    - Alineación → sólo si los dos elementos son del MISMO grupo de DINOv2 (botón con botón).
    - Lo demás (espaciado, tonos casi iguales, casi centrado) → descartado, visible en "ver por qué".
    """
    for f in findings:
        if f.category == "atencion" and f.verdict != "descartado":
            f.verdict = "descartado"
            f.verdict_reason = "Es una guía para landing pages, no un error. Consúltala en la vista «Atención (IA)»."
            continue
        if f.verdict == "descartado" or f.source == "dinov2" or f.category in ("contraste", "sistema")                 or f.title.startswith("Demasiados colores"):
            continue
        if f.category == "alineacion" and f.box and f.ref_box:
            ga, gb = group_of.get(tuple(f.box)), group_of.get(tuple(f.ref_box))
            if ga is not None and ga == gb:
                (x, y, w, h), (rx, ry, rw, rh) = f.box, f.ref_box
                vertical = "superior" in f.title or "inferior" in f.title
                d_center = abs((y + h / 2) - (ry + rh / 2)) if vertical else abs((x + w / 2) - (rx + rw / 2))
                d_size = abs(h - rh) if vertical else abs(w - rw)
                if d_size >= 2 and d_center <= 1.5:  # distinto tamaño (p. ej. el botón activo) pero centrados: está bien
                    f.verdict = "descartado"
                    f.verdict_reason = "Son del mismo componente con distinto tamaño, pero están centrados entre sí: no es error."
                continue
        f.verdict = "descartado"
        f.verdict_reason = "Modo estricto: regla clásica sin respaldo de DINOv2 (no compara elementos del mismo componente)."


def _pack(findings, rgb):
    findings.sort(key=lambda f: (f.verdict == "descartado", ORDER[f.severity]))
    dicts = [f.to_dict(i + 1) for i, f in enumerate(findings)]
    aid = feedback.remember_analysis(
        {d["id"]: (f._emb, f.category, f.title) for d, f in zip(dicts, findings) if hasattr(f, "_emb")})
    return dicts, to_data_url(annotate(rgb, [d for d in dicts if d["verdict"] != "descartado"])), aid


@router.post("/feedback")
def save_feedback(analysis_id: str = Body(...), finding_id: int = Body(...), label: str = Body(...)):
    if label not in ("no_es_error", "es_error"):
        raise HTTPException(400, "label debe ser 'no_es_error' o 'es_error'")
    if not feedback.save(analysis_id, finding_id, label):
        raise HTTPException(404, "Ese análisis ya no está en memoria; vuelve a analizar la captura.")
    return {"ok": True, **feedback.stats()}


def _email(value: str) -> str:
    email = value.strip().lower()
    if not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", email):
        raise HTTPException(400, "Escribe un correo electrónico válido.")
    return email


@router.post("/auth/register")
def register(credentials: Credentials):
    email = _email(credentials.email)
    if len(credentials.password) < 8:
        raise HTTPException(400, "La contraseña debe tener al menos 8 caracteres.")
    try:
        user_id = execute(
            "INSERT INTO users (email, password_hash) VALUES (%s, %s)",
            (email, hash_password(credentials.password)),
        )
    except Exception as error:
        if is_database_error(error) and getattr(error, "errno", None) == 1062:
            raise HTTPException(409, "Ya existe una cuenta con ese correo.")
        raise HTTPException(503, "No se pudo guardar el usuario. Verifica la conexión con MySQL.")
    return {"token": create_session(user_id), "user": {"id": user_id, "email": email}}


@router.post("/auth/login")
def login(credentials: Credentials):
    email = _email(credentials.email)
    try:
        user = fetch_one("SELECT id, email, password_hash FROM users WHERE email = %s", (email,))
    except Exception:
        raise HTTPException(503, "No se pudo conectar con MySQL. Ejecuta primero el script SQL.")
    if not user or not verify_password(credentials.password, user["password_hash"]):
        raise HTTPException(401, "Correo o contraseña incorrectos.")
    return {"token": create_session(user["id"]), "user": {"id": user["id"], "email": user["email"]}}


@router.get("/auth/me")
def me(user: dict = CurrentUser):
    return {"user": user}


@router.post("/auth/logout")
def logout(request: Request, user: dict = CurrentUser):
    authorization = request.headers.get("Authorization", "")
    token = authorization[7:].strip()
    execute("DELETE FROM user_sessions WHERE user_id = %s AND token_hash = SHA2(%s, 256)", (user["id"], token))
    return {"ok": True}


@router.get("/history")
def history(user: dict = CurrentUser):
    rows = fetch_all(
        "SELECT id, project_id, query_type, viewport, file_names, result, created_at FROM query_history "
        "WHERE user_id = %s ORDER BY created_at DESC LIMIT 50",
        (user["id"],),
    )
    return [
        {
            "id": row["id"],
            "date": row["created_at"].isoformat(),
            "mode": row["query_type"],
            "project_id": row["project_id"],
            "viewport": row["viewport"],
            "names": json.loads(row["file_names"]) if isinstance(row["file_names"], str) else row["file_names"],
            "files": {},
            "result": json.loads(row["result"]) if isinstance(row["result"], str) else row["result"],
        }
        for row in rows
    ]


@router.post("/history")
def save_history(entry: HistoryEntry, user: dict = CurrentUser):
    if entry.mode not in ("analyze", "compare") or not entry.names or not isinstance(entry.result, dict):
        raise HTTPException(400, "El registro de historial no tiene un formato válido.")
    if entry.project_id is not None:
        own_project(entry.project_id, user["id"])
    history_id = execute(
        "INSERT INTO query_history (user_id, project_id, query_type, viewport, source_url, file_names, result) "
        "VALUES (%s, %s, %s, %s, %s, %s, %s)",
        (user["id"], entry.project_id, entry.mode, (entry.viewport or None) and entry.viewport[:20],
         (entry.source_url or None) and entry.source_url[:2048], json.dumps(entry.names), json.dumps(entry.result)),
    )
    return {
        "id": history_id,
        "date": datetime.now(timezone.utc).isoformat(),
        "mode": entry.mode,
        "project_id": entry.project_id,
        "viewport": entry.viewport,
        "names": entry.names,
        "files": {},
        "result": entry.result,
    }


@router.delete("/history")
def clear_history(user: dict = CurrentUser):
    execute("DELETE FROM query_history WHERE user_id = %s", (user["id"],))
    return {"ok": True}


@router.get("/health")
def health():
    return {"status": "ok", "dino": dino.status(), "sam": sam.status(), "qwen": vlm.status()}


@router.post("/analyze")
async def analyze(file: UploadFile = File(...), estricto: bool = True):
    img = await _read_image(file)

    def run():
        t0 = time.perf_counter()
        rgb = np.asarray(img)
        elems = detect_elements(rgb)
        # detector ENTRENADO (si existe app/ai/weights/detector.pt): suma lo que el clásico no encontró
        learned = trained_elements(dino.get(), rgb)
        if learned:
            # sólo rellena huecos: si ya hay un elemento ahí (aunque la caja difiera unos px), manda el clásico
            from app.services.layout import _iou
            learned = [e for e in learned if all(_iou(e, o) < 0.3 for o in elems)]
            elems = elems + learned
        findings = []
        findings += check_alignment(elems)
        findings += check_centering(elems, rgb.shape[1])
        findings += check_spacing(elems)
        color_findings, pal = check_colors(rgb)
        findings += color_findings
        findings += check_contrast(rgb, elems)
        findings += check_button_styles(rgb, elems)

        attention, heat, groups, comps = None, None, [], None
        group_of: dict = {}
        model = dino.get()
        if model:
            sal = model.saliency(img)
            att_findings, attention = check_attention(rgb, sal, elems)
            findings += att_findings
            heat = to_data_url(heatmap_overlay(rgb, sal))
            comp_findings, groups = check_components(rgb, elems, model)
            # si un elemento es más alto/ancho que sus gemelos, su "desalineado" es consecuencia: lo quitamos
            resized = {tuple(f.box) for f in comp_findings if "alto" in f.title or "ancho" in f.title}
            findings = [f for f in findings if not (f.category == "alineacion" and f.box and tuple(f.box) in resized)]
            # DINOv2 decide qué se compara: un botón no tiene por qué alinearse con una tarjeta
            group_of.update({tuple(b): gi for gi, g in enumerate(groups) for b in g["boxes"]})
            def same_kind(f):
                if f.category != "alineacion" or not (f.box and f.ref_box):
                    return True
                ga, gb = group_of.get(tuple(f.box)), group_of.get(tuple(f.ref_box))
                return ga is None or gb is None or ga == gb
            findings = [f for f in findings if same_kind(f)]
            findings += comp_findings
            if groups:
                comps = to_data_url(components_overlay(rgb, groups))

        _verify(findings, rgb)
        if estricto:
            _strict(findings, group_of)
        dicts, annotated, aid = _pack(findings, rgb)
        fixed, applied = apply_fixes(rgb, [d for d in dicts if d["verdict"] != "descartado"], sam.get())
        return {
            "filename": file.filename,
            "size": list(img.size),
            "score": score(findings),
            "findings": dicts,
            "palette": pal,
            "attention": attention,
            "elements_detected": len(elems),
            "components": [{"label": g["label"], "count": g["count"]} for g in groups],
            "images": {"annotated": annotated, "heatmap": heat, "components": comps,
                       "fixed": to_data_url(fixed) if applied else None},
            "fixes_applied": applied,
            "analysis_id": aid,
            "learning": feedback.stats(),
            "detector": {**detector_status(), "elementos_del_modelo": len(learned)},
            "sam": sam.status(),
            "dino": dino.status(),
            "elapsed_ms": round((time.perf_counter() - t0) * 1000),
        }

    return await run_in_threadpool(run)


@router.post("/compare")
async def compare_images(design: UploadFile = File(...), implementation: UploadFile = File(...)):
    a = await _read_image(design)
    b = await _read_image(implementation)

    def run():
        t0 = time.perf_counter()
        da = np.asarray(a)
        findings, impl_rgb, stats, sem = compare(da, np.asarray(b), dino.get())
        dicts, annotated, _ = _pack(findings, impl_rgb)
        return {
            "size": list(a.size),
            "score": score(findings),
            "findings": dicts,
            "stats": stats,
            "images": {
                "annotated": annotated,
                "implementation": to_data_url(impl_rgb),  # limpia (ya reescalada al tamaño del diseño)
                "design": to_data_url(da),
                "semantic": to_data_url(heatmap_overlay(impl_rgb, sem, 0.45)) if sem is not None else None,
            },
            "dino": dino.status(),
            "elapsed_ms": round((time.perf_counter() - t0) * 1000),
        }

    return await run_in_threadpool(run)


@router.post("/explain")
async def explain(request: Request, file: UploadFile = File(...), findings: str = Form(...)):
    """Qwen3-VL explica cada hallazgo con más detalle y dice si cree que es un falso positivo.

    Va aparte de /analyze porque en CPU tarda: el frontend muestra primero lo de DINOv2 y
    luego manda los hallazgos a Qwen en lotes pequeños.
    """
    img = await _read_image(file)
    try:
        items = json.loads(findings)
        assert isinstance(items, list)
    except (json.JSONDecodeError, AssertionError):
        raise HTTPException(400, "'findings' debe ser una lista JSON de hallazgos.")
    st = vlm.status()
    if not st["available"]:
        raise HTTPException(503, st["error"])

    cancel = threading.Event()

    def run():
        t0 = time.perf_counter()
        try:
            out = vlm.explain(np.array(img), items[:3], cancel)  # np.array copia: np.asarray de PIL es de solo lectura
        except vlm.Cancelled:
            return None
        except RuntimeError as e:
            print("Qwen:", e)
            raise HTTPException(502, str(e))
        except OSError as e:  # timeout o Ollama se cerró a medio camino
            print("Qwen sin respuesta:", e)
            raise HTTPException(504, f"Qwen no respondió a tiempo ({e}).")
        except Exception as e:  # cualquier otro fallo: que se vea el motivo en el panel, no un "Error 500" mudo
            traceback.print_exc()
            raise HTTPException(500, f"Fallo al explicar ({type(e).__name__}: {e}).")
        return {"explanations": out, "model": st["model"],
                "elapsed_ms": round((time.perf_counter() - t0) * 1000)}

    # Si el navegador cancela (análisis nuevo o pestaña cerrada) cortamos a Qwen para liberarlo.
    task = asyncio.ensure_future(run_in_threadpool(run))
    while not task.done():
        await asyncio.wait({task}, timeout=1)
        if not task.done() and await request.is_disconnected():
            cancel.set()
    result = task.result()
    if result is None:
        raise HTTPException(499, "Cancelado")
    return result
