"""Proyectos (agrupan análisis para ver cómo evoluciona el puntaje) y captura desde una URL."""
import json

from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

from app.auth import CurrentUser
from app.db import execute, fetch_all, fetch_one
from app.services.capture import capture, normalize_url

router = APIRouter(prefix="/api")


class ProjectIn(BaseModel):
    name: str
    url: str | None = None


class CaptureIn(BaseModel):
    url: str
    sizes: list[str]


def _json(v):
    return json.loads(v) if isinstance(v, (str, bytes)) else v


def _summary(findings) -> dict:
    """Cuenta sólo lo que se muestra (lo descartado por la IA no cuenta)."""
    shown = [f for f in (_json(findings) or []) if f.get("verdict") != "descartado"]
    counts = {"alta": 0, "media": 0, "baja": 0}
    for f in shown:
        counts[f.get("severity")] = counts.get(f.get("severity"), 0) + 1
    return {"total": len(shown), **counts}


def own_project(project_id: int, user_id: int) -> dict:
    p = fetch_one("SELECT id, name, url, created_at FROM projects WHERE id = %s AND user_id = %s", (project_id, user_id))
    if not p:
        raise HTTPException(404, "Ese proyecto no existe.")
    return p


def _project_out(p: dict) -> dict:
    return {"id": p["id"], "name": p["name"], "url": p["url"], "created_at": p["created_at"].isoformat()}


@router.get("/projects")
def list_projects(user: dict = CurrentUser):
    projects = fetch_all("SELECT id, name, url, created_at FROM projects WHERE user_id = %s ORDER BY created_at DESC",
                         (user["id"],))
    # puntajes (sin las imágenes, que pesan) para la mini gráfica de cada tarjeta
    rows = fetch_all(
        "SELECT project_id, viewport, created_at, JSON_EXTRACT(result, '$.score') AS score FROM query_history "
        "WHERE user_id = %s AND project_id IS NOT NULL AND query_type = 'analyze' ORDER BY created_at",
        (user["id"],),
    )
    by_project: dict[int, list] = {}
    for r in rows:
        by_project.setdefault(r["project_id"], []).append(
            {"date": r["created_at"].isoformat(), "viewport": r["viewport"], "score": _json(r["score"])})
    return [{**_project_out(p), "points": by_project.get(p["id"], [])} for p in projects]


@router.post("/projects")
def create_project(body: ProjectIn, user: dict = CurrentUser):
    name = body.name.strip()
    if not name:
        raise HTTPException(400, "Ponle un nombre al proyecto.")
    if len(name) > 120:
        raise HTTPException(400, "El nombre es muy largo (máximo 120 caracteres).")
    url = None
    if body.url and body.url.strip():
        try:
            url = normalize_url(body.url)
        except ValueError as e:
            raise HTTPException(400, str(e))
    pid = execute("INSERT INTO projects (user_id, name, url) VALUES (%s, %s, %s)", (user["id"], name, url))
    return {**_project_out(own_project(pid, user["id"])), "points": []}


@router.delete("/projects/{project_id}")
def delete_project(project_id: int, user: dict = CurrentUser):
    own_project(project_id, user["id"])
    # sus análisis se quedan en el historial general (la llave foránea los deja sin proyecto)
    execute("DELETE FROM projects WHERE id = %s AND user_id = %s", (project_id, user["id"]))
    return {"ok": True}


@router.get("/projects/{project_id}")
def project_detail(project_id: int, user: dict = CurrentUser):
    p = own_project(project_id, user["id"])
    rows = fetch_all(
        "SELECT id, query_type, viewport, source_url, file_names, created_at, "
        "JSON_EXTRACT(result, '$.score') AS score, JSON_EXTRACT(result, '$.findings') AS findings "
        "FROM query_history WHERE user_id = %s AND project_id = %s ORDER BY created_at DESC",
        (user["id"], project_id),
    )
    versions = [{
        "id": r["id"], "date": r["created_at"].isoformat(), "mode": r["query_type"], "viewport": r["viewport"],
        "source_url": r["source_url"], "names": _json(r["file_names"]), "score": _json(r["score"]),
        "counts": _summary(r["findings"]),
    } for r in rows]
    return {**_project_out(p), "versions": versions}


@router.get("/history/{history_id}")
def history_entry(history_id: int, user: dict = CurrentUser):
    r = fetch_one(
        "SELECT id, project_id, query_type, viewport, source_url, file_names, result, created_at FROM query_history "
        "WHERE id = %s AND user_id = %s", (history_id, user["id"]))
    if not r:
        raise HTTPException(404, "Ese análisis ya no existe.")
    return {"id": r["id"], "date": r["created_at"].isoformat(), "mode": r["query_type"], "project_id": r["project_id"],
            "viewport": r["viewport"], "source_url": r["source_url"], "names": _json(r["file_names"]),
            "files": {}, "result": _json(r["result"])}


@router.post("/capture")
async def capture_url(body: CaptureIn, user: dict = CurrentUser):
    try:
        url = normalize_url(body.url)
    except ValueError as e:
        raise HTTPException(400, str(e))
    try:
        shots = await run_in_threadpool(capture, url, body.sizes)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except RuntimeError as e:
        raise HTTPException(502, str(e))
    return {"url": url, "shots": shots}
