"""QA visual con DINOv2.

Ejecutar desde la carpeta backend/:
    uvicorn app.main:app --reload --port 8000
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware

from app.api.projects import router as projects_router
from app.api.routes import router
from app.core.config import CORS_ORIGINS
from app.schema import ensure_schema
from app.services import dino, sam


@asynccontextmanager
async def lifespan(app: FastAPI):
    # La primera vez descarga el modelo (~90 MB) desde Hugging Face.
    await run_in_threadpool(dino.load)
    st = dino.status()
    print("DINOv2:", "cargado en " + st["device"] if st["loaded"] else "NO disponible -> " + str(st["error"]))
    # SAM 2.1 (Meta): siluetas exactas para la vista corregida. La primera vez descarga ~150 MB.
    await run_in_threadpool(sam.load)
    ss = sam.status()
    print("SAM 2.1:", "cargado en " + ss["device"] if ss["loaded"] else "NO disponible -> " + str(ss["error"]))
    # Proyectos e historial por tamaño: agrega a MySQL lo que falte (no borra nada)
    try:
        changes = await run_in_threadpool(ensure_schema)
        print("MySQL:", "se agregó " + ", ".join(changes) if changes else "al día")
    except Exception as e:  # sin MySQL la app sigue sirviendo el análisis
        print("MySQL: no se pudo revisar la base ->", e)
    yield


app = FastAPI(title="Lynx · QA visual con IA", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(router)
app.include_router(projects_router)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
