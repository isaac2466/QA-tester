"""QA visual con DINOv2.

Ejecutar desde la carpeta backend/:
    uvicorn app.main:app --reload --port 8000
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.core.config import CORS_ORIGINS
from app.services import dino


@asynccontextmanager
async def lifespan(app: FastAPI):
    # La primera vez descarga el modelo (~90 MB) desde Hugging Face.
    await run_in_threadpool(dino.load)
    st = dino.status()
    print("DINOv2:", "cargado en " + st["device"] if st["loaded"] else "NO disponible -> " + str(st["error"]))
    yield


app = FastAPI(title="QA Tester visual (DINOv2)", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(router)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
