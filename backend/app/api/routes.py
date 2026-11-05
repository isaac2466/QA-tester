import io
import time

import numpy as np
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from PIL import Image, UnidentifiedImageError

from app.core.config import MAX_IMAGE_SIDE
from app.services import dino
from app.services.attention import check_attention
from app.services.colors import check_colors, check_contrast
from app.services.compare import compare
from app.services.findings import score
from app.services.layout import check_alignment, check_centering, check_spacing, detect_elements
from app.services.render import annotate, heatmap_overlay, to_data_url

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


def _pack(findings, rgb):
    findings.sort(key=lambda f: ORDER[f.severity])
    dicts = [f.to_dict(i + 1) for i, f in enumerate(findings)]
    return dicts, to_data_url(annotate(rgb, dicts))


@router.get("/health")
def health():
    return {"status": "ok", "dino": dino.status()}


@router.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    img = await _read_image(file)

    def run():
        t0 = time.perf_counter()
        rgb = np.asarray(img)
        elems = detect_elements(rgb)
        findings = []
        findings += check_alignment(elems)
        findings += check_centering(elems, rgb.shape[1])
        findings += check_spacing(elems)
        color_findings, pal = check_colors(rgb)
        findings += color_findings
        findings += check_contrast(rgb, elems)

        attention, heat = None, None
        model = dino.get()
        if model:
            sal = model.saliency(img)
            att_findings, attention = check_attention(rgb, sal, elems)
            findings += att_findings
            heat = to_data_url(heatmap_overlay(rgb, sal))

        dicts, annotated = _pack(findings, rgb)
        return {
            "filename": file.filename,
            "size": list(img.size),
            "score": score(findings),
            "findings": dicts,
            "palette": pal,
            "attention": attention,
            "elements_detected": len(elems),
            "images": {"annotated": annotated, "heatmap": heat},
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
        dicts, annotated = _pack(findings, impl_rgb)
        return {
            "size": list(a.size),
            "score": score(findings),
            "findings": dicts,
            "stats": stats,
            "images": {
                "annotated": annotated,
                "design": to_data_url(da),
                "semantic": to_data_url(heatmap_overlay(impl_rgb, sem, 0.45)) if sem is not None else None,
            },
            "dino": dino.status(),
            "elapsed_ms": round((time.perf_counter() - t0) * 1000),
        }

    return await run_in_threadpool(run)
