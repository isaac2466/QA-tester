import io
import time

import numpy as np
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from PIL import Image, UnidentifiedImageError

from app.core.config import MAX_IMAGE_SIDE
from app.services import dino, sam
from app.services.attention import check_attention
from app.services.colors import check_colors, check_contrast
from app.services.compare import compare
from app.services.components import check_components
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


def _pack(findings, rgb):
    findings.sort(key=lambda f: ORDER[f.severity])
    dicts = [f.to_dict(i + 1) for i, f in enumerate(findings)]
    return dicts, to_data_url(annotate(rgb, dicts))


@router.get("/health")
def health():
    return {"status": "ok", "dino": dino.status(), "sam": sam.status()}


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

        attention, heat, groups, comps = None, None, [], None
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
            group_of = {tuple(b): gi for gi, g in enumerate(groups) for b in g["boxes"]}
            def same_kind(f):
                if f.category != "alineacion" or not (f.box and f.ref_box):
                    return True
                ga, gb = group_of.get(tuple(f.box)), group_of.get(tuple(f.ref_box))
                return ga is None or gb is None or ga == gb
            findings = [f for f in findings if same_kind(f)]
            findings += comp_findings
            if groups:
                comps = to_data_url(components_overlay(rgb, groups))

        dicts, annotated = _pack(findings, rgb)
        fixed, applied = apply_fixes(rgb, dicts, sam.get())
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
        dicts, annotated = _pack(findings, impl_rgb)
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
