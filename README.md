# QA Tester visual (DINOv2)

Sube una captura de tu página y el analizador te señala micro-detalles:
elementos desalineados 1-4px, espaciados inconsistentes, tonos casi duplicados,
contraste insuficiente (WCAG), qué zonas llaman la atención (DINOv2) y, comparando
contra el diseño, qué se movió exactamente y cuántos píxeles.

## Arrancar

```bash
# Backend (terminal 1)
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000

# Frontend (terminal 2)
cd frontend
npm install
npm run dev        # http://localhost:5173
```

La primera vez se descarga el modelo `facebook/dinov2-with-registers-small` (~90 MB).

Capturas de prueba con errores plantados: `python backend/tests/make_samples.py`
(quedan en `backend/tests/samples/`).

## Qué hace cada parte

| Revisión | Cómo |
|---|---|
| Alineación / centrado / espaciado | OpenCV: detecta bloques y líneas de texto y busca bordes que *casi* coinciden |
| Colores casi duplicados, choques, exceso de acentos | Colores planos exactos + ΔE en espacio Lab |
| Contraste | Ratio WCAG 2.x entre texto y fondo |
| Atención visual (qué destaca, CTA, primer pantallazo) | DINOv2: atención del token CLS + parches distintivos |
| Diseño vs implementación | Diferencia de píxeles + correlación de fase (desplazamiento exacto) + similitud por parche de DINOv2 |

Para medidas exactas usa PNG, mismo ancho de viewport y mismo zoom en ambas capturas.
