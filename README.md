# QA Tester visual

Sube una captura de tu página (o tu diseño de Figma y la implementación) y la IA te dice qué está mal: elementos desalineados, componentes con otro tamaño o color, texto difícil de leer y qué zona llama la atención. Además te muestra **cómo se vería corregida**.

Usa dos modelos de IA de Meta:

- **DINOv2**: entiende la imagen, reconoce qué elementos son "el mismo componente" y empareja el diseño con la implementación.
- **SAM 2.1** (Segment Anything): recorta cada elemento con su silueta exacta para generar la vista corregida.

## 1. Instala esto (una sola vez)

- **Python 3.12, 3.13 o 3.14** → https://www.python.org/downloads/ (marca **"Add python.exe to PATH"**)
- **Node.js 22** → https://nodejs.org/
- **Git** → https://git-scm.com/downloads
- **Solo Windows:** https://aka.ms/vs/17/release/vc_redist.x64.exe (sin esto la IA no funciona)

Después de instalar, **reinicia la PC**.

## 2. Descarga el proyecto

```bash
git clone https://github.com/isaac2466/QA-tester.git
cd QA-tester
git checkout RamaNeto
```

> La versión con la interfaz nueva y SAM está en la rama `RamaNeto`. Si ya tenías el proyecto, usa `git pull` en lugar de `git clone`.

## 3. Crea el entorno virtual e instala las librerías

Desde la carpeta `QA-tester`:

**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r backend/requirements.txt
npm install --prefix frontend
```

**Mac / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r backend/requirements.txt
npm install --prefix frontend
```

- Sabrás que el entorno está activo porque aparece `(.venv)` al inicio de la línea. **Actívalo cada vez que abras una terminal nueva.**
- El primer `pip install` tarda unos minutos (baja ~300 MB: PyTorch y las librerías de IA).
- Usamos `python -m pip` en vez de `pip` a secas porque a veces el comando `pip` no queda instalado dentro del entorno.

## 4. Ábrelo

Abre **dos terminales** en la carpeta `QA-tester` y activa el entorno en las dos (`.\.venv\Scripts\Activate.ps1`).

**Terminal 1 (backend + IA):**
```bash
cd backend
uvicorn app.main:app --reload --port 8000
```

Espera a que aparezcan estas dos líneas:

```
DINOv2: cargado en cpu
SAM 2.1: cargado en cpu
```

La **primera vez** tarda unos minutos porque descarga los modelos (DINOv2 ~90 MB y SAM 2.1 ~150 MB). Después arranca en segundos.

**Terminal 2 (página web):**
```bash
cd frontend
npm run dev
```

Entra a **http://localhost:5173**. Arriba a la derecha debe decir **"Motor listo · DINOv2 + SAM 2.1 (cpu)"**.

## 5. Cómo se usa

### Revisar captura
Sube una captura (arrastrar, clic o **Ctrl+V**) y pulsa **Analizar**. Tarda ~10-20 segundos en una PC sin GPU.

| Vista | Qué muestra |
|---|---|
| **Hallazgos** | Recuadros numerados sobre cada error. Clic en un hallazgo → **Ver zona** |
| **Vista corregida** | Antes / después con slider: tu captura con las correcciones aplicadas |
| **Atención (IA)** | Mapa de calor: qué zona llama la atención según DINOv2 |
| **Componentes (IA)** | Cada color es un grupo de elementos que DINOv2 considera el mismo componente |

Cada hallazgo indica si lo encontró la IA (etiqueta **DINOv2 · %**) y si su corrección está en la vista corregida (etiqueta verde).

### Comparar diseños
Sube el **diseño** (exporta el frame de Figma como PNG) y la **implementación** (captura de tu página al mismo ancho y con zoom 100%). DINOv2 empareja cada elemento y te dice qué se movió, qué cambió de tamaño o color, **qué falta** y **qué sobra**. Vistas: lado a lado, superponer (slider) y diferencias.

### Otras opciones
- **No es error**: quita un hallazgo de la lista y del puntaje.
- **Historial**: tus últimos análisis.
- **Exportar reporte**: descarga un HTML con la captura anotada y los hallazgos.

## 6. Entrenar la IA (opcional)

En `backend/app/ai/` está la parte que **entrenamos nosotros** (todavía no está conectada a la app):

| Archivo | Qué hace |
|---|---|
| `dataset.py` | Genera pantallas falsas con errores plantados a propósito (los datos de entrenamiento) |
| `detector.py` + `train_detector.py` | Red que aprende a **encontrar** los elementos, entrenada sobre DINOv2 |
| `judge.py` | Red que aprenderá a **decidir** si un elemento tiene un error (falta su script de entrenamiento) |
| `segment.py` | SAM en modo "encontrar todo" (en CPU tarda 3-4 min por captura; pensado para GPU) |

Para entrenar el detector (con el entorno activo):

```bash
cd backend
python -m app.ai.train_detector --train 60 --val 10 --epochs 20
```

| Tamaño | Comando | Tiempo aprox. (CPU) |
|---|---|---|
| Prueba rápida | `--train 60 --val 10 --epochs 20` | 10-15 min |
| Mediano | `--train 120 --val 20 --epochs 30` | ~25 min |
| Completo | `--train 300 --val 40 --epochs 40` | ~1 hora |

- Las pantallas generadas se guardan en `backend/ai_data/` (no se suben a git). Si lo cortas, **retoma donde se quedó**.
- **Pérdida**: qué tanto se equivoca; debe ir bajando. **Recall**: de los elementos que había, cuántos encontró. **Precisión**: de lo que encontró, cuánto era correcto.
- El modelo se guarda en `backend/app/ai/weights/detector.pt` y sus métricas en `detector_metrics.json`.

## ¿No funciona?

| Error | Solución |
|---|---|
| `WinError 126` o `shm.dll` | Instala el link de **Solo Windows** del paso 1 y reinicia |
| `python` o `uvicorn` no se reconoce | Reinstala Python marcando **"Add python.exe to PATH"**, o activa el entorno (`.\.venv\Scripts\Activate.ps1`) |
| "La ejecución de scripts está deshabilitada" al activar el entorno | Corre una vez `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` y vuelve a activarlo |
| `pip` no se reconoce | Usa `python -m pip install -r backend/requirements.txt` |
| `No module named 'app'` | Corre `uvicorn` **desde la carpeta `backend`** (`cd backend`) |
| `No module named ...` (otro) | Activa el entorno y vuelve a correr el `pip install` |
| "Backend apagado · reintentando" | Normal mientras carga la IA. Si no se quita, la Terminal 1 está cerrada o falló |
| Abre otra app o `/api/health` da 404 | El puerto 8000 lo usa otro programa. Ciérralo, o usa otro puerto (ver abajo) |
| Se ve un resultado viejo, sin "Vista corregida" | Es el último análisis guardado: pulsa **Analizar de nuevo** |
| Aviso de *symlinks* en Windows | Es solo una advertencia al descargar los modelos; se puede ignorar |
| Página en blanco | F12 → Console, y manda captura del error al grupo |

**Usar otro puerto** (si el 8000 está ocupado):

```powershell
# Terminal 1
uvicorn app.main:app --reload --port 8001

# Terminal 2 (PowerShell)
$env:BACKEND_URL="http://127.0.0.1:8001"; npm run dev
```

**Tips:**
- Usa capturas en **PNG** y con el zoom del navegador al 100% (JPG mete ruido en los colores).
- La IA funciona mejor con pantallas que tienen **componentes repetidos** (varios botones, tarjetas o etiquetas del mismo estilo).
- Los hallazgos los genera una IA: **revísalos**, no todos son errores reales.
