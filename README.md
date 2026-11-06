# QA Tester visual

Sube una captura de tu página y te dice qué está mal: botones desalineados por 1px, colores que no quedan, texto difícil de leer y qué zona llama la atención.

Usa 2 IAs: **DINOv2** (qué llama la atención) y **Qwen3-VL** con Ollama (revisa el diseño y descarta errores falsos).

## 1. Instala esto (una sola vez)

- **Python 3.12 o 3.13** → https://www.python.org/downloads/ (marca **"Add python.exe to PATH"**)
- **Node.js 22** → https://nodejs.org/
- **Git** → https://git-scm.com/downloads
- **Ollama** → https://ollama.com/download (para la IA que revisa el diseño)
- **Solo Windows:** https://aka.ms/vs/17/release/vc_redist.x64.exe (sin esto DINOv2 no funciona)

Después de instalar, **reinicia la PC**.

## 2. Descarga el proyecto

```bash
git clone https://github.com/isaac2466/QA-tester.git
cd QA-tester
```

## 3. Instala las librerías y la IA

Desde la carpeta `QA-tester`, copia y pega estos 3 comandos:

```bash
pip install -r backend/requirements.txt
npm install --prefix frontend
ollama pull qwen3-vl:4b-instruct
```

1. Instala todo lo de Python, incluido DINOv2 (~300 MB).
2. Instala lo de la página web.
3. Descarga la IA que revisa el diseño (~3 GB, tarda un rato).

## 4. Ábrelo

Abre **dos terminales** en la carpeta `QA-tester`:

**Terminal 1:**
```bash
cd backend
uvicorn app.main:app --reload --port 8000
```
Espera a que diga `DINOv2: cargado en cpu` (la primera vez tarda porque descarga la IA).

**Terminal 2:**
```bash
cd frontend
npm run dev
```

Entra a **http://localhost:5173**

Arriba a la derecha debe decir **Qwen3-VL listo** y **DINOv2 listo**.
Con Qwen3-VL cada análisis tarda **1 a 4 minutos** (corre en tu PC, sin internet). Si quieres algo rápido, desmarca **"Revisión de diseño con Qwen3-VL"** (DINOv2 se sigue usando).

> Ollama debe estar abierto (se abre solo al prender la PC; si no, ábrelo desde el menú Inicio).

## ¿No funciona?

| Error | Solución |
|---|---|
| `WinError 126` o `shm.dll` | Instala el link de **Solo Windows** del paso 1 y reinicia |
| `python` o `uvicorn` no se reconoce | Reinstala Python marcando **"Add python.exe to PATH"** |
| `No module named ...` | Vuelve a correr `pip install -r backend/requirements.txt` |
| "Qwen3-VL no disponible" | Abre Ollama y corre `ollama pull qwen3-vl:4b-instruct`. Pasa el mouse sobre el aviso para ver el motivo |
| Qwen3-VL tarda mucho o se traba | Normal en laptops sin tarjeta de video. Cierra otros programas o desmarca "Revisión de diseño con Qwen3-VL" |
| "No se pudo conectar con el backend" | La Terminal 1 no está abierta |
| Página en blanco | F12 → Console, y manda captura del error al grupo |

**Tip:** usa capturas en **PNG** y con el zoom del navegador al 100%.
