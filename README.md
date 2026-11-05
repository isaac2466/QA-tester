# QA Tester visual

Sube una captura de tu página y te dice qué está mal: botones desalineados por 1px, colores que no quedan, texto difícil de leer y qué zona llama la atención (con IA, DINOv2).

## 1. Instala esto (una sola vez)

- **Python 3.12** → https://www.python.org/downloads/ (marca **"Add python.exe to PATH"**)
- **Node.js 22** → https://nodejs.org/
- **Git** → https://git-scm.com/downloads
- **Solo Windows:** https://aka.ms/vs/17/release/vc_redist.x64.exe (sin esto la IA no funciona)

Después de instalar, **reinicia la PC**.

## 2. Descarga el proyecto

```bash
git clone https://github.com/isaac2466/QA-tester.git
cd QA-tester
```

## 3. Instala las librerías

```bash
cd backend
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
cd ../frontend
npm install
```

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

## ¿No funciona?

| Error | Solución |
|---|---|
| `WinError 126` o `shm.dll` | Instala el link de **Solo Windows** del paso 1 y reinicia |
| `python` o `uvicorn` no se reconoce | Reinstala Python marcando **"Add python.exe to PATH"** |
| "No se pudo conectar con el backend" | La Terminal 1 no está abierta |
| Página en blanco | F12 → Console, y manda captura del error al grupo |

**Tip:** usa capturas en **PNG** y con el zoom del navegador al 100%.
