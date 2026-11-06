# QA Tester visual

Sube una captura de tu página y te dice qué está mal: botones desalineados por 1px, colores que no quedan, texto difícil de leer y qué zona llama la atención (con IA, DINOv2).

## 1. Instala esto (una sola vez)

- **Python 3.12 o 3.13** → https://www.python.org/downloads/ (marca **"Add python.exe to PATH"**)
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

Desde la carpeta `QA-tester`, copia y pega estos 2 comandos:

```bash
pip install -r backend/requirements.txt
npm install --prefix frontend
```

El primero instala todo lo de Python, incluida la IA (tarda unos minutos: baja ~300 MB). El segundo instala lo de la página web.

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

Al abrir la página puedes crear una cuenta con tu correo y una contraseña de mínimo 8 caracteres.
La contraseña se guarda en el backend como un hash seguro; después podrás iniciar sesión con esas credenciales.

## ¿No funciona?

| Error | Solución |
|---|---|
| `WinError 126` o `shm.dll` | Instala el link de **Solo Windows** del paso 1 y reinicia |
| `python` o `uvicorn` no se reconoce | Reinstala Python marcando **"Add python.exe to PATH"** |
| `No module named ...` | Vuelve a correr `pip install -r backend/requirements.txt` |
| "No se pudo conectar con el backend" | La Terminal 1 no está abierta |
| Página en blanco | F12 → Console, y manda captura del error al grupo |

**Tip:** usa capturas en **PNG** y con el zoom del navegador al 100%.
