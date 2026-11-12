# Lynx · QA visual con IA

Sube una captura de tu página (o pega su URL) y la IA te dice qué está mal: elementos desalineados, componentes con otro tamaño o color, textos difíciles de leer, botones sueltos y demasiados estilos. Además te muestra **cómo se vería corregida** y **aprende de tus revisiones**.

Usa dos modelos de IA de Meta:

- **DINOv2**: entiende la imagen y reconoce qué elementos son "el mismo componente" (botones, tarjetas, etiquetas).
- **SAM 2.1** (Segment Anything): recorta cada elemento con su silueta exacta para corregir las cajas y generar la vista corregida.

- **Analizar desde una URL**: pegas la dirección de tu página y se revisa en **móvil, tablet y escritorio** de una sola vez.
- **Proyectos**: agrupan los análisis de una misma página y muestran una **gráfica de cómo evoluciona el puntaje**.
- **Comparar versiones**: eliges dos análisis de un proyecto y te dice qué errores se **arreglaron**, cuáles **siguen** y cuáles son **nuevos**.
- **Base de datos con `.env`**: los datos de MySQL ya no se ponen en la terminal; van en un archivo.
- **Diseño LYNX**: tema oscuro (negro, grises y amarillo dorado), logo nuevo y fondo de marca.
- **Animaciones de hallazgos** (de la rama `feature/diegoqa`): al pasar el mouse por un hallazgo se ilumina su zona en la captura; menús nuevos de zoom y categoría, y botón **Limpiar** para quitar la imagen cargada.
- **Vista corregida siempre visible**: si la captura no tiene nada que corregir automáticamente, la pestaña lo explica en vez de desaparecer.

> Si ya tenías el proyecto funcionando, solo sigue estos pasos.
>
> **¿Ya habías hecho los pasos 1 a 4 antes?** Para el diseño LYNX y las animaciones no hay que instalar nada: basta con `git pull` y recargar la página.

Encima de DINOv2 hay **tres redes neuronales entrenadas por el equipo** (ver [Cómo aprende la IA](#cómo-aprende-la-ia)).

---

## 1. Instala esto (una sola vez)

- **Python 3.12, 3.13 o 3.14** → https://www.python.org/downloads/ (marca **"Add python.exe to PATH"**)
- **Node.js 22** → https://nodejs.org/
- **Git** → https://git-scm.com/downloads
- **MySQL 8** → https://dev.mysql.com/downloads/installer/ (guarda la contraseña de `root`)
- **Solo Windows:** https://aka.ms/vs/17/release/vc_redist.x64.exe (sin esto la IA no funciona)
- **Opcional, para la revisión "Detallada":** **Ollama** → https://ollama.com/download

Después de instalar, **reinicia la PC**.

## 2. Descarga el proyecto

```bash
git clone https://github.com/isaac2466/QA-tester.git
cd QA-tester
git checkout RamaNeto
```

> Si ya tenías el proyecto, solo haz `git checkout RamaNeto` y `git pull`.

## 3. Crea el entorno virtual e instala las librerías

Desde la carpeta `QA-tester`:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r backend/requirements.txt
npm install --prefix frontend
```

- Sabrás que el entorno está activo porque la línea empieza con **`(.venv)`**. **Actívalo cada vez que abras una terminal nueva.**
- El `pip install` tarda unos minutos (~300 MB: PyTorch y las librerías de IA).
- Usa siempre **`python -m pip`**, no `pip` a secas.

Librerías principales: `torch` + `torchvision` + `transformers` (DINOv2 y SAM 2.1), `opencv`, `fastapi` + `uvicorn`, `mysql-connector-python` + `bcrypt` (login), `playwright` (capturas de URLs) y `python-dotenv` (lee el `.env`).

**Navegador para las capturas de URLs:** se usa el **Edge** o **Chrome** que ya tienes. Si no tienes ninguno:

```powershell
python -m playwright install chromium
```

## 4. Configura MySQL con el archivo `.env`

1. En la carpeta `backend`, copia `.env.example` y renombra la copia como **`.env`**.
2. Escribe tu contraseña de MySQL en `MYSQL_PASSWORD=` (la misma con la que entras a Workbench).

```
MYSQL_HOST=127.0.0.1
MYSQL_PORT=3306
MYSQL_USER=root
MYSQL_PASSWORD=tu_contraseña
MYSQL_DATABASE=qa_tester
```

> **No subas el `.env` a GitHub**: ya está en `.gitignore`. Escribe tu contraseña en el `.env`, **no** en el `.env.example`.

## 5. Crea o actualiza la base de datos

Desde la carpeta `backend`:

```powershell
python crear_bd.py
```

Debe decir: `Base de datos lista. Tablas: projects, query_history, user_sessions, users`.

- Si **no tenías** la base, la crea completa. Si **ya la tenías**, solo agrega lo nuevo: **no borra nada**.
- Se puede correr las veces que quieras. El backend también revisa la base al arrancar.

## 6. Ábrelo

Abre **dos terminales** en la carpeta `QA-tester` y activa el entorno en las dos (`.\.venv\Scripts\Activate.ps1`).

**Terminal 1 (backend + IA):**
```powershell
cd backend
python -m uvicorn app.main:app --reload --port 8000
```

> Usa **`python -m uvicorn`**, no `uvicorn` a secas: Windows 11 puede bloquear `uvicorn.exe` ("Una directiva de Control de aplicaciones bloqueó este archivo").

Espera a que aparezca:
```
DINOv2: cargado en cpu
SAM 2.1: cargado en cpu
MySQL: al día
Application startup complete.
```
La **primera vez** tarda unos minutos porque descarga los modelos (DINOv2 ~90 MB, SAM 2.1 ~150 MB).

**Terminal 2 (página web):**
```powershell
cd frontend
npm run dev
```

Entra a **http://localhost:5173**, crea tu cuenta y listo. Arriba debe decir **"Listo para analizar"**.

### Opcional: revisión "Detallada" con Qwen

Explica cada hallazgo con más detalle y avisa cuáles parecen falsas alarmas. Necesita Ollama:

```powershell
ollama pull qwen3-vl:4b-instruct
```

- Son ~3.3 GB. Después **reinicia el backend** y se habilita el botón **"Detallada"**.
- En CPU tarda **~1 minuto por hallazgo** y usa **3-4 GB de RAM** extra: cierra otros programas.
- Para liberar la memoria cuando termines: `ollama stop qwen3-vl:4b-instruct`.
- Nota: **Qwen no es de Meta** (es de Alibaba). Es opcional; la app funciona completa sin él.

---

## 7. Cómo se usa

Arriba hay dos secciones: **Revisar** y **Proyectos**.

### Revisar página
1. Elige **Subir captura** (arrastra, clic o **Ctrl+V**) o **Desde una URL**.
2. Con URL, elige los tamaños: Móvil (390 px), Tablet (820 px) y Escritorio (1440 px).
3. Elige **Rápida** (segundos) o **Detallada** (con Qwen) y pulsa **Analizar**.

| Vista | Qué muestra |
|---|---|
| **Hallazgos** | Recuadros numerados sobre cada error. Clic en un hallazgo para ver el detalle |
| **Vista corregida** | Antes / después con slider: tu captura con las correcciones aplicadas |
| **Atención** | Mapa de calor: qué zona llama la atención |
| **Vista de la IA** | Cómo agrupa DINOv2 tu interfaz (grupos de 3 o más elementos iguales) |

En cada hallazgo responde **"¿Es un error real? Sí / No"**: así la IA aprende (ver abajo).

Si una página mide más de 4000 px de alto, solo se revisa esa parte y aparece un aviso.

### Comparar diseños
Sube el **diseño** (frame de Figma en PNG) y la **implementación** (captura al mismo ancho, zoom 100%). DINOv2 empareja cada elemento y dice qué se movió, qué cambió y **qué falta o sobra**.

### Proyectos
1. **Proyectos** → **+ Nuevo proyecto**. Ponle nombre y, si quieres, la dirección de la página.
2. Para guardar análisis, elígelo en **Guardar en** (junto a las pestañas de Revisar), o entra al proyecto y pulsa **Analizar ahora**.
3. Verás el último puntaje, cuánto mejoró, la **gráfica de evolución** y la tabla de versiones.

### Comparar versiones
En la tabla de versiones de un proyecto, marca **2 versiones** y pulsa **Comparar** (o usa **vs anterior**). Verás **Arreglados**, **Siguen** y **Nuevos**.

> Compara siempre el mismo tamaño (móvil con móvil).

### Qué revisa
- **Alineación:** desalineados, textos fuera de línea en menús, botones sueltos debajo de otro, tarjetas a distinta altura.
- **Consistencia:** componentes iguales con otro tamaño o color (DINOv2 encuentra a los "gemelos").
- **Color:** contraste WCAG, demasiados estilos de botón, texto blanco en botones de color.
- **Modo estricto:** solo se muestra lo que tiene respaldo de la IA; lo descartado aparece en "descartados por la IA · ver".

---

## Cómo aprende la IA

| Pieza | Qué hace | Dónde está | ¿Viaja con git? |
|---|---|---|---|
| **Detector** | Encuentra botones, tarjetas y textos | `backend/app/ai/weights/detector.pt` | ✅ |
| **Verificador** | Distingue interfaz real de fotos o degradados | `backend/app/ai/weights/verifier.pt` | ✅ |
| **Marcas del equipo** | Cada "Sí / No es error" que marcan | `backend/app/ai/learned/marcas_<tu-pc>.jsonl` | ✅ |
| **Red de marcas** | Aprende de las marcas qué es error para ustedes | Se genera sola en cada PC | — (se reentrena sola) |
| Capturas analizadas | Datos reales para reentrenar en el futuro | `backend/ai_data/` | ❌ (pesan mucho) |

- Cada **5 marcas** la red se reentrena sola (menos de un segundo). A partir de **30 marcas** empieza a descartar sola lo que marcaron como "no es error".
- **Para que todo el equipo aprenda junto:** suban sus marcas con un commit normal. Cada PC tiene su propio archivo en `learned/`, así que no hay conflictos.

### Entrenar el detector (opcional)

Con el entorno activo, desde `backend`:

```powershell
python -m app.ai.train_detector --train 120 --val 20 --epochs 30
```

| Tamaño | Comando | Tiempo aprox. (CPU) |
|---|---|---|
| Prueba rápida | `--train 60 --val 10 --epochs 20` | 10-15 min |
| Mediano (el actual) | `--train 120 --val 20 --epochs 30` | ~30 min |
| Grande | `--train 400 --val 50 --epochs 50` | ~2 horas (déjalo de noche) |

- **No uses la app mientras entrena**: con 8 GB de RAM no alcanza para las dos cosas.
- Al terminar imprime recall y precisión. **Antes de quedarte con el nuevo, compara** con `detector_metrics.json` del modelo anterior (actual: bloques 0.92 / 0.90, textos 0.74 / 0.77).
- Las pantallas generadas se guardan en `backend/ai_data/` (no van a git) y se reutilizan.

---

## ¿No funciona?

| Error | Solución |
|---|---|
| `No module named ...` y la ruta dice `AppData\Local\Programs\Python` | Estás usando el Python de Windows. **Activa el entorno** (`.\.venv\Scripts\Activate.ps1`) y usa `python`, no `py` |
| `No module named 'app'` | Corre el backend **desde la carpeta `backend`** |
| "Una directiva de Control de aplicaciones bloqueó este archivo" | Usa `python -m uvicorn ...` en lugar de `uvicorn ...` |
| "La ejecución de scripts está deshabilitada" al activar el entorno | Corre una vez `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` |
| `WinError 126` o `shm.dll` | Instala el link de **Solo Windows** del paso 1 y reinicia |
| `ECONNREFUSED 127.0.0.1:8000` en la terminal del frontend | El backend no está corriendo o se cayó: revisa la Terminal 1 |
| El backend se cae al analizar / `not enough memory` | Falta RAM: cierra otros programas (otros proyectos, pestañas, Qwen con `ollama stop ...`). No entrenes mientras usas la app |
| El puerto 8000 ya está en uso | Ciérralo o usa otro puerto: `--port 8001` y en el frontend `$env:BACKEND_URL="http://127.0.0.1:8001"; npm run dev` |
| `crear_bd.py` dice "No se pudo conectar a MySQL" | Revisa `MYSQL_USER` y `MYSQL_PASSWORD` en `backend/.env` y que MySQL esté encendido |
| No puedo iniciar sesión / "Verifica la conexión con MySQL" | Falta el `.env` o tiene mal la contraseña. Corrígelo y **reinicia el backend** |
| La sección Proyectos sale vacía o con error | Corre `python crear_bd.py` y reinicia el backend |
| "No se encontró un navegador para tomar la captura" | Instala Edge o Chrome, o corre `python -m playwright install chromium` |
| "La página tardó demasiado en cargar" | Tardó más de 30 s. Prueba de nuevo o con menos tamaños |
| "Detallada" aparece apagado | Falta Qwen: `ollama pull qwen3-vl:4b-instruct` y reinicia el backend |
| "No se guardó tu marca" | Vuelve a analizar la captura y marca de nuevo |
| Aviso de *symlinks* al descargar modelos | Es solo una advertencia de Windows; se puede ignorar |
| Página en blanco | F12 → Console, y manda captura del error al grupo |

**Tips:**
- Usa capturas en **PNG, a tamaño real** (`Win + Shift + S`), no miniaturas descargadas: con imágenes borrosas las medidas al pixel fallan.
- Funciona mejor con **dashboards y landings de diseño plano**; las páginas llenas de fotos confunden más.
- Los hallazgos los genera una IA: **revísalos y márcalos**. Así mejora.
