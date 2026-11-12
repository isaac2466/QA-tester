# Lynx · Novedades

Lo nuevo en esta versión:

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

## 1. Actualiza el proyecto

```bash
git pull
```

## 2. Instala las librerías nuevas

Con el entorno activo (`.\.venv\Scripts\Activate.ps1`), desde la carpeta `QA-tester`:

```powershell
python -m pip install -r backend/requirements.txt
```

Se agregaron `playwright` (toma las capturas de las URLs), `python-dotenv` (lee el `.env`), `mysql-connector-python` y `bcrypt`.

**Navegador para las capturas:** se usa el **Edge** o **Chrome** que ya tienes instalado. En Windows, Edge viene incluido, así que no hay que descargar nada más. Si no tienes ninguno, ejecuta una vez:

```powershell
python -m playwright install chromium
```

## 3. Configura MySQL con el archivo `.env`

1. En la carpeta `backend`, copia `.env.example` y renombra la copia como **`.env`**.
2. Ábrelo y escribe tu contraseña de MySQL en `MYSQL_PASSWORD=` (la misma con la que entras a Workbench).

```
MYSQL_HOST=127.0.0.1
MYSQL_PORT=3306
MYSQL_USER=root
MYSQL_PASSWORD=tu_contraseña
MYSQL_DATABASE=qa_tester
```

> **No subas el `.env` a GitHub**: ya está en `.gitignore`. Escribe tu contraseña en el `.env`, **no** en el `.env.example`.

## 4. Crea o actualiza la base de datos

Desde la carpeta `backend`:

```powershell
python crear_bd.py
```

Debe decir: `Base de datos lista. Tablas: projects, query_history, user_sessions, users`.

- Si **no tenías** la base, la crea completa.
- Si **ya la tenías**, solo agrega lo nuevo (la tabla `projects` y tres columnas en `query_history`): **no borra nada**.
- Se puede correr las veces que quieras.

También puedes ejecutar `data/qa_tester.sql` en MySQL Workbench, pero ese archivo solo crea lo que no existe. Si ya tenías la base, usa `crear_bd.py`, que también agrega las columnas nuevas. De todos modos, el backend revisa la base al arrancar y agrega lo que falte.

## 5. Arranca como siempre

```powershell
# Terminal 1
cd backend
uvicorn app.main:app --reload --port 8000

# Terminal 2
cd frontend
npm run dev
```

Al arrancar, la Terminal 1 debe mostrar `MySQL: al día` (o `MySQL: se agregó ...` la primera vez).

## Cómo se usan las novedades

Arriba hay dos secciones: **Revisar** y **Proyectos**.

### Analizar una URL (Revisar)
1. En **Revisar página**, elige **Desde una URL**.
2. Pega la dirección (por ejemplo `misitio.com`) y elige los tamaños: Móvil (390 px), Tablet (820 px) y Escritorio (1440 px).
3. Pulsa **Analizar**. Arriba de los resultados aparecen botones con el puntaje de cada tamaño para cambiar entre ellos.

Tarda unos segundos por tamaño. Si la página mide más de 4000 px de alto, solo se revisa esa parte y aparece un aviso.

### Proyectos
1. Ve a **Proyectos** → **+ Nuevo proyecto**. Ponle nombre y, si quieres, la dirección de la página.
2. Para guardar análisis en el proyecto, elígelo en **Guardar en** (junto a las pestañas de Revisar), o entra al proyecto y pulsa **Analizar ahora**.
3. En el proyecto verás el último puntaje, cuánto mejoró, la **gráfica de evolución** (una línea por tamaño) y la tabla de versiones.

### Comparar versiones
En la tabla de versiones de un proyecto:
- marca **2 versiones** y pulsa **Comparar**, o
- usa el atajo **vs anterior** de cada fila.

Verás el puntaje antes y después y tres listas: **Arreglados**, **Siguen** y **Nuevos**, con las dos capturas lado a lado.

> Compara siempre el mismo tamaño (móvil con móvil): si comparas tamaños distintos, muchos hallazgos no van a coincidir.

## ¿No funciona?

| Error | Solución |
|---|---|
| `No module named 'playwright'` (o `mysql`, `dotenv`) | Activa el entorno y vuelve a correr el paso 2 |
| "No se encontró un navegador para tomar la captura" | Instala Edge o Chrome, o corre `python -m playwright install chromium` |
| "No se encontró ese sitio" | Revisa que la URL esté bien escrita |
| "La página tardó demasiado en cargar" | La página tardó más de 30 s. Prueba de nuevo o con menos tamaños |
| `crear_bd.py` dice "No se pudo conectar a MySQL" | Revisa `MYSQL_USER` y `MYSQL_PASSWORD` en `backend/.env` y que MySQL esté encendido |
| No puedo iniciar sesión / "Verifica la conexión con MySQL" | Falta el `.env` o tiene mal la contraseña. Corrígelo y **reinicia uvicorn** (el `.env` solo se lee al arrancar) |
| La sección Proyectos sale vacía o con error | Corre `python crear_bd.py` (paso 4) y reinicia uvicorn |
