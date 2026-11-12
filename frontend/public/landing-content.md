# Lynx · QA visual con IA

## Idea central

Lynx revisa visualmente una página web para encontrar diferencias y problemas de
interfaz antes de que lleguen a producción. Analiza capturas o toma la página
directamente desde una URL, mide la composición con precisión de píxel y convierte
los resultados en hallazgos claros, priorizados y accionables.

No sustituye al criterio del equipo de diseño o QA: lo ayuda a revisar más rápido,
detectar regresiones y mantener una referencia objetiva entre versiones.

## Propuesta de valor

- **De la captura al diagnóstico:** sube una imagen o indica una URL y obtén una
  revisión visual en pocos pasos.
- **Problemas que se pueden corregir:** cada hallazgo muestra qué ocurre, dónde
  ocurre, su severidad y una sugerencia concreta.
- **Contexto visual:** los hallazgos aparecen marcados sobre la captura, con
  acercamientos y zonas comparadas.
- **Seguimiento de calidad:** guarda análisis en proyectos y observa cómo cambia el
  puntaje con cada versión.
- **Menos ruido:** varios motores de IA y reglas de medición ayudan a separar
  problemas reales de fotos, fondos, degradados o elementos del sistema.

## ¿Qué puede hacer la app?

### 1. Analizar una captura

El usuario puede subir una captura de la interfaz. Lynx detecta elementos de la
página, revisa su composición y genera:

- Puntaje visual de 0 a 100.
- Hallazgos de severidad alta, media o baja.
- Imagen anotada con las zonas problemáticas.
- Paleta de colores dominante.
- Mapa de atención visual, cuando DINOv2 está disponible.
- Grupos de componentes repetidos.
- Vista corregida con ajustes visuales aplicados cuando SAM 2.1 está disponible.

### 2. Analizar desde una URL

La app puede capturar una página real en tres tamaños:

- **Móvil:** 390 px.
- **Tablet:** 820 px.
- **Escritorio:** 1440 px.

Así es posible revisar el comportamiento responsive con el mismo flujo y conservar
un análisis independiente por viewport.

### 3. Comparar diseño contra implementación

Permite subir el diseño o mockup junto con la implementación real. La comparación
detecta, entre otros casos:

- Elementos desplazados.
- Cambios de color.
- Contenido, texto, iconos o formas diferentes.
- Diferencias de tamaño o posición.
- Desplazamientos globales de la página.
- Similitud semántica entre ambas versiones.

La implementación se muestra junto al diseño y las diferencias se agrupan en
zonas que el equipo puede revisar.

### 4. Revisar y explicar hallazgos

Cada hallazgo incluye:

- Categoría: alineación, espaciado, color, contraste, atención visual,
  consistencia o diferencia con el diseño.
- Severidad: alta, media o baja.
- Medición y ubicación.
- Sugerencia de corrección.
- Enlace visual entre la lista y la zona marcada.

De forma opcional, Qwen3-VL puede explicar cada hallazgo en lenguaje natural:
qué pasa, por qué importa y cómo arreglarlo. También puede señalar un posible
falso positivo.

### 5. Aprender del criterio del equipo

El usuario puede marcar un hallazgo como **“Sí es error”** o **“No es error”**.
Lynx guarda esas decisiones para reconocer casos parecidos en análisis futuros y
reducir falsos positivos con el tiempo.

### 6. Gestionar proyectos y versiones

Los proyectos agrupan análisis de una misma página. Dentro de cada proyecto se
puede:

- Consultar el último puntaje.
- Ver la evolución del puntaje por tamaño de pantalla.
- Revisar el historial de versiones.
- Comparar dos análisis.
- Identificar errores **arreglados**, **pendientes** y **nuevos**.
- Mantener los análisis aunque se elimine el proyecto.

### 7. Exportar resultados

El panel permite exportar un reporte con los hallazgos para compartirlo con
desarrollo, diseño, producto o clientes.

## Cómo funciona el análisis

1. **Entrada:** se recibe una captura subida por el usuario o una captura tomada
   desde una URL.
2. **Detección:** se localizan bloques, textos y posibles elementos de interfaz.
3. **Medición:** se calculan posiciones, tamaños, espaciados, colores, contraste y
   diferencias visuales.
4. **Comprensión visual:** los modelos identifican atención, componentes
   equivalentes y similitud entre zonas.
5. **Verificación:** se descartan casos que corresponden a imágenes, fondos,
   degradados o elementos del sistema.
6. **Priorización:** se calcula un puntaje y se ordenan los hallazgos por
   severidad.
7. **Explicación y aprendizaje:** el equipo puede pedir una explicación, ignorar
   un hallazgo o confirmar que es un error.
8. **Seguimiento:** el resultado se guarda en el historial o en un proyecto para
   comparar versiones posteriores.

## Modelos y motores

### DINOv2

Es el motor visual principal basado en representaciones de imagen. En Lynx se
utiliza para:

- Extraer características visuales de la página y de cada elemento.
- Generar un mapa de atención: qué zonas destacan visualmente.
- Agrupar componentes similares, como tarjetas, botones o elementos repetidos.
- Emparejar elementos entre un diseño y su implementación.
- Calcular similitud semántica entre dos imágenes.
- Verificar si un hallazgo parece corresponder a una interfaz real.

Su ventaja es que puede reconocer elementos parecidos aunque cambien el texto,
la posición o el tamaño.

### Detector de elementos entrenado

Es una cabeza ligera entrenada sobre características de DINOv2. Predice centros,
tipo y tamaño aproximado de elementos de interfaz para complementar las reglas
clásicas de detección.

Actualmente distingue principalmente entre **bloques** y **texto**. Sus pesos
entrenados se encuentran en `backend/app/ai/weights/detector.pt`.

### Verificador basado en DINOv2

Clasifica el contenido de un recorte para decidir si pertenece a la interfaz o
debe descartarse. Considera categorías como:

- Texto.
- Componente.
- Icono.
- Imagen.
- Degradado.
- Elemento del sistema.

Esto evita contar como error una medición que realmente cayó sobre una fotografía,
un fondo, una ilustración o una marca de agua del sistema.

### SAM 2.1

Segment Anything de Meta refina la silueta de los elementos detectados. Lynx lo
usa para generar máscaras más precisas y construir la **vista corregida** sin
tener que aplicar cambios sobre rectángulos completos.

La configuración utiliza la variante ligera `facebook/sam2.1-hiera-tiny`, pensada
para poder ejecutarse también en CPU.

### Qwen3-VL

Es el modelo de visión y lenguaje opcional que se ejecuta localmente mediante
Ollama. No reemplaza las mediciones de píxel: interpreta los recortes y traduce
los resultados a una explicación útil para personas.

Puede devolver:

- Qué elemento aparece en la zona.
- Si el hallazgo parece real, falso positivo o dudoso.
- Qué está pasando.
- Por qué afecta a la experiencia.
- Cómo corregirlo.

### Reglas visuales y mediciones clásicas

Las reglas deterministas aportan mediciones precisas y reproducibles para:

- Alineación.
- Centrado.
- Espaciado.
- Paleta de colores.
- Contraste.
- Diferencias de píxel.
- Desplazamiento global.

La combinación de reglas, embeddings y modelos visuales permite equilibrar
precisión medible con comprensión de contexto.

## Qué significa el puntaje

El puntaje parte de 100 y resta puntos según la severidad de los hallazgos
visibles:

- **Alta:** resta 8 puntos.
- **Media:** resta 4 puntos.
- **Baja:** resta 1 punto.

Los hallazgos descartados por la IA no afectan el puntaje. El valor sirve como
indicador de evolución entre versiones; no pretende ser una certificación
universal de calidad o accesibilidad.

## Usuarios ideales

- Equipos frontend que quieren validar cada entrega.
- Diseñadores que necesitan comprobar que la implementación respeta el mockup.
- QA y testing visual en flujos de regresión.
- Agencias que revisan múltiples landings o sitios de clientes.
- Product managers que necesitan comunicar problemas con evidencia visual.
- Equipos pequeños que quieren automatizar una primera revisión antes de una
  validación manual.

## Mensajes para la landing

### Hero principal

**Tu diseño merece llegar a producción tal como lo imaginaste.**

Lynx encuentra errores visuales, compara versiones y te muestra exactamente qué
corregir antes de que los vea tu usuario.

CTA sugerido: **Analizar mi página**

### Alternativas de hero

1. **QA visual que ve lo que tu equipo no puede revisar a mano.**
2. **Encuentra el píxel fuera de lugar antes de publicar.**
3. **De diseño a producción, sin perder detalle.**
4. **Convierte cada captura en una revisión visual accionable.**
5. **Tu página, revisada en móvil, tablet y escritorio.**

### Bloques de beneficios

- **Detecta:** encuentra desalineaciones, contrastes débiles, espacios irregulares
  y diferencias con el diseño.
- **Entiende:** recibe mediciones, contexto visual y explicaciones claras.
- **Mejora:** compara versiones y comprueba qué errores ya desaparecieron.
- **Aprende:** enseña a la herramienta qué casos son importantes para tu equipo.

### Slogans cortos

- **Calidad visual, sin adivinar.**
- **Cada píxel cuenta.**
- **Diseña con intención. Publica con confianza.**
- **Ve el problema. Entiende el impacto. Corrígelo.**
- **Menos revisión manual. Más control visual.**
- **La lupa inteligente para tus interfaces.**
- **De la primera captura a una interfaz más precisa.**

### CTA y microcopy

- **Analizar una captura**
- **Probar con una URL**
- **Comparar con mi diseño**
- **Ver qué cambió**
- **Crear mi primer proyecto**
- **Revisar esta versión**
- **Pedir explicación a la IA**

## Frases de confianza y alcance

- “Mediciones visuales respaldadas por reglas y modelos especializados.”
- “Revisa primero; decide tú. Lynx mantiene tus hallazgos visibles para que el
  equipo conserve el control.”
- “El análisis visual es una primera capa de QA y no reemplaza pruebas
  funcionales, accesibilidad completa ni revisión humana.”
- “Qwen3-VL es opcional y se ejecuta localmente mediante Ollama.”

## Nota para el equipo de contenido

Este documento describe las capacidades implementadas actualmente y puede servir
como base para una landing page, un one-pager o el contenido de onboarding.
Antes de publicar cifras de rendimiento, compatibilidad o disponibilidad de
modelos, conviene validarlas en el entorno de despliegue final.
