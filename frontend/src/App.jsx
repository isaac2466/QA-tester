import { useEffect, useRef, useState } from "react";
import Dropzone from "./components/Dropzone.jsx";
import Results from "./components/Results.jsx";
import { analyzeImage, compareImages, health } from "./services/api.js";
import { load, save } from "./utils/storage.js";

const ANALYZE_VIEWS = [
  { key: "annotated", label: "Hallazgos" },
  { key: "heatmap", label: "Mapa de atención (DINOv2)" },
];
const COMPARE_VIEWS = [
  { key: "annotated", label: "Implementación" },
  { key: "design", label: "Diseño" },
  { key: "semantic", label: "Diferencias (DINOv2)" },
];

export default function App() {
  const [mode, setMode] = useState("analyze");
  const [file, setFileRaw] = useState(null);
  const [design, setDesignRaw] = useState(null);
  const [impl, setImplRaw] = useState(null);
  const [results, setResults] = useState({}); // { analyze: {...}, compare: {...} }
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [status, setStatus] = useState(null);
  const [hydrated, setHydrated] = useState(false);
  const resultsRef = useRef(null);
  const justRan = useRef(false);
  const result = results[mode];

  useEffect(() => {
    health().then(setStatus).catch(() => setStatus({ status: "down" }));
  }, []);

  // Restaurar la última sesión al recargar (F5)
  useEffect(() => {
    load("session").then((s) => {
      if (s) {
        setMode(s.mode ?? "analyze");
        setFileRaw(s.file ?? null);
        setDesignRaw(s.design ?? null);
        setImplRaw(s.impl ?? null);
        setResults(s.results ?? {});
      }
      setHydrated(true);
    });
  }, []);

  // Guardar cada cambio (sólo después de restaurar, para no pisar lo guardado)
  useEffect(() => {
    if (hydrated) save("session", { mode, file, design, impl, results });
  }, [hydrated, mode, file, design, impl, results]);

  // Cambiar una imagen invalida el resultado anterior de esa pestaña
  const clearResult = (m) => setResults((r) => ({ ...r, [m]: undefined }));
  const setFile = (f) => { setFileRaw(f); clearResult("analyze"); };
  const setDesign = (f) => { setDesignRaw(f); clearResult("compare"); };
  const setImpl = (f) => { setImplRaw(f); clearResult("compare"); };

  const switchMode = (m) => { setMode(m); setError(null); };

  const reset = () => {
    setError(null);
    if (mode === "analyze") setFile(null);
    else { setDesign(null); setImpl(null); }
  };

  const run = async () => {
    setLoading(true);
    setError(null);
    const m = mode;
    try {
      const data = m === "analyze" ? await analyzeImage(file) : await compareImages(design, impl);
      justRan.current = true;
      setResults((r) => ({ ...r, [m]: data }));
    } catch (e) {
      justRan.current = true;
      setError(e.message === "Failed to fetch" ? "No se pudo conectar con el backend (¿está corriendo en el puerto 8000?)" : e.message);
    } finally {
      setLoading(false);
    }
  };

  // al llegar resultados nuevos, desplazarse hasta ellos (no al restaurar con F5)
  useEffect(() => {
    if (justRan.current && (result || error)) {
      justRan.current = false;
      resultsRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
    }
  }, [result, error]);

  // En modo comparar, Ctrl+V llena primero el diseño y luego la implementación
  useEffect(() => {
    if (mode !== "compare") return;
    const onPaste = (e) => {
      const item = [...e.clipboardData.items].find((i) => i.type.startsWith("image/"));
      if (!item) return;
      const f = item.getAsFile();
      if (!design) setDesign(f);
      else setImpl(f);
    };
    window.addEventListener("paste", onPaste);
    return () => window.removeEventListener("paste", onPaste);
  }, [mode, design]);

  const ready = mode === "analyze" ? file : design && impl;
  const missing = mode === "analyze"
    ? !file && "Sube o pega una captura para analizar."
    : !design && !impl ? "Sube las 2 imágenes: el diseño y la captura de la implementación."
    : !design ? "Falta la imagen del diseño (1)."
    : !impl ? "Falta la captura de la implementación (2)." : null;

  return (
    <div className="app">
      <header className="top">
        <div>
          <h1>QA Tester visual</h1>
          <p className="muted">Sube una captura de tu página y encuentra los micro-detalles que no se ven a simple vista.</p>
        </div>
        <span className={`status ${status?.dino?.loaded ? "ok" : status?.status === "down" ? "down" : "partial"}`}>
          {!status ? "Conectando…" : status.status === "down" ? "Backend apagado"
            : status.dino.loaded ? `DINOv2 listo (${status.dino.device})` : "DINOv2 no cargado"}
        </span>
      </header>

      <div className="tabs">
        <button className={mode === "analyze" ? "active" : ""} onClick={() => switchMode("analyze")}>Analizar una captura</button>
        <button className={mode === "compare" ? "active" : ""} onClick={() => switchMode("compare")}>Comparar diseño vs implementación</button>
      </div>

      {mode === "compare" && (
        <div className="howto">
          <strong>¿Cómo funciona?</strong>
          <ol>
            <li><b>Diseño:</b> exporta el frame de Figma (o tu mockup) como PNG.</li>
            <li><b>Implementación:</b> haz una captura de tu página real <b>al mismo ancho</b> que el diseño
              (ej. 1440px) y al 100% de zoom. En Chrome: F12 → Ctrl+Shift+P → "Capture full size screenshot".</li>
            <li>Pulsa <b>Comparar</b>: se superponen ambas imágenes y te dice qué elemento se movió y cuántos px,
              qué color cambió (con su hex) y qué contenido no coincide.</li>
          </ol>
          <small className="muted">Si las imágenes tienen tamaños distintos se reescala y las medidas son aproximadas.
            Puedes pegar con Ctrl+V: la primera vez llena el diseño y la segunda la implementación.</small>
        </div>
      )}

      <div className={`inputs ${mode}`}>
        {mode === "analyze" ? (
          <Dropzone label="Captura de tu página" file={file} onFile={setFile} pasteTarget />
        ) : (
          <>
            <Dropzone label="1. Diseño (Figma / mockup)" file={design} onFile={setDesign} />
            <Dropzone label="2. Implementación (captura real)" file={impl} onFile={setImpl} />
          </>
        )}
      </div>

      <div className="actions">
        <button className="primary" disabled={!ready || loading} onClick={run}>
          {loading ? "Analizando…" : mode === "analyze" ? "Analizar" : "Comparar"}
        </button>
        {(mode === "analyze" ? file : design || impl) && (
          <button className="ghost" onClick={reset} disabled={loading}>Limpiar</button>
        )}
        {missing && <span className="muted">{missing}</span>}
      </div>

      <div ref={resultsRef}>
        {error && <p className="error">{error}</p>}
        {result && (
          <Results key={mode + result.elapsed_ms} data={result} views={mode === "analyze" ? ANALYZE_VIEWS : COMPARE_VIEWS} />
        )}
      </div>
    </div>
  );
}
