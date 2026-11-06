import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import FileSlot, { useImageInfo } from "./components/FileSlot.jsx";
import FindingsPanel from "./components/FindingsPanel.jsx";
import HistoryDrawer from "./components/HistoryDrawer.jsx";
import Viewer from "./components/Viewer.jsx";
import { Camera, Compare, History, Info, Logo, Play } from "./components/icons.jsx";
import { analyzeImage, compareImages, health } from "./services/api.js";
import { exportReport } from "./utils/report.js";
import { load, save } from "./utils/storage.js";

const HISTORY_MAX = 12;

function StatusPill({ status }) {
  const [tone, text] = !status ? ["partial", "Conectando…"]
    : status.status === "down" ? ["down", "Backend apagado · reintentando"]
    : status.dino?.loaded ? ["ok", `Motor listo · DINOv2${status.sam?.loaded ? " + SAM 2.1" : ""} (${status.dino.device})`]
    : ["partial", "IA no disponible"];
  return <span className={`status ${tone}`} title={status?.dino?.error ?? ""}><i />{text}</span>;
}

export default function App() {
  const [mode, setMode] = useState("analyze");
  const [file, setFileRaw] = useState(null);
  const [design, setDesignRaw] = useState(null);
  const [impl, setImplRaw] = useState(null);
  const [results, setResults] = useState({}); // { analyze, compare }
  const [ignored, setIgnored] = useState({}); // { analyze: [ids], compare: [ids] }
  const [history, setHistory] = useState([]);
  const [showHistory, setShowHistory] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [status, setStatus] = useState(null);
  const [hydrated, setHydrated] = useState(false);
  const [selected, setSelected] = useState(null);
  const [hovered, setHovered] = useState(null);
  const [focusTick, setFocusTick] = useState(0);
  const mainRef = useRef(null);
  const result = results[mode];
  const source = useImageInfo(mode === "analyze" && result ? file : null);

  // Estado del backend: reintenta mientras esté apagado (al arrancar tarda en cargar DINOv2)
  useEffect(() => {
    let timer;
    let alive = true;
    const check = () => health()
      .then((s) => { if (alive) setStatus(s); })
      .catch(() => { if (alive) { setStatus({ status: "down" }); timer = setTimeout(check, 3000); } });
    check();
    return () => { alive = false; clearTimeout(timer); };
  }, []);

  // Restaurar sesión e historial al recargar (F5)
  useEffect(() => {
    Promise.all([load("session"), load("history")]).then(([s, h]) => {
      if (s) {
        setMode(s.mode ?? "analyze");
        setFileRaw(s.file ?? null);
        setDesignRaw(s.design ?? null);
        setImplRaw(s.impl ?? null);
        setResults(s.results ?? {});
        setIgnored(s.ignored ?? {});
      }
      setHistory(h ?? []);
      setHydrated(true);
    });
  }, []);

  useEffect(() => {
    if (hydrated) save("session", { mode, file, design, impl, results, ignored });
  }, [hydrated, mode, file, design, impl, results, ignored]);

  // Cambiar una imagen invalida el resultado anterior de esa pestaña
  const clearResult = (m) => {
    setResults((r) => ({ ...r, [m]: undefined }));
    setIgnored((i) => ({ ...i, [m]: [] }));
    setSelected(null);
  };
  const setFile = useCallback((f) => { setFileRaw(f); clearResult("analyze"); }, []);
  const setDesign = useCallback((f) => { setDesignRaw(f); clearResult("compare"); }, []);
  const setImpl = useCallback((f) => { setImplRaw(f); clearResult("compare"); }, []);

  const switchMode = (m) => { setMode(m); setError(null); setSelected(null); };

  // Ctrl+V: en "Revisar" llena la captura; en "Comparar", primero el diseño y luego la implementación
  useEffect(() => {
    const onPaste = (e) => {
      const item = [...(e.clipboardData?.items ?? [])].find((i) => i.type.startsWith("image/"));
      if (!item) return;
      const f = item.getAsFile();
      if (mode === "analyze") setFile(f);
      else if (!design) setDesign(f);
      else setImpl(f);
    };
    window.addEventListener("paste", onPaste);
    return () => window.removeEventListener("paste", onPaste);
  }, [mode, design, setFile, setDesign, setImpl]);

  const ready = mode === "analyze" ? file : design && impl;
  const names = mode === "analyze" ? [file?.name || "captura"] : [design?.name || "diseño", impl?.name || "implementación"];

  const run = async () => {
    setLoading(true);
    setError(null);
    setSelected(null);
    const m = mode;
    try {
      const data = m === "analyze" ? await analyzeImage(file) : await compareImages(design, impl);
      setResults((r) => ({ ...r, [m]: data }));
      setIgnored((i) => ({ ...i, [m]: [] }));
      const entry = {
        id: Date.now(), date: new Date().toISOString(), mode: m, names,
        files: m === "analyze" ? { file } : { design, impl }, result: data,
      };
      setHistory((h) => {
        const next = [entry, ...h].slice(0, HISTORY_MAX);
        save("history", next);
        return next;
      });
      requestAnimationFrame(() => mainRef.current?.scrollIntoView({ behavior: "smooth", block: "start" }));
    } catch (e) {
      setError(e.message === "Failed to fetch" || /Error 50\d/.test(e.message)
        ? "No se pudo conectar con el backend. ¿Está corriendo uvicorn en el puerto 8000?" : e.message);
    } finally {
      setLoading(false);
    }
  };

  const restore = (h) => {
    setMode(h.mode);
    if (h.mode === "analyze") setFileRaw(h.files.file);
    else { setDesignRaw(h.files.design); setImplRaw(h.files.impl); }
    setResults((r) => ({ ...r, [h.mode]: h.result }));
    setIgnored((i) => ({ ...i, [h.mode]: [] }));
    setSelected(null);
    setShowHistory(false);
  };

  const ignoredIds = ignored[mode] ?? [];
  const visible = useMemo(
    () => (result ? result.findings.filter((f) => !ignoredIds.includes(f.id)) : []),
    [result, ignoredIds],
  );
  const ignore = (id) => {
    setIgnored((i) => ({ ...i, [mode]: [...(i[mode] ?? []), id] }));
    if (selected === id) setSelected(null);
  };
  const locate = (id) => { setSelected(id); setFocusTick((t) => t + 1); };

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand"><Logo /> QA Tester</div>
        <div className="topbar-right">
          <StatusPill status={status} />
          <button type="button" className="btn outline" onClick={() => setShowHistory(true)}>
            <History /> Historial
          </button>
        </div>
      </header>

      <main className="page">
        <div className="hero">
          <h1>Revisa tu interfaz</h1>
          <p className="muted">
            {mode === "analyze"
              ? "Sube una captura y encuentra los micro-detalles que no se ven a simple vista."
              : "Compara tu implementación con el diseño y descubre diferencias visuales."}
          </p>
        </div>

        <div className="tabs" role="tablist">
          <button type="button" role="tab" aria-selected={mode === "analyze"} className={mode === "analyze" ? "active" : ""}
            onClick={() => switchMode("analyze")}><Camera /> Revisar captura</button>
          <button type="button" role="tab" aria-selected={mode === "compare"} className={mode === "compare" ? "active" : ""}
            onClick={() => switchMode("compare")}><Compare /> Comparar diseños</button>
        </div>

        <section className="card filebar">
          {mode === "analyze" ? (
            <FileSlot label="Captura de tu página" file={file} onFile={setFile} />
          ) : (
            <>
              <FileSlot label="Diseño de referencia" file={design} onFile={setDesign}
                hint="Exporta el frame de Figma como PNG · arrastra, clic o Ctrl+V" />
              <span className="divider" />
              <FileSlot label="Implementación" file={impl} onFile={setImpl}
                hint="Captura de tu página al mismo ancho y zoom 100%" />
            </>
          )}
          <span className="divider" />
          <button type="button" className="btn primary lg" disabled={!ready || loading} onClick={run}>
            {loading ? <span className="spinner" /> : <Play />}
            {loading ? "Analizando…" : result ? "Analizar de nuevo" : mode === "analyze" ? "Analizar" : "Comparar"}
          </button>
        </section>

        {error && <p className="error" role="alert">{error}</p>}

        <div ref={mainRef} className="results-anchor">
          {result ? (
            <div className="workspace">
              <Viewer key={`${mode}-${result.elapsed_ms}-${result.size.join("x")}`} mode={mode} result={result}
                sourceUrl={source?.url} findings={visible} selected={selected} hovered={hovered}
                onSelect={setSelected} focusTick={focusTick} />
              <FindingsPanel key={`${mode}-${result.elapsed_ms}`} mode={mode} result={result} findings={visible}
                ignoredCount={ignoredIds.length} onIgnore={ignore}
                onRestore={() => setIgnored((i) => ({ ...i, [mode]: [] }))}
                selected={selected} onSelect={setSelected} onHover={setHovered} onLocate={locate}
                onExport={() => exportReport({ mode, result, findings: visible, names })} />
            </div>
          ) : (
            <section className="card empty-state">
              {loading ? (
                <><span className="spinner big" /><p>Analizando con DINOv2… la primera vez puede tardar unos segundos.</p></>
              ) : mode === "analyze" ? (
                <>
                  <h2>¿Qué revisa?</h2>
                  <ul className="checks">
                    <li><b>Alineación y espaciado</b> con precisión de 1px.</li>
                    <li><b>Colores</b> casi duplicados y <b>contraste</b> WCAG.</li>
                    <li><b>Consistencia (IA):</b> DINOv2 agrupa los componentes iguales (botones, tarjetas…) y encuentra el que no coincide.</li>
                    <li><b>Atención (IA):</b> qué zona llama la mirada y si tu botón principal destaca.</li>
                  </ul>
                </>
              ) : (
                <>
                  <h2>¿Cómo funciona?</h2>
                  <ol className="checks">
                    <li><b>Diseño:</b> exporta el frame de Figma (o tu mockup) como PNG.</li>
                    <li><b>Implementación:</b> captura tu página <b>al mismo ancho</b> que el diseño y con zoom al 100%.
                      En Chrome: F12 → Ctrl+Shift+P → "Capture full size screenshot".</li>
                    <li>Pulsa <b>Comparar</b>: verás qué se movió y cuántos px, qué color cambió y qué contenido no coincide.</li>
                  </ol>
                </>
              )}
            </section>
          )}
        </div>

        <p className="footnote"><Info size={16} /> Las diferencias señaladas requieren revisión humana.</p>
      </main>

      <HistoryDrawer open={showHistory} items={history} onClose={() => setShowHistory(false)} onRestore={restore}
        onClear={() => { setHistory([]); save("history", null); }} />
    </div>
  );
}
