import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import FileSlot, { useImageInfo } from "./components/FileSlot.jsx";
import FindingsPanel from "./components/FindingsPanel.jsx";
import HistoryDrawer from "./components/HistoryDrawer.jsx";
import Viewer from "./components/Viewer.jsx";
import { Camera, Compare, History, Info, Logo, Play } from "./components/icons.jsx";
import { analyzeImage, compareImages, explainFindings, health, sendFeedback } from "./services/api.js";
import { exportReport } from "./utils/report.js";
import { load, save } from "./utils/storage.js";

const HISTORY_MAX = 12;
const QWEN_BATCH = 3; // hallazgos por llamada a Qwen: lotes chicos = ~2 min cada uno y si uno falla se pierde poco
const QWEN_RETRIES = 1; // reintentos por lote antes de saltarlo

function StatusPill({ status }) {
  const [tone, text] = !status ? ["partial", "Conectando…"]
    : status.status === "down" ? ["down", "Backend apagado · reintentando"]
    : status.dino?.loaded ? ["ok", `Motor listo · DINOv2${status.sam?.loaded ? " + SAM 2.1" : ""}${status.qwen?.available ? " + Qwen" : ""} (${status.dino.device})`]
    : ["partial", "IA no disponible"];
  const tip = [status?.dino?.error, status?.qwen && !status.qwen.available && `Qwen: ${status.qwen.error}`].filter(Boolean).join(" · ");
  return <span className={`status ${tone}`} title={tip}><i />{text}</span>;
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
  const [qwen, setQwen] = useState(null); // progreso de Qwen: { done, total, error }
  // Revisión "detallada" = DINOv2 + SAM y además Qwen. Se recuerda en este navegador.
  const [detailed, setDetailedRaw] = useState(() => {
    try { return localStorage.getItem("qa-detailed") === "1"; } catch { return false; }
  });
  const setDetailed = (v) => {
    setDetailedRaw(v);
    try { localStorage.setItem("qa-detailed", v ? "1" : "0"); } catch { /* sin almacenamiento: sólo esta sesión */ }
  };
  const qwenReady = !!status?.qwen?.available;
  const qwenOn = detailed && qwenReady;
  const qwenRun = useRef(0);
  const qwenAbort = useRef(null);
  // Corta la revisión de Qwen en curso (el backend también la cancela en Ollama al ver la conexión cerrada)
  const stopQwen = () => {
    qwenRun.current += 1;
    qwenAbort.current?.abort();
    qwenAbort.current = null;
    setQwen(null);
  };
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
    if (m === "analyze") stopQwen();
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

  // Qwen explica los hallazgos y opina si son falsos positivos. Va por lotes y en segundo plano:
  // los resultados de DINOv2 se ven al instante y cada tarjeta se completa cuando Qwen responde.
  // Si un lote falla se reintenta una vez y, si vuelve a fallar, se salta y se sigue con los demás.
  const runQwen = async (img, data, historyId) => {
    stopQwen();
    const myRun = qwenRun.current;
    const ctrl = new AbortController();
    qwenAbort.current = ctrl;
    // sólo los que se muestran: los descartados por la IA no necesitan explicación
    const pending = data.findings.filter((f) => f.verdict !== "descartado" && !data.qwen?.[f.id]);
    const acc = { ...(data.qwen ?? {}) };
    let failed = 0;
    let lastError = null;
    setQwen({ done: 0, total: pending.length, failed: 0, error: null });
    for (let i = 0; i < pending.length; i += QWEN_BATCH) {
      const batch = pending.slice(i, i + QWEN_BATCH);
      for (let attempt = 0; attempt <= QWEN_RETRIES; attempt++) {
        try {
          const res = await explainFindings(img, batch, ctrl.signal);
          if (qwenRun.current !== myRun) return; // hubo un análisis nuevo: descartamos esta respuesta
          Object.assign(acc, res.explanations);
          // si la respuesta se cortó, Qwen pudo devolver sólo algunos: los demás cuentan como pendientes
          const missing = batch.filter((f) => !res.explanations[f.id]).length;
          if (missing) { failed += missing; lastError = "Qwen no alcanzó a terminar su respuesta"; }
          setResults((r) => (r.analyze?.elapsed_ms === data.elapsed_ms
            ? { ...r, analyze: { ...r.analyze, qwen: { ...acc } } } : r));
          break;
        } catch (e) {
          if (qwenRun.current !== myRun || ctrl.signal.aborted) return;
          lastError = e.message === "Failed to fetch" ? "se perdió la conexión con el backend" : e.message;
          if (attempt === QWEN_RETRIES) failed += batch.length;
        }
      }
      setQwen({ done: Math.min(i + QWEN_BATCH, pending.length), total: pending.length, failed, error: null });
    }
    if (qwenRun.current !== myRun) return;
    qwenAbort.current = null;
    if (failed) setQwen((q) => ({ ...q, error: `${failed} sin explicación: ${lastError}` }));
    if (historyId) {
      setHistory((h) => {
        const next = h.map((e) => (e.id === historyId ? { ...e, result: { ...e.result, qwen: { ...acc } } } : e));
        save("history", next);
        return next;
      });
    }
  };

  const run = async () => {
    setLoading(true);
    setError(null);
    setSelected(null);
    const m = mode;
    try {
      const data = m === "analyze" ? await analyzeImage(file) : await compareImages(design, impl);
      if (m === "analyze") stopQwen();
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
      if (m === "analyze" && qwenOn && data.findings.some((f) => f.verdict !== "descartado")) runQwen(file, data, entry.id);
    } catch (e) {
      setError(e.message === "Failed to fetch" || /Error 50\d/.test(e.message)
        ? "No se pudo conectar con el backend. ¿Está corriendo uvicorn en el puerto 8000?" : e.message);
    } finally {
      setLoading(false);
    }
  };

  const restore = (h) => {
    if (h.mode === "analyze") stopQwen();
    setMode(h.mode);
    if (h.mode === "analyze") setFileRaw(h.files.file);
    else { setDesignRaw(h.files.design); setImplRaw(h.files.impl); }
    setResults((r) => ({ ...r, [h.mode]: h.result }));
    setIgnored((i) => ({ ...i, [h.mode]: [] }));
    setSelected(null);
    setShowHistory(false);
  };

  const ignoredIds = ignored[mode] ?? [];
  // lo que la IA descartó (foto, degradado, parecido a algo que marcaste) no se dibuja ni cuenta
  const visible = useMemo(
    () => (result ? result.findings.filter((f) => !ignoredIds.includes(f.id) && f.verdict !== "descartado") : []),
    [result, ignoredIds],
  );
  const discarded = result ? result.findings.filter((f) => f.verdict === "descartado") : [];
  const [learning, setLearning] = useState(null);
  const [marked, setMarked] = useState({}); // id → "es_error"
  const teach = (id, label) => {
    if (!result?.analysis_id) return;
    sendFeedback(result.analysis_id, id, label).then(setLearning).catch((e) => setError(`No se guardó tu marca: ${e.message}`));
  };
  const ignore = (id) => {
    setIgnored((i) => ({ ...i, [mode]: [...(i[mode] ?? []), id] }));
    if (selected === id) setSelected(null);
    teach(id, "no_es_error");
  };
  const confirmError = (id) => { setMarked((m) => ({ ...m, [id]: "es_error" })); teach(id, "es_error"); };
  const ignoreMany = (ids) => setIgnored((i) => ({ ...i, [mode]: [...new Set([...(i[mode] ?? []), ...ids])] }));
  const locate = (id) => { setSelected(id); setFocusTick((t) => t + 1); };

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand"><Logo /> Lynx</div>
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
            <FileSlot label="Captura de tu página" file={file} onFile={setFile} onClear={() => setFile(null)} />
          ) : (
            <>
              <FileSlot label="Diseño de referencia" file={design} onFile={setDesign} onClear={() => setDesign(null)}
                hint="Exporta el frame de Figma como PNG · arrastra, clic o Ctrl+V" />
              <span className="divider" />
              <FileSlot label="Implementación" file={impl} onFile={setImpl} onClear={() => setImpl(null)}
                hint="Captura de tu página al mismo ancho y zoom 100%" />
            </>
          )}
          <span className="divider" />
          {mode === "analyze" && (
            <>
              <div className="engine" role="radiogroup" aria-label="Tipo de revisión">
                <span className="engine-label">Tipo de revisión</span>
                <div className="engine-options">
                  <button type="button" role="radio" aria-checked={!qwenOn} className={!qwenOn ? "active" : ""}
                    onClick={() => setDetailed(false)} title="Sólo DINOv2 + SAM 2.1: resultados en segundos">
                    <b>Rápida</b><small>DINOv2 + SAM · segundos</small>
                  </button>
                  <button type="button" role="radio" aria-checked={qwenOn} className={qwenOn ? "active" : ""}
                    disabled={!qwenReady} onClick={() => setDetailed(true)}
                    title={qwenReady
                      ? "Además, Qwen3-VL explica cada hallazgo y marca los posibles falsos positivos. En CPU tarda ~1 min por hallazgo."
                      : `Qwen no disponible: ${status?.qwen?.error ?? "revisa que Ollama esté abierto"}`}>
                    <b>Detallada</b><small>+ Qwen · ~1 min por hallazgo</small>
                  </button>
                </div>
              </div>
              <span className="divider" />
            </>
          )}
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
                ignoredCount={ignoredIds.length} onIgnore={ignore} onIgnoreMany={ignoreMany}
                onConfirm={confirmError} marked={marked} discarded={discarded} learning={learning ?? result.learning}
                qwen={mode === "analyze" ? qwen : null} qwenStatus={status?.qwen}
                onExplain={mode === "analyze" && file ? () => runQwen(file, result) : null}
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
                    <li><b>Explicación (Qwen):</b> explica cada hallazgo y te avisa cuál parece un falso positivo.</li>
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
