import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import FileSlot, { useImageInfo } from "./components/FileSlot.jsx";
import FindingsPanel from "./components/FindingsPanel.jsx";
import HistoryDrawer from "./components/HistoryDrawer.jsx";
import Viewer from "./components/Viewer.jsx";
import { Camera, Compare, History, Info, Logo, Play } from "./components/icons.jsx";
import ProjectDetail from "./pages/ProjectDetail.jsx";
import ProjectsPage from "./pages/ProjectsPage.jsx";
import VersionCompare from "./pages/VersionCompare.jsx";
import {
  analyzeImage, captureUrl, clearHistory, compareImages, explainFindings, getHistory, getHistoryEntry, getProjects,
  health, login, logout, me, register, saveHistory, sendFeedback,
} from "./services/api.js";
import { go, useRoute } from "./utils/route.js";
import { exportReport } from "./utils/report.js";
import { load, save } from "./utils/storage.js";

const HISTORY_MAX = 12;
const QWEN_BATCH = 3; // hallazgos por llamada a Qwen: lotes chicos = ~2 min cada uno y si uno falla se pierde poco
const QWEN_RETRIES = 1; // reintentos por lote antes de saltarlo
const SIZES = [["movil", "Móvil", "390 px"], ["tablet", "Tablet", "820 px"], ["escritorio", "Escritorio", "1440 px"]];

function AuthScreen({ onAuthenticated }) {
  const [mode, setMode] = useState("login");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);

  const submit = async (event) => {
    event.preventDefault();
    setLoading(true);
    setError(null);
    try {
      const response = mode === "login" ? await login(email, password) : await register(email, password);
      localStorage.setItem("qa-token", response.token);
      onAuthenticated(response.user);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  const isLogin = mode === "login";
  return (
    <main className="auth-page">
      <section className="auth-card proof" aria-labelledby="auth-title">
        <div className="brand auth-brand"><Logo /> Lynx</div>
        <div className="auth-heading">
          <h1 id="auth-title">{isLogin ? "Entra a tu cuenta" : "Crea tu cuenta"}</h1>
          <p className="muted">{isLogin ? "Tus análisis e historial te esperan." : "Guarda tus análisis y enséñale a la IA qué es un error."}</p>
        </div>
        <form className="auth-form" onSubmit={submit}>
          <label className="auth-field">
            <span>Correo</span>
            <input type="email" autoComplete="email" placeholder="tu@correo.com" value={email}
              onChange={(e) => setEmail(e.target.value)} required />
          </label>
          <label className="auth-field">
            <span>Contraseña</span>
            <input type="password" autoComplete={isLogin ? "current-password" : "new-password"} placeholder="Mínimo 8 caracteres"
              value={password} onChange={(e) => setPassword(e.target.value)} minLength={8} required />
          </label>
          {error && <p className="error" role="alert">{error}</p>}
          <button type="submit" className="btn primary lg auth-submit" disabled={loading}>
            {loading ? <><span className="spinner" /> {isLogin ? "Entrando…" : "Creando cuenta…"}</> : isLogin ? "Entrar" : "Crear cuenta"}
          </button>
        </form>
        <p className="auth-switch">
          {isLogin ? "¿Primera vez aquí?" : "¿Ya tienes cuenta?"}{" "}
          <button type="button" className="link" onClick={() => { setMode(isLogin ? "register" : "login"); setError(null); }}>
            {isLogin ? "Crea una cuenta" : "Entra"}
          </button>
        </p>
      </section>
    </main>
  );
}

/** Mientras se revisa la sesión: el esqueleto de la app, para que se note que está cargando. */
function AppSkeleton() {
  const [slow, setSlow] = useState(false);
  useEffect(() => {
    const t = setTimeout(() => setSlow(true), 4000);
    return () => clearTimeout(t);
  }, []);
  return (
    <div className="app app-skeleton" role="status" aria-label="Cargando Lynx">
      <header className="topbar">
        <span className="sk" style={{ width: 96, height: 26 }} />
        <span className="sk" style={{ width: 220, height: 22 }} />
      </header>
      <main className="page">
        <span className="sk" style={{ width: "min(420px, 80%)", height: 40 }} />
        <span className="sk" style={{ width: "min(560px, 95%)", height: 18, marginTop: 12 }} />
        <span className="sk" style={{ width: 340, height: 44, marginTop: 28 }} />
        <span className="sk" style={{ width: "100%", height: 96, marginTop: 14 }} />
        <span className="sk" style={{ width: "100%", height: 220, marginTop: 20 }} />
        
      </main>
    </div>
  );
}

function StatusPill({ status }) {
  const [tone, text] = !status ? ["partial", "Conectando…"]
    : status.status === "down" ? ["down", "Sin conexión con el servidor"]
    : status.dino?.loaded ? ["ok", "Listo para analizar"]
    : ["partial", "Iniciando…"];
  const tip = status?.dino?.loaded
    ? `DINOv2${status.sam?.loaded ? " + SAM 2.1" : ""}${status.qwen?.available ? " + Qwen" : ""} en ${status.dino.device}`
    : status?.dino?.error ?? "";
  return <span className={`status ${tone}`} title={tip}><i />{text}</span>;
}

export default function App() {
  const [user, setUser] = useState(null);
  const [authReady, setAuthReady] = useState(false);
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
  const route = useRoute();
  const [projects, setProjects] = useState(null);
  const [projectsError, setProjectsError] = useState(null);
  const [projectId, setProjectId] = useState(""); // proyecto donde se guarda el próximo análisis ("" = ninguno)
  const [origin, setOrigin] = useState("file"); // "file" = subir captura · "url" = capturar una página
  const [url, setUrl] = useState("");
  const [sizes, setSizes] = useState(["movil", "tablet", "escritorio"]);
  const [shots, setShots] = useState([]); // análisis de una URL: uno por tamaño { size, label, file, result, historyId }
  const [activeShot, setActiveShot] = useState(0);
  const [progress, setProgress] = useState(null); // texto del paso actual mientras se analiza una URL
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
    load("session").then((s) => {
      if (s) {
        setMode(s.mode ?? "analyze");
        setFileRaw(s.file ?? null);
        setDesignRaw(s.design ?? null);
        setImplRaw(s.impl ?? null);
        setResults(s.results ?? {});
        setIgnored(s.ignored ?? {});
        setOrigin(s.origin ?? "file");
        setUrl(s.url ?? "");
        setSizes(s.sizes?.length ? s.sizes : ["movil", "tablet", "escritorio"]);
        setProjectId(s.projectId ?? "");
        setShots(s.shots ?? []);
        setActiveShot(s.activeShot ?? 0);
      }
      setHydrated(true);
    });
  }, []);

  useEffect(() => {
    const token = localStorage.getItem("qa-token");
    if (!token) {
      load("history").then((h) => setHistory(h ?? [])).finally(() => setAuthReady(true));
      return;
    }
    Promise.all([me(), getHistory()])
      .then(([session, remoteHistory]) => {
        setUser(session.user);
        setHistory(remoteHistory);
      })
      .catch(() => {
        localStorage.removeItem("qa-token");
        setUser(null);
      })
      .finally(() => setAuthReady(true));
  }, []);

  useEffect(() => {
    if (hydrated) save("session", { mode, file, design, impl, results, ignored, origin, url, sizes, projectId, shots, activeShot });
  }, [hydrated, mode, file, design, impl, results, ignored, origin, url, sizes, projectId, shots, activeShot]);

  const loadProjects = useCallback(() => {
    getProjects().then((p) => { setProjects(p); setProjectsError(null); })
      .catch((e) => { setProjects((p) => p ?? []); setProjectsError(`No se pudieron cargar los proyectos: ${e.message}`); });
  }, []);
  useEffect(() => { if (user) loadProjects(); }, [user, loadProjects]);
  // al volver a la lista, refresca (puede haber análisis nuevos)
  const onProjectList = route[0] === "proyectos" && !route[1];
  useEffect(() => { if (user && onProjectList) loadProjects(); }, [user, onProjectList, loadProjects]);

  // Cambiar una imagen invalida el resultado anterior de esa pestaña
  const clearResult = (m) => {
    if (m === "analyze") stopQwen();
    setResults((r) => ({ ...r, [m]: undefined }));
    setIgnored((i) => ({ ...i, [m]: [] }));
    setSelected(null);
  };
  const setFile = useCallback((f) => { setFileRaw(f); setShots([]); clearResult("analyze"); }, []);
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

  const ready = mode === "analyze" ? (origin === "url" ? url.trim() && sizes.length : file) : design && impl;
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
          setShots((list) => list.map((x) => (x.result.analysis_id === data.analysis_id
            ? { ...x, result: { ...x.result, qwen: { ...acc } } } : x)));
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

  // Guarda en MySQL (y en el historial de este navegador). Si falla, el análisis igual se ve.
  const remember = async (m, entryNames, data, files, extra = {}) => {
    let entry = { id: Date.now(), date: new Date().toISOString(), mode: m, names: entryNames, files, result: data, ...extra };
    try {
      const saved = await saveHistory({
        mode: m, names: entryNames, result: data, project_id: projectId ? Number(projectId) : null, ...extra,
      });
      entry = { ...entry, id: saved.id, date: saved.date };
    } catch (e) {
      setError(`El análisis terminó, pero no se guardó en MySQL: ${e.message}`);
    }
    setHistory((h) => {
      const next = [entry, ...h].slice(0, HISTORY_MAX);
      save("history", next);
      return next;
    });
    return entry;
  };

  const showShot = (list, i) => {
    stopQwen();
    setActiveShot(i);
    setFileRaw(list[i].file);
    setResults((r) => ({ ...r, analyze: list[i].result }));
    setIgnored((ig) => ({ ...ig, analyze: [] }));
    setSelected(null);
  };

  // Desde una URL: captura cada tamaño, lo analiza y lo guarda. Se ve el primero en cuanto está listo.
  const runUrl = async () => {
    setLoading(true);
    setError(null);
    setSelected(null);
    stopQwen();
    try {
      setProgress(`Abriendo la página en ${sizes.length} ${sizes.length === 1 ? "tamaño" : "tamaños"}…`);
      const cap = await captureUrl(url, SIZES.map(([k]) => k).filter((k) => sizes.includes(k)));
      setUrl(cap.url);
      const list = [];
      for (const [i, shot] of cap.shots.entries()) {
        setProgress(`Analizando ${shot.label.toLowerCase()} (${i + 1} de ${cap.shots.length})…`);
        const data = await analyzeImage(shot.file);
        const entry = await remember("analyze", [shot.file.name], data, { file: shot.file },
          { viewport: shot.size, source_url: cap.url });
        list.push({ size: shot.size, label: shot.label, truncated: shot.truncated, file: shot.file, result: data, historyId: entry.id });
        setShots([...list]);
        if (i === 0) {
          showShot(list, 0);
          requestAnimationFrame(() => mainRef.current?.scrollIntoView({ behavior: "smooth", block: "start" }));
        }
      }
      if (projectId) loadProjects();
      const first = list[0];
      if (qwenOn && first?.result.findings.some((f) => f.verdict !== "descartado")) runQwen(first.file, first.result, first.historyId);
    } catch (e) {
      setError(e.message === "Failed to fetch" ? "No se pudo conectar con el backend. ¿Está corriendo uvicorn en el puerto 8000?" : e.message);
    } finally {
      setLoading(false);
      setProgress(null);
    }
  };

  const run = async () => {
    if (mode === "analyze" && origin === "url") return runUrl();
    setLoading(true);
    setError(null);
    setSelected(null);
    const m = mode;
    try {
      const data = m === "analyze" ? await analyzeImage(file) : await compareImages(design, impl);
      if (m === "analyze") stopQwen();
      setResults((r) => ({ ...r, [m]: data }));
      setIgnored((i) => ({ ...i, [m]: [] }));
      if (m === "analyze") setShots([]);
      const entry = await remember(m, names, data, m === "analyze" ? { file } : { design, impl });
      if (projectId) loadProjects();
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
    if (h.mode === "analyze") { stopQwen(); setShots([]); }
    setMode(h.mode);
    if (h.mode === "analyze") setFileRaw(h.files.file);
    else { setDesignRaw(h.files.design); setImplRaw(h.files.impl); }
    setResults((r) => ({ ...r, [h.mode]: h.result }));
    setIgnored((i) => ({ ...i, [h.mode]: [] }));
    setSelected(null);
    setShowHistory(false);
  };

  // "Abrir" una versión desde un proyecto: la carga en la página de revisión
  const openVersion = async (id) => {
    try {
      restore(await getHistoryEntry(id));
    } catch (e) {
      setError(e.message);
    }
    go("revisar");
  };
  // "Analizar ahora" desde un proyecto: deja todo listo en la página de revisión
  const analyzeProject = (p) => {
    setMode("analyze");
    setProjectId(String(p.id));
    if (p.url) { setOrigin("url"); setUrl(p.url); } else setOrigin("file");
    setError(null);
    go("revisar");
  };

  const clearStoredHistory = async () => {
    try {
      await clearHistory();
      setHistory([]);
      save("history", null);
    } catch (e) {
      setError(`No se pudo borrar el historial: ${e.message}`);
    }
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

  if (!authReady) return <AppSkeleton />;
  if (!user) return <AuthScreen onAuthenticated={(nextUser) => {
    setUser(nextUser);
    getHistory().then(setHistory).catch((e) => setError(e.message));
  }} />;

  const page = route[0] === "proyectos" ? "proyectos" : "revisar";
  const toggleSize = (k) => setSizes((s) => (s.includes(k) ? s.filter((x) => x !== k) : [...s, k]));
  const projectName = projects?.find((p) => String(p.id) === String(projectId))?.name;

  let content;
  if (page === "proyectos" && route[2] === "comparar" && route[3] && route[4]) {
    content = <VersionCompare projectId={route[1]} a={route[3]} b={route[4]} />;
  } else if (page === "proyectos" && route[1]) {
    content = (
      <ProjectDetail id={route[1]} onAnalyze={analyzeProject} onOpenVersion={openVersion}
        onDeleted={(id) => { setProjects((ps) => ps?.filter((p) => p.id !== id)); if (String(projectId) === String(id)) setProjectId(""); }} />
    );
  } else if (page === "proyectos") {
    content = <ProjectsPage projects={projects} error={projectsError} onCreated={(p) => setProjects((ps) => [p, ...(ps ?? [])])} />;
  } else {
    content = (
      <main className="page">
        <div className="hero">
          <h1>Revisa tu interfaz</h1>
          <p className="muted">
            {mode === "compare"
              ? "Compara tu implementación con el diseño y descubre diferencias visuales."
              : origin === "url"
                ? "Pega la dirección de tu página: la revisamos en móvil, tablet y escritorio."
                : "Sube una captura y encuentra los micro-detalles que no se ven a simple vista."}
          </p>
        </div>

        <div className="tabs-row">
          <div className="tabs" role="tablist">
            <button type="button" role="tab" aria-selected={mode === "analyze"} className={mode === "analyze" ? "active" : ""}
              onClick={() => switchMode("analyze")}><Camera /> Revisar página</button>
            <button type="button" role="tab" aria-selected={mode === "compare"} className={mode === "compare" ? "active" : ""}
              onClick={() => switchMode("compare")}><Compare /> Comparar diseños</button>
          </div>
          <label className="save-to">
            <span>Guardar en</span>
            <select value={projectId} onChange={(e) => {
              if (e.target.value === "__new") { go("proyectos"); return; }
              setProjectId(e.target.value);
            }}>
              <option value="">Sin proyecto</option>
              {(projects ?? []).map((p) => <option key={p.id} value={p.id}>{p.name}</option>)}
              <option value="__new">+ Nuevo proyecto…</option>
            </select>
          </label>
        </div>

        <section className="card filebar">
          {mode === "analyze" ? (
            <div className="source">
              <div className="segmented small" role="radiogroup" aria-label="Qué revisar">
                <button type="button" role="radio" aria-checked={origin === "file"} className={origin === "file" ? "active" : ""}
                  onClick={() => setOrigin("file")}>Subir captura</button>
                <button type="button" role="radio" aria-checked={origin === "url"} className={origin === "url" ? "active" : ""}
                  onClick={() => setOrigin("url")}>Desde una URL</button>
              </div>
              {origin === "file" ? (
                <FileSlot label="Captura de tu página" file={file} onFile={setFile} onClear={() => setFile(null)} />
              ) : (
                <div className="url-input">
                  <input type="url" inputMode="url" placeholder="https://misitio.com" value={url} aria-label="Dirección de la página"
                    onChange={(e) => setUrl(e.target.value)} onKeyDown={(e) => e.key === "Enter" && ready && !loading && run()} />
                  <div className="size-chips" role="group" aria-label="Tamaños de pantalla">
                    {SIZES.map(([k, label, px]) => (
                      <button key={k} type="button" aria-pressed={sizes.includes(k)} className={`chip ${sizes.includes(k) ? "active" : ""}`}
                        onClick={() => toggleSize(k)}>{label} <small>{px}</small></button>
                    ))}
                  </div>
                </div>
              )}
            </div>
          ) : (
            <>
              <FileSlot label="Diseño de referencia" file={design} onFile={setDesign} onClear={() => setDesign(null)}
                hint="Exporta el frame de Figma como PNG · arrastra, clic o Ctrl+V" />
              <span className="divider" />
              <FileSlot label="Implementación" file={impl} onFile={setImpl} onClear={() => setImpl(null)}
                hint="Captura de tu página al mismo ancho y zoom 100%" />
            </>
          )}
          <div className="filebar-actions">
          {mode === "analyze" && (
            <>
              <div className="engine" role="radiogroup" aria-label="Tipo de revisión">
                <div className="engine-options">
                  <button type="button" role="radio" aria-checked={!qwenOn} className={!qwenOn ? "active" : ""}
                    onClick={() => setDetailed(false)} title="Sólo DINOv2 + SAM 2.1: resultados en segundos">
                    <b>Rápida</b><small>Unos segundos</small>
                  </button>
                  <button type="button" role="radio" aria-checked={qwenOn} className={qwenOn ? "active" : ""}
                    disabled={!qwenReady} onClick={() => setDetailed(true)}
                    title={qwenReady
                      ? "Además, Qwen3-VL explica cada hallazgo y marca los posibles falsos positivos. En CPU tarda ~1 min por hallazgo."
                      : `Qwen no disponible: ${status?.qwen?.error ?? "revisa que Ollama esté abierto"}`}>
                    <b>Detallada</b><small>Explica cada hallazgo · ~1 min</small>
                  </button>
                </div>
              </div>
            </>
          )}
          <button type="button" className="btn primary lg" disabled={!ready || loading} onClick={run}>
            {loading ? <span className="spinner" /> : <Play />}
            {loading ? "Analizando…" : result ? "Analizar de nuevo" : mode === "analyze" ? "Analizar" : "Comparar"}
          </button>
          </div>
        </section>

        {error && <p className="error" role="alert">{error}</p>}

        <div ref={mainRef} className="results-anchor">
          {mode === "analyze" && shots.length > 0 && (
            <div className="shot-bar" role="tablist" aria-label="Tamaño de pantalla">
              {shots.map((sh, i) => (
                <button key={sh.size} type="button" role="tab" aria-selected={i === activeShot}
                  className={`shot ${i === activeShot ? "active" : ""}`} onClick={() => showShot(shots, i)}>
                  <span>{sh.label}</span>
                  <b className={sh.result.score >= 85 ? "good" : sh.result.score >= 60 ? "warn" : "bad"}>{sh.result.score}</b>
                </button>
              ))}
              {loading && progress && <span className="shot-progress"><span className="spinner" /> {progress}</span>}
              {projectName && !loading && (
                <a className="link shot-project" href={`#/proyectos/${projectId}`}>Guardado en «{projectName}» →</a>
              )}
            </div>
          )}
          {mode === "analyze" && shots[activeShot]?.truncated && (
            <p className="notice">La página es muy larga: se revisaron los primeros 4000 px.</p>
          )}
          {result ? (
            <div className="workspace enter">
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
            <section className={`card empty-state ${loading ? "is-loading" : ""}`}>
              {loading ? (
                <div className="skeleton-ws" role="status" aria-label="Analizando la captura">
                  <div className="sk-viewer"><span className="sk sk-bar" /><span className="sk sk-canvas" /></div>
                  <div className="sk-panel">
                    <span className="sk sk-title" />
                    {[0, 1, 2, 3].map((i) => <span key={i} className="sk sk-row" style={{ "--i": i }} />)}
                  </div>
                  <p className="sk-note">{progress ?? "DINOv2 está revisando cada elemento…"}</p>
                </div>
              ) : mode === "analyze" ? (
                <>
                  <h2>Esto es lo que revisamos</h2>
                  <div className="checks-grid">
                    <div><h3>Alineación y espaciado</h3><p>Elementos que se salen de línea por pocos píxeles y separaciones que no coinciden.</p></div>
                    <div><h3>Componentes consistentes</h3><p>Botones, tarjetas y etiquetas iguales que cambian de tamaño, color o altura sin razón.</p></div>
                    <div><h3>Contraste y color</h3><p>Textos difíciles de leer y demasiados colores compitiendo por atención.</p></div>
                  </div>
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
      </main>
    );
  }

  return (
    <div className="app">
      <header className="topbar">
        <div className="topbar-left">
          <a className="brand" href="#/revisar"><Logo /> Lynx</a>
          <nav className="nav" aria-label="Secciones">
            <a href="#/revisar" className={page === "revisar" ? "active" : ""} aria-current={page === "revisar" ? "page" : undefined}>Revisar</a>
            <a href="#/proyectos" className={page === "proyectos" ? "active" : ""} aria-current={page === "proyectos" ? "page" : undefined}>Proyectos</a>
          </nav>
        </div>
        <div className="topbar-right">
          <StatusPill status={status} />
          <button type="button" className="btn ghost" onClick={() => setShowHistory(true)}>
            <History /> Historial
          </button>
          <span className="user-chip" title={user.email}>{user.email}</span>
          <button type="button" className="btn ghost" onClick={async () => {
            try { await logout(); } finally { localStorage.removeItem("qa-token"); setUser(null); }
          }}>Salir</button>
        </div>
      </header>

      {content}

      <HistoryDrawer open={showHistory} items={history} onClose={() => setShowHistory(false)}
        onRestore={(h) => { restore(h); go("revisar"); }} onClear={clearStoredHistory} />
    </div>
  );
}
