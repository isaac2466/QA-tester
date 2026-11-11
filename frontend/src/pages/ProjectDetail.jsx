import { useEffect, useMemo, useState } from "react";
import ScoreChart from "../components/ScoreChart.jsx";
import { deleteProject, getProject } from "../services/api.js";
import { go } from "../utils/route.js";
import { VIEWPORTS, viewportOf } from "../utils/versions.js";

const fmt = (d) => new Date(d).toLocaleString("es-MX", { day: "numeric", month: "short", year: "numeric", hour: "2-digit", minute: "2-digit" });
const scoreTone = (s) => (s >= 85 ? "good" : s >= 60 ? "warn" : "bad");

export function ViewportTag({ viewport }) {
  const v = VIEWPORTS[viewportOf(viewport)];
  return <span className="vp-tag"><i style={{ background: v.color }} />{v.label}</span>;
}

export default function ProjectDetail({ id, onAnalyze, onOpenVersion, onDeleted }) {
  const [project, setProject] = useState(null);
  const [error, setError] = useState(null);
  const [picked, setPicked] = useState([]);

  useEffect(() => {
    setProject(null);
    setError(null);
    getProject(id).then(setProject).catch((e) => setError(e.message));
  }, [id]);

  const analyses = useMemo(() => (project?.versions ?? []).filter((v) => v.mode === "analyze"), [project]);
  // la versión anterior del mismo tamaño, para el atajo "vs anterior"
  const previous = useMemo(() => {
    const out = {};
    const lastSeen = {};
    for (const v of [...analyses].reverse()) {
      const k = viewportOf(v.viewport);
      if (lastSeen[k]) out[v.id] = lastSeen[k];
      lastSeen[k] = v.id;
    }
    return out;
  }, [analyses]);

  if (error) return <main className="page"><a className="back" href="#/proyectos">← Proyectos</a><p className="error">{error}</p></main>;
  if (!project) return <main className="page"><p className="muted">Cargando proyecto…</p></main>;

  const toggle = (vid) => setPicked((p) => (p.includes(vid) ? p.filter((x) => x !== vid) : [...p, vid].slice(-2)));
  const compare = (a, b) => {
    // siempre de la más vieja a la más nueva
    const [older, newer] = [a, b].sort((x, y) => new Date(analyses.find((v) => v.id === x).date) - new Date(analyses.find((v) => v.id === y).date));
    go(`proyectos/${project.id}/comparar/${older}/${newer}`);
  };
  const remove = async () => {
    if (!window.confirm(`¿Eliminar el proyecto «${project.name}»? Sus análisis se quedan en tu historial.`)) return;
    try {
      await deleteProject(project.id);
      onDeleted(project.id);
      go("proyectos");
    } catch (e) {
      setError(e.message);
    }
  };

  const latest = analyses[0];
  // el cambio se mide contra el primer análisis del MISMO tamaño (móvil contra móvil…)
  const first = [...analyses].reverse().find((v) => viewportOf(v.viewport) === viewportOf(latest?.viewport)) ?? latest;

  return (
    <main className="page">
      <a className="back" href="#/proyectos">← Proyectos</a>
      <div className="page-head">
        <div className="hero">
          <h1>{project.name}</h1>
          {project.url
            ? <a className="muted project-link" href={project.url} target="_blank" rel="noreferrer">{project.url}</a>
            : <p className="muted">Proyecto con capturas subidas a mano.</p>}
        </div>
        <div className="page-actions">
          <button type="button" className="btn ghost" onClick={remove}>Eliminar</button>
          <button type="button" className="btn primary" onClick={() => onAnalyze(project)}>
            {project.url ? "Analizar ahora" : "Analizar una captura"}
          </button>
        </div>
      </div>

      {!analyses.length ? (
        <section className="card empty-state">
          <h2>Aún no hay análisis en este proyecto</h2>
          <p className="muted">
            {project.url
              ? "Pulsa «Analizar ahora»: tomamos capturas de tu página en móvil, tablet y escritorio y las revisamos."
              : "Pulsa «Analizar una captura» y elige este proyecto al guardar."}
          </p>
        </section>
      ) : (
        <>
          <section className="stats-row">
            <div className="card stat">
              <span className="stat-label">Último puntaje</span>
              <b className={`stat-value ${scoreTone(latest.score)}`}>{latest.score}<small>/100</small></b>
            </div>
            <div className="card stat">
              <span className="stat-label">Desde el primer análisis · {VIEWPORTS[viewportOf(latest.viewport)].label}</span>
              <b className="stat-value">
                {latest.score - first.score > 0 ? "+" : ""}{latest.score - first.score}<small> pts</small>
              </b>
            </div>
            <div className="card stat">
              <span className="stat-label">Análisis</span>
              <b className="stat-value">{analyses.length}</b>
            </div>
          </section>

          <section className="card">
            <h2 className="section-title">Evolución del puntaje</h2>
            {analyses.length > 1
              ? <ScoreChart points={analyses} />
              : <p className="muted">Con un segundo análisis verás aquí cómo cambia el puntaje.</p>}
          </section>

          <section className="card">
            <div className="section-head">
              <h2 className="section-title">Versiones</h2>
              <div className="pick-bar">
                <span className="muted">{picked.length < 2 ? `Elige 2 para comparar (${picked.length}/2)` : "Listo para comparar"}</span>
                <button type="button" className="btn primary sm" disabled={picked.length < 2} onClick={() => compare(...picked)}>
                  Comparar
                </button>
              </div>
            </div>
            <div className="table-wrap">
              <table className="versions">
                <thead>
                  <tr><th aria-label="Elegir" /><th>Fecha</th><th>Tamaño</th><th>Puntaje</th><th>Hallazgos</th><th aria-label="Acciones" /></tr>
                </thead>
                <tbody>
                  {analyses.map((v) => (
                    <tr key={v.id} className={picked.includes(v.id) ? "picked" : ""}>
                      <td><input type="checkbox" checked={picked.includes(v.id)} onChange={() => toggle(v.id)}
                        aria-label={`Elegir versión del ${fmt(v.date)}`} /></td>
                      <td>{fmt(v.date)}</td>
                      <td><ViewportTag viewport={v.viewport} /></td>
                      <td><b className={`score-text ${scoreTone(v.score)}`}>{v.score}</b></td>
                      <td>
                        <span className="sev-counts">
                          {v.counts.alta > 0 && <span className="dot-alta">{v.counts.alta}</span>}
                          {v.counts.media > 0 && <span className="dot-media">{v.counts.media}</span>}
                          {v.counts.baja > 0 && <span className="dot-baja">{v.counts.baja}</span>}
                          {!v.counts.total && <span className="muted">ninguno</span>}
                        </span>
                      </td>
                      <td className="row-actions">
                        {previous[v.id] && (
                          <button type="button" className="link" onClick={() => compare(previous[v.id], v.id)}>vs anterior</button>
                        )}
                        <button type="button" className="link" onClick={() => onOpenVersion(v.id)}>Abrir</button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>
        </>
      )}
    </main>
  );
}
