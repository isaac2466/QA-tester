import { useState } from "react";
import { Sparkline } from "../components/ScoreChart.jsx";
import { createProject } from "../services/api.js";
import { go } from "../utils/route.js";

const scoreTone = (s) => (s >= 85 ? "good" : s >= 60 ? "warn" : "bad");

function NewProject({ onCreated, onCancel }) {
  const [name, setName] = useState("");
  const [url, setUrl] = useState("");
  const [error, setError] = useState(null);
  const [saving, setSaving] = useState(false);
  const submit = async (e) => {
    e.preventDefault();
    setSaving(true);
    setError(null);
    try {
      onCreated(await createProject(name, url || null));
    } catch (err) {
      setError(err.message);
    } finally {
      setSaving(false);
    }
  };
  return (
    <form className="card new-project" onSubmit={submit}>
      <h2>Nuevo proyecto</h2>
      <label className="field">
        <span>Nombre</span>
        <input value={name} onChange={(e) => setName(e.target.value)} placeholder="Landing de verano" maxLength={120} required autoFocus />
      </label>
      <label className="field">
        <span>Dirección de la página <small className="muted">(opcional)</small></span>
        <input value={url} onChange={(e) => setUrl(e.target.value)} placeholder="https://misitio.com" inputMode="url" />
      </label>
      {error && <p className="error" role="alert">{error}</p>}
      <div className="form-actions">
        <button type="button" className="btn ghost" onClick={onCancel}>Cancelar</button>
        <button type="submit" className="btn primary" disabled={saving}>{saving ? "Creando…" : "Crear proyecto"}</button>
      </div>
    </form>
  );
}

export default function ProjectsPage({ projects, error, onCreated }) {
  const [creating, setCreating] = useState(false);
  return (
    <main className="page">
      <div className="page-head">
        <div className="hero">
          <h1>Proyectos</h1>
          <p className="muted">Agrupa los análisis de una misma página y mira si cada versión mejora.</p>
        </div>
        {!creating && <button type="button" className="btn primary" onClick={() => setCreating(true)}>+ Nuevo proyecto</button>}
      </div>

      {creating && (
        <NewProject onCancel={() => setCreating(false)}
          onCreated={(p) => { setCreating(false); onCreated(p); go(`proyectos/${p.id}`); }} />
      )}
      {error && <p className="error" role="alert">{error}</p>}

      {projects == null ? (
        <p className="muted">Cargando proyectos…</p>
      ) : !projects.length && !creating ? (
        <section className="card empty-state">
          <h2>Todavía no tienes proyectos</h2>
          <p className="muted">Crea uno, analiza tu página y vuelve a analizarla después de cada cambio: verás cómo evoluciona el puntaje
            y qué errores se arreglaron.</p>
          <button type="button" className="btn primary" onClick={() => setCreating(true)}>Crear mi primer proyecto</button>
        </section>
      ) : (
        <ul className="project-grid">
          {projects.map((p) => {
            const last = [...p.points].sort((a, b) => new Date(b.date) - new Date(a.date))[0];
            return (
              <li key={p.id}>
                <a className="card project-card" href={`#/proyectos/${p.id}`}>
                  <span className="project-card-top">
                    <span className="project-name">{p.name}</span>
                    {last && <span className={`score-chip ${scoreTone(last.score)}`}>{last.score}</span>}
                  </span>
                  <span className="muted project-url">{p.url ? p.url.replace(/^https?:\/\//, "") : "Sin dirección · capturas subidas"}</span>
                  <span className="project-card-foot">
                    <span className="muted">
                      {p.points.length ? `${p.points.length} ${p.points.length === 1 ? "análisis" : "análisis"} · último ${new Date(last.date).toLocaleDateString("es-MX", { day: "numeric", month: "short" })}` : "Sin análisis todavía"}
                    </span>
                    <Sparkline points={p.points} />
                  </span>
                </a>
              </li>
            );
          })}
        </ul>
      )}
    </main>
  );
}
