import { useEffect, useMemo, useState } from "react";
import { getHistoryEntry } from "../services/api.js";
import { CATEGORY, SEVERITY } from "../utils/findings.js";
import { diffVersions } from "../utils/versions.js";
import { ViewportTag } from "./ProjectDetail.jsx";

const fmt = (d) => new Date(d).toLocaleString("es-MX", { day: "numeric", month: "short", hour: "2-digit", minute: "2-digit" });

function FindingRow({ f }) {
  return (
    <li className={`diff-item sev-${f.severity}`}>
      <span className="diff-title">{f.title}</span>
      <span className="diff-meta">
        <span className="cat">{CATEGORY[f.category] ?? f.category}</span>
        <span className={`pill sev-${f.severity}`}>{SEVERITY[f.severity]}</span>
      </span>
    </li>
  );
}

function Group({ tone, title, hint, items, empty }) {
  return (
    <section className={`card diff-group ${tone}`}>
      <header>
        <h2>{title} <span className="count">{items.length}</span></h2>
        <p className="muted">{hint}</p>
      </header>
      {items.length ? <ul>{items.map((f) => <FindingRow key={f.id} f={f} />)}</ul> : <p className="muted small">{empty}</p>}
    </section>
  );
}

export default function VersionCompare({ projectId, a, b }) {
  const [entries, setEntries] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    setEntries(null);
    Promise.all([getHistoryEntry(a), getHistoryEntry(b)]).then(setEntries).catch((e) => setError(e.message));
  }, [a, b]);

  const diff = useMemo(() => entries && diffVersions(entries[0].result, entries[1].result), [entries]);
  const back = <a className="back" href={`#/proyectos/${projectId}`}>← Volver al proyecto</a>;

  if (error) return <main className="page">{back}<p className="error">{error}</p></main>;
  if (!entries) return <main className="page">{back}<p className="muted">Cargando versiones…</p></main>;

  const [before, after] = entries;
  const delta = diff.scoreAfter - diff.scoreBefore;
  const sameSize = (before.viewport ?? "captura") === (after.viewport ?? "captura");

  return (
    <main className="page">
      {back}
      <div className="hero">
        <h1>Comparar versiones</h1>
        <p className="muted">Qué se arregló, qué sigue pendiente y qué apareció de nuevo.</p>
      </div>
      {!sameSize && (
        <p className="notice">Estás comparando tamaños distintos (<ViewportTag viewport={before.viewport} /> contra{" "}
          <ViewportTag viewport={after.viewport} />): es normal que muchos hallazgos no coincidan.</p>
      )}

      <section className="card score-delta">
        <div>
          <span className="stat-label">Antes · {fmt(before.date)}</span>
          <b className="stat-value">{diff.scoreBefore}</b>
        </div>
        <span className={`delta ${delta > 0 ? "up" : delta < 0 ? "down" : ""}`} aria-label={`Cambio de ${delta} puntos`}>
          {delta > 0 ? "▲ +" : delta < 0 ? "▼ " : "= "}{delta} pts
        </span>
        <div>
          <span className="stat-label">Después · {fmt(after.date)}</span>
          <b className="stat-value">{diff.scoreAfter}</b>
        </div>
      </section>

      <div className="diff-groups">
        <Group tone="fixed" title="Arreglados" hint="Estaban antes y ya no aparecen." items={diff.fixed}
          empty="Ninguno de los errores anteriores desapareció." />
        <Group tone="still" title="Siguen" hint="Aparecen en las dos versiones." items={diff.still.map((s) => s.after)}
          empty="No queda ninguno pendiente." />
        <Group tone="added" title="Nuevos" hint="No estaban en la versión anterior." items={diff.added}
          empty="No apareció ningún error nuevo. 🎉" />
      </div>

      <section className="card">
        <h2 className="section-title">Las dos versiones</h2>
        <div className="side-by-side">
          {[["Antes", before], ["Después", after]].map(([label, e]) => (
            <figure key={label}>
              <figcaption><b>{label}</b> <ViewportTag viewport={e.viewport} /> <span className="muted">{fmt(e.date)}</span></figcaption>
              <img src={e.result.images?.annotated} alt={`Captura ${label.toLowerCase()} con sus hallazgos marcados`} />
            </figure>
          ))}
        </div>
      </section>
    </main>
  );
}
