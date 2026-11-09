import { useState } from "react";
import { CATEGORY, SEVERITY, countBy, scoreOf } from "../utils/findings.js";
import { Check, Chevron, Download, EyeOff, Flag, Sparkles, Target } from "./icons.jsx";

function Score({ value }) {
  const tone = value >= 85 ? "good" : value >= 60 ? "warn" : "bad";
  return (
    <div className={`score ${tone}`} title="Puntuación: 100 menos los puntos de cada hallazgo (alta 8, media 4, baja 1)">
      <b>{value}</b><span>/100</span>
    </div>
  );
}

export default function FindingsPanel({
  mode, result, findings, ignoredCount, onIgnore, onConfirm, marked = {}, discarded = [], learning, onRestore, selected, onSelect, onHover, onLocate, onExport,
}) {
  const [sev, setSev] = useState("all");
  const [cat, setCat] = useState("all");
  const [showDiscarded, setShowDiscarded] = useState(false);
  const bySev = countBy(findings, "severity");
  const byCat = countBy(findings, "category");
  const shown = findings.filter((f) => (sev === "all" || f.severity === sev) && (cat === "all" || f.category === cat));
  const chips = [["all", "Todas", findings.length], ...["alta", "media", "baja"].filter((s) => bySev[s]).map((s) => [s, SEVERITY[s], bySev[s]])];

  return (
    <aside className="card panel">
      <header className="panel-head">
        <div>
          <h2><Flag size={22} /> {mode === "analyze" ? "Hallazgos" : "Diferencias detectadas"}</h2>
          <p className="muted">
            {findings.length ? `${findings.length} ${findings.length === 1 ? "zona" : "zonas"} para revisar` : "Todo se ve bien"}
            {` · ${result.elapsed_ms} ms`}
          </p>
        </div>
        <Score value={scoreOf(findings)} />
      </header>

      {(result.components?.length > 0 || result.stats?.semantic_similarity != null || !result.dino?.loaded) && (
        <div className="ai-box">
          <Sparkles size={16} />
          <div>
            {findings.length > 0 && (
              <p><b>{findings.filter((f) => f.source === "dinov2").length} de {findings.length}</b> hallazgos los encontró DINOv2.</p>
            )}
            {result.fixes_applied?.length > 0 && (
              <p>La <b>vista corregida</b> aplica {result.fixes_applied.length} cambios
                {result.sam?.loaded ? " con las siluetas de SAM 2.1" : " (sin SAM: recortes rectangulares)"}.</p>
            )}
            {result.stats?.matched != null && (
              <p>DINOv2 emparejó <b>{result.stats.matched}</b> de {result.stats.design_elements} elementos del diseño con la implementación.</p>
            )}
            {result.components?.length > 0 && (
              <p>DINOv2 reconoció: {result.components.map((c) => `${c.count} ${c.label}`).join(" · ")}</p>
            )}
            {result.stats?.semantic_similarity != null && (
              <p>Parecido semántico (DINOv2): <b>{(result.stats.semantic_similarity * 100).toFixed(1)}%</b> ·
                píxeles distintos: <b>{(result.stats.changed_pixels * 100).toFixed(2)}%</b></p>
            )}
            {!result.dino?.loaded && <p className="warn-text">DINOv2 no está cargado; se omitió el análisis con IA. {result.dino?.error}</p>}
          </div>
        </div>
      )}

      <div className="filters">
        {chips.map(([k, label, n]) => (
          <button key={k} type="button" className={`chip ${k !== "all" ? `dot-${k}` : ""} ${sev === k ? "active" : ""}`}
            onClick={() => setSev(k)}>
            {label} <span className="count">{n}</span>
          </button>
        ))}
        {Object.keys(byCat).length > 1 && (
          <select value={cat} onChange={(e) => setCat(e.target.value)} aria-label="Filtrar por categoría">
            <option value="all">Todas las categorías</option>
            {Object.entries(byCat).map(([c, n]) => <option key={c} value={c}>{CATEGORY[c] ?? c} ({n})</option>)}
          </select>
        )}
      </div>

      <ol className="list">
        {shown.map((f) => {
          const open = f.id === selected;
          return (
            <li key={f.id} className={`item sev-${f.severity} ${open ? "open" : ""}`}
              onMouseEnter={() => onHover(f.id)} onMouseLeave={() => onHover(null)}>
              <button type="button" className="item-main" onClick={() => onSelect(open ? null : f.id)} aria-expanded={open}>
                <span className="num">{f.id}</span>
                <span className="item-text">
                  <span className="item-title">{f.title}</span>
                  <span className="tags">
                    <span className="cat">{CATEGORY[f.category] ?? f.category}</span>
                    {f.source === "dinov2" && (
                      <span className="ai-tag" title="Hallazgo encontrado por la IA (DINOv2). El % es su parecido visual.">
                        <Sparkles size={12} /> DINOv2{f.confidence != null && ` · ${Math.round(f.confidence * 100)}%`}
                      </span>
                    )}
                    {result.fixes_applied?.includes(f.id) && (
                      <span className="fixed-tag" title="Esta corrección se ve aplicada en la pestaña «Vista corregida»">
                        <Check size={12} /> En vista corregida
                      </span>
                    )}
                  </span>
                  <span className="detail">{f.detail}</span>
                </span>
                <span className={`pill sev-${f.severity}`}>{SEVERITY[f.severity]}</span>
                <Chevron size={18} className="chev" />
              </button>
              {open && <p className="fix">→ {f.suggestion}</p>}
              <div className="item-actions">
                {f.box && (
                  <button type="button" className="btn outline sm" onClick={() => onLocate(f.id)}>
                    <Target size={16} /> Ver zona
                  </button>
                )}
                {marked[f.id] ? (
                  <span className="fixed-tag"><Check size={12} /> Marcado como error · la IA aprende</span>
                ) : (
                  <button type="button" className="btn ghost sm" onClick={() => onConfirm(f.id)} title="Enseña a la IA que esto SÍ es un error">
                    <Check size={16} /> Sí es error
                  </button>
                )}
                <button type="button" className="btn ghost sm" onClick={() => onIgnore(f.id)} title="Enseña a la IA que esto NO es un error">
                  <EyeOff size={16} /> No es error
                </button>
              </div>
            </li>
          );
        })}
        {!shown.length && <li className="empty-list">{findings.length ? "Nada con este filtro." : "Sin hallazgos 🎉"}</li>}
      </ol>

      {discarded.length > 0 && (
        <div className="discarded">
          <button type="button" className="link" onClick={() => setShowDiscarded((v) => !v)}>
            {discarded.length} {discarded.length === 1 ? "posible error descartado" : "posibles errores descartados"} por la IA · {showDiscarded ? "ocultar" : "ver por qué"}
          </button>
          {showDiscarded && (
            <ul>{discarded.map((f) => <li key={f.id}><b>{f.title}</b> — {f.verdict_reason}</li>)}</ul>
          )}
        </div>
      )}
      {learning && (
        <p className="muted small">
          🧠 La IA ha aprendido de {learning.marcas} {learning.marcas === 1 ? "marca" : "marcas"} tuyas
          {learning.red_decide ? " · la red neuronal ya decide" : ` · la red neuronal decide a partir de ${learning.minimo_para_decidir ?? 30} marcas`}
        </p>
      )}

      {ignoredCount > 0 && (
        <button type="button" className="link" onClick={onRestore}>
          {ignoredCount} {ignoredCount === 1 ? "hallazgo ignorado" : "hallazgos ignorados"} · mostrar de nuevo
        </button>
      )}

      <button type="button" className="btn outline wide" onClick={onExport}>
        <Download size={18} /> Exportar reporte
      </button>
    </aside>
  );
}
