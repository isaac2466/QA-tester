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

const VERDICT = {
  real: ["real", "Confirmado", "Qwen revisó la zona y cree que sí es un error real"],
  falso_positivo: ["fp", "Posible falso positivo", "Qwen cree que la herramienta se equivocó aquí"],
  dudoso: ["doubt", "Dudoso", "Qwen no alcanzó a verlo con claridad"],
};

const seconds = (ms) => (ms < 1000 ? `${ms} ms` : `${(ms / 1000).toFixed(1)} s`);

/** Una fila por motor de IA: nombre a la izquierda, qué hizo a la derecha. */
function Engine({ name, children, tone }) {
  return (
    <div className={`engine-row ${tone ?? ""}`}>
      <span className="engine-name">{name}</span>
      <div className="engine-info">{children}</div>
    </div>
  );
}

function QwenRow({ qwen, qwenStatus, explained, falsePositives, onExplain, onIgnoreMany }) {
  const running = qwen && !qwen.error && qwen.done < qwen.total;
  if (running) {
    const pct = qwen.total ? Math.round((qwen.done / qwen.total) * 100) : 0;
    const mins = Math.max(1, Math.round(((qwen.total - qwen.done) * 40) / 60)); // ~40 s por hallazgo en CPU
    return (
      <Engine name="Qwen" tone="busy">
        <span>Explicando… <b>{qwen.done}/{qwen.total}</b> <span className="muted">· faltan ~{mins} min</span>
          {qwen.failed > 0 && <span className="warn-text"> · {qwen.failed} fallaron</span>}</span>
        <span className="progress" role="progressbar" aria-valuenow={pct} aria-valuemin={0} aria-valuemax={100}>
          <i style={{ width: `${Math.max(pct, 3)}%` }} />
        </span>
      </Engine>
    );
  }
  if (qwen?.error && explained === 0) {
    return (
      <Engine name="Qwen" tone="warn">
        <span className="warn-text">No pudo explicar: {qwen.error}</span>
        {onExplain && <button type="button" className="link" onClick={onExplain}>Reintentar</button>}
      </Engine>
    );
  }
  if (explained > 0) {
    return (
      <Engine name="Qwen" tone={qwen?.error ? "warn" : ""}>
        <span>
          Explicó <b>{explained}</b>
          {falsePositives.length > 0
            ? <> · <b className="fp-text">{falsePositives.length} {falsePositives.length === 1 ? "posible falso positivo" : "posibles falsos positivos"}</b></>
            : " · sin falsos positivos"}
        </span>
        {falsePositives.length > 0 && (
          <button type="button" className="link" onClick={() => onIgnoreMany(falsePositives)}>Ocultarlos</button>
        )}
        {qwen?.error && (
          <span className="engine-sub">
            <span className="warn-text">{qwen.error}</span>
            {onExplain && <button type="button" className="link" onClick={onExplain}>Reintentar esos</button>}
          </span>
        )}
      </Engine>
    );
  }
  if (!qwenStatus?.available) {
    return (
      <Engine name="Qwen" tone="off">
        <span className="muted" title={qwenStatus?.error}>No disponible · abre Ollama</span>
      </Engine>
    );
  }
  return (
    <Engine name="Qwen" tone="off">
      <span className="muted">Revisión rápida: sin explicaciones.</span>
      {onExplain && <button type="button" className="link" onClick={onExplain}>Pedir a Qwen</button>}
    </Engine>
  );
}

export default function FindingsPanel({
  mode, result, findings, ignoredCount, onIgnore, onIgnoreMany, onConfirm, marked = {}, discarded = [], learning,
  onRestore, selected, onSelect, onHover, onLocate, onExport,
  qwen, qwenStatus, onExplain,
}) {
  const explanations = result.qwen ?? {};
  const explained = Object.keys(explanations).length;
  const falsePositives = findings.filter((f) => explanations[f.id]?.veredicto === "falso_positivo").map((f) => f.id);
  const [sev, setSev] = useState("all");
  const [cat, setCat] = useState("all");
  const [showDiscarded, setShowDiscarded] = useState(false);
  const bySev = countBy(findings, "severity");
  const byCat = countBy(findings, "category");
  const shown = findings.filter((f) => (sev === "all" || f.severity === sev) && (cat === "all" || f.category === cat));
  const chips = [["all", "Todas", findings.length], ...["alta", "media", "baja"].filter((s) => bySev[s]).map((s) => [s, SEVERITY[s], bySev[s]])];
  const byDino = findings.filter((f) => f.source === "dinov2").length;

  return (
    <aside className="card panel">
      <header className="panel-head">
        <div>
          <h2><Flag size={20} /> {mode === "analyze" ? "Hallazgos" : "Diferencias"}</h2>
          <p className="muted">
            {findings.length ? `${findings.length} ${findings.length === 1 ? "zona" : "zonas"} para revisar` : "Todo se ve bien"}
            {` · ${seconds(result.elapsed_ms)}`}
          </p>
        </div>
        <Score value={scoreOf(findings)} />
      </header>

      <section className="engines" aria-label="Motores de IA">
        {result.dino?.loaded ? (
          <Engine name="DINOv2">
            <span>
              {findings.length > 0 && <><b>{byDino}/{findings.length}</b> hallazgos</>}
              {result.stats?.matched != null && <> · emparejó <b>{result.stats.matched}/{result.stats.design_elements}</b></>}
              {result.stats?.semantic_similarity != null && <> · parecido <b>{(result.stats.semantic_similarity * 100).toFixed(0)}%</b></>}
              {result.components?.length > 0 && (
                <span className="muted" title={result.components.map((c) => `${c.count} ${c.label}`).join(", ")}>
                  {" "}· {result.components.length} {result.components.length === 1 ? "grupo" : "grupos"} de componentes
                </span>
              )}
            </span>
          </Engine>
        ) : (
          <Engine name="DINOv2" tone="warn"><span className="warn-text" title={result.dino?.error}>No cargado: análisis sin IA</span></Engine>
        )}
        {mode === "analyze" && (
          <Engine name="SAM 2.1" tone={result.sam?.loaded ? "" : "off"}>
            <span>
              {result.fixes_applied?.length
                ? <><b>{result.fixes_applied.length}</b> correcciones en la vista corregida</>
                : <span className="muted">Sin correcciones automáticas</span>}
              {!result.sam?.loaded && <span className="muted"> · sin siluetas</span>}
            </span>
          </Engine>
        )}
        {mode === "analyze" && (
          <QwenRow qwen={qwen} qwenStatus={qwenStatus} explained={explained} falsePositives={falsePositives}
            onExplain={onExplain} onIgnoreMany={onIgnoreMany} />
        )}
      </section>

      <div className="filters">
        {chips.map(([k, label, n]) => (
          <button key={k} type="button" className={`chip ${k !== "all" ? `dot-${k}` : ""} ${sev === k ? "active" : ""}`}
            onClick={() => setSev(k)}>
            {label} {k !== "all" && <span className="count">{n}</span>}
          </button>
        ))}
        {Object.keys(byCat).length > 1 && (
          <select value={cat} onChange={(e) => setCat(e.target.value)} aria-label="Filtrar por categoría">
            <option value="all">Categoría</option>
            {Object.entries(byCat).map(([c, n]) => <option key={c} value={c}>{CATEGORY[c] ?? c} ({n})</option>)}
          </select>
        )}
      </div>

      <ol className="list">
        {shown.map((f, idx) => {
          const open = f.id === selected;
          const q = explanations[f.id];
          const [vClass, vLabel, vTip] = (q && VERDICT[q.veredicto]) || [];
          return (
            <li key={f.id} style={{ "--i": Math.min(idx, 12) }} className={`item sev-${f.severity} ${open ? "open" : ""} ${vClass === "fp" ? "is-fp" : ""}`}
              onMouseEnter={() => onHover(f.id)} onMouseLeave={() => onHover(null)}>
              <button type="button" className="item-main" onClick={() => onSelect(open ? null : f.id)} aria-expanded={open}>
                <span className="num">{f.id}</span>
                <span className="item-text">
                  <span className="item-top">
                    <span className="item-title">{f.title}</span>
                    <span className={`pill sev-${f.severity}`}>{SEVERITY[f.severity]}</span>
                  </span>
                  <span className="tags">
                    <span className="cat">{CATEGORY[f.category] ?? f.category}</span>
                    {f.source === "dinov2" && (
                      <span className="tag ai" title="Hallazgo encontrado por la IA (DINOv2). El % es su parecido visual.">
                        <Sparkles size={11} /> DINOv2{f.confidence != null && ` ${Math.round(f.confidence * 100)}%`}
                      </span>
                    )}
                    {q && vLabel && (
                      <span className={`tag ${vClass}`} title={`${vTip}${q.razon ? `: ${q.razon}` : ""}`}>
                        <Sparkles size={11} /> {vLabel}
                      </span>
                    )}
                    {result.fixes_applied?.includes(f.id) && (
                      <span className="tag fixed icon-only" title="Corregido: se ve aplicado en la pestaña «Vista corregida»"
                        aria-label="Corregido en la vista corregida">
                        <Check size={11} />
                      </span>
                    )}
                  </span>
                  <span className="detail">{f.detail}</span>
                </span>
                <Chevron size={16} className="chev" />
              </button>

              {open && (
                <div className="item-body">
                  {q ? (
                    // Todo lo de este bloque lo escribe Qwen
                    <div className={`explain ${vClass ?? ""}`}>
                      <div className="verdict">
                        <span className="verdict-label"><Sparkles size={13} /> {vLabel ?? "Explicado por Qwen"}</span>
                        {q.razon && <p className="verdict-reason">{q.razon}</p>}
                      </div>
                      <dl className="explain-steps">
                        {(q.que_pasa || q.explicacion) && <><dt>Qué pasa</dt><dd>{q.que_pasa || q.explicacion}</dd></>}
                        {q.por_que_importa && <><dt>Por qué importa</dt><dd>{q.por_que_importa}</dd></>}
                      </dl>
                      {q.como_arreglarlo && (
                        <div className="explain-fix"><span>Cómo arreglarlo</span><p>{q.como_arreglarlo}</p></div>
                      )}
                      <p className="explain-measure"><span>Medición</span>{f.detail}</p>
                    </div>
                  ) : (
                    // Sin Qwen: sólo lo que midió la herramienta
                    <div className="explain">
                      <dl className="explain-steps"><dt>Lo que se midió</dt><dd>{f.detail}</dd></dl>
                      <div className="explain-fix"><span>Sugerencia</span><p>{f.suggestion}</p></div>
                    </div>
                  )}
                  <div className="item-actions">
                    {f.box && (
                      <button type="button" className="btn outline sm" onClick={() => onLocate(f.id)}>
                        <Target size={15} /> Ver zona
                      </button>
                    )}
                    {marked[f.id] ? (
                      <span className="fixed-tag"><Check size={12} /> Marcado como error · la IA aprende</span>
                    ) : (
                      <button type="button" className="btn ghost sm" onClick={() => onConfirm(f.id)} title="Enseña a la IA que esto SÍ es un error">
                        <Check size={15} /> Sí es error
                      </button>
                    )}
                    <button type="button" className="btn ghost sm" onClick={() => onIgnore(f.id)} title="Enseña a la IA que esto NO es un error">
                      <EyeOff size={15} /> No es error
                    </button>
                  </div>
                </div>
              )}
            </li>
          );
        })}
        {!shown.length && <li className="empty-list">{findings.length ? "Nada con este filtro." : "Sin hallazgos 🎉"}</li>}
      </ol>

      {(discarded.length > 0 || learning) && (
        <div className="panel-notes">
          {discarded.length > 0 && (
            <button type="button" className="link" onClick={() => setShowDiscarded((v) => !v)} aria-expanded={showDiscarded}>
              {discarded.length} descartados por la IA · {showDiscarded ? "ocultar" : "ver"}
            </button>
          )}
          {showDiscarded && (
            <ul>{discarded.map((f) => <li key={f.id}><b>{f.title}</b> — {f.verdict_reason}</li>)}</ul>
          )}
        </div>
      )}

      <footer className="panel-foot">
        {ignoredCount > 0 ? (
          <button type="button" className="link" onClick={onRestore}>
            {ignoredCount} {ignoredCount === 1 ? "ignorado" : "ignorados"} · mostrar
          </button>
        ) : <span />}
        <button type="button" className="btn outline sm" onClick={onExport}>
          <Download size={15} /> Exportar reporte
        </button>
      </footer>
    </aside>
  );
}
