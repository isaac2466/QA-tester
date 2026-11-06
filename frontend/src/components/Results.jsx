import { useState } from "react";

const CATEGORY = {
  alineacion: "Alineación",
  espaciado: "Espaciado",
  color: "Color",
  contraste: "Contraste",
  atencion: "Atención visual",
  diferencia: "Diferencia con el diseño",
  tipografia: "Tipografía",
  jerarquia: "Jerarquía visual",
  consistencia: "Consistencia",
};

const VERDICT = {
  confirmado: { label: "✓ Confirmado por Qwen3-VL", cls: "v-ok" },
  sin_verificar: { label: "Medición sin verificar", cls: "v-unk" },
  descartado: { label: "✕ Descartado por Qwen3-VL", cls: "v-no" },
};

function ScoreRing({ score }) {
  const tone = score >= 85 ? "good" : score >= 60 ? "warn" : "bad";
  return (
    <div className={`score ${tone}`}>
      <span className="score-num">{score}</span>
      <span className="score-label">/ 100</span>
    </div>
  );
}

function FindingItem({ f, active, setActive }) {
  const v = VERDICT[f.verdict ?? "sin_verificar"];
  return (
    <li
      className={`finding sev-${f.severity} ${active?.id === f.id ? "active" : ""} ${f.verdict === "descartado" ? "discarded" : ""}`}
      onMouseEnter={() => setActive(f)}
      onMouseLeave={() => setActive(null)}
    >
      <div className="finding-top">
        <span className="badge">{f.id}</span>
        <span className="cat">{CATEGORY[f.category] ?? f.category}</span>
        <span className="sev">{f.severity}</span>
      </div>
      <h3>{f.title}</h3>
      <p>{f.detail}</p>
      <p className="fix">→ {f.suggestion}</p>
      <div className="verdict-row">
        <span className={`verdict ${f.source === "ia" ? "v-ia" : v.cls}`}>
          {f.source === "ia" ? "★ Observación de Qwen3-VL" : v.label}
        </span>
        {f.verdict_reason && <span className="muted reason">{f.verdict_reason}</span>}
      </div>
    </li>
  );
}

export default function Results({ data, views }) {
  const available = views.filter((v) => data.images[v.key]);
  const [view, setView] = useState(available[0]?.key);
  const [active, setActive] = useState(null);
  const [showDiscarded, setShowDiscarded] = useState(false);

  const visible = data.findings.filter((f) => f.verdict !== "descartado");
  const discarded = data.findings.filter((f) => f.verdict === "descartado");
  const counts = visible.reduce((a, f) => ({ ...a, [f.severity]: (a[f.severity] ?? 0) + 1 }), {});

  return (
    <section className="results">
      <header className="results-head">
        <ScoreRing score={data.score} />
        <div>
          <h2>{visible.length ? `${visible.length} hallazgos` : "Sin hallazgos 🎉"}</h2>
          <p className="muted">
            {counts.alta ?? 0} altos · {counts.media ?? 0} medios · {counts.baja ?? 0} bajos · {(data.elapsed_ms / 1000).toFixed(1)} s
            {discarded.length > 0 && ` · ${discarded.length} falsos positivos descartados por Qwen3-VL`}
          </p>
          {data.vlm && !data.vlm.available && (
            <p className="warn-box">
              Sin revisión de Qwen3-VL: los hallazgos son mediciones automáticas y pueden incluir falsos positivos. {data.vlm.error}
            </p>
          )}
          {data.stats?.semantic_similarity != null && (
            <p className="muted">
              Similitud semántica (DINOv2): <b>{(data.stats.semantic_similarity * 100).toFixed(1)}%</b> ·
              píxeles distintos: <b>{(data.stats.changed_pixels * 100).toFixed(2)}%</b>
            </p>
          )}
        </div>
      </header>

      {data.summary && (
        <div className="summary">
          <strong>Opinión de Qwen3-VL:</strong> {data.summary}
        </div>
      )}

      <div className="results-body">
        <div className="viewer">
          <div className="tabs small">
            {available.map((v) => (
              <button key={v.key} className={view === v.key ? "active" : ""} onClick={() => setView(v.key)}>
                {v.label}
              </button>
            ))}
          </div>
          <div className="canvas">
            <img src={data.images[view]} alt={view} />
            {active?.box && view === "annotated" && (
              <div
                className="highlight"
                style={{
                  left: `${(active.box[0] / data.size[0]) * 100}%`,
                  top: `${(active.box[1] / data.size[1]) * 100}%`,
                  width: `${(active.box[2] / data.size[0]) * 100}%`,
                  height: `${(active.box[3] / data.size[1]) * 100}%`,
                }}
              />
            )}
          </div>
          {data.palette && (
            <div className="palette">
              {data.palette.map((c) => (
                <div key={c.hex} className="swatch" title={`${(c.share * 100).toFixed(1)}%`}>
                  <span style={{ background: c.hex }} />
                  <code>{c.hex}</code>
                </div>
              ))}
            </div>
          )}
        </div>

        <div>
          <ol className="findings">
            {visible.map((f) => <FindingItem key={f.id} f={f} active={active} setActive={setActive} />)}
          </ol>
          {discarded.length > 0 && (
            <>
              <button className="ghost discard-toggle" onClick={() => setShowDiscarded((s) => !s)}>
                {showDiscarded ? "Ocultar" : "Ver"} {discarded.length} descartados por Qwen3-VL
              </button>
              {showDiscarded && (
                <ol className="findings">
                  {discarded.map((f) => <FindingItem key={f.id} f={f} active={active} setActive={setActive} />)}
                </ol>
              )}
            </>
          )}
        </div>
      </div>
    </section>
  );
}
