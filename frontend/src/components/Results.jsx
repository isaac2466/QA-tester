import { useState } from "react";
import { motion } from "framer-motion";

const CATEGORY = {
  alineacion: "Alineación",
  espaciado: "Espaciado",
  color: "Color",
  contraste: "Contraste",
  atencion: "Atención visual",
  diferencia: "Diferencia con el diseño",
};

function ScoreRing({ score }) {
  const tone = score >= 85 ? "good" : score >= 60 ? "warn" : "bad";
  return (
    <motion.div className={`score ${tone}`} initial={{ scale: 0.85, opacity: 0 }} animate={{ scale: 1, opacity: 1 }} transition={{ type: "spring", stiffness: 300, damping: 25 }}>
      <span className="score-num">{score}</span>
      <span className="score-label">/ 100</span>
    </motion.div>
  );
}

export default function Results({ data, views }) {
  const available = views.filter((v) => data.images[v.key]);
  const [view, setView] = useState(available[0]?.key);
  const [active, setActive] = useState(null);
  const counts = data.findings.reduce((a, f) => ({ ...a, [f.severity]: (a[f.severity] ?? 0) + 1 }), {});

  return (
    <motion.section className="results" initial={{ opacity: 0, y: 18 }} animate={{ opacity: 1, y: 0 }} transition={{ type: "spring", stiffness: 300, damping: 25 }}>
      <header className="results-head">
        <ScoreRing score={data.score} />
        <div>
          <h2>{data.findings.length ? `${data.findings.length} hallazgos` : "Sin hallazgos 🎉"}</h2>
          <p className="muted">
            {counts.alta ?? 0} altos · {counts.media ?? 0} medios · {counts.baja ?? 0} bajos · {data.elapsed_ms} ms
            {data.elements_detected != null && ` · ${data.elements_detected} elementos detectados`}
          </p>
          {!data.dino?.loaded && (
            <p className="warn-box">
              DINOv2 no está cargado: se omitió el análisis de atención/semántico. Motivo: {data.dino?.error ?? "desconocido"}
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

      <div className="results-body">
        <div className="viewer">
          <div className="tabs small">
            {available.map((v) => (
              <motion.button key={v.key} className={view === v.key ? "active" : ""} whileTap={{ scale: 0.95 }} onClick={() => setView(v.key)}>
                {v.label}
              </motion.button>
            ))}
          </div>
          <motion.div className="canvas" whileTap={{ scale: 0.995 }}>
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
          </motion.div>
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

        <ol className="findings">
          {data.findings.map((f) => (
            <motion.li
              key={f.id}
              className={`finding sev-${f.severity} ${active?.id === f.id ? "active" : ""}`}
              whileTap={{ scale: 0.98 }}
              onMouseEnter={() => setActive(f)}
              onMouseLeave={() => setActive(null)}
            >
              <div className="finding-top">
                <span className="badge">{f.id}</span>
                <span className="cat">{CATEGORY[f.category] ?? f.category}</span>
                <span className={`sev`}>{f.severity}</span>
              </div>
              <h3>{f.title}</h3>
              <p>{f.detail}</p>
              <p className="fix">→ {f.suggestion}</p>
            </motion.li>
          ))}
        </ol>
      </div>
    </motion.section>
  );
}
