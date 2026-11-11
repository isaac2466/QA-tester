import { useEffect, useMemo, useRef, useState } from "react";
import { VIEWPORTS, viewportOf } from "../utils/versions.js";

const H = 220;
const PAD = { top: 16, right: 100, bottom: 28, left: 34 };

const fmt = (d) => new Date(d).toLocaleString("es-MX", { day: "numeric", month: "short", hour: "2-digit", minute: "2-digit" });

/**
 * Puntaje de cada análisis a lo largo del tiempo, una línea por tamaño (móvil, tablet, escritorio…).
 * El eje X es el orden de los análisis (no el tiempo real), para que varios análisis el mismo día no se encimen.
 * Los análisis tomados juntos (mismo minuto, p. ej. una URL en 3 tamaños) comparten columna.
 */
export default function ScoreChart({ points }) {
  const [hover, setHover] = useState(null);
  // el SVG usa el ancho real en px: así el texto no se agranda ni se encoge con la pantalla
  const box = useRef(null);
  const [W, setW] = useState(640);
  useEffect(() => {
    const el = box.current;
    if (!el) return undefined;
    const ro = new ResizeObserver(([e]) => setW(Math.max(280, Math.round(e.contentRect.width))));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  const { runs, series } = useMemo(() => {
    const sorted = [...points].filter((p) => p.score != null).sort((a, b) => new Date(a.date) - new Date(b.date));
    const runs = [];
    for (const p of sorted) {
      const t = new Date(p.date).getTime();
      const last = runs[runs.length - 1];
      if (last && t - last.t < 120000 && !last.items.some((x) => viewportOf(x.viewport) === viewportOf(p.viewport))) {
        last.items.push(p);
      } else runs.push({ t, date: p.date, items: [p] });
    }
    const series = {};
    runs.forEach((r, i) => r.items.forEach((p) => {
      const v = viewportOf(p.viewport);
      (series[v] ??= []).push({ i, score: p.score, date: p.date });
    }));
    return { runs, series };
  }, [points]);

  if (!runs.length) return null;
  const n = runs.length;
  const iw = W - PAD.left - PAD.right;
  const ih = H - PAD.top - PAD.bottom;
  const x = (i) => PAD.left + (n === 1 ? iw / 2 : (i / (n - 1)) * iw);
  const y = (s) => PAD.top + ih - (s / 100) * ih;
  const keys = Object.keys(VIEWPORTS).filter((k) => series[k]); // orden fijo: el color sigue al tamaño
  const hovered = hover != null ? runs[hover] : null;
  // en pantallas angostas las fechas completas no caben: si todo fue el mismo día basta la hora
  const sameDay = new Date(runs[0].date).toDateString() === new Date(runs[n - 1].date).toDateString();
  const axisFmt = (d) => (W >= 520 ? fmt(d) : sameDay
    ? new Date(d).toLocaleTimeString("es-MX", { hour: "2-digit", minute: "2-digit" })
    : new Date(d).toLocaleDateString("es-MX", { day: "numeric", month: "short" }));
  // el tooltip se pega al borde en vez de salirse de la tarjeta
  const tipPct = hovered ? (x(hover) / W) * 100 : 0;
  const tipShift = tipPct < 20 ? "0%" : tipPct > 80 ? "-100%" : "-50%";

  // etiquetas al final de cada línea, separadas para que no se encimen
  const ends = keys.map((k) => ({ k, s: series[k][series[k].length - 1] })).sort((a, b) => y(a.s.score) - y(b.s.score));
  for (let j = 1; j < ends.length; j++) {
    ends[j].ly = Math.max(y(ends[j].s.score), (ends[j - 1].ly ?? y(ends[j - 1].s.score)) + 14);
  }

  const onMove = (e) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const px = ((e.clientX - rect.left) / rect.width) * W;
    let best = 0;
    for (let i = 1; i < n; i++) if (Math.abs(x(i) - px) < Math.abs(x(best) - px)) best = i;
    setHover(best);
  };

  return (
    <figure className="chart">
      {keys.length > 1 && (
        <figcaption className="chart-legend">
          {keys.map((k) => <span key={k}><i style={{ background: VIEWPORTS[k].color }} />{VIEWPORTS[k].label}</span>)}
        </figcaption>
      )}
      <div className="chart-box" ref={box}>
        <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Evolución del puntaje por análisis"
          onMouseMove={onMove} onMouseLeave={() => setHover(null)}>
          {[0, 50, 100].map((t) => (
            <g key={t}>
              <line x1={PAD.left} x2={W - PAD.right} y1={y(t)} y2={y(t)} className="chart-grid" />
              <text x={PAD.left - 8} y={y(t) + 4} className="chart-axis" textAnchor="end">{t}</text>
            </g>
          ))}
          <text x={PAD.left} y={H - 8} className="chart-axis">{axisFmt(runs[0].date)}</text>
          {n > 1 && <text x={W - PAD.right} y={H - 8} className="chart-axis" textAnchor="end">{axisFmt(runs[n - 1].date)}</text>}
          {hovered && <line x1={x(hover)} x2={x(hover)} y1={PAD.top} y2={PAD.top + ih} className="chart-cross" />}
          {keys.map((k) => (
            <g key={k} style={{ color: VIEWPORTS[k].color }}>
              {series[k].length > 1 && (
                <polyline fill="none" stroke="currentColor" strokeWidth="2" strokeLinejoin="round" strokeLinecap="round"
                  points={series[k].map((p) => `${x(p.i)},${y(p.score)}`).join(" ")} />
              )}
              {series[k].map((p) => (
                <circle key={p.i} cx={x(p.i)} cy={y(p.score)} r={hover === p.i ? 5.5 : 4} fill="currentColor"
                  className="chart-dot" />
              ))}
            </g>
          ))}
          {ends.map(({ k, s, ly }) => (
            <text key={k} x={x(s.i) + 10} y={(ly ?? y(s.score)) + 4} className="chart-end">
              {VIEWPORTS[k].label} {s.score}
            </text>
          ))}
        </svg>
        {hovered && (
          <div className="chart-tip" style={{ left: `${tipPct}%`, transform: `translateX(${tipShift})` }}>
            <b>{fmt(hovered.date)}</b>
            {keys.filter((k) => hovered.items.some((p) => viewportOf(p.viewport) === k)).map((k) => (
              <span key={k}><i style={{ background: VIEWPORTS[k].color }} />{VIEWPORTS[k].label}
                <b>{hovered.items.find((p) => viewportOf(p.viewport) === k).score}</b></span>
            ))}
          </div>
        )}
      </div>
    </figure>
  );
}

/** Mini gráfica para las tarjetas de proyecto: una sola línea con todos los análisis. */
export function Sparkline({ points }) {
  const pts = [...points].filter((p) => p.score != null).sort((a, b) => new Date(a.date) - new Date(b.date)).slice(-12);
  if (pts.length < 2) return null;
  const w = 120;
  const h = 32;
  const px = (i) => 3 + (i / (pts.length - 1)) * (w - 6);
  const py = (s) => 3 + (h - 6) - (s / 100) * (h - 6);
  return (
    <svg className="spark" viewBox={`0 0 ${w} ${h}`} aria-hidden="true">
      <polyline fill="none" stroke="currentColor" strokeWidth="2" strokeLinejoin="round" strokeLinecap="round"
        points={pts.map((p, i) => `${px(i)},${py(p.score)}`).join(" ")} />
      <circle cx={px(pts.length - 1)} cy={py(pts[pts.length - 1].score)} r="3" fill="currentColor" />
    </svg>
  );
}
