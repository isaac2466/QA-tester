import { useEffect, useRef, useState } from "react";
import { Columns, Diff, Eye, Layers, Sparkles, Target, Wand, Zoom } from "./icons.jsx";
import Dropdown from "./Dropdown.jsx";

const VIEWS = {
  analyze: [
    { key: "findings", label: "Hallazgos", icon: Target, boxes: true },
    { key: "fixed", label: "Vista corregida", icon: Wand, needs: "fixed" },
    { key: "heatmap", label: "Atención (IA)", icon: Eye, needs: "heatmap" },
    { key: "components", label: "Componentes (IA)", icon: Sparkles, needs: "components" },
  ],
  compare: [
    { key: "side", label: "Lado a lado", icon: Columns, boxes: true },
    { key: "overlay", label: "Superponer", icon: Layers, boxes: true },
    { key: "semantic", label: "Diferencias (IA)", icon: Diff, needs: "semantic", boxes: true },
  ],
};
const ZOOMS = [["fit", "Ajustar"], [0.5, "50 %"], [1, "100 %"], [1.5, "150 %"], [2, "200 %"]];

const pct = (v, total) => `${(v / total) * 100}%`;
const rectStyle = ([x, y, w, h], [W, H]) => ({ left: pct(x, W), top: pct(y, H), width: pct(w, W), height: pct(h, H) });

/** Recuadros numerados encima de la imagen, en % para que sigan al zoom. */
function Boxes({ findings, size, selected, hovered, onSelect }) {
  const sel = findings.find((f) => f.id === selected);
  const hoverTarget = findings.find((f) => f.id === hovered && f.box);
  const selectedTarget = findings.find((f) => f.id === selected && f.box);
  const hoverTargetId = hoverTarget?.id;
  const [lastHovered, setLastHovered] = useState(null);
  useEffect(() => {
    if (hoverTargetId != null) setLastHovered(hoverTargetId);
  }, [hoverTargetId]);
  const highlighted = hoverTarget ?? selectedTarget ?? findings.find((f) => f.id === lastHovered && f.box);

  return (
    <div className="boxes">
      {sel?.ref_box && <div className="box ref" style={rectStyle(sel.ref_box, size)}><span>referencia</span></div>}
      {highlighted && (
        <div
          className={`box-highlight sev-${highlighted.severity}`}
          style={rectStyle(highlighted.box, size)}
          aria-hidden="true"
        />
      )}
      {findings.filter((f) => f.box).map((f, _, all) => {
        // varios hallazgos sobre el mismo elemento: separa sus números en vez de encimarlos
        const stack = all.filter((o) => o.box.join() === f.box.join());
        const offset = stack.indexOf(f) * 26;
        return (
        <button
          key={f.id}
          data-i={Math.min(all.indexOf(f), 12)}
          type="button"
          data-box={f.id}
          className={`box sev-${f.severity} ${f.id === selected ? "selected" : ""} ${f.id === hovered ? "hovered" : ""}`}
          style={rectStyle(f.box, size)}
          onClick={(e) => { e.stopPropagation(); onSelect(f.id); }}
          title={f.title}
        >
          <span className="num" style={offset ? { left: offset - 12 } : undefined}>{f.id}</span>
        </button>
        );
      })}
    </div>
  );
}

function Stage({ src, size, zoom, boxes, children, label }) {
  return (
    <div className="stage" style={zoom === "fit" ? { width: "100%", maxWidth: size[0], marginInline: "auto" } : { width: size[0] * zoom }}>
      {label && <span className="stage-label">{label}</span>}
      <img src={src} alt={label ?? ""} draggable={false} />
      {children}
      {boxes}
    </div>
  );
}

/** Implementación debajo, diseño encima recortado hasta la barra que se arrastra. */
function OverlayStage({ design, impl, size, zoom, boxes, labels = ["Diseño de referencia", "Implementación"] }) {
  const [pos, setPos] = useState(50);
  const ref = useRef(null);
  const move = (e) => {
    const r = ref.current.getBoundingClientRect();
    setPos(Math.min(100, Math.max(0, ((e.clientX - r.left) / r.width) * 100)));
  };
  return (
    <div ref={ref} className="stage overlay" style={zoom === "fit" ? { width: "100%", maxWidth: size[0], marginInline: "auto" } : { width: size[0] * zoom }}>
      <img src={impl} alt="Implementación" draggable={false} />
      <img src={design} alt="Diseño" draggable={false} className="top-img" style={{ clipPath: `inset(0 ${100 - pos}% 0 0)` }} />
      <span className="stage-label">{labels[0]}</span>
      <span className="stage-label right">{labels[1]}</span>
      {boxes}
      <div
        className="handle"
        style={{ left: `${pos}%` }}
        role="slider"
        aria-label="Comparar diseño e implementación"
        aria-valuenow={Math.round(pos)}
        tabIndex={0}
        onClick={(e) => e.stopPropagation()}
        onPointerDown={(e) => { e.currentTarget.setPointerCapture(e.pointerId); move(e); }}
        onPointerMove={(e) => e.currentTarget.hasPointerCapture(e.pointerId) && move(e)}
        onKeyDown={(e) => {
          if (e.key === "ArrowLeft") setPos((p) => Math.max(0, p - 2));
          if (e.key === "ArrowRight") setPos((p) => Math.min(100, p + 2));
        }}
      >
        <span className="knob">‹ ›</span>
      </div>
    </div>
  );
}

export default function Viewer({ mode, result, sourceUrl, findings, selected, hovered, onSelect, focusTick }) {
  const views = VIEWS[mode].filter((v) => !v.needs || result.images[v.needs]);
  const [view, setView] = useState(views[0].key);
  const [zoom, setZoom] = useState("fit");
  const scroller = useRef(null);
  const current = views.find((v) => v.key === view) ?? views[0];
  const size = result.size;

  // "Ver zona": pasar a una vista con recuadros, acercar si está muy chico y centrarlo
  useEffect(() => {
    if (!focusTick || selected == null) return;
    if (!current.boxes) setView(views[0].key);
    const f = findings.find((x) => x.id === selected);
    if (zoom === "fit" && f?.box && scroller.current && f.box[2] < size[0] * 0.08) setZoom(1);
    requestAnimationFrame(() => requestAnimationFrame(() => {
      scroller.current?.querySelector(`[data-box="${selected}"]`)
        ?.scrollIntoView({ behavior: "smooth", block: "center", inline: "center" });
    }));
  }, [focusTick]); // eslint-disable-line react-hooks/exhaustive-deps

  const boxes = current.boxes && (
    <Boxes findings={findings} size={size} selected={selected} hovered={hovered} onSelect={onSelect} />
  );

  let content;
  if (view === "fixed") {
    // izquierda = tu captura (antes); derecha = con las correcciones de DINOv2 + SAM (después)
    content = (
      <OverlayStage design={sourceUrl ?? result.images.annotated} impl={result.images.fixed} size={size} zoom={zoom}
        labels={["Antes", "Después (DINOv2 + SAM)"]} />
    );
  } else if (mode === "analyze") {
    const src = view === "findings" ? (sourceUrl ?? result.images.annotated) : result.images[view];
    content = <Stage src={src} size={size} zoom={zoom} boxes={boxes} />;
  } else if (view === "side") {
    content = (
      <div className="side">
        <Stage src={result.images.design} size={size} zoom={zoom} label="Diseño de referencia" />
        <Stage src={result.images.implementation ?? result.images.annotated} size={size} zoom={zoom}
          label="Implementación" boxes={boxes} />
      </div>
    );
  } else if (view === "overlay") {
    content = (
      <OverlayStage design={result.images.design} impl={result.images.implementation ?? result.images.annotated}
        size={size} zoom={zoom} boxes={boxes} />
    );
  } else {
    content = <Stage src={result.images.semantic} size={size} zoom={zoom} boxes={boxes} label="Rojo = más distinto según DINOv2" />;
  }

  return (
    <section className="card viewer">
      <header className="viewer-head">
        <h2><Eye size={22} /> {mode === "analyze" ? "Vista de la captura" : "Comparación visual"}</h2>
        <div className="viewer-tools">
          <div className="segmented" role="tablist">
            {views.map(({ key, label, icon: Icon }) => (
              <button key={key} type="button" role="tab" aria-selected={view === key}
                className={view === key ? "active" : ""} onClick={() => setView(key)}>
                <Icon size={16} /> {label}
              </button>
            ))}
          </div>
          <div className="zoom">
            <Zoom size={16} />
            <Dropdown
              label="Zoom"
              value={zoom}
              options={ZOOMS.map(([value, label]) => ({ value, label }))}
              onChange={setZoom}
              className="zoom-select"
            />
          </div>
        </div>
      </header>
      <div className="canvas-scroll" ref={scroller} onClick={() => onSelect(null)}>
        {content}
      </div>
      {view === "fixed" && (
        <p className="legend">Arrastra la barra: a la izquierda tu captura, a la derecha la misma captura con {result.fixes_applied?.length ?? 0} correcciones
          aplicadas. SAM recorta cada elemento con su silueta exacta para moverlo o ajustarlo sin parches. No es una imagen inventada: son tus píxeles corregidos.</p>
      )}
      {view === "heatmap" && <p className="legend">Rojo = lo que más llama la atención según DINOv2 · Azul = lo que pasa desapercibido.</p>}
      {view === "components" && (
        <p className="legend">Cada color es un grupo de elementos que DINOv2 reconoce como el mismo componente (A1, A2… son del mismo grupo).</p>
      )}
    </section>
  );
}
