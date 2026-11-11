import { useEffect, useState } from "react";
import { Camera, Close, Compare } from "./icons.jsx";

export default function HistoryDrawer({ open, items, onClose, onRestore, onClear }) {
  useEffect(() => {
    if (!open) return;
    const onKey = (e) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, onClose]);

  // se queda montado un momento al cerrar para que se vea la animación de salida
  const [shown, setShown] = useState(open);
  useEffect(() => {
    if (open) return setShown(true);
    const t = setTimeout(() => setShown(false), 220);
    return () => clearTimeout(t);
  }, [open]);

  if (!shown) return null;
  return (
    <div className={`drawer-backdrop ${open ? "" : "leaving"}`} onClick={onClose}>
      <aside className="drawer" role="dialog" aria-label="Historial" onClick={(e) => e.stopPropagation()}>
        <header>
          <h2>Historial</h2>
          <button type="button" className="icon-btn" onClick={onClose} aria-label="Cerrar"><Close /></button>
        </header>
        {!items.length && <p className="muted">Aquí aparecerán tus últimos análisis.</p>}
        <ul>
          {items.map((h) => (
            <li key={h.id}>
              <button type="button" onClick={() => onRestore(h)}>
                <img src={h.result.images.annotated} alt="" />
                <span>
                  <strong>{h.mode === "analyze" ? <Camera size={14} /> : <Compare size={14} />} {h.names.join(" vs ")}</strong>
                  <small>{new Date(h.date).toLocaleString("es-MX")} · {h.result.findings.length} hallazgos · {h.result.score}/100</small>
                </span>
              </button>
            </li>
          ))}
        </ul>
        {items.length > 0 && <button type="button" className="btn ghost" onClick={onClear}>Borrar historial</button>}
      </aside>
    </div>
  );
}
