import { useEffect, useRef, useState } from "react";
import { formatBytes } from "../utils/findings.js";
import { Broom, Upload } from "./icons.jsx";

/** Hook: URL temporal + dimensiones de una imagen (File/Blob). */
export function useImageInfo(file) {
  const [info, setInfo] = useState(null);
  useEffect(() => {
    if (!file) return setInfo(null);
    const url = URL.createObjectURL(file);
    const img = new Image();
    img.onload = () => setInfo({ url, w: img.naturalWidth, h: img.naturalHeight });
    img.onerror = () => setInfo({ url });
    img.src = url;
    return () => URL.revokeObjectURL(url);
  }, [file]);
  return info;
}

/** Un archivo de entrada: vacío = zona para soltar; lleno = miniatura + datos y acciones. */
export default function FileSlot({ label, file, onFile, onClear, hint }) {
  const input = useRef(null);
  const [over, setOver] = useState(false);
  const info = useImageInfo(file);

  const pick = (files) => {
    const f = [...files].find((x) => x.type.startsWith("image/"));
    if (f) onFile(f);
  };
  const dnd = {
    onDragOver: (e) => { e.preventDefault(); setOver(true); },
    onDragLeave: () => setOver(false),
    onDrop: (e) => { e.preventDefault(); setOver(false); pick(e.dataTransfer.files); },
  };
  const open = () => input.current.click();
  const fileInput = (
    <input ref={input} type="file" accept="image/*" hidden onChange={(e) => { pick(e.target.files); e.target.value = ""; }} />
  );

  if (!file) {
    return (
      <button type="button" className={`slot empty ${over ? "over" : ""}`} onClick={open} {...dnd}>
        {fileInput}
        <span className="slot-icon"><Upload size={22} /></span>
        <span>
          <strong>{label}</strong>
          <small>{hint ?? "Arrastra una imagen, haz clic o pega con Ctrl+V · PNG recomendado"}</small>
        </span>
      </button>
    );
  }

  return (
    <div className={`slot ${over ? "over" : ""}`} {...dnd}>
      {fileInput}
      <div className="thumb">{info?.url && <img src={info.url} alt="" />}</div>
      <div className="slot-meta">
        <strong>{label}</strong>
        <span className="file-name" title={file.name}>{file.name || "imagen pegada"}</span>
        <small>{info?.w ? `${info.w} × ${info.h}` : "…"} · {formatBytes(file.size)}</small>
      </div>
      <div className="slot-actions">
        <button type="button" className="btn outline" onClick={open}><Upload size={16} /> Reemplazar</button>
        {onClear && <button type="button" className="btn outline" onClick={onClear}><Broom size={16} /> Limpiar</button>}
      </div>
    </div>
  );
}
