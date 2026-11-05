import { useEffect, useRef, useState } from "react";

/** Zona para soltar, elegir o pegar (Ctrl+V) una captura de pantalla. */
export default function Dropzone({ label, file, onFile, pasteTarget = false }) {
  const input = useRef(null);
  const [over, setOver] = useState(false);
  const [preview, setPreview] = useState(null);

  useEffect(() => {
    if (!file) return setPreview(null);
    const url = URL.createObjectURL(file);
    setPreview(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  useEffect(() => {
    if (!pasteTarget) return;
    const onPaste = (e) => {
      const item = [...e.clipboardData.items].find((i) => i.type.startsWith("image/"));
      if (item) onFile(item.getAsFile());
    };
    window.addEventListener("paste", onPaste);
    return () => window.removeEventListener("paste", onPaste);
  }, [pasteTarget, onFile]);

  const pick = (files) => {
    const f = [...files].find((x) => x.type.startsWith("image/"));
    if (f) onFile(f);
  };

  return (
    <div
      className={`dropzone ${over ? "over" : ""} ${file ? "has-file" : ""}`}
      onClick={() => input.current.click()}
      onDragOver={(e) => { e.preventDefault(); setOver(true); }}
      onDragLeave={() => setOver(false)}
      onDrop={(e) => { e.preventDefault(); setOver(false); pick(e.dataTransfer.files); }}
      role="button"
      tabIndex={0}
      onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && input.current.click()}
    >
      <input ref={input} type="file" accept="image/*" hidden onChange={(e) => { pick(e.target.files); e.target.value = ""; }} />
      {file && preview ? (
        <>
          <img src={preview} alt="" />
          <span className="file-name">{file.name}</span>
        </>
      ) : (
        <div className="dz-empty">
          <strong>{label}</strong>
          <span>Arrastra una captura, haz clic{pasteTarget ? " o pega con Ctrl+V" : ""}</span>
          <small>PNG recomendado (JPG añade ruido a los colores)</small>
        </div>
      )}
    </div>
  );
}
