import { CATEGORY, SEVERITY, scoreOf } from "./findings.js";

const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);

/** Descarga un reporte HTML autocontenido (imagen anotada + hallazgos no ignorados). */
export function exportReport({ mode, result, findings, names }) {
  const date = new Date().toLocaleString("es-MX");
  const title = mode === "analyze" ? "Revisión de captura" : "Comparación diseño vs implementación";
  const items = findings.map((f) => `
    <li class="sev-${f.severity}">
      <div class="top"><b>#${f.id}</b> <span class="pill">${SEVERITY[f.severity]}</span>
        <span class="cat">${esc(CATEGORY[f.category] ?? f.category)}</span></div>
      <h3>${esc(f.title)}</h3>
      <p>${esc(f.detail)}</p>
      <p class="fix">→ ${esc(f.suggestion)}</p>
    </li>`).join("");

  const html = `<!doctype html><html lang="es"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>${title} · QA Tester</title>
<style>
body{font:15px/1.5 system-ui,sans-serif;color:#1f2330;max-width:1100px;margin:32px auto;padding:0 16px;background:#f7f7fb}
h1{margin:0 0 4px}.muted{color:#646a7a}img{max-width:100%;border:1px solid #e4e4ec;border-radius:10px;margin:16px 0;background:#fff}
ol{list-style:none;padding:0;display:grid;gap:10px}li{background:#fff;border:1px solid #e4e4ec;border-left:4px solid;border-radius:10px;padding:12px 16px}
.sev-alta{border-left-color:#dc2626}.sev-media{border-left-color:#d97706}.sev-baja{border-left-color:#2563eb}
h3{margin:4px 0;font-size:16px}p{margin:4px 0}.fix{color:#5b21b6;font-weight:600}.pill{font-size:12px;padding:1px 8px;border-radius:99px;background:#f1effc}
.cat{font-size:12px;color:#646a7a}.score{font-size:28px;font-weight:700}
</style></head><body>
<h1>${title}</h1>
<p class="muted">${esc(names.join(" vs "))} · ${date}</p>
<p><span class="score">${scoreOf(findings)}/100</span> · ${findings.length} hallazgos</p>
<img src="${result.images.annotated}" alt="Captura anotada">
<ol>${items}</ol>
<p class="muted">Generado con QA Tester (DINOv2). Las diferencias señaladas requieren revisión humana.</p>
</body></html>`;

  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([html], { type: "text/html" }));
  a.download = `reporte-qa-${new Date().toISOString().slice(0, 10)}.html`;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}
