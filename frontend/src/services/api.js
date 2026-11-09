// En desarrollo, Vite reenvía /api → http://127.0.0.1:8000 (ver vite.config.js).
const API = import.meta.env.VITE_API_URL ?? "";

async function post(path, formData, signal) {
  const res = await fetch(`${API}${path}`, { method: "POST", body: formData, signal });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.detail ?? `Error ${res.status} del servidor`);
  return data;
}

export async function analyzeImage(file, { ia = true } = {}) {
  const formData = new FormData();
  formData.append("file", file);
  return post(`/api/analyze?ia=${ia}`, formData);
}

export async function compareImages(design, implementation) {
  const formData = new FormData();
  formData.append("design", design);
  formData.append("implementation", implementation);
  return post("/api/compare", formData);
}

export async function health() {
  const res = await fetch(`${API}/api/health`);
  if (!res.ok) throw new Error("Backend no disponible");
  return res.json();
}

// Qwen3-VL explica los hallazgos y opina si son falsos positivos. En CPU tarda: se llama por lotes.
export async function explainFindings(file, findings, signal) {
  const formData = new FormData();
  formData.append("file", file);
  formData.append("findings", JSON.stringify(findings.map(
    ({ id, category, severity, title, detail, box, ref_box }) => ({ id, category, severity, title, detail, box, ref_box }),
  )));
  return post("/api/explain", formData, signal);
}
