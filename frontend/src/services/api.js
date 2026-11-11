// En desarrollo, Vite reenvía /api → http://127.0.0.1:8000 (ver vite.config.js).
const API = import.meta.env.VITE_API_URL ?? "";

function authHeaders() {
  const token = localStorage.getItem("qa-token");
  return token ? { Authorization: `Bearer ${token}` } : {};
}

async function post(path, formData, signal) {
  const res = await fetch(`${API}${path}`, { method: "POST", body: formData, signal, headers: authHeaders() });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.detail ?? `Error ${res.status} del servidor`);
  return data;
}

async function jsonRequest(path, method, body) {
  const res = await fetch(`${API}${path}`, {
    method,
    headers: { "Content-Type": "application/json", ...authHeaders() },
    body: body == null ? undefined : JSON.stringify(body),
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.detail ?? `Error ${res.status}`);
  return data;
}

export const register = (email, password) => jsonRequest("/api/auth/register", "POST", { email, password });
export const login = (email, password) => jsonRequest("/api/auth/login", "POST", { email, password });
export const me = () => jsonRequest("/api/auth/me", "GET");
export const logout = () => jsonRequest("/api/auth/logout", "POST");
export const getHistory = () => jsonRequest("/api/history", "GET");
export const saveHistory = (entry) => jsonRequest("/api/history", "POST", entry);
export const clearHistory = () => jsonRequest("/api/history", "DELETE");

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

/** Enseña a la IA: guarda tu marca ("no_es_error" | "es_error") de un hallazgo. */
export async function sendFeedback(analysisId, findingId, label) {
  const res = await fetch(`${API}/api/feedback`, {
    method: "POST",
    headers: { "Content-Type": "application/json", ...authHeaders() },
    body: JSON.stringify({ analysis_id: analysisId, finding_id: findingId, label }),
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.detail ?? `Error ${res.status}`);
  return data;
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

// Proyectos: agrupan análisis para ver cómo evoluciona el puntaje
export const getProjects = () => jsonRequest("/api/projects", "GET");
export const createProject = (name, url) => jsonRequest("/api/projects", "POST", { name, url });
export const deleteProject = (id) => jsonRequest(`/api/projects/${id}`, "DELETE");
export const getProject = (id) => jsonRequest(`/api/projects/${id}`, "GET");
export const getHistoryEntry = (id) => jsonRequest(`/api/history/${id}`, "GET");

/** Toma capturas de una página en varios tamaños. Devuelve { url, shots: [{ size, label, width, height, file }] }. */
export async function captureUrl(url, sizes) {
  const data = await jsonRequest("/api/capture", "POST", { url, sizes });
  const host = new URL(data.url).hostname.replace(/^www\./, "");
  const shots = await Promise.all(data.shots.map(async ({ png_b64, ...s }) => {
    const blob = await (await fetch(`data:image/png;base64,${png_b64}`)).blob();
    return { ...s, file: new File([blob], `${host}-${s.size}.png`, { type: "image/png" }) };
  }));
  return { url: data.url, shots };
}
