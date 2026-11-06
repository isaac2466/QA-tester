// En desarrollo, Vite reenvía /api → http://127.0.0.1:8000 (ver vite.config.js).
const API = import.meta.env.VITE_API_URL ?? "";

async function post(path, formData) {
  const token = localStorage.getItem("qa_token");
  const headers = token ? { Authorization: `Bearer ${token}` } : {};
  const res = await fetch(`${API}${path}`, { method: "POST", body: formData, headers });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.detail ?? `Error ${res.status} del servidor`);
  return data;
}

async function authPost(path, body) {
  const res = await fetch(`${API}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.detail ?? `Error ${res.status} del servidor`);
  localStorage.setItem("qa_token", data.token);
  return data;
}

export function login(email, password) {
  return authPost("/api/auth/login", { email, password });
}

export function register(email, password) {
  return authPost("/api/auth/register", { email, password });
}

export async function currentUser() {
  const token = localStorage.getItem("qa_token");
  if (!token) return null;
  const res = await fetch(`${API}/api/auth/me`, { headers: { Authorization: `Bearer ${token}` } });
  if (!res.ok) {
    localStorage.removeItem("qa_token");
    return null;
  }
  return res.json();
}

export async function logout() {
  const token = localStorage.getItem("qa_token");
  if (token) await fetch(`${API}/api/auth/logout`, { method: "POST", headers: { Authorization: `Bearer ${token}` } });
  localStorage.removeItem("qa_token");
}

export async function analyzeImage(file) {
  const formData = new FormData();
  formData.append("file", file);
  return post("/api/analyze", formData);
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
