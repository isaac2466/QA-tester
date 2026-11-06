import { useState } from "react";
import { login, register } from "../services/api.js";

export default function Login({ onAuthenticated }) {
  const [mode, setMode] = useState("login");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);

  const submit = async (event) => {
    event.preventDefault();
    setLoading(true);
    setError(null);
    try {
      const session = mode === "login"
        ? await login(email, password)
        : await register(email, password);
      onAuthenticated(session.user);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="auth-card">
      <h1>QA Tester visual</h1>
      <p className="muted">{mode === "login" ? "Accede a tu cuenta" : "Crea tu cuenta"}</p>
      <form onSubmit={submit} className="auth-form">
        <label>Correo electrónico
          <input type="email" value={email} onChange={(e) => setEmail(e.target.value)} required />
        </label>
        <label>Contraseña (mínimo 8 caracteres)
          <input type="password" value={password} onChange={(e) => setPassword(e.target.value)} minLength="8" required />
        </label>
        {error && <p className="error">{error}</p>}
        <button className="primary" disabled={loading}>
          {loading ? "Procesando…" : mode === "login" ? "Iniciar sesión" : "Registrarme"}
        </button>
      </form>
      <button className="ghost" onClick={() => { setMode(mode === "login" ? "register" : "login"); setError(null); }}>
        {mode === "login" ? "Crear una cuenta" : "Ya tengo una cuenta"}
      </button>
    </main>
  );
}
