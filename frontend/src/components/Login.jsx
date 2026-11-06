import { useState } from "react";
import { motion } from "framer-motion";
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
    <motion.main
      className="auth-card"
      initial={{ opacity: 0, y: 18, scale: 0.98 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      transition={{ type: "spring", stiffness: 300, damping: 25 }}
    >
      <h1>QA Tester visual</h1>
      <p className="muted">{mode === "login" ? "Accede a tu cuenta" : "Crea tu cuenta"}</p>
      <motion.form onSubmit={submit} className="auth-form" initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.12 }}>
        <label>Correo electrónico
          <input type="email" value={email} onChange={(e) => setEmail(e.target.value)} required />
        </label>
        <label>Contraseña (mínimo 8 caracteres)
          <input type="password" value={password} onChange={(e) => setPassword(e.target.value)} minLength="8" required />
        </label>
        {error && <p className="error">{error}</p>}
        <motion.button className="primary" whileTap={{ scale: 0.95 }} disabled={loading}>
          {loading ? "Procesando…" : mode === "login" ? "Iniciar sesión" : "Registrarme"}
        </motion.button>
      </motion.form>
      <motion.button className="ghost" whileTap={{ scale: 0.95 }} onClick={() => { setMode(mode === "login" ? "register" : "login"); setError(null); }}>
        {mode === "login" ? "Crear una cuenta" : "Ya tengo una cuenta"}
      </motion.button>
    </motion.main>
  );
}
