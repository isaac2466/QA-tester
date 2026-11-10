import { Component } from "react";
import { save } from "../utils/storage.js";

/** Si algo falla al pintar, muestra el error en vez de dejar la página en blanco. */
export default class ErrorBoundary extends Component {
  state = { error: null };

  static getDerivedStateFromError(error) {
    return { error };
  }

  componentDidCatch(error, info) {
    console.error(error, info.componentStack);
  }

  reset = async () => {
    await save("session", null); // por si la sesión guardada es la que provoca el fallo
    location.reload();
  };

  render() {
    if (!this.state.error) return this.props.children;
    return (
      <div className="page">
        <h1>Algo salió mal</h1>
        <p className="error">{String(this.state.error?.message ?? this.state.error)}</p>
        <div className="actions">
          <button className="btn primary" onClick={() => location.reload()}>Recargar</button>
          <button className="btn ghost" onClick={this.reset}>Borrar datos guardados y recargar</button>
        </div>
      </div>
    );
  }
}
