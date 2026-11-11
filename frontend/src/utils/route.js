import { useEffect, useState } from "react";

/**
 * Navegación por hash (#/proyectos/3) sin librerías: el botón "atrás" del navegador funciona
 * y al recargar se queda en la misma página.
 */
const parse = () => (window.location.hash.replace(/^#\/?/, "") || "revisar").split("/").filter(Boolean);

export function useRoute() {
  const [parts, setParts] = useState(parse);
  useEffect(() => {
    const onChange = () => { setParts(parse()); window.scrollTo(0, 0); };
    window.addEventListener("hashchange", onChange);
    return () => window.removeEventListener("hashchange", onChange);
  }, []);
  return parts;
}

export const go = (path) => { window.location.hash = `/${path}`; };
