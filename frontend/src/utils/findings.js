export const CATEGORY = {
  alineacion: "Alineación",
  espaciado: "Espaciado",
  color: "Color",
  contraste: "Contraste",
  atencion: "Atención",
  consistencia: "Consistencia",
  diferencia: "Diferencia",
};

export const SEVERITY = { alta: "Alta", media: "Media", baja: "Baja" };

// Mismo cálculo que el backend (findings.py) para recalcular al ignorar hallazgos
const WEIGHT = { alta: 8, media: 4, baja: 1 };
export const scoreOf = (findings) => Math.max(0, 100 - findings.reduce((s, f) => s + WEIGHT[f.severity], 0));

export const countBy = (findings, key) =>
  findings.reduce((acc, f) => ({ ...acc, [f[key]]: (acc[f[key]] ?? 0) + 1 }), {});

export function formatBytes(n) {
  if (n == null) return "";
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${Math.round(n / 1024)} KB`;
  return `${(n / 1024 / 1024).toFixed(1)} MB`;
}

