// Compara los hallazgos de dos análisis de la misma página: qué se arregló, qué sigue y qué es nuevo.

export const VIEWPORTS = {
  escritorio: { label: "Escritorio", color: "var(--series-1)" },
  movil: { label: "Móvil", color: "var(--series-2)" },
  tablet: { label: "Tablet", color: "var(--series-3)" },
  captura: { label: "Captura", color: "var(--series-4)" },
};
export const viewportOf = (v) => (v && VIEWPORTS[v] ? v : "captura");

const shown = (result) => (result?.findings ?? []).filter((f) => f.verdict !== "descartado");
// "Desalineado 3px (borde superior)" y "Desalineado 2px (borde superior)" son el mismo tipo de problema
const kind = (f) => `${f.category}|${(f.title ?? "").replace(/[\d.,:#]+/g, "").replace(/\s+/g, " ").trim().toLowerCase()}`;

/** Posición del centro relativa al ancho de la imagen: así sirve aunque la página cambie un poco de tamaño. */
function center(f, size) {
  if (!f.box) return null;
  const w = size?.[0] || 1;
  return [(f.box[0] + f.box[2] / 2) / w, (f.box[1] + f.box[3] / 2) / w];
}

const TOLERANCE = 0.06; // 6% del ancho: ~86 px en escritorio, ~23 px en móvil

export function diffVersions(before, after) {
  const a = shown(before);
  const b = shown(after);
  const pairs = [];
  for (const fa of a) {
    for (const fb of b) {
      if (fa.category !== fb.category) continue;
      const ca = center(fa, before.size);
      const cb = center(fb, after.size);
      let d;
      if (ca && cb) d = Math.hypot(ca[0] - cb[0], ca[1] - cb[1]);
      else if (!ca && !cb) d = kind(fa) === kind(fb) ? 0 : Infinity; // hallazgos de toda la página (paleta, atención)
      else d = Infinity;
      // mismo tipo de problema → se acepta un poco más lejos (el elemento se pudo mover al rediseñar)
      const limit = kind(fa) === kind(fb) ? TOLERANCE * 1.5 : TOLERANCE;
      if (d <= limit) pairs.push([d, fa, fb]);
    }
  }
  pairs.sort((x, y) => x[0] - y[0]);
  const usedA = new Set();
  const usedB = new Set();
  const still = [];
  for (const [, fa, fb] of pairs) {
    if (usedA.has(fa.id) || usedB.has(fb.id)) continue;
    usedA.add(fa.id);
    usedB.add(fb.id);
    still.push({ before: fa, after: fb });
  }
  return {
    fixed: a.filter((f) => !usedA.has(f.id)),
    still,
    added: b.filter((f) => !usedB.has(f.id)),
    scoreBefore: before.score,
    scoreAfter: after.score,
  };
}
