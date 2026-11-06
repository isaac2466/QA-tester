from dataclasses import asdict, dataclass

SEVERITY_WEIGHT = {"alta": 8, "media": 4, "baja": 1}


@dataclass
class Finding:
    category: str  # alineacion | espaciado | color | contraste | atencion | consistencia | diferencia
    severity: str  # alta | media | baja
    title: str
    detail: str
    suggestion: str
    box: list[int] | None = None  # [x, y, w, h] en píxeles de la imagen original
    ref_box: list[int] | None = None
    source: str = "pixeles"  # "dinov2" si lo encontró la IA | "pixeles" si fue una regla clásica
    confidence: float | None = None  # similitud de DINOv2 (0..1) que respalda el hallazgo
    # corrección concreta para la vista corregida, p.ej. {"type": "move", "box": [...], "dx": -3, "dy": 0}
    fix: dict | None = None

    def to_dict(self, idx: int) -> dict:
        d = {"id": idx, **asdict(self)}
        if d["confidence"] is not None:
            d["confidence"] = round(float(d["confidence"]), 3)
        return d


def score(findings: list[Finding]) -> int:
    return max(0, 100 - sum(SEVERITY_WEIGHT[f.severity] for f in findings))
