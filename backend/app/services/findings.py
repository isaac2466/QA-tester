from dataclasses import asdict, dataclass

SEVERITY_WEIGHT = {"alta": 8, "media": 4, "baja": 1}


@dataclass
class Finding:
    category: str  # alineacion | espaciado | color | contraste | atencion | diferencia
    severity: str  # alta | media | baja
    title: str
    detail: str
    suggestion: str
    box: list[int] | None = None  # [x, y, w, h] en píxeles de la imagen original
    ref_box: list[int] | None = None

    def to_dict(self, idx: int) -> dict:
        return {"id": idx, **asdict(self)}


def score(findings: list[Finding]) -> int:
    return max(0, 100 - sum(SEVERITY_WEIGHT[f.severity] for f in findings))
