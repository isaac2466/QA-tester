from dataclasses import asdict, dataclass

SEVERITY_WEIGHT = {"alta": 8, "media": 4, "baja": 1}


@dataclass
class Finding:
    category: str  # alineacion | espaciado | color | contraste | atencion | diferencia | tipografia | ...
    severity: str  # alta | media | baja
    title: str
    detail: str
    suggestion: str
    box: list[int] | None = None  # [x, y, w, h] en píxeles de la imagen original
    ref_box: list[int] | None = None
    source: str = "medicion"  # medicion (reglas OpenCV) | ia (modelo de visión)
    # sin_verificar | confirmado (la IA lo vio y es real) | descartado (la IA dijo que no)
    verdict: str = "sin_verificar"
    verdict_reason: str = ""

    def to_dict(self, idx: int) -> dict:
        return {"id": idx, **asdict(self)}


def score(findings: list[Finding]) -> int:
    """Puntuación 0-100. Los hallazgos sin verificar pesan la mitad y los descartados no cuentan."""
    total = 0.0
    for f in findings:
        if f.verdict == "descartado":
            continue
        w = SEVERITY_WEIGHT.get(f.severity, 1)
        # las mediciones sin verificar y las opiniones de la IA (subjetivas) pesan la mitad
        if (f.source == "medicion" and f.verdict == "sin_verificar") or f.source == "ia":
            w /= 2
        total += w
    return max(0, round(100 - total))
