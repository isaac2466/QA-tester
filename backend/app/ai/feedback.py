"""La IA aprende de tus revisiones ("Es error" / "No es error").

Cada hallazgo tiene una "huella" de DINOv2 (vector que describe cómo se ve ese recorte).
Cuando marcas un hallazgo, su huella se guarda en ai_data/feedback.jsonl. En los siguientes
análisis, si un hallazgo nuevo se parece mucho (similitud ≥ 0.92) a uno que marcaste como
"No es error" de la misma categoría, se descarta solo y se explica por qué.

Hay dos formas de aprender, que trabajan juntas:
1. Memoria (vecino más cercano): funciona desde la PRIMERA marca. Descarta lo casi idéntico.
2. Red neuronal (2 capas): se reentrena sola cada 5 marcas con TODAS tus marcas y aprende a
   generalizar ("estos casos suelen no ser error"), aunque no sean idénticos.
   Pesos en ai_data/feedback_net.pt.
"""
from __future__ import annotations

import json
import threading
import uuid
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

CATEGORIES = ["alineacion", "espaciado", "color", "contraste", "atencion", "consistencia", "diferencia"]

FILE = Path(__file__).resolve().parents[2] / "ai_data" / "feedback.jsonl"
NET_FILE = FILE.parent / "feedback_net.pt"
SIMILAR = 0.97  # sólo descarta casi idénticos (con 0.92 confundía botones parecidos de otras páginas)
MIN_MARKS_TO_DECIDE = 30  # con menos ejemplos la red memoriza en vez de aprender: no decide todavía
RETRAIN_EVERY = 5
_net = None
_lock = threading.Lock()
_cache: OrderedDict = OrderedDict()  # análisis recientes: id → {id_hallazgo: (huella, categoría, título)}
_memory: list[dict] | None = None


def _load():
    global _memory
    if _memory is None:
        _memory = []
        if FILE.exists():
            for line in FILE.read_text(encoding="utf-8").splitlines():
                try:
                    r = json.loads(line)
                    r["emb"] = np.asarray(r["emb"], dtype=np.float32)
                    _memory.append(r)
                except (ValueError, KeyError):
                    continue
    return _memory


CACHE_DIR = FILE.parent / "analisis"


def _from_disk(analysis_id: str, finding_id: int):
    p = CACHE_DIR / f"{analysis_id}.npz"
    if not analysis_id.isalnum() or not p.exists():
        return None
    d = np.load(p)
    hit = np.nonzero(d["ids"] == finding_id)[0]
    if not len(hit):
        return None
    category, title = json.loads(str(d["meta"][hit[0]]))
    return d["emb"][hit[0]].astype(np.float32), category, title


def remember_analysis(items: dict) -> str:
    """Guarda las huellas de un análisis para poder recibir sus marcas después."""
    aid = uuid.uuid4().hex[:12]
    with _lock:
        _cache[aid] = items
        try:  # también en disco: si el servidor se reinicia, tus marcas sobre este análisis siguen sirviendo
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(CACHE_DIR / f"{aid}.npz",
                                ids=np.array(list(items), dtype=np.int32),
                                emb=np.stack([v[0] for v in items.values()]) if items else np.zeros((0, 1)),
                                meta=np.array([json.dumps([v[1], v[2]], ensure_ascii=False) for v in items.values()]))
        except OSError:
            pass
        while len(_cache) > 50:
            _cache.popitem(last=False)
    return aid


def save(analysis_id: str, finding_id: int, label: str) -> bool:
    with _lock:
        item = _cache.get(analysis_id, {}).get(finding_id)
        if item is None:
            item = _from_disk(analysis_id, finding_id)
        if item is None:
            return False
        emb, category, title = item
        rec = {"label": label, "category": category, "title": title, "analysis_id": analysis_id,
               "finding_id": finding_id, "emb": [round(float(v), 5) for v in emb]}
        FILE.parent.mkdir(parents=True, exist_ok=True)
        with FILE.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        mem = _load()
        mem.append({**rec, "emb": np.asarray(emb, dtype=np.float32)})
    if len(mem) % RETRAIN_EVERY == 0:
        train_net()
    return True


def match(emb: np.ndarray, category: str):
    """(similitud, etiqueta) de la marca más parecida de la misma categoría, o None."""
    mem = [r for r in _load() if r["category"] == category]
    if not mem:
        return None
    sims = np.stack([r["emb"] for r in mem]) @ emb
    k = int(np.argmax(sims))
    return (float(sims[k]), mem[k]["label"]) if sims[k] >= SIMILAR else None


def stats() -> dict:
    mem = _load()
    return {"marcas": len(mem), "no_es_error": sum(r["label"] == "no_es_error" for r in mem),
            "red_entrenada": NET_FILE.exists(), "red_decide": NET_FILE.exists() and len(mem) >= MIN_MARKS_TO_DECIDE,
            "minimo_para_decidir": MIN_MARKS_TO_DECIDE}


# ---------- red neuronal que aprende de tus marcas ----------

class FeedbackNet(nn.Module):
    """Entrada: huella DINOv2 (768) + categoría (one-hot). Salida: probabilidad de que SÍ sea error."""

    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim + len(CATEGORIES), 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def _x(emb, category):
    cat = np.zeros(len(CATEGORIES), dtype=np.float32)
    if category in CATEGORIES:
        cat[CATEGORIES.index(category)] = 1
    return np.concatenate([np.asarray(emb, dtype=np.float32), cat])


def train_net(epochs=200) -> dict | None:
    """Reentrena la red con todas las marcas. Necesita al menos 3 de cada tipo."""
    global _net
    mem = _load()
    pos = sum(r["label"] == "es_error" for r in mem)
    if pos < 3 or len(mem) - pos < 3:
        return None
    X = torch.from_numpy(np.stack([_x(r["emb"], r["category"]) for r in mem]))
    y = torch.tensor([1.0 if r["label"] == "es_error" else 0.0 for r in mem])
    torch.manual_seed(0)
    net = FeedbackNet(X.shape[1] - len(CATEGORIES))
    opt = torch.optim.AdamW(net.parameters(), lr=5e-3, weight_decay=1e-3)
    lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor((len(y) - pos) / pos))
    net.train()
    for _ in range(epochs):
        opt.zero_grad()
        loss = lossf(net(X), y)
        loss.backward()
        opt.step()
    net.eval()
    with torch.no_grad():
        acc = float(((torch.sigmoid(net(X)) > 0.5).float() == y).float().mean())
    torch.save({"state": net.state_dict(), "dim": X.shape[1] - len(CATEGORIES), "n": len(mem)}, NET_FILE)
    _net = net
    return {"marcas": len(mem), "exactitud_entrenamiento": round(acc, 3)}


def net_predict(emb, category) -> float | None:
    """Probabilidad (0-1) de que SÍ sea error según la red, o None si aún no hay red entrenada."""
    global _net
    if _net is None and NET_FILE.exists():
        ck = torch.load(NET_FILE, weights_only=False)
        _net = FeedbackNet(ck["dim"])
        _net.load_state_dict(ck["state"])
        _net.eval()
    if _net is None or len(_load()) < MIN_MARKS_TO_DECIDE:
        return None
    with torch.no_grad():
        return float(torch.sigmoid(_net(torch.from_numpy(_x(emb, category))[None]))[0])
