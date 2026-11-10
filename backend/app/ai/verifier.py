"""Verificador con DINOv2 (reemplaza al modelo de visión externo; sólo usa modelos de Meta).

Mira el recorte de cada hallazgo y decide QUÉ hay dentro:
    texto | componente | icono   → es interfaz real: el hallazgo se CONFIRMA
    imagen | degradado | sistema → no es interfaz (foto, fondo, "Activar Windows"): se DESCARTA

Cómo aprende: DINOv2 (congelado) convierte cada recorte en un vector; encima entrenamos una capa
lineal con ejemplos etiquetados que generamos solos:
    - texto/componente/icono: del generador de pantallas sintéticas (dataset.py)
    - imagen: recortes de las fotos de fondo de pantalla que trae Windows
    - degradado: degradados y fondos lisos al azar
    - sistema: la marca de agua "Activar Windows" sobre distintos fondos

Entrenar:   cd backend  →  python -m app.ai.verifier --per-class 250
Guarda:     app/ai/weights/verifier.pt  (+ verifier_metrics.json)
"""
from __future__ import annotations

import argparse
import glob
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont

CLASSES = ["texto", "componente", "icono", "imagen", "degradado", "sistema"]
UI = {"texto", "componente", "icono"}
WEIGHTS = Path(__file__).parent / "weights" / "verifier.pt"
WALLPAPERS = "C:/Windows/Web/**/*.jpg"


# ---------- ejemplos de entrenamiento ----------

def _rand_box(W, H, kind):
    if kind == "texto":
        w, h = random.randint(40, 420), random.randint(10, 40)
    elif kind == "icono":
        s = random.randint(14, 48)
        w, h = s, s
    else:
        w, h = random.randint(40, 500), random.randint(24, 260)
    w, h = min(w, W - 2), min(h, H - 2)
    return [random.randint(0, W - w), random.randint(0, H - h), w, h]


def _ui_examples(n_per_class, seed=7):
    from app.ai.dataset import make_page
    pools = {"texto": [], "componente": [], "icono": []}
    i = 0
    while min(len(v) for v in pools.values()) < n_per_class and i < 400:
        img, meta = make_page(seed * 1000 + i)
        rgb = np.asarray(img)
        for e in meta["elements"]:
            c = "texto" if e["type"] == "texto" else "icono" if e["type"] == "icono" else "componente"
            if len(pools[c]) < n_per_class and random.random() < 0.5:
                pools[c].append((rgb, e["box"]))
        i += 1
    return pools


def _photo_examples(n):
    files = glob.glob(WALLPAPERS, recursive=True)
    out = []
    if not files:
        return out
    imgs = [np.asarray(Image.open(f).convert("RGB").resize((1536, 864))) for f in files]
    for _ in range(n):
        rgb = random.choice(imgs)
        out.append((rgb, _rand_box(1536, 864, random.choice(["texto", "componente", "icono"]))))
    return out


def _gradient_examples(n):
    out = []
    for _ in range(n):
        W, H = 600, 400
        a = np.array([random.randint(0, 255) for _ in range(3)], np.float32)
        b = np.array([random.randint(0, 255) for _ in range(3)], np.float32)
        if random.random() < 0.25:  # fondo liso (con algo de ruido de compresión)
            b = a + np.random.uniform(-6, 6, 3)
        t = np.linspace(0, 1, W)[None, :, None] if random.random() < .5 else np.linspace(0, 1, H)[:, None, None]
        img = a * (1 - t) + b * t
        img = np.broadcast_to(img, (H, W, 3)) + np.random.normal(0, 1.5, (H, W, 3))
        rgb = np.clip(img, 0, 255).astype(np.uint8)
        out.append((rgb, _rand_box(W, H, random.choice(["texto", "componente", "icono"]))))
    return out


def _system_examples(n):
    out = []
    for _ in range(n):
        W, H = 520, 120
        bg = tuple(random.randint(180, 255) if random.random() < .7 else random.randint(0, 80) for _ in range(3))
        img = Image.new("RGB", (W, H), bg)
        d = ImageDraw.Draw(img)
        gray = 140 if sum(bg) > 380 else 170
        try:
            f1, f2 = ImageFont.truetype("segoeui.ttf", random.randint(20, 26)), ImageFont.truetype("segoeui.ttf", random.randint(13, 16))
        except OSError:
            f1 = f2 = ImageFont.load_default()
        d.text((12, 14), "Activar Windows", fill=(gray, gray, gray), font=f1)
        d.text((12, 52), "Ve a Configuración para activar Windows.", fill=(gray, gray, gray), font=f2)
        out.append((np.asarray(img), [4, 6, W - 12, H - 30]))
    return out


# ---------- modelo ----------

def embed(dino, items, batch=48):
    """Vectores DINOv2 de cada (imagen, caja), en lotes para ir rápido."""
    vecs = []
    for i in range(0, len(items), batch):
        chunk = items[i:i + batch]
        x = torch.stack([dino._crop_tensor(rgb, box) for rgb, box in chunk]).to(dino.device)
        start = 1 + dino.num_registers
        with dino._lock, torch.inference_mode():
            h = dino.model(pixel_values=x).last_hidden_state
        e = torch.cat([h[:, 0], h[:, start:].mean(1)], dim=-1)
        vecs.append(torch.nn.functional.normalize(e, dim=-1).cpu().numpy())
    return np.concatenate(vecs).astype(np.float32)


class Verifier:
    def __init__(self, dino, path=WEIGHTS):
        ck = torch.load(path, weights_only=False)
        self.dino = dino
        self.head = nn.Linear(ck["dim"], len(CLASSES))
        self.head.load_state_dict(ck["head"])
        self.head.eval()
        self.metrics = ck.get("metrics", {})

    @torch.inference_mode()
    def classify(self, rgb, boxes):
        return self.classify_emb(self.dino.embed_boxes(rgb, boxes)) if boxes else []

    @torch.inference_mode()
    def classify_emb(self, emb):
        p = torch.softmax(self.head(torch.from_numpy(emb)), -1).numpy()
        return [(CLASSES[int(k)], float(p[n, k])) for n, k in enumerate(p.argmax(1))]


def load(dino):
    try:
        return Verifier(dino) if WEIGHTS.exists() else None
    except Exception:
        return None


def train(per_class=250, epochs=300):
    from app.services import dino as dino_service
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    dino_service.load()
    dino = dino_service.get()
    print("Generando ejemplos...", flush=True)
    data = _ui_examples(per_class)
    data["imagen"] = _photo_examples(per_class)
    data["degradado"] = _gradient_examples(per_class)
    data["sistema"] = _system_examples(per_class // 2)
    X, y = [], []
    for c in CLASSES:
        print(f"  {c}: {len(data[c])} ejemplos → DINOv2...", flush=True)
        X.append(embed(dino, data[c]))
        y += [CLASSES.index(c)] * len(data[c])
    X = torch.from_numpy(np.concatenate(X))
    y = torch.tensor(y)
    idx = torch.randperm(len(y))
    n_val = len(y) // 5
    val, tr = idx[:n_val], idx[n_val:]
    head = nn.Linear(X.shape[1], len(CLASSES))
    opt = torch.optim.AdamW(head.parameters(), lr=1e-2, weight_decay=1e-3)
    w = 1.0 / torch.bincount(y[tr], minlength=len(CLASSES)).float().clamp(min=1)
    lossf = nn.CrossEntropyLoss(weight=w / w.sum() * len(CLASSES))
    for ep in range(epochs):
        opt.zero_grad()
        loss = lossf(head(X[tr]), y[tr])
        loss.backward()
        opt.step()
    with torch.no_grad():
        pred = head(X[val]).argmax(1)
    acc = float((pred == y[val]).float().mean())
    per = {c: round(float((pred[y[val] == k] == k).float().mean()), 3) for k, c in enumerate(CLASSES) if (y[val] == k).any()}
    metrics = {"ejemplos": int(len(y)), "validacion": int(n_val), "exactitud": round(acc, 3), "por_clase": per}
    WEIGHTS.parent.mkdir(exist_ok=True)
    torch.save({"head": head.state_dict(), "dim": X.shape[1], "metrics": metrics}, WEIGHTS)
    (WEIGHTS.parent / "verifier_metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Listo:", json.dumps(metrics, ensure_ascii=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=250)
    a = ap.parse_args()
    train(a.per_class)
